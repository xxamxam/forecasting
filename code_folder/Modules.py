"""
модули общие для остальных модейлей

The given code defines several neural network classes.
Here is a brief description of each class and its purpose:

1. RNN: A recurrent neural network class that takes an input of shape (1, batch_size, input_dim),
 and returns the output of a GRU layer applied to the input.

2. NeuralODE: A class that defines a neural ODE, which applies the ODE function 
defined in ODE_func to a given input x and returns the output of the ODE integration.

"""
import torch
import torch.nn as nn
import torchdiffeq
import torch.nn.functional as F

class Hooks:
    def add_hooks(self):
        for p in self.ode_func.parameters():
            p.register_hook(lambda grad: grad / torch.norm(grad))
            
class NeuralODE(nn.Module):
    def __init__(self, ode_func, tolerance = 1e-5):
        super(NeuralODE, self).__init__()
        self.func = ode_func
        self.tolerance = tolerance
    
    def forward(self, x, t):
        # x.shape = (batch_size, ...)
        t = t.view(t.shape[0])
        batch_size = x.shape[0]
        device = next(self.parameters()).device

        t1 = t[0] if t.shape[0] > 1 else t.item()
        out = torchdiffeq.odeint_adjoint(self.func, x, torch.tensor([0, t1]).to(device), method = 'euler')
        out = out[-1]
        return out
        

class ode_func_interface(nn.Module):
    def __init__(self, ): 
        super(ode_func_interface, self).__init__()
        self.device = None  
    def compose_x_with_h(self, x, h): raise NotImplementedError()
    def decompose_x_with_h(self, inp): raise NotImplementedError()
    def inp2hid(self, t, x): raise NotImplementedError()
    def forward(self, t , x): raise NotImplementedError()

class LSTM_ODE_func(ode_func_interface):
    def __init__(self, input_dim, hidden_dim, proj_dim = 0, num_layers = 1):
        super(LSTM_ODE_func, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = proj_dim if proj_dim > 0 else hidden_dim - 1
        self.num_layers = num_layers
        self.out2inp = nn.Linear(self.out_dim, input_dim)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, proj_size = self.out_dim)
        self.device = torch.device("cpu")
        self._inp2hid_ = nn.Linear(input_dim, hidden_dim * num_layers)
        self.inp2out = nn.Linear(input_dim, self.out_dim * num_layers)
    
    def compose_x_with_h(self, x, h,c):
        x = torch.cat([x, *[h[i] for i in range(self.num_layers)],*[c[i] for i in range(self.num_layers)]], dim = -1)
        return x.to(self.device)

    def decompose_x_with_h(self, inp):
        inp = inp.unsqueeze(0)
        x = inp[..., : self.input_dim].to(self.device)
        h_s = torch.cat( 
                [ inp[..., self.input_dim + self.out_dim * i 
                        :self.input_dim + self.out_dim * (i + 1) ] 
                        for i in range(self.num_layers)], dim = 0 )
        start_pos = self.input_dim + self.out_dim * self.num_layers
        c_s = torch.cat(
            [ inp[..., start_pos + self.hidden_dim* i:
                    start_pos + self.hidden_dim*(i + 1)]
                    for i in range(self.num_layers)], dim = 0 )
        return x, [h_s.to(self.device), c_s.to(self.device)]
    
    def inp2hid(self, t, x):
        c = self._inp2hid_(x[..., 0])
        c = torch.cat([c[..., self.hidden_dim * i : self.hidden_dim * (i + 1)].unsqueeze(0) 
            for i in range(self.num_layers)], dim = 0)
        h = self.inp2out(x[..., 0])
        h = torch.cat([h[..., self.out_dim * i : self.out_dim * (i + 1)].unsqueeze(0) 
            for i in range(self.num_layers)], dim = 0)
        return [h.to(self.device), c.to(self.device)]

    def forward(self, t, x):
        device = next(self.parameters()).device
        input_, [h_s, c_s] = self.decompose_x_with_h(x)
        output_, (h_s, c_s) = self.rnn(input_ , (h_s, c_s))
        output_ = output_.view(output_.shape[1:])
        output_ = self.out2inp(output_)
        return self.compose_x_with_h(output_, h_s, c_s)




class ODE_on_RNN(nn.Module, Hooks):
    def __init__(self, ode_func, tolerance = 1e-5, internal_loss_dim = None):
        super(ODE_on_RNN, self).__init__()
        self.ode_func = ode_func
        self.ode = NeuralODE(self.ode_func, tolerance)
        self.loss_function = nn.MSELoss()
        self.internal_loss_dim = internal_loss_dim
    

    def forward(self, x, t, return_hidden = False):
        internal_loss = 0
        device = next(self.parameters()).device
        self.ode_func.device = device
        h = self.ode_func.inp2hid(t ,x)
        h_s = []
        for i in range(x.shape[2]):
            if return_hidden:
                h_s.append(h)
            x_i = x[..., i]
            inp = self.ode_func.compose_x_with_h(x_i, *h)
            out = self.ode(inp, t[..., i])
            out, h = self.ode_func.decompose_x_with_h(out)
            if self.internal_loss_dim is not None and i != x.shape[2] -1 :
                internal_loss += self.loss_function(out[..., : self.internal_loss_dim], 
                                                    x[..., i + 1][..., :self.internal_loss_dim])
        return out, h_s, internal_loss


def ode_on_lstm( input_dim, hidden_dim, proj_dim = 0, num_layers = 1, tolerance = 1e-5, internal_loss_dim = None):
    ode_func = LSTM_ODE_func( input_dim, hidden_dim, proj_dim, num_layers)
    net = ODE_on_RNN(ode_func, tolerance, internal_loss_dim)
    return net

