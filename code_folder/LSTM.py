import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module): 
    def __init__(self, input_dim, hidden_dim, proj_dim = 0, num_layers = 1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = proj_dim if proj_dim > 0 else hidden_dim - 1
        self.num_layers = num_layers
        self.out2inp = nn.Linear(self.out_dim, input_dim)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, proj_size = self.out_dim)
    def forward(self, x, t = None):
        # device = next(self.parameters()).device
        h_s, c_s = self.get_hidden(x.shape[0])
        x = torch.permute(x, (2, 0 ,1))
        output_, (h_s, c_s) = self.rnn(x , (h_s, c_s))
        output_ = output_[-1][None, :]
        output_ = self.out2inp(output_)
        return output_, 0, 0
    
    def get_hidden(self, batch_size):
        h0 = torch.zeros(1 * self.num_layers, batch_size, self.out_dim)
        c0 = torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim)
        return h0, c0

