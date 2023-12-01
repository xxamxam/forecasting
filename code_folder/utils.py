import torch
from sklearn.metrics import r2_score
import numpy as np

def energy_dataloader(dataloader, epochs = 1, scale_factor = 1.):
    def get_dataloader():
        for x, y in dataloader:
            b_s = x.shape[0]
            time_steps = x.shape[1]
            yield x.unsqueeze(1), y.unsqueeze(1), torch.ones((b_s, 1, time_steps)) / scale_factor
    for _ in range(epochs):
        yield get_dataloader()
        
def energy_dataloader_with_time_axis(dataloader, epochs = 1, scale_factor = 1.):
    for _ in range(epochs):
        def get_dataloader():
            for x, y in dataloader:
                b_s = x.shape[0]
                time_steps = x.shape[-1]
                time_delta = torch.ones((b_s, 1, time_steps)) / scale_factor
                yield x, y, time_delta 
        yield get_dataloader()


def compute_metrics(net, test_data, energy):
    predictions = []
    y_true = []
    net = net.to("cpu")
    for x, y , t in test_data:
        y_pred, h, _ = net(x, t) #,  return_hidden =False)           # сетка приняла только x , t
        y_pred = y_pred.detach().squeeze() 
        predictions.append(y_pred) # добавили в predictions

        y = y.detach().squeeze()
        y_true.append(y)
        
    if len(predictions[0].shape) == 1:
        predictions = torch.hstack(predictions).numpy()
        y_true = torch.hstack(y_true).numpy()

        predictions = energy.target_scaler.inverse_transform(predictions[:,None]).squeeze()
        y_true = energy.target_scaler.inverse_transform(y_true[:, None]).squeeze()
    else:
        predictions = torch.vstack(predictions).numpy()
        y_true = torch.vstack(y_true).numpy()
        print(predictions.shape)
        predictions = energy.target_scaler.inverse_transform(predictions).squeeze()
        y_true = energy.target_scaler.inverse_transform(y_true).squeeze()
        predictions = predictions[:, 0]
        y_true = y_true[:, 0]

    deltas = predictions - y_true

    rmse = np.sqrt(np.mean(np.square(deltas)))
    mae = np.mean(np.abs(deltas))
    r2 = r2_score(y_true, predictions)

    rez = {"RMSE": rmse, "MAE": mae, "R2": r2, \
           "#parameters": sum(p.numel() for p in net.parameters() if p.requires_grad)}
    
    return rez, predictions, y_true