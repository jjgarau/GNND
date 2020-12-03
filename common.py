import torch
import numpy as np

def all_optimizers():
    optim = torch.optim
    optimizers = [
        optim.Adam,
        optim.Adadelta,
        optim.Adagrad,
        optim.AdamW,
        optim.Adamax,
        optim.ASGD,
        optim.RMSprop,
        optim.Rprop,
        optim.SGD
    ]
    return optimizers

def mape_loss(output, label):
    loss = torch.mean(torch.abs(torch.div((output - label), label)))
    return loss

def train_gnn(model, loader, optimizer, loss_func, device):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        output = torch.reshape(output, label.shape)
        loss = loss_func(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all

def evaluate_gnn(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()
            label = data.y.detach().cpu().numpy()
            pred = pred.reshape(label.shape)
            predictions.append(pred)
            labels.append(label)
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    return np.mean(abs((labels - predictions) / labels))