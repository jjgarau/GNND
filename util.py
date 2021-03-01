import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric_temporal.nn import *

def mape_loss(output, label):
    return torch.mean(torch.abs(torch.div((output - label), label)))

def mse_loss(output, label):
    return torch.mean(torch.square(output - label))

def rmse_loss(output, label):
    return torch.sqrt(torch.mean(torch.square(output - label)))

def mae_loss(output, label):
    return torch.mean(torch.abs(output - label))

def mase_loss(output, label):
    label_mean = torch.mean(label)
    if label_mean == 0:
        return torch.mean(torch.abs(output - label))
    else:
        return torch.mean(torch.abs(output - label)) / label_mean

def inv_reg_mase_loss(output, label):
    return mase_loss(output, label) + torch.mean(torch.div(1, output))

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
    p = np.vstack(predictions)
    l = np.vstack(labels)
    return np.mean(np.abs(p - l)) / np.mean(l) #np.mean(abs((labels - predictions) / labels))  #reporting loss function, different from training


def evaluate_gnn_recurrent(model, dataset, lookback_pattern, loss_func):
    predictions, labels, losses = [], [], []

    def forward(snapshot, h, c, detach=False):
        if type(model) is GConvLSTM or type(model) is GConvGRU:
            h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr[:, 0], h, c)
            if detach:
                h = h.detach()
                c = c.detach()
            return h, h, c
        else:
            return model(snapshot, h, c)

    model.eval()
    with torch.no_grad():
        cost = 0
        for time, snapshot in enumerate(dataset):
            h, c = None, None
            for sub_time in range(len(lookback_pattern)):
                sub_snapshot = Data(x=snapshot.x[:, sub_time:sub_time + 1], edge_index=snapshot.edge_index,
                                    edge_attr=snapshot.edge_attr)
                y_hat, h, c = forward(sub_snapshot, h, c, detach=True)
            predictions.append(y_hat)
            labels.append(snapshot.y)
            cost += loss_func(y_hat, snapshot.y)
        cost /= time + 1
        cost = cost.item()
        losses.append(cost)
    return predictions, labels, losses

def show_predictions(predictions, labels):
    # Plot predictions and labels over time
    x = np.arange(0, len(predictions))
    plt.title('COVID Europe Dataset')
    plt.xlabel("Time (days)")
    plt.ylabel("New Cases")
    plt.plot(x, [torch.mean(p) for p in predictions], label="Predictions")
    plt.plot(x, [torch.mean(l) for l in labels], label="Labels")
    # plt.plot(x, [1000*mase_loss(predictions[i], labels[i]) for i in range(len(predictions))], label="Loss")
    plt.legend()
    plt.show()

def show_loss_by_country(predictions, labels, nations, plot=True):
    # Plot loss by country over time
    x = np.arange(0, len(predictions))
    plt.title('Loss by Country')
    plt.xlabel("Time (days)")
    plt.ylabel("MASE Loss")
    losses = {}

    for i in range(len(nations)):
        # Compute MAE loss for each example
        loss = [float(mae_loss(predictions[time][i], labels[time][i])) for time in range(len(predictions))]
        losses[nations[i]] = loss
        if plot:
            plt.plot(x, loss, label=nations[i])
    if plot:
        plt.show()
    return losses

def show_labels_by_country(labels, nations):
    # Plot labels by country over time
    x = np.arange(0, len(labels))
    plt.title('New Cases by Country')
    plt.xlabel("Time (days)")
    plt.ylabel("New COVID Cases")
    for i in range(5):
        label = [torch.mean(l[i]) for l in labels]
        plt.plot(x, label, label=nations[i])
        print(nations[i] + ": " + str(int(sum(label)/len(label))))
    plt.show()