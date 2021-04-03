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

def mase_loss(output, label, mean=None):
    label = label[:, 0]
    label_mean = torch.mean(label)
    if not mean is None:
        return torch.mean(torch.abs(output - label) / mean)
    elif label_mean == 0:
        return torch.mean(torch.abs(output - label))
    else:
        return torch.mean(torch.abs(output - label)) / label_mean


def mase1_loss(output, label, mean=None):
    # Extreme 1: all countries equal
    # L_i = (x_i - y_i)^2 / y_i
    # L = (L_1 + L_2 + … + L_N) / N
    label = label[:, 0]
    label_mean = torch.mean(label)
    if not mean is None:
        return torch.mean(torch.abs(output - label) / mean)
    elif label_mean == 0:
        return torch.mean(torch.abs(output - label))
    else:
        return torch.mean(torch.abs(output - label)) / label_mean

def mase2_loss(output, label, mean=None):
    # Extreme 2: all people equal
    # X = (x_1 + x_2 + … + x_N)
    # Y = (y_1 + y_2 + … + y_N)
    # L = (X - Y)^2 / Y
    label = label[:, 0]
    X = torch.sum(output)
    Y = torch.sum(label)
    if not mean is None:
        return torch.abs(X - Y) / torch.sum(mean)
    elif Y == 0:
        return torch.abs(X - Y)
    else:
        return torch.abs(X - Y) / Y

def mase3_loss(output, label, populations, mean=None, k=500000):
    # Middle point: consider a population threshold k
    # x_k = sum(x_i) such that country i has less than k population
    # y_k = sum(y_i) such that country i has less than k population
    # L_i = (x_i - y_i)^2 / y_i   for countries i with more than k population
    # L_k = (x_k - y_k)^2 / y_k
    # L = L_k + sum(L_i)
    label = label[:, 0]

    if mean is None:
        mean = torch.mean(label)
    if sum(mean) == 0:
        mean = 1

    large_outputs = []
    large_labels = []
    large_means = []

    small_outputs = []
    small_labels = []
    small_means = []
    for i in range(len(populations)):
        if populations[i] < k:
            small_outputs.append(output[i])
            small_labels.append(label[i])
            small_means.append(mean[i])
        else:
            large_outputs.append(output[i])
            large_labels.append(label[i])
            large_means.append(mean[i])

    x_k = sum(small_outputs)
    y_k = sum(small_labels)
    L_i = torch.abs(torch.FloatTensor(large_outputs) - torch.FloatTensor(large_labels)) / torch.FloatTensor(large_means)
    L_k = abs(x_k - y_k) / sum(small_means)
    return L_k + torch.sum(L_i)


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
    x = np.arange(0, len(predictions['train']))
    plt.title('COVID Europe Dataset')
    plt.xlabel("Time (days)")
    plt.ylabel("New Cases")
    plt.plot(x, [torch.mean(p) for p in predictions['train']], label="Predictions")
    plt.plot(x, [torch.mean(l) for l in labels['train']], label="Labels")
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