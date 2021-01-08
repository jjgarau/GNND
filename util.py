import torch
import numpy as np
import matplotlib.pyplot as plt

def mape_loss(output, label):
    return torch.mean(torch.abs(torch.div((output - label), label)))

def mse_loss(output, label):
    return torch.mean(torch.square(output - label))

def rmse_loss(output, label):
    return torch.sqrt(torch.mean(torch.square(output - label)))

def mae_loss(output, label):
    return torch.mean(torch.abs(output - label))

def mase_loss(output, label):
    return torch.mean(torch.abs(output - label)) / torch.mean(label)

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
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    return np.mean(np.abs(predictions - labels)) / np.mean(labels) #np.mean(abs((labels - predictions) / labels))  #reporting loss function, different from training


def show_predictions(predictions, labels):
    # Plot predictions and labels over time
    x = np.arange(0, len(predictions))
    plt.title('Predictions')
    plt.xlabel("Time")
    plt.ylabel("Prediction")
    plt.plot(x, [torch.mean(p) for p in predictions], label="Predictions")
    plt.plot(x, [torch.mean(l) for l in labels], label="Labels")
    plt.plot(x, [1000*mase_loss(predictions[i], labels[i]) for i in range(len(predictions))], label="Loss")
    plt.legend()
    plt.show()

def show_loss_by_country(predictions, labels, nations):
    # Plot loss by country over time
    x = np.arange(0, len(predictions))
    plt.title('Loss by Country')
    plt.xlabel("Time")
    plt.ylabel("MAE Loss")
    for i in range(len(nations)):
        loss = [mae_loss(predictions[time][i], labels[time][i]) for time in range(len(predictions))]
        plt.plot(x, loss, label=nations[i])
        print(nations[i] + ": " + str(int(sum(loss)/len(loss))))
    plt.show()

def show_labels_by_country(labels, nations):
    # Plot labels by country over time
    x = np.arange(0, len(labels))
    plt.title('Labels by Country')
    plt.xlabel("Time")
    plt.ylabel("New COVID Cases")
    for i in range(len(nations)):
        label = [torch.mean(l[i]) for l in labels]
        plt.plot(x, label, label=nations[i])
        print(nations[i] + ": " + str(int(sum(label)/len(label))))
    plt.show()