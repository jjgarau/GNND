from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.data.splitter import discrete_train_test_split
from util import *
from rnn import LSTM, GRU, RNN
from tqdm import tqdm

#Testing for the builtin Chickenpox Dataset from PyTorch Geometric Temporal


#Get dataset
loader = ChickenpoxDatasetLoader()
dataset = loader.get_dataset()
train_dataset, test_dataset = discrete_train_test_split(dataset, train_ratio=0.2)


lookback = 4
output = 1


#List of models to test
models = [RNN(lookback)]

for i in range(len(models)):

    model = models[i]
    print(model)

    #Select optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #Set model in training mode
    model.train()

    #Repeat for 200 epochs
    for epoch in tqdm(range(200)):
        cost = 0
        for time, snapshot in enumerate(train_dataset):
            #Calculate cost for each snapshot in dataset
            y_hat = model(snapshot)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)

        # Calculate total cost and gradients, then update parameters
        cost = cost / (time+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()


        #In each epoch, evaluate on testing portion of the dataset
        model.eval()
        cost = 0
        for time, snapshot in enumerate(test_dataset):
            y_hat = model(snapshot)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost = cost.item()
        print("MSE: {:.4f}".format(cost))