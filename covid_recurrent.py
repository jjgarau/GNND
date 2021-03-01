import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import pandas as pd
import json
import os.path
import datetime
from tqdm import tqdm
from util import *
import graph_nets
from weight_sage import WeightedSAGEConv
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import bisect
import numpy as np
from rnn import RNN, LSTM, GRU, VanillaRNN, PGT_DCRNN, PGT_GConvLSTM, PGT_GConvGRU, SimpleRNN
from torch_geometric.nn import LEConv, SAGEConv
from torch_geometric_temporal.nn import GConvGRU, GConvLSTM, GCLSTM, LRGCN
from torch_geometric.nn import ASAPooling, TopKPooling, EdgePooling, SAGPooling
from torchvision import transforms

#Test Recurrent Neural Networks on COVID Dataset
#There is a separate file because the training pattern is slightly different,
#and I am almost exclusively using RNNs at this point.

lookback = 5
lookback_pattern = [11, 10, 9, 8, 7]
edge_count = 0

class COVIDDatasetSpaced(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(COVIDDatasetSpaced , self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['covid_dataset_spaced.dataset']

    def download(self):
        pass

    def process(self):

        #Determine edge_index: Closest 3 countries are connected

        DISTANCE_THRESHOLD = 250 #km
        EDGES_PER_NODE = 3

        data_list = []
        n = len(nations)
        source_nodes = []
        target_nodes = []
        edge_attrs = []
        for i in range(n):
            c1 = country_centroids[nations[i]]
            distances = []
            countries = []
            for j in range(n):
                c2 = country_centroids[nations[j]]
                dist = geodesic(c1, c2)
                index = bisect.bisect(distances, dist)
                if index < EDGES_PER_NODE:
                    distances.insert(index, dist.km)
                    countries.insert(index, j)
                    # if distances[len(distances) - 1] > DISTANCE_THRESHOLD:
                    #     distances.pop()
                    #     countries.pop() #Uncomment to create edge between all countries within a distance threshold, or at least a minimum of EDGES_PER_NODE nearest countries
            source_nodes += [i]*EDGES_PER_NODE
            target_nodes += countries[:EDGES_PER_NODE]
            edge_attrs += distances[:EDGES_PER_NODE]

        torch_def = torch.cuda if torch.cuda.is_available() else torch

        node_mask = torch.ones(len(df)).bool()
        edge_mask = torch.ones(len(source_nodes)).bool()

        edge_count = len(source_nodes)

        for i in tqdm(range(len(df) - lookback_pattern[0])):
            # !Masking currently not being used!
            edge_mask = torch.logical_not(torch.logical_xor(edge_mask, torch.bernoulli(0.95 * torch.ones(len(source_nodes))).bool()))
            node_mask = torch.logical_not(torch.logical_xor(node_mask, torch.bernoulli(0.95 * torch.ones(len(df))).bool()))
            inv_node_mask = ~node_mask
            nodes_to_drop = set(torch.arange(len(df))[inv_node_mask].tolist())

            temp_edge_mask = edge_mask.clone()
            for j in range(len(source_nodes)):
                if source_nodes[j] in nodes_to_drop or target_nodes[j] in nodes_to_drop:
                    temp_edge_mask[j] = False

            # Node Features
            values_x = []
            for n in lookback_pattern:
                m = i + lookback_pattern[0] - n
                values_x.append(df.iloc[m, 1:])
            values_x = pd.DataFrame(values_x).to_numpy().T
            x = torch_def.FloatTensor(values_x)
            # x = x[node_mask, :]

            # Labels
            values_y = df.iloc[(i + lookback_pattern[0]):(i + lookback_pattern[0] + 1), 1:].to_numpy().T
            y = torch_def.FloatTensor(values_y)
            # y = y[node_mask, :]

            # Edge Index
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])
            # edge_index = edge_index[:, temp_edge_mask]
            # Edge Weights
            edge_attr = torch_def.FloatTensor([[weight] for weight in edge_attrs])
            edge_attr = edge_attr / torch.mean(edge_attr)
            # edge_attr = edge_attr[temp_edge_mask, :]


            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def gnn_predictor_single():
    # Load, shuffle, and split dataset
    dataset = COVIDDatasetSpaced(root='data/covid-data/')
    dataset = dataset.shuffle()

    sample = len(dataset)
    sample *= 1.0 # Optionally, choose a frame of the dataset to work with
    train_dataset = dataset[:int(0.8 * sample)]
    val_dataset = dataset[int(0.8 * sample):int(0.9 * sample)]
    test_dataset = dataset[int(0.9 * sample):int(sample)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set loss function
    loss_func = mae_loss

    # CONSTRUCT MODELS
    WSC = WeightedSAGEConv
    USC = lambda in_channels, out_channels, bias=True: WeightedSAGEConv(in_channels, out_channels, weighted=False)
    linear_module = lambda in_channels, out_channels, bias: graph_nets.GraphLinear(in_channels, out_channels, bias=bias)
    DeepUSC = lambda lookback, dim: graph_nets.GNNModule(USC, 3, lookback, dim=dim, res_factors=[1, 0, 1], dropouts=[1])
    DeepWSC = lambda lookback, dim: graph_nets.GNNModule(WSC, 3, lookback, dim=dim, res_factors=None, dropouts=[])

    models = [
        graph_nets.LagPredictor(),
        RNN(module=WSC, gnn=WSC, rnn=LSTM, dim=16, gnn_2=WSC, rnn_depth=1, name="Our Model", edge_count=edge_count),
        # graph_nets.RecurrentGraphNet(GConvLSTM),
        # graph_nets.RecurrentGraphNet(GConvGRU),
        # graph_nets.RecurrentGraphNet(DCRNN),
        # graph_nets.RecurrentGraphNet(GCLSTM),
    ]

    # Setup for results
    experiment_descr = "Trying to fix the lag data by using MAE loss rather than MASE Loss for training."
    print(experiment_descr)
    results = {
        "Description": experiment_descr,
        "Models": {},
    }
    train_losseses, train_eval_losseses, val_losseses, test_losseses = [], [], [], []

    # For each model...
    for i in range(len(models)):
        model = models[i]
        print(model)

        # Setup for results
        train_losses, train_eval_losses, val_losses, test_losses = [], [], [], []
        results["Models"][model.name] = {
            "Architecture": str(model),
            "Loss by Epoch": [],
            "Loss by Country": {}
        }

        def forward(snapshot, h, c):
            """
            Deals with slight differences in forward calls between models
            """
            if model.name == 'Lag':
                return model(snapshot, h, c)

            if h is None:
                out = model(snapshot, h, c)
            elif c is None:
                out = model(snapshot, h.detach(), c)
            else:
                out = model(snapshot, h.detach(), c.detach())
            if len(out) == 3:
                x, h, c = out
                h = h.detach()
                return x, h, c
            else:
                x, h = out
                h = h.detach()
                return x, h

        # Lag model does not optimize
        if model.name != "Lag":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


        # For each epoch...
        num_epochs = 10
        for epoch in range(num_epochs):

            # Setup for results
            predictions, labels = [], []

            # TRAIN MODEL
            model.train()
            train_cost = 0
            # For each training example...
            for time, snapshot in enumerate(train_dataset):
                # Reset cell and hidden states
                h, c = None, None

                # For each snapshot in the example lookback
                for sub_time in range(len(lookback_pattern)):
                    # Get output and new cell/hidden states for prediction on example
                    sub_snapshot = Data(x=snapshot.x[:, sub_time:sub_time+1], edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr)
                    y_hat, h, c = forward(sub_snapshot, h, c)

                # Calculate the loss from the final prediction of the sequence
                train_cost += loss_func(y_hat, snapshot.y)

            #Take average of loss from all training examples
            train_cost /= time + 1
            train_losses.append(train_cost)  # Keep list of training loss from each epoch

            # Backpropagate, unless lag model
            if model.name != "Lag":
                train_cost.backward()
                optimizer.step()
                optimizer.zero_grad()


            # Evaluate perforamance on train/val/test datasets
            with torch.no_grad():
                model.eval()

                # EVALUATE MODEL - TRAINING
                train_eval_cost = 0
                for time, snapshot in enumerate(train_dataset):
                    h, c = None, None
                    for sub_time in range(len(lookback_pattern)):
                        sub_snapshot = Data(x=snapshot.x[:, sub_time:sub_time + 1], edge_index=snapshot.edge_index,
                                            edge_attr=snapshot.edge_attr)
                        y_hat, h, c = forward(sub_snapshot, h, c)

                    # Keep a list of the predictions and labels across the entire epoch
                    predictions.append(y_hat)
                    labels.append(snapshot.y)
                    train_eval_cost += loss_func(y_hat, snapshot.y)
                train_eval_cost /= time + 1
                train_eval_cost = train_eval_cost.item()
                train_eval_losses.append(train_eval_cost)

                # EVALUATE MODEL - VALIDATION
                val_cost = 0
                for time, snapshot in enumerate(val_dataset):
                    h, c = None, None
                    for sub_time in range(len(lookback_pattern)):
                        sub_snapshot = Data(x=snapshot.x[:, sub_time:sub_time+1], edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr)
                        y_hat, h, c = forward(sub_snapshot, h, c)
                    predictions.append(y_hat)
                    labels.append(snapshot.y)
                    val_cost += loss_func(y_hat, snapshot.y)
                val_cost /= time + 1
                val_cost = val_cost.item()
                val_losses.append(val_cost)

                # EVALUATE MODEL - TEST
                test_cost = 0
                for time, snapshot in enumerate(test_dataset):
                    h, c = None, None
                    for sub_time in range(len(lookback_pattern)):
                        sub_snapshot = Data(x=snapshot.x[:, sub_time:sub_time+1], edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr)
                        y_hat, h, c = forward(sub_snapshot, h, c)
                    predictions.append(y_hat)
                    labels.append(snapshot.y)
                    test_cost += loss_func(y_hat, snapshot.y)
                test_cost /= time + 1
                test_cost = test_cost.item()
                test_losses.append(test_cost)

            # Save to results and display losses for this epoch
            results["Models"][model.name]["Loss by Epoch"].append({
                "Train": float(train_cost),
                "Train Evaluation": float(train_eval_cost),
                "Validation": float(val_cost),
                "Test": float(test_cost)
            })
            print('Epoch: {:03d}, Train Loss: {:.5f}, Train Eval Loss: {:.5f}, Val Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch,
                                                                                                      train_cost, train_eval_cost,
                                                                                                      val_cost,
                                                                                                      test_cost))
        # Keep a list of losses from each epoch for every model
        train_losseses.append(train_losses)
        train_eval_losseses.append(train_eval_losses)
        val_losseses.append(val_losses)
        test_losseses.append(test_losses)

        # Calculate and save loss per country to results. Optionally, visualize data
        # show_predictions(predictions, labels)
        results["Models"][model.name]['Loss by Country'] = show_loss_by_country(predictions, labels, nations, plot=False)
        # show_labels_by_country(labels, nations)




    # Set labels and plot loss curves for validation
    x = np.arange(0, num_epochs)
    plt.title('Model Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MASE Loss')
    for i in range(len(models)):
        label = models[i].name
        # plt.plot(x, train_losseses[i], label=str(label) + " (train)")
        # plt.plot(x, train_eval_losseses[i], label=str(label) + " (train eval)")
        plt.plot(x, val_losseses[i], label=str(label) + " (val)")
        # plt.plot(x, test_losseses[i], label=str(label) + " (test)")
    plt.legend()
    plt.show()

    #Save results into a .json file
    date = datetime.datetime.now().isoformat().split(".")[0]
    with open(f'results/results{date}.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    #Get country centroids data
    df2 = pd.read_csv("country_centroids.csv")
    columns = ['name_long', 'Longitude', 'Latitude', 'continent']
    df2 = df2.filter(columns)
    # Choose only a single continent for smaller testing (can choose any continent)
    df2 = df2[df2.continent == 'Europe']




    df = pd.read_csv('data/covid-data/covid-19-world-cases-deaths-testing.csv')
    columns = ['location', 'date', 'new_cases']
    df = df.filter(columns)
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)

    df = df[df.location.isin(df2.name_long.values)]
    nations = df.location.unique()
    nations = np.delete(nations, [len(nations)-1, len(nations)-2]) #Remove World and International

    # Only needed if remaking the dataset
    if not os.path.exists('data/covid-data/processed/covid_dataset_spaced.dataset'):
        dates = sorted(df.date.unique())
        new_data = {'Time': range(len(dates))}
        for i in range(len(nations)):
            nation = nations[i]
            nation_data = df.loc[df.location == nation]
            new_cases = []
            last_value = 0.0
            for date in dates:
                date_row = nation_data.loc[nation_data.date == date]
                if not date_row.empty:
                    new_cases.append(date_row.new_cases.iloc[0])
                    last_value = date_row.iloc[0].new_cases
                else:
                    new_cases.append(last_value)
            new_data[nation + '_new_cases'] = new_cases
        df = pd.DataFrame(data=new_data)

        print('Dataset preprocessed')
        df.to_csv("df.csv")
        print(df.head())
        print(df.columns)
        print(df.shape)

    #Get centroid of each country
    country_centroids = {}
    for nation in nations:
        match = df2.loc[df2.name_long == nation]
        if len(match):
            lon = match.Longitude.values[0]
            lat = match.Latitude.values[0]
            country_centroids[nation] = (lat, lon)
        else:
            print("Missing coordinates for country", nation)

    # Make predictions:
    gnn_predictor_single()