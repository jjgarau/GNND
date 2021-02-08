import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import pandas as pd
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

#Test Recurrent Neural Networks on COVID Dataset
#There is a separate file because the training pattern is slightly different,
#and I am almost exclusively using RNNs at this point.

lookback = 5
lookback_pattern = [25, 10, 5, 4, 3, 2, 1]

class COVIDDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(COVIDDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['covid_dataset.dataset']

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
        for i in tqdm(range(len(df) - lookback)):
            # Node Features
            values_x = df.iloc[i:(i + lookback), 1:].to_numpy().T
            x = torch_def.FloatTensor(values_x)

            # Labels
            values_y = df.iloc[(i + lookback):(i + lookback + 1), 1:].to_numpy().T
            y = torch_def.FloatTensor(values_y)

            # Edge Index
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])
            # Edge Weights
            edge_attr = torch_def.FloatTensor([[weight] for weight in edge_attrs])


            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class COVIDDatasetSingle(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(COVIDDatasetSingle , self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['covid_dataset_single.dataset']

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

        lookback = 1 #This is the only difference from the other dataset
        for i in tqdm(range(len(df) - lookback)):
            # Node Features
            values_x = df.iloc[i:(i + lookback), 1:].to_numpy().T
            x = torch_def.FloatTensor(values_x)

            # Labels
            values_y = df.iloc[(i + lookback):(i + lookback + 1), 1:].to_numpy().T
            y = torch_def.FloatTensor(values_y)

            # Edge Index
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])
            # Edge Weights
            edge_attr = torch_def.FloatTensor([[weight] for weight in edge_attrs])


            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


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

        for i in tqdm(range(len(df) - lookback_pattern[0])):
            edge_mask = torch.logical_not(torch.logical_xor(edge_mask, torch.bernoulli(0.95 * torch.ones(len(source_nodes))).bool()))
            node_mask = torch.logical_not(torch.logical_xor(node_mask, torch.bernoulli(0.95 * torch.ones(len(df))).bool()))
            inv_node_mask = ~node_mask
            nodes_to_drop = set(torch.arange(len(df))[inv_node_mask].tolist())

            temp_edge_mask = edge_mask.clone()
            for i in range(len(source_nodes)):
                if source_nodes[i] in nodes_to_drop or target_nodes[i] in nodes_to_drop:
                    temp_edge_mask[i] = False

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
            edge_index = edge_index[:, temp_edge_mask]
            # Edge Weights
            edge_attr = torch_def.FloatTensor([[weight] for weight in edge_attrs])
            edge_attr = edge_attr[temp_edge_mask, :]


            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def gnn_predictor():

    #Load and split dataset

    dataset = COVIDDatasetSpaced(root='data/covid-data/')
    dataset = dataset.shuffle()

    sample = len(dataset)
    # Optionally, make dataset smaller for quick testing
    sample *= 0.7
    train_dataset = dataset[:int(0.8 * sample)]
    val_dataset = dataset[int(0.8 * sample):int(0.9 * sample)]
    test_dataset = dataset[int(0.9 * sample):int(sample)]



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_func = mase_loss


    # Design models for testing
    output_gnns = [
        lambda lookback, dim: graph_nets.GNNModule(WeightedSAGEConv, 1, lookback, dim=dim, res_factors=None, dropouts=[]),
        lambda lookback, dim: graph_nets.GNNModule(WeightedSAGEConv, 3, lookback, dim=dim, res_factors=None, dropouts=[])
    ]

    linear_module = lambda in_channels, out_channels, bias: graph_nets.GraphLinear(in_channels, out_channels, bias=bias)

    WSC = WeightedSAGEConv
    models = [
        RNN(module=WSC, gnn=WSC, rnn=LSTM, dim=16, gnn_2=WSC, rnn_depth=1, node_features=len(lookback_pattern)),
        graph_nets.RecurrentGraphNet(GConvLSTM, lookback=len(lookback_pattern)),
        graph_nets.RecurrentGraphNet(GConvGRU, lookback=len(lookback_pattern)),
        graph_nets.RecurrentGraphNet(DCRNN, lookback=len(lookback_pattern)),
        graph_nets.RecurrentGraphNet(GCLSTM, lookback=len(lookback_pattern)),
    ]

    for i in range(len(models)):

        model = models[i]
        print(model)


        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


        num_epochs = 40
        val_losses = []
        h, c = None, None
        for epoch in range(num_epochs):
            h, c = None, None
            # TRAIN MODEL
            model.train()
            train_cost = 0
            for time, snapshot in enumerate(train_dataset):
                h, c = None, None
                y_hat, h, c = model(snapshot, h, c)
                train_cost += loss_func(y_hat, snapshot.y)
            train_cost /=  time + 1
            train_cost.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():

                # EVALUATE MODEL - TRAINING
                h, c = None, None
                model.eval()
                train_eval_cost = 0
                for time, snapshot in enumerate(train_dataset):
                    h, c = None, None
                    y_hat, h, c = model(snapshot, h, c)
                    train_eval_cost += loss_func(y_hat, snapshot.y)
                train_eval_cost /= time + 1
                train_eval_cost = train_eval_cost.item()

                # EVALUATE MODEL - VALIDATION
                h, c = None, None
                val_cost = 0
                for time, snapshot in enumerate(val_dataset):
                    h, c = None, None
                    y_hat, h, c = model(snapshot, h, c)
                    val_cost += loss_func(y_hat, snapshot.y)
                val_cost /= time + 1
                val_cost = val_cost.item()
                val_losses.append(val_cost)

                # EVALUATE MODEL - TEST
                h, c = None, None
                test_cost = 0
                for time, snapshot in enumerate(test_dataset):
                    h, c = None, None
                    y_hat, h, c = model(snapshot, h, c)
                    test_cost += loss_func(y_hat, snapshot.y)
                test_cost /= time + 1
                test_cost = test_cost.item()

            #Display losses for this epoch
            print('Epoch: {:03d}, Loss: {:.5f}, Train Loss: {:5f}, Val Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch,
                                                                                                                train_cost,
                                                                                                                train_eval_cost,
                                                                                                                val_cost,
                                                                                                                test_cost))

        # Set labels and plot loss curves for validation
        x = np.arange(0, num_epochs)
        lbls = ['GConvLSTM', 'GConvGRU', 'DCRNN', 'GCLSTM', 'Our Model']
        label = lbls[i]
        plt.title('Model Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('MASE Loss')
        plt.plot(x, val_losses, label=str(label))


    plt.legend()
    plt.show()

def gnn_predictor_single():
    # Load and split dataset
    dataset = COVIDDatasetSpaced(root='data/covid-data/')
    dataset = dataset.shuffle()

    sample = len(dataset)
    # Choose a frame of the dataset to work with
    sample *= 0.7
    train_dataset = dataset[:int(0.8 * sample)]
    val_dataset = dataset[int(0.8 * sample):int(0.9 * sample)]
    test_dataset = dataset[int(0.9 * sample):int(sample)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_func = mase_loss

    WSC = WeightedSAGEConv
    USC = lambda in_channels, out_channels, bias=True: WeightedSAGEConv(in_channels, out_channels, weighted=False)
    # Design models for testing
    output_gnns = [
        lambda lookback, dim: graph_nets.GNNModule(USC, 1, lookback, dim=dim, res_factors=[0],
                                                   dropouts=[0]),
        lambda lookback, dim: graph_nets.GNNModule(USC, 3, lookback, dim=dim, res_factors=[1, 0, 1],
                                                   dropouts=[1]),
        lambda lookback, dim: graph_nets.GNNModule(USC, 5, lookback, dim=dim, res_factors=[1, 0, 1, 0, 1],
                                                   dropouts=[0, 3]),
        lambda lookback, dim: graph_nets.GNNModule(USC, 8, lookback, dim=dim, res_factors=[1, 0, 1, 0, 1, 0, 1, 0],
                                                   dropouts=[0, 3, 5]),
        lambda lookback, dim: graph_nets.GNNModule(USC, 10, lookback, dim=dim, res_factors=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
                                                   dropouts=[0, 3, 5, 8]),
    ]

    linear_module = lambda in_channels, out_channels, bias: graph_nets.GraphLinear(in_channels, out_channels,
                                                                                   bias=bias)

    models = [
        RNN(module=SAGEConv, gnn=SAGEConv, rnn=LSTM, dim=16, gnn_2=SAGEConv, rnn_depth=1),
    ]

    train_losseses, train_eval_losseses, val_losseses, test_losseses = [], [], [], []
    for i in range(len(models)):

        model = models[i]
        print(model)

        for p in model.parameters():
            # p.data.fill_(100)
            pass

        def forward(snapshot, h, c):
            if h is None:
                out = model(snapshot, h, c)
            else:
                out = model(snapshot, h.detach(), c.detach())
            if len(out) == 3:
                x, h, c = out
                h = h.detach()
                c = c.detach()
                return x, h, c
            else:
                x, h = out
                h = h.detach()
                return x, h

        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        num_epochs = 50
        train_losses, train_eval_losses, val_losses, test_losses = [], [], [], []

        for epoch in range(num_epochs):
            h, c = None, None

            predictions, labels = [], []

            # TRAIN MODEL
            model.train()
            train_cost = 0
            for time, snapshot in enumerate(train_dataset):
                h, c = None, None
                for sub_time in range(len(lookback_pattern)):
                    sub_snapshot = Data(x=snapshot.x[:, sub_time:sub_time+1], edge_index=snapshot.edge_index, edge_attr=snapshot.edge_attr)
                    y_hat, h, c = forward(sub_snapshot, h, c)
                predictions.append(y_hat)
                labels.append(snapshot.y)
                train_cost += loss_func(y_hat, snapshot.y)
            train_cost /= time + 1
            train_losses.append(train_cost)
            train_cost.backward()
            optimizer.step()
            optimizer.zero_grad()

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

            # Display losses for this epoch
            print('Epoch: {:03d}, Train Loss: {:.5f}, Train Eval Loss: {:.5f}, Val Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch,
                                                                                                      train_cost, train_eval_cost,
                                                                                                      val_cost,
                                                                                                      test_cost))
            if 1 == 0:
                show_predictions(predictions, labels)

        train_losseses.append(train_losses)
        train_eval_losseses.append(train_eval_losses)
        val_losseses.append(val_losses)
        test_losseses.append(test_losses)

    # Set labels and plot loss curves for validation
    x = np.arange(0, num_epochs)
    lbls = ['GConvLSTM', 'GConvGRU', 'DCRNN', 'GCLSTM', 'Our Model']
    plt.title('Model Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MASE Loss')
    for i in range(len(models)):
        label = lbls[i]
        # plt.plot(x, train_losseses[i], label=str(label) + " (train)")
        # plt.plot(x, train_eval_losseses[i], label=str(label) + " (train eval)")
        plt.plot(x, val_losseses[i], label=str(label) + " (val)")
        # plt.plot(x, test_losseses[i], label=str(label) + " (test)")
    plt.legend()
    plt.show()

    show_predictions(predictions, labels)
    # show_loss_by_country(predictions, labels, nations)
    show_labels_by_country(labels, nations)

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

    # Commented unless remaking dataset
    dates = df.date.unique()
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

    gnn_predictor_single()