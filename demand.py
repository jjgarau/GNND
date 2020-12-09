import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.nn import SAGEConv, GCNConv, GENConv, GatedGraphConv, GraphConv, HypergraphConv, LEConv, SGConv, TAGConv
import torch_geometric.nn
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import max_pool_x as map, avg_pool_x as avp
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import random
import matplotlib.pyplot as plt
# import xgboost as xgb
import torch_geometric_temporal.nn
from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.data import discrete_train_test_split
from torch_geometric_temporal import GConvGRU, GConvLSTM, GCLSTM, DCRNN
from common import *
import graph_nets
from geopy.distance import geodesic

drop_columns = ['cet_cest_timestamp', 'DE_50hertz_load_actual_entsoe_transparency',
                'DE_AT_LU_load_actual_entsoe_transparency', 'DE_LU_load_actual_entsoe_transparency',
                'DE_amprion_load_actual_entsoe_transparency', 'DE_tennet_load_actual_entsoe_transparency',
                'DE_transnetbw_load_actual_entsoe_transparency']
country_codes = ['AT', 'BE', 'DE', 'HU', 'LU', 'NL']
output_size = len(country_codes)
pop_centroids_2000 = {'AT': (47.765386201318, 14.645625300333),
                      'BE': (50.844005826061, 4.4332869095216),
                      'DE': (50.855573924694, 9.6963409646128),
                      'HU': (47.288770753717, 19.388772968949),
                      'LU': (49.643734947502, 6.0837996175026),
                      'NL': (52.072871145825, 5.2875541627667)
                      } #Based on populations from year 2000 from http://cs.ecs.baylor.edu/~hamerly/software/europe_population_weighted_centers.txt
lookback = 5

class DemandDatasetGeodesic(InMemoryDataset):
    #Edge features are geodesic distance between population centroids
    def __init__(self, root, transform=None, pre_transform=None):
        super(DemandDatasetGeodesic, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['time_series_15min_singleindex_filtered.csv']

    @property
    def processed_file_names(self):
        return ['demand_dataset_geodesic.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        combs = list(itertools.combinations(range(len(country_codes)), 2))
        source_nodes = [e[0] for e in combs]
        target_nodes = [e[1] for e in combs]
        torch_def = torch.cuda if torch.cuda.is_available() else torch
        print(len(df))
        for i in tqdm(range(len(df) - lookback)):
            #Node Features
            values_x = df.iloc[i:(i+lookback), 1:].to_numpy().T
            x = torch_def.FloatTensor(values_x)
            # print(x.shape) [6, 5]

            #Labels
            values_y = df.iloc[(i+lookback):(i+lookback+1), 1:].to_numpy().T
            y = torch_def.FloatTensor(values_y)
            # print(y.shape) [6, 1]

            #Edge Features
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])
            # print(edge_index.shape) [2, 15]
            edge_attr = torch_def.FloatTensor([[geodesic(pop_centroids_2000[country_codes[comb[0]]], pop_centroids_2000[country_codes[comb[1]]]).km] for comb in combs])
            edge_attr = torch.nn.functional.normalize(edge_attr, dim=0)
            # print(edge_attr.shape) [15, 1]

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DemandDataset(InMemoryDataset):
    #Edge features are difference in longitude between population centroids
    def __init__(self, root, transform=None, pre_transform=None):
        super(DemandDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['time_series_15min_singleindex_filtered.csv']

    @property
    def processed_file_names(self):
        return ['demand_dataset.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        combs = list(itertools.combinations(range(len(country_codes)), 2))
        source_nodes = [e[0] for e in combs]
        target_nodes = [e[1] for e in combs]
        torch_def = torch.cuda if torch.cuda.is_available() else torch
        print(len(df))
        for i in tqdm(range(len(df) - lookback)):
            #Node Features
            values_x = df.iloc[i:(i+lookback), 1:].to_numpy().T
            x = torch_def.FloatTensor(values_x)
            # print(x.shape) [6, 5]

            #Labels
            values_y = df.iloc[(i+lookback):(i+lookback+1), 1:].to_numpy().T
            y = torch_def.FloatTensor(values_y)
            # print(y.shape) [6, 1]

            #Edge Features
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])
            # print(edge_index.shape) [2, 15]
            edge_attrs = []
            for comb in combs:
                coords = [pop_centroids_2000[country_codes[country]] for country in comb]
                comb_attr = [coords[0][1] - coords[1][1]]
                edge_attrs.append(comb_attr)
            edge_attr = torch_def.FloatTensor(edge_attrs)
            edge_attr = torch.nn.functional.normalize(edge_attr, dim=0)
            # print(edge_attr.shape) [15, 1]

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DemandDatasetDirectional(InMemoryDataset):
    #Edges are directional, so there are 2 per combination
    #Edge features are difference in longitude between population centroids
    def __init__(self, root, transform=None, pre_transform=None):
        super(DemandDatasetDirectional, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['time_series_15min_singleindex_filtered.csv']

    @property
    def processed_file_names(self):
        return ['demand_dataset_directional.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        combs = list(itertools.permutations(range(len(country_codes)), 2))
        source_nodes = [e[0] for e in combs]
        target_nodes = [e[1] for e in combs]
        torch_def = torch.cuda if torch.cuda.is_available() else torch
        print(len(df))
        for i in tqdm(range(len(df) - lookback)):
            #Node Features
            values_x = df.iloc[i:(i+lookback), 1:].to_numpy().T
            x = torch_def.FloatTensor(values_x)
            # print(x.shape) [6, 5]

            #Labels
            values_y = df.iloc[(i+lookback):(i+lookback+1), 1:].to_numpy().T
            y = torch_def.FloatTensor(values_y)
            # print(y.shape) [6, 1]

            #Edge Features
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])
            # print(edge_index.shape) [2, 15]
            edge_attrs = []
            for comb in combs:
                coords = [pop_centroids_2000[country_codes[country]] for country in comb]
                comb_attr = [coords[0][1] - coords[1][1]]
                edge_attrs.append(comb_attr)
            edge_attr = torch_def.FloatTensor(edge_attrs)
            edge_attr = torch.nn.functional.normalize(edge_attr, dim=0)
            # print(edge_attr.shape) [15, 1]

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class DemandDatasetNEF(InMemoryDataset):
    #No edge features
    def __init__(self, root, transform=None, pre_transform=None):
        super(DemandDatasetNEF, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['time_series_15min_singleindex_filtered.csv']

    @property
    def processed_file_names(self):
        return ['demand_dataset_nef.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        combs = list(itertools.combinations(range(len(country_codes)), 2))
        source_nodes = [e[0] for e in combs]
        target_nodes = [e[1] for e in combs]
        torch_def = torch.cuda if torch.cuda.is_available() else torch
        print(len(df))
        for i in tqdm(range(len(df) - lookback)):
            #Node Features
            values_x = df.iloc[i:(i+lookback), 1:].to_numpy().T
            x = torch_def.FloatTensor(values_x)
            # print(x.shape) [6, 5]

            #Labels
            values_y = df.iloc[(i+lookback):(i+lookback+1), 1:].to_numpy().T
            y = torch_def.FloatTensor(values_y)
            # print(y.shape) [6, 1]

            #Edge Features
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])
            # print(edge_index.shape) [2, 15]

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def gnn_predictor():
    dataset = DemandDatasetNEF(root='data/demand-data/')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_func = mape_loss

    layers = [SAGEConv]
    recurrent_layers = [GConvLSTM, DCRNN, GConvGRU]

    models = []

    for layer in recurrent_layers:
        models.append(graph_nets.RecurrentGraphNet(layer, lookback, output_size))
    for layer in layers:
        models.append(graph_nets.GraphNet(layer, lookback, output_size))

    models = [graph_nets.GraphNet(SAGEConv, lookback, output_size)]
    for i in range(len(models)): #Grid search loop
        model = models[i]
        dataset = dataset.shuffle()

        sample = len(dataset)
        # Make dataset smaller for quick testing
        sample *= 1.0
        train_dataset = dataset[:int(0.8 * sample)]
        val_dataset = dataset[int(0.8 * sample):int(0.9 * sample)]
        test_dataset = dataset[int(0.9 * sample):int(sample)]

        print(model)

        batch_size = 256
        lr = 0.005
        if type(model) is graph_nets.RecurrentGraphNet:
            # alter filter size, what else can I do to make these things work? lookback?
            lr = 0.05

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        num_epochs = 20
        val_losses = []
        for epoch in range(num_epochs):
            loss = train_gnn(model, train_loader, optimizer, loss_func, device)
            loss /= len(train_dataset)
            train_acc = evaluate_gnn(model, train_loader, device)
            val_acc = evaluate_gnn(model, val_loader, device)
            test_acc = evaluate_gnn(model, test_loader, device)
            val_losses.append(val_acc)
            print('Epoch: {:03d}, Loss: {:.5f}, Train Loss: {:.5f}, Val Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch, loss,
                                                                                                                train_acc,
                                                                                                                val_acc,
                                                                                                                  test_acc))
        x = np.arange(0, num_epochs)
        plt.title("Adding Residual to LEPooling")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        labels = ["LEPooling", "LEPooling w/ Residual"]
        plt.plot(x, val_losses, label=labels[i])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('data/demand-data/time_series_15min_singleindex_filtered.csv')
    df = df.drop(columns=drop_columns)
    df = df.fillna(method='pad')
    df.columns = ['time'] + country_codes
    print('Dataset preprocessed')

    print(df.head())
    print(df.columns)
    print(df.shape)

    gnn_predictor()
