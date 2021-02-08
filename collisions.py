import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import torch
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from geopy.distance import geodesic
from bisect import bisect
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from util import *
from weight_sage import WeightedSAGEConv
import graph_nets
from rnn import *
from torch_geometric.nn import SAGEConv

id_to_street = {}
street_to_location = defaultdict(list)
location_to_id = {}
id_to_location = {}
lookback_pattern = [12, 11, 10, 9, 8]

def draw_map(m, scale=0.2):
    """Utility function to draw map on top of matplotlib pyplot"""
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)

    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = itertools.chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = itertools.chain(*(tup[1][0] for tup in lons.items()))
    all_lines = itertools.chain(lat_lines, lon_lines)

    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')

class CollisionDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, should_visualize_data=False, num_nodes=None):
        self.num_nodes = num_nodes
        super(CollisionDataset, self).__init__(root, transform, pre_transform)
        self.should_visualize_data = should_visualize_data
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['collision_dataset.dataset']

    def download(self):
        pass

    def process(self):
        borough = "STATEN ISLAND"
        process_df(num_nodes=self.num_nodes, borough=borough)
        df = pd.read_csv(f'data/collision-data/{borough}_collisions.csv')
        print(df.head())

        # visualize_dataset()

        source_nodes = []
        target_nodes = []
        edge_attrs = []
        print("Processing edges...")
        for source_id, source_cross_streets in tqdm(id_to_street.items()):
            all_target_ids = []
            all_distances = []
            for source_cross_street in source_cross_streets:
                distances = []
                near_locations = []
                intersection_locations = street_to_location[source_cross_street]
                for intersection_location in intersection_locations:
                    dist = geodesic(intersection_location, id_to_location[source_id])
                    index = bisect(distances, dist)
                    if index < 4:
                        distances.insert(index, dist.m)
                        near_locations.insert(index, location_to_id[intersection_location])
                all_target_ids += near_locations
                all_distances += distances

            source_nodes += [source_id] * min(len(all_target_ids), 4)
            target_nodes += all_target_ids[:4]
            edge_attrs += all_distances[:4]

        torch_def = torch.cuda if torch.cuda.is_available() else torch
        data_list = []

        print("Processing CollisionDataset...")
        for i in tqdm(range(len(df) - lookback_pattern[0])):

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
            # edge_index = edge_index[:, edge_mask]
            # Edge Weights
            edge_attr = torch_def.FloatTensor([[weight] for weight in edge_attrs])
            # edge_attr = edge_attr[edge_mask, :]

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class LargeCollisionDataset(CollisionDataset):
    def __init__(self, root, transform=None, pre_transform=None, should_visualize_data=False):
        super(LargeCollisionDataset, self).__init__(root, transform, pre_transform, should_visualize_data=should_visualize_data, num_nodes=5000)

    @property
    def processed_file_names(self):
        return ['large_collision_dataset.dataset']

    @property
    def raw_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        super(LargeCollisionDataset, self).process()

class SmallCollisionDataset(CollisionDataset):
    def __init__(self, root, transform=None, pre_transform=None, should_visualize_data=False):
        super(SmallCollisionDataset, self).__init__(root, transform, pre_transform, should_visualize_data=should_visualize_data, num_nodes=500)

    @property
    def processed_file_names(self):
        return ['small_collision_dataset.dataset']

    @property
    def raw_file_names(self):
        return []

    def process(self):
        super(SmallCollisionDataset, self).process()


def gnn_predictor(should_train=True, models=None, dataset=None):
    # Load and split dataset
    if not dataset:
        dataset = SmallCollisionDataset(root='data/collision-data/')
    # dataset = dataset.shuffle()

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

    linear_module = lambda in_channels, out_channels, bias: graph_nets.GraphLinear(in_channels, out_channels, bias=bias)

    if models is None:
        models = [
            RNN(module=USC, gnn=USC, rnn=LSTM, dim=8, gnn_2=USC, rnn_depth=1),
        ]

    train_losseses, train_eval_losseses, val_losseses, test_losseses = [], [], [], []
    for i in range(len(models)):

        model = models[i]
        print(model)

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

        num_epochs = 15
        train_losses, train_eval_losses, val_losses, test_losses = [], [], [], []

        for epoch in range(num_epochs):
            h, c = None, None

            predictions, labels = [], []

            if should_train:
                # TRAIN MODEL
                model.train()
                train_cost = 0
                for time, snapshot in enumerate(train_dataset):
                    h, c = None, None
                    for sub_time in range(len(lookback_pattern)):
                        sub_snapshot = Data(x=snapshot.x[:, sub_time:sub_time + 1], edge_index=snapshot.edge_index,
                                            edge_attr=snapshot.edge_attr)
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
                        sub_snapshot = Data(x=snapshot.x[:, sub_time:sub_time + 1], edge_index=snapshot.edge_index,
                                            edge_attr=snapshot.edge_attr)
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
                        sub_snapshot = Data(x=snapshot.x[:, sub_time:sub_time + 1], edge_index=snapshot.edge_index,
                                            edge_attr=snapshot.edge_attr)
                        y_hat, h, c = forward(sub_snapshot, h, c)
                    predictions.append(y_hat)
                    labels.append(snapshot.y)
                    test_cost += loss_func(y_hat, snapshot.y)
                test_cost /= time + 1
                test_cost = test_cost.item()
                test_losses.append(test_cost)

            # Display losses for this epoch
            print(
                'Epoch: {:03d}, Train Loss: {:.5f}, Train Eval Loss: {:.5f}, Val Loss: {:.5f}, Test Loss: {:.5f}'.format(
                    epoch,
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
    lbls = ['SAGEConv', 'GConvGRU', 'DCRNN', 'GCLSTM', 'Our Model']
    plt.title('Collision Prediction')
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

    return models


def process_df(no_save=False, num_nodes=None, borough=None):
    df = pd.read_csv('collisions.csv')
    columns = ['CRASH DATE', 'CRASH TIME', 'BOROUGH', 'LOCATION', 'ON STREET NAME', 'CROSS STREET NAME']
    df = df.filter(columns)

    if borough:
        df = df[df.BOROUGH == borough]
    df = df.dropna()

    #Label unique locations
    df.LOCATION = df.LOCATION.apply(lambda loc: tuple(map(lambda coordinate: float(coordinate), loc.replace('(', '').replace(')', '').split(', '))))
    locations = df.LOCATION.unique()
    #Select a subset of the map
    if num_nodes:
        locations = locations[:num_nodes]
        df = df[df.LOCATION.isin(locations)]

    i = 0
    for location in tqdm(locations):

        location_to_id[location] = i
        id_to_location[i] = location

        loc_repr_row = df[df.LOCATION == location]
        street_names = set()
        for name in loc_repr_row['ON STREET NAME'].unique():
            n = name.strip()
            street_names.add(n)
            street_to_location[n].append(location)
        id_to_street[i] = street_names
        i += 1

    if no_save:
        return

    df.LOCATION = df.LOCATION.apply(lambda x: location_to_id[x])

    # Group by timestamp
    def parse_date(date, time):
        date_parts = date.split('/')
        time_parts = time.split(':')
        return datetime(int(date_parts[2]), int(date_parts[0]), int(date_parts[1]), int(time_parts[0]),
                        int(time_parts[1]))

    df['TIMESTAMP'] = df.apply(lambda x: parse_date(x['CRASH DATE'], x['CRASH TIME']), axis=1)
    df = df.sort_values('TIMESTAMP')
    df = df.set_index('TIMESTAMP')
    df['COUNT'] = 0

    #Split by borough
    boroughs = df.BOROUGH.unique()
    for borough in boroughs:
        df_borough = df[df.BOROUGH == borough]
        df_borough = df_borough.groupby([pd.Grouper(freq='M'), 'LOCATION']).agg({'COUNT': 'count', 'ON STREET NAME': 'first', 'CROSS STREET NAME': 'first'})

        df_borough = df_borough.reset_index()
        new_data = []
        for timestamp in tqdm(df_borough.TIMESTAMP.unique()):
            row = np.zeros(len(df_borough))
            for location in df_borough[df_borough.TIMESTAMP == timestamp].iterrows():
                row[location[1].LOCATION] = location[1].COUNT

            new_data.append(row)

        df_borough = pd.DataFrame(new_data)
        df_borough.to_csv(f"data/collision-data/{borough}_collisions.csv")


def visualize_dataset():
    dataset = CollisionDataset(root='data/collision-data/', should_visualize_data=True)
    source_nodes = dataset[0].edge_index[0].tolist()
    target_nodes = dataset[0].edge_index[1].tolist()

    # Create Basemap over Staten Island
    fig = plt.figure(figsize=(8, 8))
    m = Basemap(projection='lcc', resolution=None, lat_0=30, lon_0=-20,
                llcrnrlat=40, llcrnrlon=-75, urcrnrlat=41, urcrnrlon=-74)

    draw_map(m)

    # Collect coordinates
    coordinates = [id_to_location[i] for i in source_nodes]
    coordinates2 = [id_to_location[i] for i in target_nodes]
    for i in range(len(coordinates)):
        m.drawgreatcircle(coordinates[i][1], coordinates[i][0], coordinates2[i][1], coordinates2[i][0])

    # Draw coordinates on map
    # m.scatter([centroid[1] for centroid in id_to_location.values()],
    #           [centroid[0] for centroid in id_to_location.values()], latlon=True, zorder=10)
    plt.show()


if __name__ == '__main__':
    models = gnn_predictor()
    large_dataset = LargeCollisionDataset(root='data/collision-data/')
    gnn_predictor(models=models, dataset=large_dataset)
