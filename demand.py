import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.nn import SAGEConv
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


drop_columns = ['cet_cest_timestamp', 'DE_50hertz_load_actual_entsoe_transparency',
                'DE_AT_LU_load_actual_entsoe_transparency', 'DE_LU_load_actual_entsoe_transparency',
                'DE_amprion_load_actual_entsoe_transparency', 'DE_tennet_load_actual_entsoe_transparency',
                'DE_transnetbw_load_actual_entsoe_transparency']
country_codes = ['AT', 'BE', 'DE', 'HU', 'LU', 'NL']
lookback = 5

class GraphNet(torch.nn.Module):
    def __init__(self):
        super(GraphNet, self).__init__()

        dim = 64
        self.conv1 = SAGEConv(lookback, dim)
        self.pool1 = TopKPooling(dim, ratio=0.8)
        self.conv2 = SAGEConv(dim, dim)
        self.pool2 = TopKPooling(dim, ratio=0.8)
        self.lin1 = torch.nn.Linear(dim * 2, dim)
        self.lin2 = torch.nn.Linear(dim, len(country_codes))
        self.act1 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)

        return x

class DeeperGraphNet(torch.nn.Module):
    def __init__(self):
        super(DeeperGraphNet, self).__init__()

        dim = 64
        self.conv1 = SAGEConv(lookback, dim)
        self.pool1 = TopKPooling(dim, ratio=0.8)
        self.conv2 = SAGEConv(dim, dim)
        self.pool2 = TopKPooling(dim, ratio=0.8)
        self.conv3 = SAGEConv(dim, dim)
        self.pool3 = TopKPooling(dim, ratio=0.8)
        self.lin1 = torch.nn.Linear(dim * 2, dim)
        self.lin2 = torch.nn.Linear(dim, len(country_codes))
        self.act1 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return x

class GNN(torch.nn.Module):
    def __init__(self, conv1, conv2):
        super(GNN, self).__init__()

        dim = 64
        self.conv1 = conv1
        self.pool1 = TopKPooling(dim, ratio=0.8)
        self.conv2 = conv2
        self.pool2 = TopKPooling(dim, ratio=0.8)
        self.lin1 = torch.nn.Linear(dim * 2, dim)
        self.lin2 = torch.nn.Linear(dim, len(country_codes))
        self.act1 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return x


class DemandDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DemandDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['demand_dataset.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        perms = list(itertools.permutations(range(len(country_codes)), 2))
        source_nodes = [e[0] for e in perms]
        target_nodes = [e[1] for e in perms]
        torch_def = torch.cuda if torch.cuda.is_available() else torch
        for i in tqdm(range(len(df) - lookback)):
            values_x = df.iloc[i:(i+lookback), 1:].to_numpy().T
            x = torch_def.FloatTensor(values_x)
            values_y = df.iloc[(i+lookback):(i+lookback+1), 1:].to_numpy().T
            y = torch_def.FloatTensor(values_y)
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def train_gnn(model, loader, optimizer, crit, device):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        output = torch.reshape(output, label.shape)
        loss = crit(output, label)
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
    return np.mean(abs(labels - predictions) / labels)

def mape_loss(output, label):
    loss = torch.mean(torch.div(torch.abs(output - label), label))
    return loss

def grid_search_generator(optimizer=True, crit=True):
    """Yields combinations for (random) exhaustive search over space of hyperparameters/loss functions/optimizers/layer types/etc..."""
    nn_dir = dir(torch.nn)
    optim_dir = dir(torch.optim)

    optimizers = [optimizer]
    if optimizer:
        optimizers = []
        for func in optim_dir:
            attr = getattr(torch.optim, func)
            if callable(attr) and issubclass(attr, torch.optim.Optimizer):
                optimizers.append(attr)

    crits = [crit]
    if crit:
        crits = []
        for func in nn_dir:
            if "Loss" in func and not func == 'AdaptiveLogSoftmaxWithLoss':
                attr = getattr(torch.nn, func)
                if callable(attr):
                    crits.append(attr)

    while True:
        yield {'optimizer': random.choice(optimizers), 'crit': random.choice(crits)}


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

def choose_crit():
    crit = random.choice([
        mape_loss,
        torch.nn.L1Loss(), #reduction=mean
        torch.nn.MSELoss(),
        torch.nn.SmoothL1Loss()
    ])
    return crit

def gnn_predictor():
    dataset = DemandDataset(root='demand-data/')
    dataset = dataset.shuffle()

    sample = len(dataset)
    # Make dataset smaller for quick testing
    sample *= 0.2

    train_dataset = dataset[:int(0.8 * sample)]
    val_dataset = dataset[int(0.8 * sample):int(0.9 * sample)]
    test_dataset = dataset[int(0.9 * sample):int(sample)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    crit = mape_loss
    dim = 64
    models = [DeeperGraphNet().to(device),
              GNN(torch_geometric.nn.MFConv(lookback, dim), torch_geometric.nn.MFConv(dim, dim)).to(device),
              GNN(torch_geometric.nn.ChebConv(lookback, dim, 16), torch_geometric.nn.ChebConv(dim, dim, 16)).to(device),
              GNN(torch_geometric.nn.GATConv(lookback, dim), torch_geometric.nn.GATConv(dim, dim)).to(device),
              GraphNet().to(device),
              GNN(torch_geometric.nn.GatedGraphConv(dim, 2), torch_geometric.nn.GatedGraphConv(dim, 2)).to(device),
              GNN(torch_geometric.nn.GraphConv(lookback, dim), torch_geometric.nn.GraphConv(dim, dim)).to(device)
              ]

    for model in models: #Grid search loop
        print(model)
        optimizer = torch.optim.Rprop(model.parameters(), lr=0.005)

        batch_size = 256
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        num_epochs = 10
        val_losses = []
        for epoch in range(num_epochs):
            loss = train_gnn(model, train_loader, optimizer, crit, device)
            loss /= len(train_dataset)
            train_acc = evaluate_gnn(model, train_loader, device)
            val_acc = evaluate_gnn(model, val_loader, device)
            test_acc = evaluate_gnn(model, test_loader, device)
            val_losses.append(val_acc)
            print('Epoch: {:03d}, Loss: {:.5f}, Train MAPE: {:.5f}, Val MAPE: {:.5f}, Test MAPE: {:.5f}'.format(epoch, loss,
                                                                                                                train_acc,
                                                                                                                val_acc,
                                                                                                                  test_acc))
        x = np.arange(0, num_epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(x, val_losses, label=str(model).split('(')[0])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('demand-data/time_series_15min_singleindex_filtered.csv')
    df = df.drop(columns=drop_columns)
    df = df.fillna(method='pad')
    df.columns = ['time'] + country_codes
    print('Dataset preprocessed')

    print(df.head())
    print(df.columns)
    print(df.shape)

    gnn_predictor()
