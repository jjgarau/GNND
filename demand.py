import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import max_pool_x as map, avg_pool_x as avp
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
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


def gnn_predictor():
    dataset = DemandDataset(root='demand-data/')
    dataset = dataset.shuffle()

    sample = len(dataset)
    train_dataset = dataset[:int(0.8 * sample)]
    val_dataset = dataset[int(0.8 * sample):int(0.9 * sample)]
    test_dataset = dataset[int(0.9 * sample):]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    # crit = torch.nn.L1Loss(reduction='mean')
    crit = mape_loss

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    num_epochs = 20
    for epoch in range(num_epochs):
        loss = train_gnn(model, train_loader, optimizer, crit, device)
        loss /= len(train_dataset)
        train_acc = evaluate_gnn(model, train_loader, device)
        val_acc = evaluate_gnn(model, val_loader, device)
        test_acc = evaluate_gnn(model, test_loader, device)
        print('Epoch: {:03d}, Loss: {:.5f}, Train MAPE: {:.5f}, Val MAPE: {:.5f}, Test MAPE: {:.5f}'.format(epoch, loss,
                                                                                                            train_acc,
                                                                                                            val_acc,
                                                                                                            test_acc))


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
