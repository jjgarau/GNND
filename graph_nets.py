import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric_temporal.nn import DCRNN, GConvGRU, GConvLSTM

country_codes = ['AT', 'BE', 'DE', 'HU', 'LU', 'NL']


class DeeperGraphNet(torch.nn.Module):
    def __init__(self, lookback):
        super(DeeperGraphNet, self).__init__()

        dim = 64
        self.conv1 = SAGEConv(lookback, dim)
        self.pool1 = TopKPooling(dim, ratio=0.8)
        self.conv2 = SAGEConv(dim, dim)
        self.pool2 = TopKPooling(dim, ratio=0.8)
        self.conv3 = SAGEConv(dim, dim)
        self.pool3 = TopKPooling(dim, ratio=0.8)
        self.conv4 = SAGEConv(dim, dim)
        self.pool4 = TopKPooling(dim, ratio=0.8)
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
        x = F.relu(self.conv4(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3 + x4
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return x

class GNN(torch.nn.Module):
    def __init__(self, layer, num_layers, lookback):
        """Layer is a lambda function taking input_channels and output_channels and returning a conv layer."""
        super(GNN, self).__init__()

        dim = 64
        self.hidden = torch.nn.ModuleList([layer(lookback, dim)])
        self.pools = torch.nn.ModuleList([TopKPooling(dim, ratio=0.8)])
        self.num_layers = num_layers
        for n in range(1, num_layers):
            self.hidden.append(layer(dim, dim))
            self.pools.append(TopKPooling(dim, ratio=0.8))
        self.lin1 = torch.nn.Linear(dim * 2, dim)
        self.lin2 = torch.nn.Linear(dim, len(country_codes))
        self.act1 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.hidden[0](x, edge_index)[0])
        x, edge_index, _, batch, _, _ = self.pools[0](x, edge_index, None, batch)
        summation = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        for n in range(1, self.num_layers):
            x = F.relu(self.hidden[n].forward(x, edge_index)[0])
            x, edge_index, _, batch, _, _ = self.pools[n].forward(x, edge_index, None, batch)
            summation += torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = summation
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return x

class SAGEConvNet(torch.nn.Module):
    def __init__(self, lookback, output_size, dim=64):
        super(SAGEConvNet, self).__init__()

        self.dim = dim
        self.lookback = lookback
        self.conv1 = SAGEConv(lookback, dim)
        self.pool1 = TopKPooling(dim, ratio=0.8)
        self.conv2 = SAGEConv(dim, dim)
        self.pool2 = TopKPooling(dim, ratio=0.8)
        self.lin1 = torch.nn.Linear(dim * 2, dim)
        self.lin2 = torch.nn.Linear(dim, output_size) #len(country_codes)) for demand, 20 for chickenpox
        self.act1 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)

        return x

class GraphNet(torch.nn.Module):
    def __init__(self, layer, lookback, output_size, dim=64):
        super(GraphNet, self).__init__()

        self.dim = dim
        self.lookback = lookback
        self.conv1 = layer(lookback, dim)
        self.pool1 = TopKPooling(dim, ratio=0.8)
        self.conv2 = layer(dim, dim)
        self.pool2 = TopKPooling(dim, ratio=0.8)
        self.lin1 = torch.nn.Linear(dim * 2, dim)
        self.lin2 = torch.nn.Linear(dim, output_size) #len(country_codes)) for demand, 20 for chickenpox
        self.act1 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)

        return x

class RecurrentGraphNet(torch.nn.Module):
    def __init__(self, layer, lookback, output_size, dim=64):
        super(RecurrentGraphNet, self).__init__()

        self.recurrent = layer(lookback, dim, 1) #last param is Chebyshev Filter Size
        self.pool1 = TopKPooling(dim, ratio=0.8)
        self.lin1 = torch.nn.Linear(dim * 2, dim)
        self.lin2 = torch.nn.Linear(dim, output_size)
        self.act1 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.recurrent(x, edge_index, edge_attr.reshape([edge_attr.shape[0]]))
        if type(x) is tuple:
            x = x[0]
        x = F.relu(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)

        return x