import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch_geometric_temporal.nn import GConvLSTM, GConvGRU, DCRNN, GCLSTM, LRGCN

#Abstracted classes for Graph Neural Networks

class GraphLinear(torch.nn.Linear):
    """This is the exact same as torch.nn.Linear,
    except that it can take edge_index, edge_attr and do nothing with them.
    Makes it interchangeable with graph neural network modules."""

    def forward(self, input, edge_index, edge_attr):
        return super(GraphLinear, self).forward(input)


class GNN(torch.nn.Module):
    """
    Generalized Graph Neural Network whose parameters allow for the full range of testing.
    Parameters:
        layer: torch.nn.Module - type of GNN from torch_geometric_temporal.nn to use (can also be a lambda function taking (input_channels, output_channels) and returning torch.nn.Module
        num_layers: int - number of repetitions of layer to use in sequence (depth of GNN)
        lookback: int - number of input node features
        output_size: int - number of nodes to predict
        dim: int - length of hidden embedding vectors
        res_factors: [int] - Array of length num_layers, containing coefficient to residual at corresponding layers
        dropouts: [int] - Indices of layers in which to include dropout during testing
    """

    def __init__(self, layer, num_layers, lookback, output_size, dim=64, res_factors=None, dropouts=[]):
        super(GNN, self).__init__()

        self.dim = dim
        if res_factors is None:
            self.res_factors = [0.0] * num_layers
        else:
            self.res_factors = res_factors
        self.dropouts = dropouts
        self.lookback = lookback
        self.output_size = output_size
        self.hidden = torch.nn.ModuleList([layer(lookback, dim)])
        self.pools = torch.nn.ModuleList([TopKPooling(dim, ratio=0.8)])
        self.num_layers = num_layers
        for n in range(1, num_layers):
            self.hidden.append(layer(dim, dim))
            self.pools.append(TopKPooling(dim, ratio=0.8))
        self.lin1 = torch.nn.Linear(dim * 2, dim)
        self.lin2 = torch.nn.Linear(dim, output_size)
        self.act1 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Pad input and multiply by res_factor to calculate residual
        residual = self.res_factors[0] * F.pad(x, (0, self.dim - self.lookback), value=0)

        # Forward first layer
        x = residual + F.relu(self.hidden[0](x, edge_index))

        # Pooling
        x, edge_index, edge_attr, batch, _, _ = self.pools[0](x, edge_index, edge_attr, batch)
        summation = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Repeat for depth - 1
        for n in range(1, self.num_layers):

            # Update edges if there is a dropout layer
            if n in self.dropouts:
                edge_index, edge_attr = dropout_adj(edge_index, edge_attr=edge_attr, training=self.training)

            # Calculate residual
            residual = self.res_factors[n] * x

            # Forward nth layer
            x = residual + F.relu(self.hidden[n](x, edge_index))

            # Pooling
            x, edge_index, edge_attr, batch, _, _ = self.pools[n](x, edge_index, edge_attr, batch)
            summation += torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = summation

        # Pass through final linear transformations and activation
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return x


class GNNModule(torch.nn.Module):
    """
    Generalized Graph Neural Network whose parameters allow for the full range of testing,
    without the linear transformations and activation at the end, for use in cell of RNNs.
    Parameters:
        layer: torch.nn.Module - type of GNN from torch_geometric_temporal.nn to use (can also be a lambda function taking (input_channels, output_channels) and returning torch.nn.Module
        num_layers: int - number of repetitions of layer to use in sequence (depth of GNN)
        lookback: int - number of input node features
        output_size: int - number of nodes to predict
        dim: int - length of hidden embedding vectors
        res_factors: [int] - Array of length num_layers, containing coefficient to residual at corresponding layers
        dropouts: [int] - Indices of layers in which to include dropout during testing
        bias: bool
    """
    def __init__(self, layer, num_layers, lookback, dim=64, res_factors=None, dropouts=[], bias=True):
        super(GNNModule, self).__init__()

        self.dim = dim
        if res_factors is None:
            self.res_factors = [0.0] * num_layers
        else:
            self.res_factors = res_factors
        self.res_factors = torch.nn.Parameter(torch.randn(num_layers))
        self.dropouts = dropouts
        self.lookback = lookback
        self.hidden = torch.nn.ModuleList([layer(lookback, dim)])
        self.pools = torch.nn.ModuleList([TopKPooling(dim, ratio=0.8)])
        self.num_layers = num_layers
        for n in range(1, num_layers):
            self.hidden.append(layer(dim, dim))
            self.pools.append(TopKPooling(dim, ratio=0.8))

    def forward(self, x, edge_index, edge_attr=None, batch=None, residual=None):

        if residual is None:
            residual = torch.clone(x)
        # Pad input and multiply by res_factor to calculate residual
        res = torch.mul(F.pad(residual, (0, self.dim - self.lookback), value=0), self.res_factors[0])

        # Forward first layer
        x = res + F.relu(self.hidden[0](x, edge_index, edge_attr))

        #Pooling currently commented out as hotfix
        # x, edge_index, edge_attr, batch, _, _ = self.pools[0](x, edge_index, edge_attr, batch)
        # summation = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Repeat for depth - 1
        for n in range(1, self.num_layers):

            # Update edges if there is a dropout layer
            if n in self.dropouts:
                edge_index, edge_attr = dropout_adj(edge_index, edge_attr=edge_attr, training=self.training)

            # Calculate residual
            res = torch.mul(residual, self.res_factors[n])

            # Forward nth layer
            x = residual + F.relu(self.hidden[n](x, edge_index, edge_attr))

            #Pooling
            # x, edge_index, edge_attr, batch, _, _ = self.pools[n](x, edge_index, edge_attr, batch)
            # summation += torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = summation

        #Return x without final transformations and activation
        return x


class GraphNet(torch.nn.Module):
    """Vanilla GNN -
    No longer used, as it has been better abstracted to the GNN class.
    This is essentially the original best performer on the demand dataset.
    Parameters:
        layer: torch.nn.Module - type of GNN to use
        lookback: int - number of input node features
        output_size: int - number of nodes to predict
        dim: int - length of hidden embedding vectors
        res_factor: int - value of 0 indicates not to use residual, value of 1 indicates to use residual
    """
    def __init__(self, layer, lookback, output_size, dim=64, res_factor=0):
        super(GraphNet, self).__init__()

        self.dim = dim
        self.res_factor = res_factor
        self.lookback = lookback
        self.conv1 = layer(lookback, dim)
        self.pool1 = TopKPooling(dim, ratio=0.8)
        self.conv2 = layer(dim, dim)
        self.pool2 = TopKPooling(dim, ratio=0.8)
        self.lin1 = torch.nn.Linear(dim * 2, dim)
        self.lin2 = torch.nn.Linear(dim, output_size)
        self.act1 = torch.nn.ReLU()

    def forward(self, data, residual=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if residual is None:
            residual = torch.clone(x)
        x = torch.mul(F.pad(residual, (0, self.dim - self.lookback), value=0), self.res_factor) + F.relu(self.conv1(x, edge_index))
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
    """GNNs from PyTorch Geometric Temporal -
    This is the previous and incorrect implementation, kept for historical record.
    Parameters:
        layer: torch.nn.Module - type of GNN from torch_geometric_temporal.nn to use
        lookback: int - number of input node features
        output_size: int - number of nodes to predict
        dim: int - length of hidden embedding vectors
        filter_size: int - Chebyshev filter size
    """
<<<<<<< HEAD
    def __init__(self, layer, lookback=1, output_size=1, dim=128, filter_size=1, rnn_depth=1, name=None):
=======
    def __init__(self, layer, lookback=1, output_size=1, dim=128, filter_size=1, rnn_depth=1):
>>>>>>> 2a8a690ed5b1c10139c198e6939a4579af3a2410
        super(RecurrentGraphNet, self).__init__()

        self.layer = layer
        self.rnn_depth = rnn_depth
        self.filter_size = filter_size
        self.recurrent = layer(lookback, dim, filter_size)
<<<<<<< HEAD
        if name is None:
            self.name = layer.__name__
        else:
            self.name = name
=======
>>>>>>> 2a8a690ed5b1c10139c198e6939a4579af3a2410
        if type(self.recurrent) is GConvLSTM or type(self.recurrent) is GCLSTM:
            self.has_c = True
        else:
            self.has_c = False
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin2 = torch.nn.Linear(dim, output_size)
        self.act1 = torch.nn.ReLU()

    def forward(self, data, h=None, c=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.rnn_depth):
            if self.has_c:
                h, c = self.recurrent(x, edge_index, edge_attr.reshape([edge_attr.shape[0]]), h, c)
            else:
                h = self.recurrent(x, edge_index, edge_attr.reshape([edge_attr.shape[0]]), h)
        x = F.relu(h)
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)

        return x, h, c


class LagPredictor(torch.nn.Module):
    """GNNs from PyTorch Geometric Temporal -
    This is the previous and incorrect implementation, kept for historical record.
    Parameters:
        layer: torch.nn.Module - type of GNN from torch_geometric_temporal.nn to use
        lookback: int - number of input node features
        output_size: int - number of nodes to predict
        dim: int - length of hidden embedding vectors
        filter_size: int - Chebyshev filter size
    """
    def __init__(self):
        super(LagPredictor, self).__init__()
<<<<<<< HEAD
        self.name = "Lag"
=======
>>>>>>> 2a8a690ed5b1c10139c198e6939a4579af3a2410

    def forward(self, data, h=None, c=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        return x[:, -1], h, c