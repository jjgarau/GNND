import torch
from torch.nn import Parameter
from weight_sage import WeightedSAGEConv
from torch_geometric.nn.inits import glorot, zeros
import torch.nn.functional as F
from torch.nn import LSTMCell, GRUCell, RNNCell, LSTM as TorchLSTM
from graph_nets import GraphLinear
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import ASAPooling, TopKPooling, EdgePooling, SAGPooling
from torch_geometric_temporal.nn import DCRNN, GConvLSTM, GConvGRU
from torch.nn.init import xavier_uniform

#Recurrent Neural Network Modules

class LSTM(torch.nn.Module):
    # This is an adaptation of torch_geometric_temporal.nn.GConvLSTM, with ChebConv replaced by the given model.
    """
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        module (torch.nn.Module, optional): The layer or set of layers used to calculate each gate.
            Could also be a lambda function returning a torch.nn.Module when given the parameters in_channels: int, out_channels: int, and bias: bool
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool=True, module=WeightedSAGEConv):
        super(LSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.module = module
        self._create_parameters_and_layers()
        self._set_parameters()


    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = self.module(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)

        self.conv_h_i = self.module(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))


    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = self.module(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)

        self.conv_h_f = self.module(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))


    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = self.module(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)

        self.conv_h_c = self.module(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))


    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = self.module(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)

        self.conv_h_o = self.module(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))


    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()


    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)


    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H


    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels)
        return C


    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C):
        I = self.conv_x_i(X, edge_index, edge_weight)
        I = I + self.conv_h_i(H, edge_index, edge_weight)
        I = I + (self.w_c_i*C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I


    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C):
        F = self.conv_x_f(X, edge_index, edge_weight)
        F = F + self.conv_h_f(H, edge_index, edge_weight)
        F = F + (self.w_c_f*C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F


    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F):
        T = self.conv_x_c(X, edge_index, edge_weight)
        T = T + self.conv_h_c(H, edge_index, edge_weight)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F*C + I*T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C):
        O = self.conv_x_o(X, edge_index, edge_weight)
        O = O + self.conv_h_o(H, edge_index, edge_weight)
        O = O + (self.w_c_o*C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O


    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H


    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None,
                H: torch.FloatTensor=None, C: torch.FloatTensor=None) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C)
        H = self._calculate_hidden_state(O, C)
        return H, C



class GRU(torch.nn.Module):
    #This is an adaptation of torch_geometric_temporal.nn.GConvGRU, with ChebConv replaced by the given model.
    r"""An implementation of the Chebyshev Graph Convolutional Gated Recurrent Unit
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        module (torch.nn.Module, optional): The layer or set of layers used to calculate each gate.
            Could also be a lambda function returning a torch.nn.Module when given the parameters in_channels: int, out_channels: int, and bias: bool
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool=True, module=WeightedSAGEConv):
        super(GRU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.module = module
        self._create_parameters_and_layers()


    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = self.module(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)

        self.conv_h_z = self.module(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)


    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = self.module(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)

        self.conv_h_r = self.module(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)


    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = self.module(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)

        self.conv_h_h = self.module(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 bias=self.bias)


    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()


    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels)
        return H


    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = self.conv_x_z(X, edge_index, edge_weight)
        Z = Z + self.conv_h_z(H, edge_index, edge_weight)
        Z = torch.sigmoid(Z)
        return Z


    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = self.conv_x_r(X, edge_index, edge_weight)
        R = R + self.conv_h_r(H, edge_index, edge_weight)
        R = torch.sigmoid(R)
        return R


    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight)
        H_tilde = H_tilde + self.conv_h_h(H*R, edge_index, edge_weight)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde


    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z*H + (1-Z)*H_tilde
        return H


    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None, C: torch.FloatTensor=None) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H, C

class VanillaRNN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool=True, module=WeightedSAGEConv):
        super(VanillaRNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.module = module
        self.bias = bias

        #Hidden input
        self.conv_h_i = self.module(in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    bias=self.bias)

        #Hidden hidden
        self.conv_h_h = self.module(in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    bias=self.bias)

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None, C: torch.FloatTensor=None):
        input = self.conv_h_i(X, edge_index, edge_weight)
        hidden = self.conv_h_h(H, edge_index, edge_weight)
        H = torch.tanh(input + hidden)

        return H, C


class RNN(torch.nn.Module):
    """
    Base class for Recurrent Neural Networks (LSTM or GRU).
    Initialization to this class contains all variables for variation of the model.
    Consists of one of the above RNN architectures followed by an optional GNN on the final hidden state.
    Parameters:
        node_features: int - number of features per node
        output: int - length of the output vector on each node
        dim: int - number of features of embedding for each node
        module: torch.nn.Module - to be used in the LSTM to calculate each gate
    """
    def __init__(self, node_features=1, output=1, dim=32, module=GraphLinear, rnn=LSTM, gnn=WeightedSAGEConv, gnn_2=WeightedSAGEConv, rnn_depth=1):
        super(RNN, self).__init__()
        self.dim = dim
        self.rnn_depth = rnn_depth

        if gnn:
            self.gnn = gnn(node_features, dim)
            if rnn:
                self.recurrent = rnn(dim, dim, module=module)
            else:
                self.recurrent = None
        else:
            self.gnn = None
            if rnn:
                self.recurrent = rnn(node_features, dim, module=module)
            else:
                self.recurrent = None
        if gnn_2:
            if gnn:
                self.gnn_2 = gnn_2(2 * dim, 2 * dim)
            else:
                self.gnn_2 = gnn_2(dim + node_features, 2 * dim)
        else:
            self.gnn_2 = None

        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, output)
        self.act1 = torch.nn.ReLU()

    def forward(self, data, h=None, c=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.gnn:
            x = self.gnn(x, edge_index, edge_attr)
            x = F.relu(x)

        if h is None:
            h = torch.zeros(x.shape[0], self.dim)
        if c is None:
            c = torch.zeros(x.shape[0], self.dim)

        if self.recurrent:
            for i in range(self.rnn_depth):
                h, c = self.recurrent(x, edge_index, edge_attr, h, c)

        x = torch.cat((x, h), 1)
        if self.gnn_2:
            x = self.gnn_2(x, edge_index, edge_attr)
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)

        return x, h, c



class PGT_DCRNN(torch.nn.Module):
    def __init__(self, node_features, dim=16):
        super(PGT_DCRNN, self).__init__()
        self.recurrent = DCRNN(node_features, dim, 1)
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, data, h=None, c=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.recurrent(x, edge_index, edge_attr, h)
        x = F.relu(h)
        x = self.linear(x)
        return x, h, None

class PGT_GConvLSTM(torch.nn.Module):
    def __init__(self, node_features, dim=16):
        super(PGT_GConvLSTM, self).__init__()
        self.recurrent = GConvLSTM(node_features, dim, 1)
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, data, h=None, c=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_attr = torch.FloatTensor([attr[0] for attr in edge_attr])
        h, c = self.recurrent(x, edge_index, edge_attr, h)
        x = F.relu(h)
        x = self.linear(x)
        return x, h, c

class PGT_GConvGRU(torch.nn.Module):
    def __init__(self, node_features, dim=16):
        super(PGT_GConvGRU, self).__init__()
        self.recurrent = GConvGRU(node_features, dim, 1)
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, data, h=None, c=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_attr = torch.FloatTensor([attr[0] for attr in edge_attr])
        h = self.recurrent(x, edge_index, edge_attr, h)
        x = F.relu(h)
        x = self.linear(x)
        return x, h, None


class SimpleRNN(torch.nn.Module):

    def __init__(self, node_features=1, output=1, dim=32, module=GraphLinear, rnn=LSTM, rnn_depth=1):
        super(SimpleRNN, self).__init__()
        self.dim = dim
        self.rnn_depth = rnn_depth

        self.recurrent = rnn(node_features, dim, module=module)

        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin2 = torch.nn.Linear(dim, output)
        self.act1 = torch.nn.ReLU()

    def forward(self, data, h=None, c=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i in range(self.rnn_depth):
            h, c = self.recurrent(x, edge_index, edge_attr, h, c)

        x = self.lin1(h)
        x = self.act1(x)
        x = self.lin2(x)

        return x, h, c