import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import SAGEConv, GCNConv, GENConv, GatedGraphConv, GraphConv, HypergraphConv, LEConv, SGConv, TAGConv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader
import graph_nets
from common import *
from torch_geometric_temporal import GConvGRU, GConvLSTM, GCLSTM, DCRNN

#Original file for testing the Chickenpox dataset - later moved to torch_geometric_temporal_example.py

lookback = 5
output_size = 20

class ChickenpoxDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ChickenpoxDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['chickenpox_dataset.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        torch_def = torch.cuda if torch.cuda.is_available() else torch
        for i in tqdm(range(df.snapshot_count - lookback)):
            #Node Features
            values_x = df.features[i:(i+lookback)]
            x = torch_def.FloatTensor(values_x).transpose(0, 1)[:, :, 0]

            #Labels
            values_y = df.targets[(i+lookback):(i+lookback+1)]
            y = torch_def.FloatTensor(values_y).transpose(0, 1)

            #Edge Features
            edge_index = torch_def.LongTensor(df.edge_index)
            edge_attr = torch_def.FloatTensor(df.edge_weight)
            edge_attr = edge_attr.reshape([edge_attr.shape[0], 1])

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])




def gnn_predictor():
    dataset = ChickenpoxDataset(root='data/chickenpox-data/')
    dataset = dataset.shuffle()

    sample = len(dataset)
    # Make dataset smaller for quick testing
    sample *= 1.0
    train_dataset = dataset[:int(0.8 * sample)]
    val_dataset = dataset[int(0.8 * sample):int(0.9 * sample)]
    test_dataset = dataset[int(0.9 * sample):int(sample)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_func = mape_loss
    loss_funcs = [mape_loss, mse_loss]


    layers = [SAGEConv, LEConv, TAGConv]
    recurrent_layers = [GConvLSTM, DCRNN, GConvGRU]

    models = []

    for layer in recurrent_layers:
        models.append(graph_nets.RecurrentGraphNet(layer, lookback, output_size))
    for layer in layers:
        models.append(graph_nets.GraphNet(layer, lookback, output_size))

    model = graph_nets.GraphNet(SAGEConv, lookback, output_size)
    for loss_func in loss_funcs: #Grid search loop

        print(model)
        batch_size = 16
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
            print('Epoch: {:03d}, Loss: {:.5f}, Train MAPE: {:.5f}, Val MAPE: {:.5f}, Test MAPE: {:.5f}'.format(epoch, loss,
                                                                                                                train_acc,
                                                                                                                val_acc,
                                                                                                                  test_acc))
        x = np.arange(0, num_epochs)
        plt.title("Chickenpox MSE vs. MAPE Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        lbl = "?"
        if type(model) is graph_nets.GraphNet:
            lbl = str(model.conv1)
        elif type(model) is graph_nets.RecurrentGraphNet:
            lbl = str(model.recurrent)
        plt.plot(x, val_losses, label=str(loss_func))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    loader = ChickenpoxDatasetLoader()
    df = loader.get_dataset()

    gnn_predictor()