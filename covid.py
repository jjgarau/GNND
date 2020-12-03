import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import pandas as pd
import itertools
from tqdm import tqdm
import random
from common import *
import graph_nets
from torch_geometric.nn import SAGEConv

lookback = 5

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
        data_list = []
        combs = list(itertools.combinations(range(len(nations)), 2))
        combs = random.choices(combs, k=len(nations)*3)
        source_nodes = [e[0] for e in combs]
        target_nodes = [e[1] for e in combs]
        torch_def = torch.cuda if torch.cuda.is_available() else torch
        for i in tqdm(range(len(df) - lookback)):
            # Node Features
            values_x = df.iloc[i:(i + lookback), 1:].to_numpy().T
            x = torch_def.FloatTensor(values_x)

            # Labels
            values_y = df.iloc[(i + lookback):(i + lookback + 1), 1:].to_numpy().T
            y = torch_def.FloatTensor(values_y)

            # Edge Features
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def gnn_predictor():
    dataset = COVIDDataset(root='data/covid-data/')
    dataset = dataset.shuffle()

    sample = len(dataset)
    # Make dataset smaller for quick testing
    sample *= 1.0
    train_dataset = dataset[:int(0.8 * sample)]
    val_dataset = dataset[int(0.8 * sample):int(0.9 * sample)]
    test_dataset = dataset[int(0.9 * sample):int(sample)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_func = mape_loss
    dim = 64
    models = [graph_nets.GraphNet(SAGEConv, lookback, len(nations))]

    for model in models: #Grid search loop

        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        batch_size = 256
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
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(x, val_losses, label=str(model).split('(')[0])
    plt.legend()
    plt.show()
if __name__ == '__main__':
    df = pd.read_csv('data/covid-data/covid-19-world-cases-deaths-testing.csv')
    columns = ['location', 'date', 'new_cases']
    df = df.filter(columns)
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)

    nations = df.location.unique()
    # dates = df.date.unique()
    # new_data = dict()
    # for nation in nations:
    #     nation_data = df.loc[df.location == nation]
    #     new_cases = []
    #     last_value = 0.0
    #     for date in dates:
    #         date_row = nation_data.loc[nation_data.date == date]
    #         if not date_row.empty:
    #             new_cases.append(date_row.new_cases.iloc[0])
    #             last_value = date_row.iloc[0].new_cases
    #         else:
    #             new_cases.append(last_value)
    #     new_data[nation + '_new_cases'] = new_cases
    # df = pd.DataFrame(data=new_data)

    # print('Dataset preprocessed')
    #
    # print(df.head())
    # print(df.columns)
    # print(df.shape)

    gnn_predictor()