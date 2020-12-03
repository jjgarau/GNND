from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.data.splitter import discrete_train_test_split

loader = ChickenpoxDatasetLoader()

dataset = loader.get_dataset()

train_dataset, test_dataset = discrete_train_test_split(dataset, train_ratio=0.2)




import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        print(x.shape, edge_index.shape, edge_weight.shape) #[20, 4] [2, 102] [102]
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h



model = RecurrentGCN(node_features = 4)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for time, snapshot in enumerate(train_dataset):
    print("Time: ", time)
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    print("PREDICTED[0]: ", y_hat[0])
    print("----------------------------------------")