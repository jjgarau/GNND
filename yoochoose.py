import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import MessagePassing, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from tqdm import tqdm


embed_dim = 128


class SAGEConv(MessagePassing):
	def __init__(self, in_channels, out_channels):
		super(SAGEConv, self).__init__(aggr='max')
		self.lin = torch.nn.Linear(in_channels, out_channels)
		self.act = torch.nn.ReLU()
		self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
		self.update_act = torch.nn.ReLU()

	def forward(self, x, edge_index):
		# x has shape [N, in_channels]
		# edge_index has shape [2, E]
		edge_index, _ = remove_self_loops(edge_index)
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

	def message(self, x_j):
		# x_j has shape [E, in_channels]
		x_j = self.lin(x_j)
		x_j = self.act(x_j)
		return x_j

	def update(self, aggr_out, x):
		# aggr_out has shape [N, out_channels]
		new_embedding = torch.cat([aggr_out, x], dim=1)
		new_embedding = self.update_lin(new_embedding)
		new_embedding = self.update_act(new_embedding)
		return new_embedding


class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.conv1 = SAGEConv(embed_dim, 128)
		self.pool1 = TopKPooling(128, ratio=0.8)
		self.conv2 = SAGEConv(128, 128)
		self.pool2 = TopKPooling(128, ratio=0.8)
		self.conv3 = SAGEConv(128, 128)
		self.pool3 = TopKPooling(128, ratio=0.8)
		self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() + 1, embedding_dim=embed_dim)
		self.lin1 = torch.nn.Linear(256, 128)
		self.lin2 = torch.nn.Linear(128, 64)
		self.lin3 = torch.nn.Linear(64, 1)
		self.bn1 = torch.nn.BatchNorm1d(128)
		self.bn2 = torch.nn.BatchNorm1d(64)
		self.act1 = torch.nn.ReLU()
		self.act2 = torch.nn.ReLU()

	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch
		x = self.item_embedding(x)
		x = x.squeeze(1)
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
		x = self.act2(x)
		x = F.dropout(x, p=0.5, training=self.training)
		x = torch.sigmoid(self.lin3(x)).squeeze(1)
		return x


class YooChooseBinaryDataset(InMemoryDataset):
	def __init__(self, root, transform=None, pre_transform=None):
		super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return []

	@property
	def processed_file_names(self):
		return ['yoochoose_click_binary_1M_sess.dataset']

	def download(self):
		pass

	def process(self):
		data_list = []

		# process by session_id
		grouped = df.groupby('session_id')
		for session_id, group in tqdm(grouped):
			sess_item_id = LabelEncoder().fit_transform(group.item_id)
			group = group.reset_index(drop=True)
			group['sess_item_id'] = sess_item_id
			node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
				'sess_item_id').item_id.drop_duplicates().values

			node_features = torch.LongTensor(node_features).unsqueeze(1)
			target_nodes = group.sess_item_id.values[1:]
			source_nodes = group.sess_item_id.values[:-1]

			edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
			x = node_features
			y = torch.FloatTensor([group.label.values[0]])

			data = Data(x=x, edge_index=edge_index, y=y)
			data_list.append(data)

		data, slices = self.collate(data_list)
		torch.save((data, slices), self.processed_paths[0])


def train():
	model.train()
	loss_all = 0
	for data in train_loader:
		data = data.to(device)
		optimizer.zero_grad()
		output = model(data)
		label = data.y.to(device)
		loss = crit(output, label)
		loss.backward()
		loss_all += data.num_graphs * loss.item()
		optimizer.step()
	return loss_all / len(train_dataset)


def evaluate(loader):
	model.eval()
	predictions, labels = [], []
	with torch.no_grad():
		for data in loader:
			data = data.to(device)
			pred = model(data).detach().cpu().numpy()
			label = data.y.detach().cpu().numpy()
			predictions.append(pred)
			labels.append(label)
	predictions = np.hstack(predictions)
	labels = np.hstack(labels)
	return roc_auc_score(labels, predictions)


if __name__ == "__main__":
	df = pd.read_csv('yoochoose-data/yoochoose-clicks.dat', header=None)
	df.columns = ['session_id', 'timestamp', 'item_id', 'category']

	buy_df = pd.read_csv('yoochoose-data/yoochoose-buys.dat')
	buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

	item_encoder = LabelEncoder()
	df['item_id'] = item_encoder.fit_transform(df.item_id)

	# randomly sample a couple of them
	sample = 10000
	sampled_session_id = np.random.choice(df.session_id.unique(), sample, replace=False)
	df = df.loc[df.session_id.isin(sampled_session_id)]
	df['label'] = df.session_id.isin(buy_df.session_id)
	print('Dataset preprocessed')

	dataset = YooChooseBinaryDataset(root='yoochoose-data/')
	dataset = dataset.shuffle()

	train_dataset = dataset[:int(0.8 * sample)]
	val_dataset = dataset[int(0.8 * sample):int(0.9 * sample)]
	test_dataset = dataset[int(0.9 * sample):]

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Net().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
	crit = torch.nn.BCELoss()

	batch_size = 1024
	train_loader = DataLoader(train_dataset, batch_size=batch_size)
	val_loader = DataLoader(val_dataset, batch_size=batch_size)
	test_loader = DataLoader(test_dataset, batch_size=batch_size)

	num_epochs = 5
	for epoch in range(num_epochs):
		loss = train()
		train_acc = evaluate(train_loader)
		val_acc = evaluate(val_loader)
		test_acc = evaluate(test_loader)
		print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.format(epoch, loss, train_acc, val_acc, test_acc))
