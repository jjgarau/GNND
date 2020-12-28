import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import pandas as pd
import itertools
from tqdm import tqdm
from util import *
import graph_nets
from torch_geometric.nn import SAGEConv, LEConv
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import bisect
from torch_geometric_temporal.nn import DCRNN, GConvLSTM, GConvGRU


#Testing for COVID Dataset


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


lookback = 5

class COVIDDataset(InMemoryDataset):
    """
    Each snapshot in this dataset is a graph with
    node features being new cases in the country from previous 5 dates,
    edges being between any countries within threshold distance, or else a min number of connections per country,
    and edge weights being geodesic distance between land mass centroids of the countries
    """
    def __init__(self, root, transform=None, pre_transform=None, should_visualize_data=False):
        super(COVIDDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.should_visualize_data = should_visualize_data

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['covid_dataset.dataset']

    def download(self):
        pass

    def process(self):

        # Determine edge index

        DISTANCE_THRESHOLD = 250 #km
        EDGES_PER_NODE = 3

        data_list = []
        n = len(nations)
        source_nodes = []
        target_nodes = []
        edge_attrs = []
        for i in range(n):
            c1 = country_centroids[nations[i]]
            distances = []
            countries = []
            for j in range(n):
                c2 = country_centroids[nations[j]]
                dist = geodesic(c1, c2)
                index = bisect.bisect(distances, dist)
                if index < EDGES_PER_NODE:
                    distances.insert(index, dist.km)
                    countries.insert(index, j)
                    # if distances[len(distances) - 1] > DISTANCE_THRESHOLD:
                    #     distances.pop()
                    #     countries.pop() #Uncomment to create edge between all countries within a distance threshold, or at least a minimum of EDGES_PER_NODE nearest countries
            source_nodes += [i]*EDGES_PER_NODE
            target_nodes += countries[:EDGES_PER_NODE]
            edge_attrs += distances[:EDGES_PER_NODE]



        torch_def = torch.cuda if torch.cuda.is_available() else torch
        for i in tqdm(range(len(df) - lookback)):
            # Node Features
            values_x = df.iloc[i:(i + lookback), 1:].to_numpy().T
            x = torch_def.FloatTensor(values_x)

            # Labels
            values_y = df.iloc[(i + lookback):(i + lookback + 1), 1:].to_numpy().T
            y = torch_def.FloatTensor(values_y)

            # Edge Index
            edge_index = torch_def.LongTensor([source_nodes.copy(), target_nodes.copy()])
            # Edge Weights
            edge_attr = torch_def.FloatTensor([[weight] for weight in edge_attrs])


            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # Optionally, visualize the graph on a map
        if self.should_visualize_data:
            # Create Basemap
            fig = plt.figure(figsize=(8, 8))
            # For Europe dataset
            m = Basemap(projection='lcc', resolution=None, lat_0=30, lon_0=-20,
                        llcrnrlat=30, llcrnrlon=-20, urcrnrlat=70, urcrnrlon=80)
            # For global data
            # m = Basemap(projection='cyl', resolution=None,
            #             llcrnrlat=-90, urcrnrlat=90,
            #             llcrnrlon=-180, urcrnrlon=180, )
            draw_map(m)

            #Collect coordinates
            coordinates = [country_centroids[nations[i]] for i in source_nodes]
            coordinates2 = [country_centroids[nations[i]] for i in target_nodes]
            for i in range(len(coordinates)):
                m.drawgreatcircle(coordinates[i][1], coordinates[i][0], coordinates2[i][1], coordinates2[i][0])

            #Draw coordinates on map
            m.scatter([centroid[1] for centroid in country_centroids.values()],
                      [centroid[0] for centroid in country_centroids.values()], latlon=True, zorder=10)
            plt.show()


def gnn_predictor():

    #Load and split dataset

    dataset = COVIDDataset(root='data/covid-data/')
    dataset = dataset.shuffle()

    sample = len(dataset)
    # Optionally, make dataset smaller for quick testing
    sample *= 1.0
    train_dataset = dataset[:int(0.8 * sample)]
    val_dataset = dataset[int(0.8 * sample):int(0.9 * sample)]
    test_dataset = dataset[int(0.9 * sample):int(sample)]



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_func = mase_loss


    #Design models to test
    models = [
        graph_nets.GNN(SAGEConv, 3, lookback, len(nations), dim=128, res_factors=[0.0, 1.0, 0.0, 0.0, 1.0])
    ]

    for i in range(len(models)):

        model = models[i]
        print(model)


        optimizer = torch.optim.Adam(model.parameters(), lr=0.005*(1+i))


        #Split dataset into batches
        batch_size = 1
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)


        # Evaluate losses for each epoch
        num_epochs = 100
        val_losses = []
        for epoch in range(num_epochs):
            loss = train_gnn(model, train_loader, optimizer, loss_func, device)
            loss /= len(train_dataset)
            train_acc = evaluate_gnn(model, train_loader, device)
            val_acc = evaluate_gnn(model, val_loader, device)
            test_acc = evaluate_gnn(model, test_loader, device)
            val_losses.append(val_acc)
            print('Epoch: {:03d}, Loss: {:.5f}, Train Loss: {:.5f}, Val Loss: {:.5f}, Test Loss: {:.5f}'.format(epoch, loss,
                                                                                                                train_acc,
                                                                                                                val_acc,
                                                                                                                     test_acc))
        # Set labels and display loss curves for validation
        x = np.arange(0, num_epochs)
        labels = ["SOTA", "GConvLSTM"]
        plt.title('Recurrent Networks')
        plt.xlabel('Epoch')
        plt.ylabel('MASE Loss')
        plt.plot(x, val_losses, label=str(labels[i]))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #Get country centroids data
    df2 = pd.read_csv("country_centroids.csv")
    columns = ['name_long', 'Longitude', 'Latitude', 'continent']
    df2 = df2.filter(columns)
    # Choose only a single continent for smaller testing (can choose any continent)
    df2 = df2[df2.continent == 'Europe']


    df = pd.read_csv('data/covid-data/covid-19-world-cases-deaths-testing.csv')
    columns = ['location', 'date', 'new_cases']
    df = df.filter(columns)
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)

    df = df[df.location.isin(df2.name_long.values)]
    nations = df.location.unique()
    nations = np.delete(nations, [len(nations)-1, len(nations)-2]) #Remove World and International columns

    #Commented out unless creating the dataset again
    # dates = df.date.unique()
    # new_data = {'Time': range(len(dates))}
    # for i in range(len(nations)):
    #     nation = nations[i]
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
    #
    # print('Dataset preprocessed')
    # df.to_csv("df.csv")
    # print(df.head())
    # print(df.columns)
    # print(df.shape)

    #Get centroid of each country
    country_centroids = {}
    for nation in nations:
        match = df2.loc[df2.name_long == nation]
        if len(match):
            lon = match.Longitude.values[0]
            lat = match.Latitude.values[0]
            country_centroids[nation] = (lat, lon)
        else:
            print("Missing coordinates for country", nation)

    #Run training, validation, and testing on the dataset
    gnn_predictor()