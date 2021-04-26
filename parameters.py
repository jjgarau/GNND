import argparse
import json
import torch
from util import *
from weight_sage import WeightedSAGEConv
import graph_nets
from rnn import *

# CONSTRUCT MODELS
WSC = WeightedSAGEConv
USC = lambda in_channels, out_channels, bias=True: WeightedSAGEConv(in_channels, out_channels, weighted=False)
linear_module = lambda in_channels, out_channels, bias: graph_nets.GraphLinear(in_channels, out_channels, bias=bias)
DeepUSC = lambda lookback, dim: graph_nets.GNNModule(USC, 3, lookback, dim=dim, res_factors=[1, 0, 1], dropouts=[1])
DeepWSC = lambda lookback, dim: graph_nets.GNNModule(WSC, 3, lookback, dim=dim, res_factors=None, dropouts=[])

args = {
  # Number of previous timesteps to use for prediction
  "lookback": 5,

  # Pattern of previous timesteps to use for prediction
  "lookback_pattern": [11, 10, 9, 8, 7],

  # Number of edges in the graph - this was a hotfix, needs to be deleted/resolved
  "edge_count": 0,

  # Number of folds in K-fold cross validation
  "K": 5,

  # Should perform K-fold cross validation instead of normal training
  "CROSS_VALIDATE": False,

  # Threshold for creation of edges based on geodesic distance
  "DISTANCE_THRESHOLD": 250,  # km

  # Minimum number of edges per node in graph
  "EDGES_PER_NODE": 3,

  # Name of loss function to be used in training
  "loss_func": mase2_loss,

  # Name of reporting metric
  "reporting_metric": mase1_loss,

  # A description to be included in the results output file
  "experiment_description": "DESCRIPTION NOT FOUND",

  # Number of epochs to train models
  "num_epochs": 100,

  # Name of optimizer used in training
  "optimizer": torch.optim.Adam,

  # Learning rate of optimizer used in training
  "learning_rate": 0.05,

  # Percentage of dataset to use for training (less than 1.0 to speed up training)
  "sample": 1.0,

  # Features to train on
  "features": ["new_cases",
                "new_deaths",
                # "icu_patients",
                # "tests_per_case",
                # "stringency_index",

                # "reproduction_rate",
                # "icu_patients", "hosp_patients", "weekly_icu_admissions",
                # "weekly_hosp_admissions", "new_tests", "tests_per_case",
                # "positive_rate", "stringency_index",
                # "population", "population_density", "median_age",
                # "aged_65_older", "aged_70_older", "gdp_per_capita",
                # "extreme_poverty", "cardiovasc_death_rate", "diabetes_prevalence",
                # "female_smokers", "male_smokers", "handwashing_facilities",
                # "hospital_beds_per_thousand", "life_expectancy", "human_development_index"
                ],
  "models": []
}

models = [
    graph_nets.LagPredictor(),
    RNN(module=WSC, gnn=WSC, rnn=LSTM, dim=128, gnn_2=WSC, rnn_depth=1, name="Our Model", edge_count=args['edge_count'],
        node_features=len(args['features'])),
    # graph_nets.RecurrentGraphNet(GConvLSTM),
    # graph_nets.RecurrentGraphNet(GConvGRU),
    # graph_nets.RecurrentGraphNet(DCRNN),
    # graph_nets.RecurrentGraphNet(GCLSTM),
    ]

args['models'] = models

class Parameters:
    def __init__(self):
        # parser = argparse.ArgumentParser('Recurrent GNN COVID Prediction')
        #
        # try:
        #     args = parser.parse_args()
        #     with open('parameters.json', 'rt') as f:
        #         t_args = argparse.Namespace()
        #         t_args.__dict__.update(json.load(f))
        #         args = parser.parse_args(namespace=t_args)
        # except:
        #     parser.print_help()

        self.lookback  = args['lookback']
        self.lookback_pattern = args['lookback_pattern']
        self.edge_count = args['edge_count']
        self.K = args['K']
        self.CROSS_VALIDATE = args['CROSS_VALIDATE']
        self.DISTANCE_THRESHOLD = args['DISTANCE_THRESHOLD']
        self.EDGES_PER_NODE = args['EDGES_PER_NODE']
        self.experiment_description = args['experiment_description']
        self.num_epochs = args['num_epochs']
        self.learning_rate = args['learning_rate']
        self.sample = args['sample']
        self.features = args['features']

        self.loss_func = args['loss_func']
        self.reporting_metric = args['reporting_metric']
        self.optimizer = args['optimizer']
        self.models = args['models']
    
    def get_optimizer(self, model_params):
        return self.optimizer(model_params, self.learning_rate)


# loss_func/reporting_metric, models, and optimizer (get_optimizer()?) need special initializing or maybe you could change the json to a python file
