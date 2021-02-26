import json
import os
from matplotlib import pyplot as plt
from numpy import array as nparray
import countryinfo
import pandas as pd
import datetime

def compare_models():
    files = os.listdir('results')

    losses = {}

    for file in files:
        with open('results/' + file, 'r') as f:
            data = json.load(f)

            for model, model_data in data['Models'].items():
                if not model in losses:
                    losses[model] = {
                        'Train': [],
                        'Train Evaluation': [],
                        'Validation': [],
                        'Test': []
                    }
                for segment, loss in model_data['Loss by Epoch'][-1].items():
                    losses[model][segment].append(loss)
    for model, model_data in losses.items():
        for segment, segment_losses in model_data.items():
            losses[model][segment] = "%.3f" % (sum(segment_losses) / len(segment_losses))

    cellText = list(map(lambda x: list(x.values()), list(losses.values())))

    rows = list(losses.keys())
    columns = ['Train', 'Train Evaluation', 'Validation', 'Test']
    table = plt.table(
        cellText=cellText,
        rowLabels=rows,
        colLabels=columns,
        loc='center'
    )
    table.scale(1, 1.5)

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)
    plt.title("Comparison of Models on Week Ahead COVID Case Prediction")

    plt.show()

    models = list(losses.keys())
    for i in range(len(cellText)):
        cellText[i].insert(0, models[i])
    df = pd.DataFrame(cellText)
    df.columns = ["Model"] + columns
    df.to_csv("reports/Loss By Epoch Report - " + datetime.datetime.now().isoformat().split(".")[0] + ".csv")

def loss_by_country():
    files = os.listdir('results')
    files = ['results2021-02-22T12:01:45.json']
    losses = {}

    for file in files:
        with open('results/' + file, 'r') as f:
            data = json.load(f)

            for model, model_data in data['Models'].items():
                if not model in losses:
                    losses[model] = {}
                    for country in model_data['Loss by Country'].keys():
                        losses[model][country] = []
                for country, country_losses in model_data['Loss by Country'].items():
                    losses[model][country] += country_losses
    for model, model_data in losses.items():
        for country, country_losses in model_data.items():
            losses[model][country] = "%.3f" % (sum(country_losses) / len(country_losses))

    losses["Population"] = {}
    countries = list(model_data.keys())
    for country in countries:
        try:
            losses["Population"][country] = countryinfo.CountryInfo(country).population()
        except KeyError:
            losses["Population"][country] = "?"


    losses["Total Cases"] = {}
    df = pd.read_csv("df.csv")
    df = dict(df.sum())
    for country in countries:
        losses["Total Cases"][country] = df[country + "_new_cases"]

    cellText = list(map(lambda x: list(x.values()), list(losses.values())))
    cellText = nparray(cellText).T.tolist()
    countries = list(list(losses.values())[0].keys())


    columns = list(losses.keys())
    table = plt.table(
        cellText=cellText,
        rowLabels=countries,
        colLabels=columns,
        loc='center'
    )
    table.scale(0.4, 0.6)

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)

    plt.show()


    for i in range(len(cellText)):
        cellText[i].insert(0, countries[i])
    df = pd.DataFrame(cellText)
    df.columns = ["Country"] + columns
    df.to_csv("reports/Loss By Countries Report - " + datetime.datetime.now().isoformat().split(".")[0] + ".csv")

if __name__ == "__main__":
    loss_by_country()