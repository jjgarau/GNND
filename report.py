import json
import os
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import array as nparray
import countryinfo
import pandas as pd
import datetime
import folium
import torch
import numpy as np

def compare_models():
    """
    Output a table showing final train/val/test loss by model.
    """
    files = os.listdir('results')
    files = ['results2021-03-04T16:05:09.json',
             'results2021-03-04T16:10:42.json',
             'results2021-03-04T16:11:19.json',
             'results2021-03-04T16:11:28.json']
    losses = {}

    # Average final losses across several results files to get a more accurate average...
    for file in files:
        with open('results/' + file, 'r') as f:
            data = json.load(f)

            # For each model in the file...
            for model, model_data in data['Models'].items():

                # Setup keys for new models
                if not model in losses:
                    losses[model] = {
                        'Train': [],
                        'Train Evaluation': [],
                        'Validation': [],
                        'Test': []
                    }

                # Append the loss for each segment
                for segment, loss in model_data['Loss by Epoch'][-1].items():
                    losses[model][segment].append(loss)

    # Calculate the average loss per segment over all the files
    for model, model_data in losses.items():
        for segment, segment_losses in model_data.items():
            losses[model][segment] = "%.3f" % (sum(segment_losses) / len(segment_losses))

    # Display in a pyplot table
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

    # Output the table as a CSV file
    models = list(losses.keys())
    for i in range(len(cellText)):
        cellText[i].insert(0, models[i])
    df = pd.DataFrame(cellText)
    df.columns = ["Model"] + columns
    df.to_csv("reports/Loss By Epoch Report - " + datetime.datetime.now().isoformat().split(".")[0] + ".csv")


def draw_map(data):
    url = (
        "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
    )
    state_geo = 'europe.json'

    m = folium.Map(location=[48, -102], zoom_start=3, tiles='https://api.mapbox.com/styles/v1/sestinj/cko1roam616pw18pgzg1kv1yh/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1Ijoic2VzdGluaiIsImEiOiJjanAzYjF6bWcwNXl4M3Bxd2xzdDFyZ2JlIn0.hjznaFqS-tQOab08WCb5ug',
        attr='Mapbox attribution')

    folium.Choropleth(
        geo_data=state_geo,
        name="choropleth",
        data=data,
        columns=["Country", "Our Model"],
        key_on="feature.properties.name",
        fill_color="YlGnBu",
        nan_fill_opacity=1.0,
        nan_fill_color="#ffffff",
        fill_opacity=1.0,
        line_opacity=1.0,
        legend_name="Fraction of Cases Missed",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    m.save("map.html")


def loss_by_country():
    """
    Output a table showing average loss by country for multiple models, along with population and total cases stats
    """
    files = os.listdir('results')
    files = ['results2021-05-21T11:16:52.json',
             ]
    all_losses = {'train': {}, 'val': {}, 'test': {}}

    for segment, losses in all_losses.items():
        if not segment == 'val':
            continue

        # Aggregate across several specified files
        for file in files:
            with open('results/' + file, 'r') as f:
                data = json.load(f)

                # For each model...
                for model, model_data in data['Models'].items():

                    # Create keys in dictionary
                    if not model in losses:
                        losses[model] = {}
                        for country in model_data['Loss by Country'][segment].keys():
                            losses[model][country] = []

                    # Append losses per country from this file
                    for country, country_losses in model_data['Loss by Country'][segment].items():
                        losses[model][country] += country_losses

        # Calculate average over all losses per country
        for model, model_data in losses.items():
            for country, country_losses in model_data.items():
                losses[model][country] = "%.3f" % (sum(country_losses) / len(country_losses))

        # Get population from countryinfo library
        losses["Population"] = {}
        countries = list(model_data.keys())
        for country in countries:
            try:
                losses["Population"][country] = countryinfo.CountryInfo(country).population()
            except KeyError:
                losses["Population"][country] = "?"

        # Get total cases per country from original dataset CSV
        losses["Total Cases"] = {}
        df = pd.read_csv("df.csv")
        df = dict(df.sum())
        for country in countries:
            if country + "_new_cases" in df:
                losses["Total Cases"][country] = df[country + "_new_cases"]
            else:
                losses["Total Cases"][country] = 100000000

        # Display in a pyplot table
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

        # plt.show()

        # Save this table to a CSV file
        for i in range(len(cellText)):
            cellText[i].insert(0, countries[i])
        df = pd.DataFrame(cellText)
        df.columns = ["Country"] + columns
        # df.to_csv("reports/" + segment + " Loss By Countries Report - " + datetime.datetime.now().isoformat().split(".")[0] + ".csv")

        df = df.filter(["Country", "Our Model"])
        df.set_index("Country", inplace=True)
        df.to_csv("testing.csv")
        df = pd.read_csv('testing.csv')
        draw_map(df)


def generate_testing_results_map():
    x = np.arange(0, 5)
    plt.plot(x, x, label="Predictions")
    plt.plot(x, x, label="Labels")
    plt.legend()
    plt.show()
    quit()
    """
    Output a table showing average loss by country for multiple models, along with population and total cases stats
    """
    file = 'results2021-05-25T21:43:45.json'
    model = 'Our Model'

    with open('results/' + file, 'r') as f:
        data = json.load(f)
        model_data = data['Models'][model]

        countries = []

        for country in model_data['Loss by Country']['test'].keys():
            countries.append(country)

        predictions = torch.FloatTensor(model_data['best_epoch']["Test Predictions"])
        labels = torch.FloatTensor(model_data['best_epoch']["Test Labels"])
        # 5 day rolling average to avoid zero in the denominator
        rolling_labels = labels.unfold(dimension=0, size=5, step=1).mean(dim=2)
        # Because the first two and last two don't have full windows
        rolling_labels = torch.cat((rolling_labels[:2, :], rolling_labels, rolling_labels[rolling_labels.shape[0] - 2:rolling_labels.shape[0]]), 0)
        raw_diffs = predictions - labels
        missed_cases = (raw_diffs < 0).float() * raw_diffs * -1
        normalized_diffs = torch.div(missed_cases, rolling_labels)
        sum_diffs = torch.sum(raw_diffs, dim=0)
        other_mean = sum_diffs/torch.sum(labels, dim=0)
        mean_normalized_diff = torch.mean(normalized_diffs, dim=0)
        median_normalized_diff = torch.median(normalized_diffs, dim=0).values

        data = [[countries[i], float(mean_normalized_diff[i])] for i in range(len(countries))]
        df = pd.DataFrame(data)
        df.columns = ["Country", "Our Model"]
        df = df.filter(["Country", "Our Model"])
        df.set_index("Country", inplace=True)
        df.to_csv("testing.csv")
        df = pd.read_csv('testing.csv')
        draw_map(df)

        fig, ax = plt.subplots(7, 6, sharex='col')
        fig.tight_layout(pad=1.0)
        plt.rcParams['font.size'] = '8'
        plt.rcParams['xtick.labelsize'] = '4'
        plt.rcParams['ytick.labelsize'] = '4'

        for i in range(7):
            for j in range(6):
                if i*6+j >= predictions.shape[1]:
                    continue
                x = np.arange(0, 45)
                ax[i, j].set_title(countries[i*6+j], fontsize=8)
                ax[i, j].plot(x, predictions[:, i*6+j], label="Predictions")
                ax[i, j].plot(x, labels[:, i*6+j], label="Labels")
                for label in (ax[i, j].get_xticklabels() + ax[i, j].get_yticklabels()):
                    label.set_fontsize(8)

        plt.legend()
        plt.show()
        fig.savefig('grid_plot.pdf', bbox_inches='tight')

if __name__ == "__main__":
    generate_testing_results_map()
    # loss_by_country()
    # compare_models()