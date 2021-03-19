import json
import os
from matplotlib import pyplot as plt
from numpy import array as nparray
import countryinfo
import pandas as pd
import datetime

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


def loss_by_country():
    """
    Output a table showing average loss by country for multiple models, along with population and total cases stats
    """
    files = os.listdir('results')
    files = ['results2021-03-04T16:05:09.json',
'results2021-03-04T16:10:42.json',
'results2021-03-04T16:11:19.json',
'results2021-03-04T16:11:28.json'
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
            losses["Total Cases"][country] = df[country + "_new_cases"]

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

        plt.show()

        # Save this table to a CSV file
        for i in range(len(cellText)):
            cellText[i].insert(0, countries[i])
        df = pd.DataFrame(cellText)
        df.columns = ["Country"] + columns
        df.to_csv("reports/" + segment + " Loss By Countries Report - " + datetime.datetime.now().isoformat().split(".")[0] + ".csv")

if __name__ == "__main__":
    loss_by_country()
    # compare_models()