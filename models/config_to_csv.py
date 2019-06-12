from pathlib import Path
import json
import csv
import pandas as pd
from pandas.io.json import json_normalize
from prettytable import PrettyTable

def remove_struct(df):
    L_to_keep = ['0_name', 'trainingset', 'learning_rate', 'dropout', 'depth', 'features_per_convolution']
    for param in df.columns.tolist():
        if param not in L_to_keep:
            df = df.drop(param, axis=1, errors='ignore')
    return df

def config_decode(array_config, path_model, type_):
    # Loading and flattening json file
    path_config = path_model / 'config_network.json'
    model_name = path_model.parts[-1]

    with open(path_config, 'r') as config_file:
        config = json.loads(config_file.read())
        config_flatten = json_normalize(config)
        config_flatten.rename(columns=lambda x: x[8:], inplace=True)
        if type_ == 'describe':
            config_flatten = remove_struct(config_flatten)
        config_flatten.insert(0,'0_name',model_name)
        # Appending to existing dataframe
        res = array_config.append(config_flatten)
        return res

def describe(write_model):
    models = pd.DataFrame()

    for item_path in Path.cwd().iterdir():
        if item_path.is_dir():
            for data in item_path.iterdir():
                if data.match('*config_network*'):
                    models = config_decode(models, item_path, 'describe')

    models.reindex_axis(sorted(models.columns), axis=1)
    models = models.fillna('None')
    if write_model:
        models.to_csv('models_description.csv', index=False)

    # Now we display the differences between the dataframes
    t = PrettyTable(models.columns.tolist())
    for index, rows in models.iterrows():
        t.add_row(rows)
    print(t)

def compare(compare_models):
    models = pd.DataFrame()

    L_models = []
    L_models.append(compare_models[0])
    L_models.append(compare_models[1])

    for model_name in L_models:
        for item_path in Path.cwd().iterdir():
            if item_path.is_dir() and item_path.match("*{0}".format(model_name)):
                for data in item_path.iterdir():
                    if data.match('*config_network*'):
                        models = config_decode(models, item_path, 'compare')
    models.index = [0,1]

    # Now we display the differences between the dataframes
    t = PrettyTable(['param', L_models[0], L_models[1]])
    for col in models.columns:
        if (models.loc[0,col] != models.loc[1,col]) and (str(col) != '0_name'):
            t.add_row([str(col), str(models.loc[0,col]), str(models.loc[1,col])])
    print(t)

def describe_model(model_name):
    models = pd.DataFrame()
    for item_path in Path.cwd().iterdir():
        if item_path.is_dir() and item_path.match("*{0}".format(model_name)):
            for data in item_path.iterdir():
                if data.match('*config_network*'):
                    models = config_decode(models, item_path, 'compare')
    # Now we display the differences between the dataframes
    t = PrettyTable(['param', model_name])
    for col in models.columns:
        if (str(col) != '0_name'):
            t.add_row([str(col), str(models.loc[0,col])])
    print(t)
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--describe", required=False, action='store_true', help="")
    ap.add_argument("-w", "--write", required=False, action='store_true', help="")
    ap.add_argument("-c", "--compare", required=False, nargs=2, help="")
    ap.add_argument("-l", "--listconfig", required=False, nargs=1, help="")

    args = vars(ap.parse_args())
    describe_models = args["describe"]
    write_model = args["write"]
    compare_models = args["compare"]
    model_config = args["listconfig"]

    if describe_models:
        describe(write_model)
    elif model_config is not None:
        describe_model(model_config[0])
    else:
        compare(compare_models)


if __name__ == '__main__':
    main()
