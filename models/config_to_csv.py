import os
import json
import csv
from pandas.io.json import json_normalize
import pandas as pd

def remove_struct(df):
    L_to_remove = ['n_classes', 'size_of_convolutions_per_layer', 'features_per_convolution', 'depth', 'convolution_per_layer',
                   'thresholds', 'data_augmentation.transformations.elastic', 'data_augmentation.transformations.flipping',
                   'data_augmentation.transformations.noise_addition',           
                   'data_augmentation.transformations.random_rotation',
                   'data_augmentation.transformations.rescaling',
                   'data_augmentation.transformations.shifting']
    for param in L_to_remove:
        df = df.drop(param, axis=1, errors='ignore')
    return df

def config_decode(array_config, path_model):
    # Loading and flattening json file
    path_config = os.path.join(path_model, 'config_network.json')
    model_name = path_model.split('/')[-1]

    with open(path_config, 'r') as config_file:
        config = json.loads(config_file.read())
        config_flatten = json_normalize(config)
        config_flatten.rename(columns=lambda x: x[8:], inplace=True)
        config_flatten = remove_struct(config_flatten)
        config_flatten.insert(0,'0_name',model_name)

        # Appending to existing dataframe
        res = array_config.append(config_flatten)
        return res

def main():
    models = pd.DataFrame()

    for root in os.listdir(os.path.curdir)[:]:
        if os.path.isdir(root):
            subpath_data = os.path.join(os.path.curdir, root)
            for data in os.listdir(subpath_data):
                if 'config_network' in data:
                    models = config_decode(models, subpath_data)
    
    models.reindex_axis(sorted(models.columns), axis=1)
    models = models.fillna('None')
    models.to_csv('models_description.csv', index=False)

if __name__ == '__main__':
    main()