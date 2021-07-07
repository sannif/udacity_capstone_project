# from azureml.core import Dataset
# from azureml.core.run import Run

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import argparse
import pandas as pd
import numpy as np
import joblib
import os




# run = Run.get_context()

def preprocess(df):
    # Drop columns with a single value
    df_clean = df.drop(['gill-attachment', 'veil-type'], axis=1)

    df_clean['bruises'] = df_clean['bruises'].replace({'t': 1 , 'f': 0})
    # df_clean['bruises'] = df_clean['bruises'].replace({True: 1 , False: 0})
    df_clean['gill-spacing'] = df_clean['gill-spacing'].replace({'c' : 1, 'w' : 0})
    df_clean['gill-size'] = df_clean['gill-size'].replace({'n' : 1, 'b' : 0})
    df_clean['stalk-shape'] = df_clean['stalk-shape'].replace({'e' : 1, 't' : 0})

    # df_clean['class'] = df_clean['class'].replace({"p": 1 , "e": 0})
    
    # one-hot encoding
    dummy_cols = ['cap-shape', 'cap-surface', 'cap-color', 'odor', 'gill-color', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
              'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
    df_clean = pd.get_dummies(df_clean, columns=dummy_cols)
    return df_clean

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max_depth', type=int, default=6, help='Depth of the tree')
    parser.add_argument('--criterion', type=str, default='gini', help='The function to measure the quality of a split')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='The minimum number of samples required to be at a leaf node')

   
    args = parser.parse_args()
    # run.log('Estimators', np.int(args.n_estimators))
    # run.log('Depth', np.int(args.max_depth))
    # run.log('Criterion', str(args.criterion))
    # run.log('Min sample leaf', np.int(args.min_samples_leaf))

    # Load data
    # dataset = Dataset.Tabular.from_delimited_files('https://raw.githubusercontent.com/sannif/udacity_capstone_project/main/dataset/mushrooms.csv')
    # df = dataset.to_pandas_dataframe()
    df = pd.read_csv('https://raw.githubusercontent.com/sannif/udacity_capstone_project/main/dataset/mushrooms.csv')
    df_clean = preprocess(df)

    y = df_clean.pop('class')
    x = df_clean

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print(x_train.head(1))
    
    #  Train the model
    model = RandomForestClassifier(n_estimators = args.n_estimators,
                                   max_depth = args.max_depth,
                                   criterion = args.criterion,
                                   min_samples_leaf = args.min_samples_leaf,
                                   n_jobs=-1,
                                   random_state = 1234)
    model.fit(x_train, y_train)
    # add feature names to the model object
    model.feature_names = list(x.columns.values)
    accuracy = model.score(x_test, y_test)
    # run.log('accuracy', accuracy)

    # Save model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.pkl')

if __name__ == '__main__':
    main()




    