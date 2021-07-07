from azureml.core import Dataset
from azureml.core.run import Run

from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import argparse
import joblib
import os



run = Run.get_context()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--iterations', type=int, default=100, help='Number of trees')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--l2_leaf_reg', type=float, default=3, help='Coefficient at the L2 regularization term of the cost function')
    parser.add_argument('--depth', type=int, default=6, help='Depth of the tree')

    args = parser.parse_args()
    run.log('Iterations', np.int(args.iterations))
    run.log('Learning rate', np.float(args.learning_rate))
    run.log('L2 coefficient', np.float(args.l2_leaf_reg))
    run.log('Depth', np.int(args.depth))

    # Load data
    dataset = Dataset.Tabular.from_delimited_files('https://raw.githubusercontent.com/sannif/udacity_capstone_project/main/dataset/mushrooms.csv')
    df = dataset.to_pandas_dataframe()

    # Process the data
    # Drop columns with a single value
    df_clean = df.drop(['gill-attachment', 'veil-type'], axis=1)

    df_clean['bruises'] = df_clean['bruises'].replace({True: 1 , False: 0})
    df_clean['class'] = df_clean['class'].replace({"p": 1 , "e": 0})

    y = df_clean.pop('class')
    x = df_clean

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    #  Train the model
    model = CatBoostClassifier(loss_function='CrossEntropy', eval_metric='Accuracy', 
                               iterations=args.iterations, learning_rate = args.learning_rate,
                               l2_leaf_reg = args.l2_leaf_reg, depth=args.depth)
    model.fit(x_train, y_train, cat_features=list(range(x_train.shape[1])))
    accuracy = model.score(x_test, y_test)
    run.log('accuracy', accuracy)

    # Save model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.pkl')

if __name__ == '__main__':
    main()




    