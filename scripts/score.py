import pandas as pd
import joblib
import json
import os


def init():
    global model
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], 'random_forest.pkl')
    model = joblib.load(model_path)


def run(request):
    text = json.loads(request)
    data = process(pd.DataFrame(text['data']))    
    result = model.predict(data)
    return result.tolist()

def process(df):
    df['bruises'] = df['bruises'].replace({'t': 1 , 'f': 0})
    df['gill-spacing'] = df['gill-spacing'].replace({'c' : 1, 'w' : 0})
    df['gill-attachment'] = df['gill-attachment'].replace({'f': 1 , 'a': 0})
    df['gill-size'] = df['gill-size'].replace({'n' : 1, 'b' : 0})
    df['stalk-shape'] = df['stalk-shape'].replace({'e' : 1, 't' : 0})
    
    # one-hot encoding
    dummy_cols = ['cap-shape', 'cap-surface', 'cap-color', 'odor', 'gill-color', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
              'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
    df = pd.get_dummies(df, columns=dummy_cols)
    
    missing_cols = set(model.feature_names) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    df = df[model.feature_names]
    return df
    