# Udacity capstone project
In this project, we create two models: one using Azure AutoML and a second one leveraging HyperDrive to tune its hyperparameters.

## Dataset
### Overview
The dataset is the [Mushrooms](https://archive.ics.uci.edu/ml/datasets/mushroom)'s dataset from UCI. It includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible or definitely poisonous. It is a classification task consisting of predicting whether a mushroom is edible or poisonous.

### Task
There are 21 variables and 8124 observations. All the variables are categorical. It is a binary classification task. The two classes are balanced.  

### Access
The data is downloaded from [Kaggle](https://www.kaggle.com/uciml/mushroom-classification) and then stored in the folder [dataset](https://github.com/sannif/udacity_capstone_project/blob/bae713dfb6b071da6282cc004f1400e8a8131ffc/dataset/mushrooms.csv). Then, we get a link to the dataset that is used in Azure ML.

## Automated ML
The AutoML experiment is created and ran using the notebook [automl.ipynb](https://github.com/sannif/udacity_capstone_project/blob/bae713dfb6b071da6282cc004f1400e8a8131ffc/automl.ipynb). Here are the parameters we chose for the `AutoMLConfig`:
* `task`: it is a classification task
* `primary_metric`: we choose *accuracy* because we have balanced classes
* `training_data`: the Mushroom dataset
* `label_column_name`: `class` is the name of the target column
* `featurization` is set to `auto`
* We enabled the early stopping by setting `enable_early_stopping` to `True`
* The experiment timeout is set to 20 minutes meaning that the experiment will stop after 20 minutes

Below are the screenshots of the `RunDetails` widget showing the runs.  

![run_details_automl1](https://github.com/sannif/udacity_capstone_project/blob/b8e2b6d67ff705894c4c491df092ac7b45019767/images/automl_run_details_1.PNG)
![run_details_automl1](https://github.com/sannif/udacity_capstone_project/blob/b8e2b6d67ff705894c4c491df092ac7b45019767/images/automl_run_details_2.PNG)


### Results
*XGBoostClassifier*, *LightGBM*, *Logistic Regression*, *Random Forest*, *ExtremeRandomTrees* are the models that have been tested in combination with different processing such as *StandardScalerWrapper*, *MaxAbsScaler*. More than half of the models reached 100% of accuracy. *RandomForest* and *ExtremeRandomTrees* are the two algorithms that didn't reach 100% accuracy. Moreover *XGBoostClassifier* with a *StandardScalerWrapper* as processing produced models with 100% accuracy but also models with 52% accuracy. It demonstrates the importance of the hyperparameters.  

The model we kept as best is *LightGBM* with *MaxAbsScaler* processing. *min_data_in_leaf* is the only hyperparameter that has been changed from its default value to 20. Below is the screenshot of the best model.  

![best_automl](https://github.com/sannif/udacity_capstone_project/blob/b8e2b6d67ff705894c4c491df092ac7b45019767/images/automl_best_run.PNG)
![best_automl2](https://github.com/sannif/udacity_capstone_project/blob/362cb273a3f28ddfbfdb0be0c6c44eca9f8614e8/images/automl_best_model_id.PNG)
![best_automl_params](https://github.com/sannif/udacity_capstone_project/blob/362cb273a3f28ddfbfdb0be0c6c44eca9f8614e8/images/automl_best_model_params.PNG)


## Hyperparameter Tuning
We fit a Random Forest classifier to the data. 4 hyperparameters are tuned:  
* n_estimators: the number of trees ```choice([50, 100, 250, 500])```
* max_depth: the depth of a tree ```choice([5, 6, 7, 8, 9, 10, 15])```
* criterion: the function to measure the quality of a split ```choice(['gini', 'entropy'])```
* min_samples_leaf: the minimum number of samples required to be at a leaf node ```choice([1, 2, 3, 4]```  

More information on the role of each hyperparameter can be found here.  

We choose a *MedianStoppingPolicy* as the termination policy. It permits to stop non promising runs and save costs. A Bayesian sampling is used to sample the hyperparameter space. Finally, we limited the total number of runs to 25.

### Results
All the runs performed very well with an accuracy between 99.6 and 100%. It seems like the values of the hyperparameters didn't impact much the performance of the models. Below are the screenshots of the `RunDetails` widget and the best model trained with its parameters.

![run_hyper](https://github.com/sannif/udacity_capstone_project/blob/68a36537213552cc3147d761afa51fb16cd5c869/images/hyperdrive_run_details.PNG)

![best_hyper1](https://github.com/sannif/udacity_capstone_project/blob/68a36537213552cc3147d761afa51fb16cd5c869/images/best_model.PNG)

![best_hyper2](https://github.com/sannif/udacity_capstone_project/blob/68a36537213552cc3147d761afa51fb16cd5c869/images/best_hyperdrive_2.PNG)

## Model Deployment
We deploy the best model from Hyperparameter tuning. It is a *Random Forest* model. Here are the steps we follow:
1. Register the model
2. Create the entry script [score.py](https://github.com/sannif/udacity_capstone_project/blob/68a36537213552cc3147d761afa51fb16cd5c869/scripts/score.py)
3. Define an inference configuartion
4. Define the deployment configuration  

The code is available in the notebook [hyperparameter_tuning.ipynb](https://github.com/sannif/udacity_capstone_project/blob/68a36537213552cc3147d761afa51fb16cd5c869/hyperparameter_tuning.ipynb).

The endpoint can be queried using the piece of code below that comes from the notebook `hyperparameter_tuning.ipynb`. In addition a demo is done in the screencast.
```
service = Webservice(workspace=ws, name="mushroom-service")
scoring_uri = service.scoring_uri

# Get the key for authentication
key, _ = service.get_keys()

headers = {"Content-Type": "application/json"}
headers["Authorization"] = f"Bearer {key}"
data = json.dumps({
    'data':
    [
        {'cap-shape': 'x',
         'cap-surface': 's',
         'cap-color': 'n',
         'bruises': 't',
         'odor': 'p',
         'gill-attachment': 'f',
         'gill-spacing': 'c',
         'gill-size': 'n',
         'gill-color': 'k',
         'stalk-shape': 'e',
         'stalk-root': 'e',
         'stalk-surface-above-ring': 's',
         'stalk-surface-below-ring': 's',
         'stalk-color-above-ring': 'w',
         'stalk-color-below-ring': 'w',
         'veil-color': 'w',
         'ring-number': 'o',
         'ring-type': 'p',
         'spore-print-color': 'k',
         'population': 's',
         'habitat': 'u'
        }
    ]
})
resp = requests.post(scoring_uri, data=data, headers=headers)
print(resp.text)
```

## Improvement
The classification problem was quite simple and we already reached 100% of accuracy on a test set. There is no improvement possible on that point. However, for the model deployment, it may be interesting to convert the Sklearn model into ONNX in order to use the `onnxruntime` module.

## Screen Recording
We did a screen recording of the project in action that demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response.

The screencast is available [here](https://youtu.be/By0VDrHl00I).
