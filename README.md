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
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
The AutoML experiment is created and run using the notebook [automl.ipynb](https://github.com/sannif/udacity_capstone_project/blob/bae713dfb6b071da6282cc004f1400e8a8131ffc/automl.ipynb). We choose the accuracy as the primary metric because we have balanced classes. Also, the experiment timeout is set to 20 minutes meaning that the experiment will stop after 20 minutes. We select classification for the task parameter. More details are available in the notebook. Below is the screenshot of the *RunDetails* wigdget showing the runs.  
![run_details_automl]()

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
