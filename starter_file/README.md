# The Capstone Project for Azure Machine Learning Engineer of Patients with Heart Failure

This is a final project for the Udacity Nanodegree which uses two machine learning models (AutoML and HyperDrive tuned model) to predict patients survival based on medical data obtained from Kaggle public dataset called Heart Failure Prediction (https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

In the project, I have created and trained two models, one using Azure AutoML and another using HyperDrive, and deploy one of the models as ACI (Azure Container Instances).

## Project Set Up and Installation

The GitHub repo contains the following files which are required to run experiments:

- automl.ipynb: the notebook for AutoML experiment
- config.json: the fie contains settings for the subscription, resource group and name of the workspace used for the experiments
- hyperparameter_tuning.ipynb: the notebook for HyperDrive experiment
- train.py: the script for training the HyperDrive model based on different hyperparameters
- hyper-outputs/model.pkl: the best model from the HyperDrive experiment
- heart-failure-clinical-data/heart-failure-clinical-records_dataset.csv: downloaded dataset from Kaggle
- automl-outputs/env.yml: Conda environment specification
- automl-outputs/model.pkl: the best model from the AutoML experiment
- automl-outputs/scoring.py: the script downloaded from Azure Machine Learning Studio used for loading and running the model

I am using Python 3.8 - AzureML as the environment for running notebooks.

In order to run the project, you have to:

- register for a free account from Kaggle to be able to download the dataset
- update the config.json file to your subscription id, resource group and workspace name. The file used is used in the automl.ipynb notebook to register the downloaded dataset in the Data Store of the Azure Machine Learning

## Dataset

### Overview

The selected dataset Heart Failure Prediction (https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) contains medical data of patient with with cardiovascular disease. Details of the dataset can be found on Kaggle website under the link listed above. The columns and their descriptions can be found on the Kaggle website as well as in the automl.ipynb notebook, in the Dataset section.
![Dataset1](./screenshots/dataset.PNG)

### Task

The task of this project was to create two machine learning models (AutoML and HyperDived tuned model) to classify patients based on their odds of survival.
The prediction is based on features from the dataset while the label used for training the models is the last column named DEATH_EVENT being either 0 (no) or 1 (yes) (binary classification).

### Access

To get the data, you need to create a free account in Kaggle. Next you need to run the automl.ipynb notebook - the first and the second code cells. The first code cell imports dependencies while the second code cell actually downloads the dataset (if required) and stores it in the heart-failure-clinical-data folder. I am using the opendatasets package to download the dataset. When running the second code cell, the provider will ask you for your username and password which will be used to access the dataset in Kaggle.
I am also registering the dataset in Azure Machine Learning Studio:
![Dataset2](./screenshots/registered_dataset.PNG)

As noted before, the dataset is also stored in the heart-failure-clinical-data folder.
![Dataset3](./screenshots/downloaded_dataset.PNG)

I am using the stored dataset in the train.py script when training the HyperDrive model.

## Automated ML

An overview of the AutoML settings and configuration:

Settings:

- n_cross_validations: this parameter sets how many cross validations to perform based on the number of subsets. I set that value to 2
- primary_metric: I chose 'accuracy' as the default metric for the experiment
- enable_early_stopping: It enables early termination of the experiment when the score is not improving between runs thus saving resources
- max_concurrent_iterations: Defines max number of iterations executed in parallel. I set it to 4
- experiment_timeout_minutes: Defines how long the experiment should continue. I set it to 30 minutes
- verbosity: logging verbosity level. I set it to INFO
  
Configuration:

- compute_target: Azure ML compute cluster which will be used for the experiment
- task: Type of experiment. In our project we use 'classification'
- training_dataset: Datset to be used during the experiment
- label_column_name: Target column name for which the prediction is done
- path: Project path
- debug_log: Log file name
- featurization: Defines if the featurization step should be performed automatically 'auto' or not 'off'. I set it to 'auto'

In the config we could also set:

- enable_onnx_compatible_models: Whether to enable or disable enforcing the ONNX-compatible models (defaults to False)
- blocked_models: collection of algorithms we want to skip from AutoML during training e.g. due to the compatibility issues

The AutoML settings with config descriptions I have used:
![AutoML Settings](./screenshots/automl_settings.PNG)

### Results

The best AutoML model algorithm: VotingEnsemble.
The best AutoML model metrics with charts:
![AutoML Best Model Metrics](./screenshots/automl_best_model_metrics.PNG)
![AutoML Best Model Explanations](./screenshots/automl_best_model_explanations.PNG)
As we can see from the explanations chart, time is the most important factor, followed next by serum creatinine, ejection fraction and age. Surprisingly, at least for me, the smoking feature as well as the high blood pressure are the least contributing ones.

Best AutoML model details from the notebook:
![AutoML Best Model](./screenshots/automl_best_model_01.PNG)

Best AutoML model from Azure Machine Learning Studio:
![AutoML Best Model 2](./screenshots/automl_best_model_02.PNG)

I also ran the RunDetails widget:
![AutoML Best Model RunDetails2](./screenshots//automl_rundetails2.png)
Unfortunately, due to some package incompatibilities, I was not able to make the widget working as expected. I had exactly the same issue with the Runtime widget in the previous project within this Nanodegree, in the Machine Learning Operations course. I tried to upgrade existing packages, downgrade them, use different combinations of versions of packages as suggested on Stackoverflow, followed Microsoft examples on https://github.com/Azure/MachineLearningNotebooks as well as Microsoft recommendations on https://learn.microsoft.com/en-us/python/api/azureml-widgets/azureml.widgets.rundetails?view=azure-ml-py but without any luck. I was using latest Azure Machine Learning Studio with Python 3.8 and AutoML which can be seen on the screenshot.

The RunDetails widget details of the best AutoML model:
![AutoML Best Model RunDetails](./screenshots/automl_rundetails.PNG)

I believe we could improve the results by:

- setting bigger n_cross_validation
- removing/disabling early stopping policy
- increasing experiment timeout

## Hyperparameter Tuning

Tha algorithm used for the HyperParameter training is Logistic Regression optimized by HyperDrive.
There are two hyperparameters which are tuned to find the best model (with the best accuracy):

- c : the inverse of the reqularization strength
- max_iter : Maximum number of iterations

The HyperDrive config with settings' descriptions:
![HyperDrive Config](./screenshots/hyperdrive_config.PNG)

### Results

The best Logistic Regression model tuned by HyperDrive has 0.88(3) Accuracy with c=1000 and max_iterations=100.
![HyperDrive Best Model Metrics](./screenshots/hyperdrive_metrics.PNG)

Best model details from the notebook:
![HyperDrive Best Model](./screenshots/hyperdrive_best_model.PNG)
![HyperDrive Best Model2](./screenshots/hyperdrive_best_model_02.PNG)

I also ran the RunDetails widget:
![HyperDrive Best Model RunDetails2](./screenshots//hyperdrive_rundetails2.png)
Unfortunately, due to some package incompatibilities, I was not able to make the widget working as expected. I had exactly the same issue with the Runtime widget in the previous project within this Nanodegree, in the Machine Learning Operations course. I tried to upgrade existing packages, downgrade them, use different combinations of versions of packages as suggested on Stackoverflow, followed Microsoft examples on https://github.com/Azure/MachineLearningNotebooks as well as Microsoft recommendations on https://learn.microsoft.com/en-us/python/api/azureml-widgets/azureml.widgets.rundetails?view=azure-ml-py but without any luck. I was using latest Azure Machine Learning Studio with Python 3.8 and AutoML which can be seen on the screenshot.

The RunDetails widget details of the best model:
![HyperDrive Best Model RunDetails](./screenshots/hyperdrive_rundetails.PNG)

## Model Deployment

I decided to deploy the AutoML best model (the VotingEnsemble model):
![AutoML Best Model Endpoint 1](./screenshots/automl_endpoint_02.PNG)

The deployment created the ACI container:
![AutoML Best Model ACI](./screenshots/automl_best_model_aci_deployment.PNG)

Using the browser and the Swagger endpoint, I was able to get the swagger.json file:
![AutoML Best Model Endpoint 2](./screenshots/automl_endpoint_03.PNG)

I was able also to test the endpoint using the Requests Python library in the automl.ipynb notebook.
![AutoML Best Model Endpoint 3](./screenshots/automl_endpoint_01.PNG)

## Screen Recording
The screencast for this project can be found here: //TODO

## Comments and future improvements

- increase the training time for AutoML model. I think it would be useful to increase the training time of this model. I suspect this is one of the reasons why the AutoML model had a slightly lower score, though the difference is very small, compering to the HyperDrive model
- increase the sample dataset. Currently the dataset has only 299 rows which is quite small. With bigger dataset, models could be better trained and get better scoring. I think this is one of the reasons why, to my surprise, features like smoking and high blood pressure were the least contributing ones.
- use 'Weighted Accuracy' as the primary metrics instead of 'Accuracy'. We could try to use the 'Weighted Accuracy' metrics that could help with the class imbalance we have in the dataset (as per https://www.sciencedirect.com/topics/computer-science/weighted-accuracy)
- investigate the Runtime widget incompatibility issues with the latest Azure Machine Learning Studio when using Python 3.8 and AzureML. It does seem that, despite of number of tries with different versions of packages, the Runtime widget has some incompatibility issues when running in notebooks in Azure ML Studio with Python 3.8 and AzureML. I think it would be useful to investigate it further, perhaps by an experience machine learning engineer and give some hints for students in the course.
