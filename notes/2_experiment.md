# Intro to Experiment Tracking

_[Video source](https://www.youtube.com/watch?v=MiA7LQin9c8&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=12)_

## Definition of Experiment Tracking

***Experiment tracking*** (ET) is the process of keeping track of all the ***relevant information*** from an ***ML experiment***, which may include:
* Source code
* Environemtn
* Data
* Hyperparameters
* Metrics
* ...

_Relevant information_ may vary depending on the nature of the experiment: you may be experimenting with different data sources, or perhaps tuning the hyperparameters, or any other kind of experiment with different kinds of results to track.

## Why is Experiment Tracking important?

There are 3 main reasons for tracking experiments:
* ***Reproducibility***: as data scientists, we want our experiments to be repeatable in the future in order to verify our results.
* ***Organization***: if you're collaborating with a team, it's important that everyone knows where to find what. And even if you're working alone, you may want to go back to previous work, and having it organized it helps tremendously.
* ***Optimization***: the usage of experiment tracking tools allows us to create more efficient workflows and automate certain steps which are usually performed manually.

## Basic experiment tracking: spreadsheets

One of the most basic and common forms of ET is using a spreadsheet to paste your results onto, usually in the form of metrics from multiple tests.

For any remotely sophisticated project this isn't enough:
* ***Error prone***: manually copying and pasting results is awkward and bound to introduce errors in the spreadsheet in the long term. And even if you automate the data input, you may end up with issues down the road.
* ***No standard format***: each data scientist may use different conventions for writing down their results, and some vital information such as hyperparameters could be missing or misinterpreted if the data scientist isn't careful with their template.
* ***Bad visibility and difficult collaboration***: using spreadsheets for collaboration is difficult and joining 2 separate spreadsheets may be an exercise in frustration. A manager may go insane trying to understand each of the team members' different spreadsheets, and looking for specific results among different spreadsheets may be difficult.

## Intro to MLflow

***[MLflow](https://mlflow.org/)*** is an _open source platform for the machine learning lifecycle_, according to the official website.

In practice, it's a Python package that can be installed with `pip` which contains 4 main modules:
* ***[Tracking](https://mlflow.org/docs/latest/tracking.html)***: the module we will focus on on this module, for ET.
* ***[Models](https://mlflow.org/docs/latest/models.html)***: a standard format for packaging ML models that can be used in diverse serving environments. Will be covered in future modules.
* ***[Model Registry](https://mlflow.org/docs/latest/model-registry.html)***: useful for managing models in MLflow. Will be covered in future modules.
* ***[Projects](https://mlflow.org/docs/latest/projects.html)***: a format for packaging code in a reusable and reproducible way. Out of scope for this course but it is recommended to check it out.

## Tracking experiments with MLflow

The Tracking module allows us to organize our _experiments_ into _runs_. A ***run*** is a trial in a ML experiment. An ***experiment*** is made of runs, and each run may keep track of:
* Parameters (both hyperparameters as well as any other relevant parameters such as data paths, etc)
* Evaluation metrics
* Metadata (e.g. tags for organizing your runs)
* Artifacts (e.g. visualizations)
* Models

Along with all of the above data, MLflow also logs additional information abour the run:
* Source code (name of the file used for the experiment)
* Code version (git commit)
* Start and end time
* Author

# Getting started with MLflow

_[Video source](https://www.youtube.com/watch?v=cESCQE9J3ZE&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=12)_

## Installing and running MLflow

For this block we will use a Python 3.9 Conda environment with the requirements listed [in this file](../2_experiment/requirements.txt). Check the [Conda cheatsheet](https://gist.github.com/ziritrion/8024025672ea92b8bdeb320d6015aa0d) if you need a refresher on how to set it up.

Once you've installed MLflow, You may access the MLflow web UI by running the  `mlflow ui` command from a terminal using your Conda environment; however, you will need to provide a backend in order to save and retrieve experiment data. We can use SQLite as a backend with the following command:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

This will create a `mlflow.db` file in the folder your run the command from. You should now be able to access the web UI by browsing to `http://127.0.0.1:5000` .

## Using MLflow with Jupyter Notebook

As mentioned before, MLflow is a library that you import to your code. We will use it now to track and store our models.

You will need to create a `models` subdirectory in the work directory that you intend to run your notebook from, so that MLflow can store the models.

You may copy [this notebook from the last lesson's ML refresher](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/01-intro/duration-prediction.ipynb) and modify it accordingly if you want to follow along.

>Note: make sure that the directory you run your notebook is the same in which you run the `mlflow ui` command. There should be a `mlflow.db` file in the folder.

The first step is importing the MLflow library and setting a couple of important parameters:

```python
import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")
```
* `set_tracking_uri` is used to point to our backend. The provided uri should match the uri you provided when you run the MLflow command from console.
* `set_experiment` is used to define the name of our experiments. You can use whatever is convenient for you.

When you run this cell, if the experiment did not exist before, MLflow will create it.

Write and run your code to load and prepare any data you may need noramlly. Then, once you're ready to fit a model, use a `with` statement block to wrap your code like this:

```python
#wrap your code with this
with mlflow.start_run():

    #tags are optional. They are useful for large teams and organization purposes
    #first param is the key and the second is the value
    mlflow.set_tag("developer", "cristian")

    #log any param that may be significant for your experiment.
    #We've decided to track the datasets we're using.
    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")

    alpha = 0.1
    #we're also logging hyperparams; alpha in this example
    mlflow.log_param("alpha", alpha)

    lr = Lasso(alpha)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    #and we're also logging metrics
    mlflow.log_metric("rmse", rmse)
```
* `mlflow.start_run()` returns the current active run, if one exists. The returned object is a [Python context manager](https://docs.python.org/2.5/whatsnew/pep-343.html), which means that we can use the `with` statement to wrap our experiment code and the run will automatically close once the statement exits.
* `mlflow.set_tag()` creates a key-value tag in the current active run.
* `mlflow.log_param()` logs a single key-value param in the current active run.
* `mlflow.log_metrics()` logs a single key-value metric, which must always be a number. MLflow will remember the value history for each metric.

## Viewing experiments on the web UI

Once you're run your code, you may check your experiment and its runs on the web UI. The left column displays your experiments and each row in the main window displays a run. You may change parameters in your notebook and rerun the code, and after refreshing the UI you should see new runs added to your experiment.

You may download a finished notebook with the changes described above [in this link](../2_experiment/duration-prediction_1.ipynb) so that you can test by yourself.

# Experiment tracking with MLflow

_[Video source](https://www.youtube.com/watch?v=iaJz-T7VWec&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=13)_

We will now see how to track experiments rather than single runs.

## Using hyperopt for hyperparameter tuning

[Hyperopt](https://hyperopt.github.io/hyperopt/) is a library for distributed hyperparameter optimisation. We will use it for tuning our experiments. It can be installed with `pip` or Conda.

The way Hyperopt works is the following:
1. We define an _objective function_ that we want to minimize.
2. We define the _space_ over which to search.
3. We define a _database_ in which to store all the point evaluations of the search.
4. We define a _search algorithm_ to use.

For our example we use the following imports:

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
```
* `fmin` is for minimizing the objective function.
* `tpe` is the algorithm that we will provide for minimizing.
* `hp` is for defining the search space for hyperparameter tuning.
* `STATUS_OK` is a signal for letting hyperopt know that the objective function has run successfully.
* The `Trials` object keeps track of information from each run.
* `scope` is for defining ranges.

## Defining the objective function

We now have to define an _objective function_. The objective function essentially consists of the model training and validation code and returning metrics. We will use XGBoost for our experiment:

```python
import xgboost as xgb

#assuming we already have the dataframes in memory, we create the matrices for xgboost
train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)

#params contains the hyperparameters for xgboost for a specific run
def objective(params):

    with mlflow.start_run():

        #set a tag for easier classification and log the hyperparameters
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)

        #model definition and training
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        #predicting with the validation set
        y_pred = booster.predict(valid)

        #rmse metric and logging
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    #we return a dict with the metric and the OK signal
    return {'loss': rmse, 'status': STATUS_OK}
```
* Hyperopt will try to optimize whatever metric that the objective function is returning. We're returning the RMSE, so we will minimize RMSE.
* We're also returning the `STATUS_OK` signal to let hyperopt know if the optimization was successful.
* We return a dictionary because we will need to provide one later on when trying to minimize the function. The `loss` and `status` key-value pairs are mandatory.

## Defining the search space

The ***search space*** refers to the ranges in which we want Hyperopt to explore the hyperparameters.

[Hyperopt's official docs](https://hyperopt.github.io/hyperopt/getting-started/search_spaces/) have a very detailed guide on [how to define search spaces](https://hyperopt.github.io/hyperopt/getting-started/search_spaces/). Here's an example for our exercise:

```python
search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}
```
* We define a dict with all the hyperparameters we want to explore and finetune.
* We use _stochastic expressions_ to define our parameter search. All expressions require a _label_ parameter (the first param in all functions above).
* `hp.quniform()` returns a "quantized" value uniformly between `a` and `b`. In other words: it returns **discreet** values in intervals of `q` following a uniform distribution between `a` and `b`.
  * In this example, we're searching for values between 4 and 100 for our `max_depth` hyperparam.
* `scope.int()` is for defining a range of integers. `hp.quniform()` returns floats, so we need `scope.int()` if we want to use integers.
* `hp.loguniform()` returns the exponential of a number between `a` and `b` following a uniform distribution.
  * `hp.loguniform('learning_rate', -3, 0)` returns a value between `0.05` and `1` which grows exponentially, similar to how we usually test learning rates in decimal increments such as `0.001`, then `0.01` and `0.1`.

## Passing information to `fmin`

We've already defined the objective function and the search space; now we need to minimize the objective function by calling the `fmin` method and passing all of the necessary info:

```python
best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)
```
* `fn` receives our objective function.
* `space` receives our search space.
* `algo` defines the _search algorithm_ that we wil use.
  * `tpe` stands for [Tree of Parzen Estimators](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf), an algorithm for hyperparameter optimization.
  * `tpe.suggest` is the default choice for almost all projects. For dummy projects, you could use `hyperopt.random.suggest` instead.
* `max_evals` defines the maximum amount of evaluation iterations for hyperparameter search.
* `trials` receives and object which will store all of the experiment info.
  * `Trials()` is a method that returns such an object for us.
  * The object receives all of the dictionaries returned by the objective function and stores them.

## Running the experiment and comparing results

Executing the last code block will start the experiment. Hyperopt will perform multiple runs trying to seach for the best hyperparameters, and you can see each of them in the MLflow UI.

By searching for the `xgboost` tag you can select all runs and compare them in a graph to see how the different hyperparams affect the RMSE:
* The _Parallel Coordinates Plot_ draws a line chart with all the hyperparam values. Very useful to quickly see any possible correlations between hyperparam values and RMSE.
* The _Scatter Plot_ is for comparing 2 specific variables, such as a hyperparm and the RMSE. It can also be helpful to uncover patterns.
* _Contour Plot_ is similar to _Scatter Plot_ but it allows you to add an additional variable to the comparison in the form of a contour map.

You can also sort the search results to see which model has the lowest RMSE. Keep in mind that you might not always want the lowest error: for complex models, the lowest error might be too heavy and complex for your needs, so choose according to your needs and the hyperparam results.

## Retraining with the optimal hyperparams and automatic logging

We now know the best hyperparams for our model but we have not saved the weights; we've only tracked the hyperparam values. But since we already know the best hyperparams, we can retrain the model with them and save the model with MLflow. Simply define the params and the training code wrapped with the `with mlflow.start_run()` statement, define a tag and track all metrics and parameters that you need.

However, there is a better way. MLflow has support for ***automatic logging*** for [a few ML frameworks](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging). Automatic logging allows us to track pretty much any info that we may need without having to manually specify all the data and artifacts to track. Here's an example:

```python
params = {
  'learning_rate': 0.09585355369315604,
  'max_depth': 30,
  'min_child_weight': 1.060597050922164,
  'objective': 'reg:linear',
  'reg_alpha': 0.018060244040060163,
  'reg_lambda': 0.011658731377413597,
  'seed': 42
}

mlflow.xgboost.autolog()

booster = xgb.train(
  params=best_params,
  dtrain=train,
  num_boost_round=1000,
  evals=[(valid, 'validation')],
  early_stopping_rounds=50
)
```

Running this code will create a new run and track all relevant hyperparams, metrics and artifacts for XGBoost.

Keep in mind that training time will likely be slightly longer than the run with the same hyperparams when searching with Hyperopt due to MLflow logging lots of data.

In the MLflow web UI you can see all of the logged data, including a `requirements.txt` file for replicating the environment used for the experiment and code snippets to load the stored model and run predictions.

You may find a notebook with all of the code we've seen in this block [in this link](../2_experiment/duration-prediction_2.ipynb).

# Model management

_[Video source](https://www.youtube.com/watch?v=OVUPIX88q88&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=14)_

So far we've seen _Experiment Tracking_, which handles model architecture, training and evaluation. ***Model Management*** would be the next step in MLops and handles model versioning and deployment, as well as hardware scaling. In this block we will see how to use MLflow for Model Management.

![Source: https://neptune.ai/blog/mlops](https://i0.wp.com/neptune.ai/wp-content/uploads/MLOps_cycle.jpg?resize=1024%2C576&ssl=1)

## Model tracking

Just like we saw with using a spreadsheet for experiment tracking, you could simply use a folder hyerarchy for model management. However, this simple management technique has the same weaknesses as the spreadsheet did:
* ***Error prone***: humans are messy and manually renaming folders and files and moving them around will surely result in mistakes down the line.
* ***No versioning***: you could use the filenames for versioning but it's likely you will mix the numbers up.
* ***No model lineage***: it's not easy to understand how all of your models were created, what the hyperparams were, etc.

In order to solve these issues, we can use MLflow for tracking our models.

## Tracking artifacts

Besides parameters and metrics, MLflow can also track ***artifacts***, such as our model weights.

Tracking an artifact is just like tracking any other element in MLflow. The simplest way of model management is simply to track the model as an artifact. Here's the same Scikit-Learn code we saw before with a new line at the end:

```python
with mlflow.start_run():

    mlflow.set_tag("developer", "cristian")

    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
    mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")

    alpha = 0.1
    mlflow.log_param("alpha", alpha)

    lr = Lasso(alpha)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    #Tracking our model
    mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")
```
* `mlflow.log_artifact()` logs a local file or directory as an artifact.
    * `local_path` is the path of the file to write.
    * `artifact_path` is the directory in `artifact_uri` to write to.
    * In this example, `models/lin_reg.bin` existed beforehand because we had created it in a code block similar to this:
        ```python
        with open('models/lin_reg.bin', 'wb') as f_out:
            pickle.dump((dv, lr), f_out)
        ```

## Model logging

The limitation with artifact tracking for model management is that it's cumbersome to search for a specific model, download the bin file and create the code to load it and run predictions.

There's a better way of managing models. We'll use the XGBoost code from before as an example; pay attention to the last line:

```python
#For the purpose of this example, let's turn off autologging
mlflow.xgboost.autolog(disable=True)

with mlflow.start_run():
    
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_params = {
        'learning_rate': 0.09585355369315604,
        'max_depth': 30,
        'min_child_weight': 1.060597050922164,
        'objective': 'reg:linear',
        'reg_alpha': 0.018060244040060163,
        'reg_lambda': 0.011658731377413597,
        'seed': 42
    }

    mlflow.log_params(best_params)

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(valid, 'validation')],
        early_stopping_rounds=50
    )

    y_pred = booster.predict(valid)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    #Model tracking
    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
```

MLflow offers model logging for specific frameworks. `mlflow.xgboost.log_model()` takes our model object and stores it in the provided path.

>Note: we disabled autologging and manually tracked all the relevant data for our purposes. You could enable autolog instead and the model would also be logged.

We can also log the DictVectorizers we used to preprocess our data because we will need them if we want to run new predictions later. Just add this code right before the last line:

```python
with open("models/preprocessor.b", "wb") as f_out:
    pickle.dump(dv, f_out)
mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
```

If you run the code again, you should now see a new folder in the Artifacts section on the web UI with the preprocessors.

## Making predictions

You may have noticed that there are code snippets next to the model artifacts that teach you how to make predictions with the logged models.

MLflow actually stores the model in a format that allows us to load it in different "flavors". For example, our XGBoost model can be loaded as an XGBoost model or as a PyFuncModel:

```python
logged_model = 'runs:/.../models_mlflow'

#Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

#Load model as a XGBoost model
xgboost_model = mlflow.xgboost.load_model(logged_model)
```

The loaded models are regular objects of their respective framework "flavor" and thus can use all of their usual methods:

```python
y_pred = xgboost_model.predict(valid)
```

You may download a finished notebook with all of the code [from this link](../2_experiment/duration-prediction_3.ipynb).

# Model registry

_[Video source](https://www.youtube.com/watch?v=TKHU7HAvGH8&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=15)_

## What is a model registry?

A ***model registry*** is a tool that allows us to track models along with all of the necessary info for reuse and deployment, such as preprocessors, training datasets, etc. Model registries are a sophisticated solution for sharing models between colleagues and documenting all of the necessary metadata for us to keep experimenting, building upon previous work and deploy successfully.

In the previous sections we made use of MLflow's _local tracking server_ for tracking all of our experiments. MLflow also offers a model registry: a Data Scientist can experiment and develop models using the tracking server to track them, and once she feels the model is ready for deployment, she can ***register*** the model in the registry so that the Deployment Engineer can deploy it.

MLflow's Model Registry offers different _stages_ (labels) for the models, such as ***staging***, ***production*** and ***archive***. The Deployment Engineer may inspect a newly registered model and assign it to a stage as needed.

>Note: regarding model analysis, a model with the lowest error or highest accuracy may not necessarily be the best model for your needs; there are other parameters such as model size and training/inference time that may affect the final deployment. You (or the Deployment Engineer) should choose according to the needs and goals of the project. MLflow gives you these values in order to make the choice easier.

Keep in mind that the model registry does not take care of deployment; it's simply there for easy storage and retrieval. You may have to combine the registry with CI/CD practices in order to achive a more mature MLOps level.

## Promoting models to the model registry

Once you've analyzed the different models of your experiment, you may decide to promote one or more to the Model Registry.

In the Runs page of the MLflow web UI, in the Artifacts section, after choosing a model you should see a blue button appear with the text `Register Model` in it. Once you click it, a new window will appear asking for a model name; give it any of your liking. If you're promoting multiple models of the same experiment, you may choose the same name for all of them (a drop-down menu should display any existing models) and the registry will track them as different versions.

You may access the Registry by clicking on the `Models` tab at the top of the web UI. Clicking on it will display all of the registered models; you may click on one to see all of the available versions for that model. From this model page you may also add a description to provide any relevant info as well as tags for each version.

## Transitioning models to stages

By default, any registered model will be assigned to the `None` stage.

From the model version page, you may transition a model version to a different stage with the _Stage_ drop-down menu near the top of the page. Clicking on any of the available options will show a pop-up window giving you the option to transition older models in the chosen stage to the `archived` stage.

## Interacting with the tracking and registry server with Python

We can use the MLflow library to access the model registry and load any promoted models by means of a `MLflowClient` object provided by MLflow.

```python
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
```
* We define the tracking server URI like in previous code examples.
* The `client` object can interact with experiments, runs and models in the registry.

The `MLflowClient` object allows us to interact with a tracking server as well as a registry server. Since we're running both the tracking and registry server locally on the same instance, a single URI allows us to access both because they're actually a single process.

### Interacting with the tracking server

Now that we've instantiated the `client` object, we can access the tracking server data, such as our experiments:

```python
# List all available experiments
client.list_experiments()
```

We could create a new experiment as well:
```python
client.create_experiment(name="my-cool-experiment")
```
* You should now see a new experiment in the web UI, similarly to how we created an experiment with `mlflow.set_experiment("nyc-taxi-experiment")`.

Let's now search and display our previous runs:
```python
from mlflow.entities import ViewType

runs = client.search_runs(
    experiment_ids='1',
    filter_string="metrics.rmse < 7",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.rmse ASC"]
)

for run in runs:
    print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['rmse']:.4f}")
```
* `experiment_ids` receives a string with the experiment ID we want to search runs in. In this case, the ID `1` should be the `nyc-taxi-experiment` from the beginning of the lesson.
* `filter_string` is a filtering query.
  *  In this case, we only want those runs with a RMSE value smaller than 7.
* `run_view_type` receives an enum with the type of runs we want to see.
  * The alternatives to `ViewType.ACTIVE_ONLY` (active runs, the default value) would be `ViewType.ALL` and `ViewType.DELETED_ONLY`.
* `max_results` limits the amount of results that `search_runs()` will return.
* `order_by` receives a list because you can include multiple order criteria.
  * In this case, we order by RMSE in ascending order.

This block of code should return something similar to this:
```
run id: 7db08e4f93af4ee1bcbce1d8a763e23a, rmse: 6.3040
run id: a06a6b594fff409cb0d34e203b49f33f, rmse: 6.7423
run id: b8904012c84343b5bf8ee72aa8f0f402, rmse: 6.9047
run id: 54493fed643c4952be5232279e309053, rmse: 6.9213
```

You can check the web UI and see that the run ID's should match these.

### Promoting a model to the registry

We can also programatically promote a model to the registry. We don't actually need the `client` object to do this:

```python
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

run_id = "b8904012c84343b5bf8ee72aa8f0f402"
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri=model_uri, name="nyc-taxi-regressor")
```
* We need to provide a URI to register the model, so we compose it based on the model path as stored by the tracking server.
* We also provide the model name in order to promote the model as a version of a previously existing model, if desired so.

### Transitioning a model to a new stage

We can also transition a model to a new stage.

First, we need to find which model version we want to transition:

```python
model_name = "nyc-taxi-regressor"
latest_versions = client.get_latest_versions(name=model_name)

for version in latest_versions:
    print(f"version: {version.version}, stage: {version.current_stage}")
```

This code block should return the version ID's as well as the stage they are currently assigned to.

Let's assume we want to transition model ID 4:

```python
model_version = 4
new_stage = "Staging"
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)
```

As you can see, the parameters in the `transition_model_version_stage()` function are pretty self-explanatory and mirror the options available in the web UI.

We can also annotate the model version, like in the web UI. Let's add a description in which we say that we transitioned the model today:

```python
from datetime import datetime
date = datetime.today().date()

client.update_model_version(
    name=model_name,
    version=model_version,
    description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
)
```

### Comparing models

We can also do the same model analysis we did in the web UI to choose the best model for our needs.

Let's assume that we want to use the [March 2021 green taxi data](https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2021-03.parquet) from the [NYC TLC Trip Record dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The following code will load the dataframe as well as preprocess it and test a model with the preprocessed data:

```python
from sklearn.metrics import mean_squared_error
import pandas as pd


def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


def preprocess(df, dv):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    train_dicts = df[categorical + numerical].to_dict(orient='records')
    return dv.transform(train_dicts)


def test_model(name, stage, X_test, y_test):
    model = mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
    y_pred = model.predict(X_test)
    return {"rmse": mean_squared_error(y_test, y_pred, squared=False)}


df = read_dataframe("data/green_tripdata_2021-03.parquet")
```
* Note that `test_model()` will load the provided model as a pyfunc model.
  * Remember that MLflow models are stored in a generic format and may be loaded by different means.

Let's now get the preprocessor we want to use from the registry:
```python
#run_id was defined previously as this:
#run_id = "b8904012c84343b5bf8ee72aa8f0f402"
client.download_artifacts(run_id=run_id, path='preprocessor', dst_path='.')
```
* We download an artifact with `download_artifacts()`.
  * We provide a run id that we previously defined.
  * When providing an ID we also need to provide a path relative to the run's root directory containing the artifacts to download.
  * `dst_path` is the path of the local filesystem destination directory to which to download the specified artifacts. In this example, we download to the same folder that contains our code.

And now let's load the preprocessor and prepare our data:
```python
import pickle

with open("preprocessor/preprocessor.b", "rb") as f_in:
    dv = pickle.load(f_in)

X_test = preprocess(df, dv)

target = "duration"
y_test = df[target].values
```

We are now ready to test our models. Since we're using a Jupyter Notebook, we can use some magic commands such as `%time` to benchmark our models

```python
#We defined model_name in a previous code block:
#model_name = "nyc-taxi-regressor"
```

```python
%time test_model(name=model_name, stage="Production", X_test=X_test, y_test=y_test)
```

```python
%time test_model(name=model_name, stage="Staging", X_test=X_test, y_test=y_test)
```

Once we've got our results, we can decide which model to keep and use `transition_model_version_stage()` to transition the chosen model to production while archiving any previous production models.

>Note: you may download a completed notebook with all of the code in this block [from this link](../2_experiment/model-registry.ipynb)

# MLflow in practice

_[Video source](https://www.youtube.com/watch?v=1ykg4YmbFVA&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=18)_

Not every ML workflow is the same and the usage of tools such as MLflow may vary depending on context. Remember: ***pragmatism is key***.

## Configuring MLflow

There are 3 main components that you need to configure in MLflow in order to adapt it to your needs:

* **Backend store**: the place where MLflow stores all the metadata of your experiments (metrics, parameters, tasks...).
  * _Local filesystem_: by default, the metadata will be saved in your local filesystem.
  * _SQLAlchemy compatible DB_: you may want to configure a DB such as SQLite [like we did before](#installing-and-running-mlflow) for storing the metadata. Using a DB enables us to use the **model registry**.
* **Artifacts store**: where your models and artifacts are stored.
  * _Local filesystem_: the default option.
  * _Remote_: e.g. an S3 bucket.
* **Tracking server**: you need to decide whether you need to run a tracking server or not.
  * _No tracking server_: enough for one-man teams for tasks such as competitions.
  * _Localhost_: good for single data scientists in dev teams that don´t need to share results with other scientists.
  * _Remote_: better for teamwork and sharing.

Let´s see how to configure these 3 components in different scenarios.

## Scenario 1: one man team, informal

(The actual scenario in the video is _A single data scientits participating in a ML competition_).

This scenario is defined as follows:

* Backend store: local filesystem
* Artifacts store: local filesystem
* Tracking server: no
  * Remember: we cannot use the tracking server unless we define a DB for our backend store. We can still explore the experiments locally by launching the MLflow UI.

Since we're using the default option (local filesystem) for both backend and artifact stores, we don't need to do any extra steps after importing the library:

```python
import mlflow

# This line will print the path to the local folder in which the bakend and artifacts will be stored
# The folder will be inside the same folder in which you run this code
print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
```

You may note that the folder printed in the previous code block (something like `mlruns`) may not exist; the folder will be created once we start interacting with MLflow. For example, you may run this line:

```python
mlflow.list_experiments()
```

And the folder will be created. You may also see that only one experiment exists (the default one), which is created as soon as you run the previous line.

Inside this folder you will see another folder named after the experiment ID which contains a YAML file with all the experiment data. This is how MLflow keeps track of data when the local filesystem is used as a backend store. Alongside the YAML file you may find a number of folders, one for each run of the experiment; we have not made any runs for the default experiment so there should be zero folders inside the default experiment folder.

After running some code and storing some artifacts, you may end up with multiple experiments with multiple runs. Here's what the folder structure may end up looking like:

* **mlruns**
  * **0**
    * meta.yaml
  * **1**
    * _49441ac587fb43728a03b9760896042a_ (or any other number)
      * **artifacts** - contains models and other artifacts. Filled with the `log_model()` method, among others.
      * **metrics** - contains the logged metrics. Filled with the `log_metric()` method, among others.
      * **params** - contains any logged parameters of interest. Filled with the `log_params()` method, among others.
      * **tags**
      * meta.yaml
    * meta.yaml

Since we did not set up a backend store DB, we cannot access the model regisgtry. However, we can still access the web UI to explore our experiments. In a terminal, go to the same folder where we've run the code and run the following command to enable it:

```bash
mlflow ui
```

Now browse to `127.0.0.1:5000` to see the experiments on the left sidebar.

>Note: if you only see the default experiment, you did not execute the command from the correct directory. Make sure that you run the command from the same directory in which the `mlruns` folder was created when you run your code.

You may find a finished notebook with all of the code explained above [in this link](../2_experiment/scenarios/scenario-1.ipynb).

## Scenario 2: cross-functional team, single data scientist

(The actual scenario in the video is _A cross-functional team with one data scientist working on an ML model_).

This scenario is defined as follows:

* Backend store: sqlite database
* Artifacts store: local filesystem
* Tracking server: yes, local filesystem

A single data scientist in a cross-functional dev team may have to interact with backend or frontend developers, or even with the product manager, and show them the progress of the model and how it's being built. A local tracking server will be enough for this setup.

In order to use the sqlite database as a backend store, we need to initialize MLflow from a terminal with the following command:

```bash
mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root ./artifacts_local
```
* This command will initialize the actual server rather than just the UI.
* The `--default-artifact-root` param is needed in order to let MLflow know where in the local filesystem to store the artifacts.
  * As written above, the chosen directory to store the artifacts will be `artifacts_local` within the same directory in which you run the command.

With the server running, you now need to connect the MLflow library to the server from your code:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

When you run your experiments, you will see that the `artifacts_local` folder will be created but the `mlruns` folder with all of the experiment metadata will be missing; all that metadata is now stored in the sqlite database, which should now appear as the `backend.db` file in your work directory.

In order to interact with the model registry we need to connect to it first:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient("http://127.0.0.1:5000")
```

We can list all of our registered models:

```python
client.list_registered_models()
```

And we can also register new models:

```python
run_id = client.list_run_infos(experiment_id='1')[0].run_id
mlflow.register_model(
    model_uri=f"runs:/{run_id}/models",
    name='iris-classifier'
)
```
* Note that `experiment_id` in the first line should contain the experiment ID you want to interact with.

You can also access the MLflow UI by browsing to `127.0.0.1:5000` as usual

>Note: if you already run the MLflow UI in scenario 1 and cannot load the web UI due to an error, delete your browser's browsing data (cookies, etc) and reload the page.

You may find a finished notebook with all of the code explained above [in this link](../2_experiment/scenarios/scenario-2.ipynb).

## Scenario 3: multiple data scientists working on multiple ML models

The last scenario we will cover is as follow:

* Backend store: Postgresql database (AWS RDS)
* Artifacts store: remote (AWS S3 bucket)
* Tracking server: yes, remote (AWS EC2)

(This block assumes you've got some knowledge on AWS and you can setup an EC2 instance, an S3 bucket and an AWS RDS Postgresql database. You may find instructions on how to set up all of these [in this link from the official DataTalks ML-OPS repo](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md).)

When you have multiple people experimenting and building models, setting up a remote tracking server is worth the effort to improve collaboration.

Starting a remote MLflow server is similar to a local one. Assuming you've properly set up an S3 bucket, a Postgresql database and allowed remote access from your EC2 instance to the database, log in to your instance and start the server:

```bash
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root s3://S3_BUCKET_NAME
```
* `-h` is short for `--host`. Since we're running the command on the remote server, `0.0.0.0` is the proper address.
* `-p` is short for `--port`. The default port for Postgresql is `5432`.
* Change all the names in all caps for your values.
  * `DB_USER` and `DB_PASSWORD` should be the _master username_ and password you set up when creating the RDS database.
  * `DB_ENDPOINT` should be the endpoint URL for the database. You can find it under _Connectivity & security_ in _Amazon RDS_ > _Databases_.
  * `DB_NAME` should be the initial database name used during RDS setup.
  * `S3_BUCKET_NAME` should be the path to your bucket.

Once the command has run successfully, you can get the URL for your EC2 instance in the AWS EC2 console and browse to `ec2-instance-url.amazonaws.com:5000` to load the MLflow UI.

Before running any code, it's recommended that you install the [AWS CLI tools](https://aws.amazon.com/es/cli/) to your computer and set up [default credentials and region](https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup.html#setup-credentials). With that out of the way, connecting our code to the tracking server only requires 2 extra lines:

```python
import mlflow
import os

os.environ["AWS_PROFILE"] = "" # fill in with your AWS profile. More info: https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup.html#setup-credentials

TRACKING_SERVER_HOST = "" # fill in with the public DNS of the EC2 instance
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
```

Just add the AWS profile name you defined with the AWS CLI tools and your tracking server URL to the code above.

The rest of the MLflow code (listing experiments, logging metrics, parameters and artifacts, registering models) is identical to what we've seen in previous scenarios, but you might experience slight delays when interacting with the server due to it being a remote instance rather than a local one.

You may find a finished notebook with all of the code explained above [in this link](../2_experiment/scenarios/scenario-3.ipynb).