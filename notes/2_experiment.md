# Intro to Experiment Tracking

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

    #finally, we log our artifacts (our models in this case)
    mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")
```
* `mlflow.start_run()` returns the current active run, if one exists. The returned object is a [Python context manager](https://docs.python.org/2.5/whatsnew/pep-343.html), which means that we can use the `with` statement to wrap our experiment code and the run will automatically close once the statement exits.
* `mlflow.set_tag()` creates a key-value tag in the current active run.
* `mlflow.log_param()` logs a single key-value param in the current active run.
* `mlflow.log_metrics()` logs a single key-value metric, which must always be a number. MLflow will remember the value history for each metric.
* `mlflow.log_artifact()` logs a local file or directory as an artifact.
    * `local_path` is the path of the file to write.
    * `artifact_path` is the directory in `artifact_uri` to write to.
    * In this example, `models/lin_reg.bin` existed beforehand because we had created it in a code block similar to this:
        ```python
        with open('models/lin_reg.bin', 'wb') as f_out:
            pickle.dump((dv, lr), f_out)
        ```

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

We now know the best hyperparams for our model but we have not saved the weights; we've only tracked the hyperparam values. But since we already know the best hyperparams, we can retrain the model with them and save the model with MLflow. Simply define the params and the training code wrapped with the `with mlflow.start_run()` statement, define a tag and track all metrics and artifacts that you need.

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

>Note: you may find a notebook with all of the code we've seen in this block [in this link](../2_experiment/duration-prediction_2.ipynb)