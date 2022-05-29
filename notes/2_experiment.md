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

>Note: make sure that the directory you run your notebook is the same in which you run the `mlflow ui` command.

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