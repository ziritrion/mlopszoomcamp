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