# Introduction to MLOps

_[Video source](https://www.youtube.com/watch?v=s0uaFZSzwfI&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=2)_

***MLOps*** is a _set of best practices_ for bringing Machine Learning to production.

Machine Learning projects can be simpplified to just 3 steps:

1. ***Design*** - is ML the right tool for solving our problem?
   * _We want to predict the duration of a taxi trip. Do we need to use ML or can we used a simpler rule-based model?_
2. ***Train*** - if we do need ML, then we train and evaluate the best model.
3. ***Operate*** - model deployment, management and monitoring.

MLOps is helpful in all 3 stages.

# Environment setup

_[Video source](https://www.youtube.com/watch?v=IXSiYkP23zo&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=3)_

You may check the link above to watch the video in order to learn how to set up a Linux VM instance in Amazon Web Services.

If you'd rather work with Google Cloud Platform, you may check out [the isntructions in this gist](https://gist.github.com/ziritrion/3214aa570e15ae09bf72c4587cb9d686). Please note that the gist was meant for the [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp) and assumes that the reader has some familiarity with GCP and Linux shell commands. You may check out [my Data Engineering Zoomcamp notes](https://github.com/ziritrion/dataeng-zoomcamp/blob/main/notes/1_intro.md#terraform-and-google-cloud-platform) for a refresher on GCP.

Alternatively, you may also use any other cloud vendor or set up a local environment. The requirements for this course are:

* [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
  * If you're using a local environment with a GUI, then [Docker Desktop](https://www.docker.com/products/docker-desktop/) is the recommended download for both components.
* [Anaconda](https://www.anaconda.com/)
  * We will use Python 3.9 for this course.
  * We will also need Jupyter Notebook.
  * You may check out my [Python environments cheatsheet](https://gist.github.com/ziritrion/8024025672ea92b8bdeb320d6015aa0d) for a refresher on how to use Anaconda to install Python.
* (Optional) [Visual Studio Code](https://code.visualstudio.com/) and the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension
  * These requirements are not necessary but they make it much easier to connect to remote instances and redirect ports. These notes will assume that you're using both.

>Note: Any additional requirements will be listed as needed during the course.