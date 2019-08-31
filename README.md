# Predicting Well Production and Dealing with Missing Data

This code is used to perform data exploration, data munging, model-building, and model evaluation, while paying special attention to the problem of missing data.

The point wasn't to build the most accurate model, but to demonstrate an approach to the data science workflow and to build a suite of tools to support it.

You can read more about the project, including the goals, methods, and business context [here](more_info.md).

## Highlighted Contents
Some things I hope are helpful to others:
- `/notebooks/EDA.ipynb`: A jupyter notebook with a supporting library for comprehensive exploratory data analysis
- `/scripts/data_munging_tools.py`: Data-munging scripts using modern pandas style with method-chaining
- `/notebooks/model_eval.ipynb`: Dynamic, interactive visualizations of model residuals, using the altair library.
- `/notebooks/imputation_exploration.ipynb`: A jupyter notebook for describing patterns of data missingness
- `/scripts/grid_search.py`: A script for showing how to use modern multiple-imputation techniques with auxillary features inside of a scikit pipeline within a cross-validated grid search

## How to use this repo
Activate the conda environment provided in `environment.yaml` file in the root directory of this repo.
You can do this by entering the following command into the terminal (when current directory is the root of the repo):
```
$ conda env create -f environment.yaml
```

If you just want to see some of the results and data visualizations, here are some nbviewer links:
- [Exploratory Data Analysis](https://nbviewer.jupyter.org/github/noproblem-james/data_imputation/blob/develop/notebooks/EDA.ipynb)
- [Missing Data Imputation](https://nbviewer.jupyter.org/github/noproblem-james/data_imputation/blob/develop/notebooks/imputation_experiment.ipynb)
- [Model Evaluation](https://nbviewer.jupyter.org/github/noproblem-james/data_imputation/blob/develop/notebooks/model_eval.ipynb)
![](images/resid_dash.gif)

