## Overview
This was completed as a capstone project satisfying a principal graduation
requirement for the Data Science Immersive Program offered by Galvanize, Inc.

I had three weeks to complete this project.

I wanted to tackle a problem with clear business value, and to that end, I partnered with NoviLabs, a company which does predictive modeling for the oil and gas industry.


## Objective
The overarching goal was to improve on an existing model that predicts oil production for new wells at different time horizons, in different geologies, under different economic conditions. For the sake of simplicity, I did the following things:
* Ignored geological features, except the overall formation (Bakken or Three Forks), and used latitude and longitude and depth as a proxy measure.
* Ignored economic conditions, as these factors (such as the forecasted price of oil) are typically modeled by the end user (drilling companies).
* Focused on predicting cumulative oil production for newly drilled wells at three discrete time intervals: 90 days (“IP90”, i.e., initial production at 90 days), 180 days (“IP180”), and 365 days (“IP365”).

I considered several different approaches to improving on the existing model, including intensive feature engineering and hyperparamter tuning. Although I did derive some features and trained a different type of machine learning model to significantly improve predictive accuracy, I focused my efforts on investigating methods for dealing with missing data. I felt this was both inherently interesting and had broad applicability to other domains.

## Business Context

 * The predictions are used to evaluate the return on investment of “completion designs” for newly drilled wells.
* “Completion designs” are essentially blueprints for treating (i.e., "stimulating") newly drilled wells to enable liquid oil recovery.
* There are certain parameters that are operator-controlled. The most important of these are:
  * Fluid amount (the amount of fluid pumped into the well to pressurize and fracture the rock around the bore hole)
   * Proppant amount (the amount of sand or similar material which is pumped in to "prop
open" the fractures, to allow oil to flow through)
   * Stage count (the number of sections, or segments, a lateral well is divided into and then pressurized individually).

* The predictions need to be generated on a pre-drilling basis - in other words, all data used to generate a prediction must be available before the well is drilled.

### Model Scoring
There are many possible answers to the question: "What constitutes a good model?"
Some examples are:

* Improved accuracy: Obviously the more accurate the model is, the better. Models are generally scored by RMSE (root mean squared error). A model with a lower RMSE score
constitutes an improvement.

* Improved ‘stability’ against key features: An important use case is to
predict the impact, or return on investment (ROI), of operator-controlled parameters described above. A model that is not stable relative to these parameters causes the end user to doubt the model and/or methodology.

* Improved 'responsiveness' against key features: As above, computing ROI against controllable parameters is the primary way users interact with the model. A model where the impacts of stage count, proppant, and fluid are most accurately represented is probably superior to a model where those features are ‘masked’ by other proxy signals.

* Improved performance across time intervals: the code provided only generates a
point-in-time prediction, but the purpose of the software, is to predict the “production curve” of a well - its performance across time. The current approach is to model different points in time independently, and apply a naive
‘smoothing’ step to enforce production increasing with time. Perhaps a different
approach is better. We have provided actual (observed) values as well as our predictions
to support analysis of your model versus ours in this context.


### Data
* Training and test sets came pre-defined. There is a separate test set (input.test.tsv) and training set (input.training.tsv ) of relevant data extracted from the NDIC (North Dakota
Industrial Commission) website.

* A description of the fields can be found in an HTML file.

......................................
## EDA
* Create master data dictionary, with field descriptions, proportion of null values for each
feature, and number of unique values for each feature.
* Examine Descriptive statistics of numerical features (mean, median, variance, etc.).
* Create a scatter matrix for numerical features.
* Create plots for each categorical feature: unique value counts and median target value for each unique value.
* Convert dates to useable format.

## Data Munging
* Drop features with high percentage of null values.
* Drop features with high cardinality (i.e., those categorical features with a high number of unique values).
* Drop other target variables.
* Deal with missing values.
* Binarize categorical features.

## Model Fitting
* Increase number of trees in Extra Random Forest Model.
* Fit scikit-learn's Gradient Boosted Trees Model.
* Fit an XG Boost trees model.
* Perform Grid Search to tune hyperparameters

......................................


## Investigating Fancy Imputation Methods

After EDA and Munging...

### Phase One: Mask

#### Procedure:
* Nonnumeric Features dropped
* Dropped rows with null values in that numeric feature column
* Retained rows with null values in other null columns
* Removed 10% of data at random, checked mean squared error
* Performing the above tests for every feature in the dataframe
* Remove larger percentages of data for every feature 10%, 20%, 30%, etc.

[Insert gifs with demo dataframe]

#### Result: Fancy imputation methods perform significantly better

### Phase Two: Fit
Fit a model to data after imputation and compare error scores for the model, and not the imputation method directly.

#### Procedure:
* Including categorical features this time, which must be binarized for both the train and the test sets in a consistent way.
* Nonnumeric features binarized and rejoined to filled dataframe.
* Also, the training set must be filled on its own, the test set must be filled attached to the training set.
* Fitting model on two different training sets: one where important feature is missing and one where it isn't.
* Add the feature back in, a few rows at a time, fit a model 50 times.
* Do this for every imputation method.

!(https://github.com/noproblem-james/data_imputation/blob/master/images/imputation_process.gif)

#### Result: Fancy imputation methods perform marginally better, depending on time horizon.
