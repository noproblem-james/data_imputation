# Prediciting Oil Production for Unconventional Wells in the Williston Basin
## Problem Description and Methods Used

### Objective
The overarching goal is to build a model for operators of oil and gas wells to predict oil production for new wells at different time horizons, in different geologies, under different economic conditions.

### Business Context

* The predictions are used to evaluate the return on investment of “completion designs” for newly drilled wells.
 * “Completion designs” are essentially blueprints for treating (i.e., "stimulating") newly drilled wells to enable liquid oil recovery.
* There are certain parameters that are operator-controlled. The most important of these are:
  * Fluid amount (the amount of fluid pumped into the well to pressurize and fracture the rock around the bore hole)
  * Proppant amount (the amount of sand or similar material which is pumped in to "prop
open" the fractures, to allow oil to flow through)
  * Stage count (the number of sections, or segments, a lateral well is divided into and then pressurized individually).

* The predictions need to be generated on a pre-drilling basis - in other words, all data used to generate a prediction must be available before the well is drilled.

### Model Evaluation
There are many possible answers to the question: "What constitutes a good model?"
Some examples are:

* Improved **accuracy**: Obviously the more accurate the model is, the better. Models are generally scored by RMSE (root mean squared error). A model with a lower RMSE score
constitutes an improvement.

* Improved **stability** against key features: An important use case is to
predict the impact, or return on investment (ROI), of operator-controlled parameters described above. A model that is not stable relative to these parameters causes the end user to doubt the model and/or methodology.

* Improved **responsiveness** against key features: As above, computing ROI against controllable parameters is the primary way users interact with the model. A model where the impacts of stage count, proppant, and fluid are most accurately represented is probably superior to a model where those features are ‘masked’ by other proxy signals.

* Improved **performance across time intervals**: the code provided only generates a
point-in-time prediction, but the goal is to predict the “production curve” of a well - its performance across time.

......................................
## EDA
* Create master data dictionary, with field descriptions, proportion of null values for each
feature, number of unique values for each feature, variation of each feature, and covariance of each feature with the target.
* Examine Descriptive statistics of numerical features (mean, median, variance, etc.).
* Create a scatter matrix for numerical features.
* Create plots for each categorical feature: unique value counts and median target value for each unique value.
* Convert dates to useable format.

## Data Munging
* Drop features with high cardinality (i.e., those categorical features with a high number of unique values) or that may provide proxy signals that are beyond the control of the operator (field name, date of frac job, latitude and longitude of the well itself)
* Binarize `stimulated_formation` (only two values are really important: Bakken and Three Forks)
* Parse `choke_size` to convert it from free text (a string representing a fraction) into a numeric float value.

## Feature Engineering
* Estimate the lateral length of each well (distance from top hole to bottom hole)
* Calculate each well's distance to nearest neighbor (distance from midpoint of lateral of Well A to to midpoint of lateral for Well B)

## Missing Data Treatment
- Pattern to missingness: Missingness is highly correlated with date the well was completed (`spud_year`)
- Drop all observations before 2010.
    - data quality issue and/or a relevance issue
    - relatively small number of observations pre-2010, and approximately 90% of those are missing values
- Imputation Strategy:
  - Use auxiliary features (date, lat/lon) to impute missing data (but not within model)
  - Use a multiple imputation technique

## Model Fitting
* Define a target: For the sake of simplicity, the model predicts cumulative oil production for newly drilled wells at 180 days (“IP180”)
* Fit scikit-learn's Gradient Boosted Trees Model.
* Perform Grid Search to tune hyperparameters

## Model Evaluation
- Scatterplots of residuals for exploration
- Feature importances

......................................
