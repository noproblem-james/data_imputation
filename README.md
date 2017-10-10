# Capstone

## Objective
The goal is to improve on a preexisting model that predicts oil production for new oil wells at different time horizons, in different geologies, under different economic conditions. For the sake of simplicity, we will focus on predicting cumulative oil production for newly drilled wells at 90 days (“IP90”, i.e., initial production at 90 days), and also at 180 days (“IP180”) and 365 days (“IP365”).

A secondary goal is to investigate methods for dealing with missing data.

## Business Context

 * The predictions are used to evaluate the return on investment of “completions designs” for newly drilled wells.
* “Completions designs” are essentially blueprints for treating (i.e., "stimulating") newly drilled wells to enable liquid oil recovery.
* There are certain parameters that are operator-controlled. The most important of these are:
  * fluid amount (the amount of fluid pumped into the well to pressurize and fracture the rock around the bore hole)
   * proppant amount (the amount of sand or similar material which is pumped in to "prop
open" the fractures, to allow oil to flow through)
   * stage count (the number of sections, or segments, a lateral well is divided into and then pressurized individually).

The predictions need to be generated on a pre-drilling basis - in other words, all data used to generate a prediction must be available before the well is drilled.

### Model Scoring
There are many possible answers to the question: "What constitutes a good model?"
Some examples are:

* Improved accuracy: Obviously the more accurate the model is, the better. Models are generally scored by RMSE (root mean squared error). A model with better RMSE score
would be an improvement.

* Improved ‘stability’ against key features: An important use case is to
predict the impact, or return on investment (ROI), of operator-controlled parameters described above. A model that is not stable relative to these
parameters looks ‘bad’ to the end users, and causes them to lose trust in the model &
methodology.

* Improved ‘accuracy’ against key features: As above, computing ROI against controllable
parameters is the primary way users interact with the model. A model where the impacts of stage count, proppant, and fluid are most accurately represented is probably superior to a model where those features are ‘masked’ by other proxy signals.

* Improved performance across time intervals: the code we provided only generates a
point-in-time prediction, but the purpose of the software, is to predict the “production curve” of a well - it’s performance across time. The current approach is to model different points in time independently, and apply a naive
‘smoothing’ step to enforce production increasing with time. Perhaps a different
approach is better. We have provided actual (observed) values as well as our predictions
to support analysis of your model versus ours in this context.


### Data
* There is a separate test set (input.test.tsv) and training set
(input.training.tsv ) TSV of relevant data extracted from the NDIC (North Dakota
Industrial Commission) website.

* A description of the fields can be found in an HTML file.

......................................
## EDA
* Create master data dictionary, with field descriptions, proportion of null values for each
feature, and number of unique values for each feature.
* Descriptive statistics of numerical features.
* Scatter matrix for numerical features
* Create date-time variables

## Data Munging
* Drop features with high percentage of null values
* Drop features with high cardinality (i.e., those categorical features with a high number of unique values)
* Drop other target variables.

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

[insert gifs of demo dataframe, and of train-test-split methods]

#### Result: Fancy imputation methods perform marginally better, depending on time horizon.
