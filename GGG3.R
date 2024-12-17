# Load in Libraries
library(tidyverse)
library(vroom)
library(DataExplorer)
library(patchwork)
library(tidymodels)
library(dplyr)
library(poissonreg)
library(glmnet)
library(ggplot2)
library(gridExtra)
library(embed)
library(discrim)
library(naivebayes)
library(bonsai)
library(lightgbm)



# Load in Data
data <- vroom("STAT348/ghosts/train.csv") 
testdata <- vroom("STAT348/ghosts/test.csv") 

data$type <- as.factor(data$type)




my_recipe <- recipe(type ~ ., data = data) %>%
  step_mutate(color = as.factor(color)) %>%  # Ensure color is a factor (if necessary)
  step_dummy(all_nominal(), -all_outcomes()) %>%  # Convert categorical variables to dummies
  step_normalize(all_numeric_predictors()) %>%  # Normalize numeric predictors
  step_range(all_numeric_predictors(), min = 0, max = 1)  # Scale to [0,1] for numeric predictors




prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet
bake(prepped_recipe, new_data=testdata)



## nb model
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)



## Tune smoothness and Laplace here
## Grid of values to tune over
tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 5) 

## Split data for CV
folds <- vfold_cv(data, v = 5, repeats=1)

## Run the CV
CV_results <- wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL


bestTune <- CV_results %>%
  select_best()

## Finalize the Workflow & fit it
final_wf <- wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=data)













## Make predictions
ggg_predictions <- predict(final_wf,
                           new_data=testdata,
                           type= "class") 




kaggle_submission <- ggg_predictions %>%
  bind_cols(., testdata) %>% #Bind predictions with test data
  select(id, .pred_class) %>% #Just keep datetime and prediction variables
  rename(type=.pred_class) #rename pred to count (for submission to Kaggle)


## Write out the file
vroom_write(x=kaggle_submission, file="./GGG2.csv", delim=",")

