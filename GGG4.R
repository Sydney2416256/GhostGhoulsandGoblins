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
library(themis)



# Load in Data
data <- vroom("STAT348/ghosts/train.csv") 
testdata <- vroom("STAT348/ghosts/test.csv") 

data$type <- as.factor(data$type)




my_recipe <- recipe(type ~. , data=data) %>%
  
  step_mutate(color = as.factor(color)) |> 
  step_mutate(id, features=id) |> 
  
  # step_lencode_mixed(all_nominal_predictors(), outcome = vars(type)) |> 
  step_normalize(all_numeric_predictors()) 




prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myDataSet
bake(prepped_recipe, new_data=testdata)



## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naiveb

wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)



## Tune smoothness and Laplace here
## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(range = c(.1, 2)),
                            levels = 14) 

## Split data for CV
folds <- vfold_cv(data, v = 9, repeats=1)

## Run the CV
CV_results <- wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(f_meas)) #Or leave metrics NULL


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
vroom_write(x=kaggle_submission, file="./GGG16.csv", delim=",")

