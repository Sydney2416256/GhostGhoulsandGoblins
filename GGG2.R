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
library(naivebayes)
library(remotes)
library(keras)

remotes::install_github("rstudio/tensorflow")
reticulate::install_python()

# Load Data
data <- vroom("STAT348/ghosts/train.csv") 
testdata <- vroom("STAT348/ghosts/test.csv") 

data$type <- as.factor(data$type)

# Define Recipe
my_recipe <- recipe(type ~ ., data = data) %>%
  step_mutate(color = as.factor(color)) %>%  # Ensure color is a factor (if necessary)
  step_dummy(all_nominal(), -all_outcomes()) %>%  # Convert categorical variables to dummies
  step_normalize(all_numeric_predictors()) %>%  # Normalize numeric predictors
  step_range(all_numeric_predictors(), min = 0, max = 1)  # Scale to [0,1] for numeric predictors

# Prepare and bake the recipe
prepped_recipe <- prep(my_recipe)
train_data <- bake(prepped_recipe, new_data = data)
test_data <- bake(prepped_recipe, new_data = testdata)

# Define Neural Network Model
nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
  set_engine("keras") %>%  # Use the keras engine
  set_mode("classification")

# Create Workflow
wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nn_model)

# Define Tuning Grid for hidden_units
tuning_grid <- grid_regular(
  hidden_units(range = c(1, 100)), # Define grid search space for hidden_units
  levels = 5
)

# Split data for cross-validation
folds <- vfold_cv(data, v = 5, repeats = 1)

# Tune the Model using Cross-Validation
CV_results <- wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))  # You can add other metrics here

# Plot results of tuning
CV_results %>% 
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x = hidden_units, y = mean)) +
  geom_line() +
  labs(title = "Model Performance vs Hidden Units")

# Get the best tuning parameters
bestTune <- select_best(CV_results, metric = "accuracy")

# Finalize the Workflow & Fit the Model
final_wf <- wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = data)

# Make Predictions on Test Data
ggg_predictions <- predict(final_wf, 
                           new_data = testdata,
                           type = "class") 

# Prepare Kaggle Submission
kaggle_submission <- ggg_predictions %>%
  bind_cols(testdata) %>%  # Bind predictions with test data
  select(id, .pred_class) %>%  # Select columns for submission
  rename(type = .pred_class)  # Rename prediction to 'type'

# Write out the file for submission
vroom_write(kaggle_submission, file = "./GGG2.csv", delim = ",")