## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment  = "#>",
  eval     = requireNamespace("parsnip", quietly = TRUE)
)

## -----------------------------------------------------------------------------
library(ggmlR)
library(parsnip)

## -----------------------------------------------------------------------------
spec <- mlp(
  hidden_units = c(64L, 32L),
  epochs       = 20L,
  dropout      = 0.1
) |>
  set_engine("ggml") |>
  set_mode("classification")

fit_obj <- fit(spec, Species ~ ., data = iris)

# Class predictions
preds <- predict(fit_obj, new_data = iris)
head(preds)

# Probability predictions
probs <- predict(fit_obj, new_data = iris, type = "prob")
head(probs)

# Accuracy
cat(sprintf("Accuracy: %.4f\n", mean(preds$.pred_class == iris$Species)))

## -----------------------------------------------------------------------------
spec_reg <- mlp(
  hidden_units = c(64L, 32L),
  epochs       = 50L
) |>
  set_engine("ggml") |>
  set_mode("regression")

fit_reg <- fit(spec_reg, mpg ~ ., data = mtcars)

preds_reg <- predict(fit_reg, new_data = mtcars)
head(preds_reg)

## -----------------------------------------------------------------------------
# Customize architecture
spec_custom <- mlp(
  hidden_units = c(128L, 64L, 32L),
  epochs       = 30L,
  dropout      = 0.3,
  activation   = "relu"
) |>
  set_engine("ggml") |>
  set_mode("classification")

## ----eval=FALSE---------------------------------------------------------------
# library(rsample)
# 
# folds <- vfold_cv(iris, v = 5L)
# 
# spec <- mlp(hidden_units = c(32L), epochs = 10L) |>
#   set_engine("ggml") |>
#   set_mode("classification")
# 
# library(tune)
# library(yardstick)
# library(workflows)
# 
# wf <- workflow() |>
#   add_model(spec) |>
#   add_formula(Species ~ .)
# 
# results <- fit_resamples(wf, resamples = folds)
# collect_metrics(results)

## ----eval=FALSE---------------------------------------------------------------
# library(recipes)
# library(workflows)
# 
# rec <- recipe(Species ~ ., data = iris) |>
#   step_normalize(all_numeric_predictors())
# 
# spec <- mlp(hidden_units = c(32L), epochs = 10L) |>
#   set_engine("ggml") |>
#   set_mode("classification")
# 
# wf <- workflow() |>
#   add_recipe(rec) |>
#   add_model(spec)
# 
# fit_obj <- fit(wf, data = iris)
# predict(fit_obj, new_data = iris)

## ----eval=FALSE---------------------------------------------------------------
# rec <- recipe(Species ~ ., data = iris) |>
#   step_dummy(all_nominal_predictors()) |>
#   step_normalize(all_numeric_predictors())

## ----eval=FALSE---------------------------------------------------------------
# library(tune)
# library(dials)
# library(workflows)
# 
# spec <- mlp(
#   hidden_units = tune(),
#   epochs       = tune(),
#   dropout      = tune()
# ) |>
#   set_engine("ggml") |>
#   set_mode("classification")
# 
# wf <- workflow() |>
#   add_model(spec) |>
#   add_formula(Species ~ .)
# 
# grid <- grid_regular(
#   hidden_units(range = c(16L, 128L)),
#   epochs(range = c(10L, 50L)),
#   dropout(range = c(0, 0.4)),
#   levels = 3L
# )
# 
# folds <- vfold_cv(iris, v = 3L)
# results <- tune_grid(wf, resamples = folds, grid = grid)
# show_best(results, metric = "accuracy")

## ----eval=FALSE---------------------------------------------------------------
# library(workflows)
# library(workflowsets)
# 
# specs <- workflow_set(
#   preproc = list(basic = Species ~ .),
#   models  = list(
#     ggml  = mlp(hidden_units = c(32L), epochs = 20L) |> set_engine("ggml"),
#     nnet  = mlp(hidden_units = 32L,    epochs = 200L) |> set_engine("nnet")
#   )
# ) |>
#   workflow_map("fit_resamples",
#                resamples = vfold_cv(iris, v = 5L))
# 
# rank_results(specs, rank_metric = "accuracy")

