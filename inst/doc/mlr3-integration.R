## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment  = "#>",
  eval     = requireNamespace("mlr3", quietly = TRUE) &&
             requireNamespace("paradox", quietly = TRUE)
)
library(ggmlR)
if (requireNamespace("mlr3", quietly = TRUE)) {
  library(mlr3)
  ggmlR:::.register_mlr3()
}

## -----------------------------------------------------------------------------
library(ggmlR)
library(mlr3)

task    <- tsk("iris")
learner <- lrn("classif.ggml",
               epochs     = 20L,
               batch_size = 16L,
               predict_type = "prob")

learner$train(task)
pred <- learner$predict(task)
pred$score(msr("classif.acc"))

## -----------------------------------------------------------------------------
task    <- tsk("mtcars")
learner <- lrn("regr.ggml",
               epochs     = 50L,
               batch_size = 8L)

learner$train(task)
pred <- learner$predict(task)
pred$score(msr("regr.rmse"))

## -----------------------------------------------------------------------------
learner <- lrn("classif.ggml")
learner$param_set$values$epochs        <- 30L
learner$param_set$values$hidden_layers <- c(256L, 128L, 64L)
learner$param_set$values$dropout       <- 0.3
learner$param_set$values$backend       <- "gpu"

## ----eval=ggml_vulkan_available()---------------------------------------------
learner <- lrn("classif.ggml",
               backend = "gpu",
               epochs  = 100L)
learner$train(tsk("iris"))

## -----------------------------------------------------------------------------
learner <- lrn("classif.ggml",
               epochs     = 50L,
               batch_size = 16L)

learner$model_fn <- function(task, n_features, n_out, pars) {
  ggml_model_sequential() |>
    ggml_layer_dense(64L, activation = "relu", input_shape = n_features) |>
    ggml_layer_dropout(rate = 0.3) |>
    ggml_layer_dense(32L, activation = "relu") |>
    ggml_layer_dense(n_out, activation = "softmax")
}

learner$train(tsk("iris"))

## -----------------------------------------------------------------------------
task    <- tsk("iris")
learner <- lrn("classif.ggml",
               epochs     = 20L,
               batch_size = 16L,
               backend    = "cpu")

rr <- resample(task, learner, rsmp("cv", folds = 5L))
rr$aggregate(msr("classif.acc"))

## ----eval=FALSE---------------------------------------------------------------
# design <- benchmark_grid(
#   tasks    = tsk("iris"),
#   learners = list(
#     lrn("classif.ggml", epochs = 20L, batch_size = 16L, backend = "cpu"),
#     lrn("classif.ggml", epochs = 20L, batch_size = 16L, backend = "gpu")
#   ),
#   resamplings = rsmp("cv", folds = 5L)
# )
# bmr <- benchmark(design)
# bmr$aggregate(msr("classif.acc"))

## ----eval=FALSE---------------------------------------------------------------
# library(mlr3tuning)
# 
# learner <- lrn("classif.ggml", backend = "gpu")
# 
# search_space <- ps(
#   epochs     = p_int(lower = 10L, upper = 100L),
#   batch_size = p_int(lower = 8L,  upper = 64L),
#   dropout    = p_dbl(lower = 0,   upper = 0.5)
# )
# 
# instance <- ti(
#   task       = tsk("iris"),
#   learner    = learner,
#   resampling = rsmp("cv", folds = 3L),
#   measures   = msr("classif.acc"),
#   terminator = trm("evals", n_evals = 20L)
# )
# 
# tuner <- tnr("random_search")
# tuner$optimize(instance)
# 
# instance$result

## -----------------------------------------------------------------------------
learner <- lrn("classif.ggml",
               epochs     = 200L,
               batch_size = 16L,
               callbacks  = list(
                 ggml_callback_early_stopping(
                   monitor  = "val_loss",
                   patience = 10L
                 )
               ),
               validation_split = 0.2)

learner$train(tsk("iris"))

## -----------------------------------------------------------------------------
d <- data.frame(
  x1 = rnorm(100),
  x2 = rnorm(100),
  y  = factor(rep(c("a", "b"), each = 50)),
  w  = c(rep(2.0, 50), rep(0.5, 50))
)
task <- as_task_classif(d, target = "y")
task$set_col_roles("w", roles = "weights_learner")

learner <- lrn("classif.ggml", epochs = 20L)
learner$train(task)

## -----------------------------------------------------------------------------
learner <- lrn("classif.ggml", epochs = 10L, backend = "cpu")
learner$train(tsk("iris"))

learner$marshal()
learner$marshaled
#> [1] TRUE

learner$unmarshal()
learner$marshaled
#> [1] FALSE

# Predictions are identical after roundtrip
pred <- learner$predict(tsk("iris"))

## -----------------------------------------------------------------------------
model <- ggml_model_sequential() |>
  ggml_layer_dense(16L, activation = "relu", input_shape = 4L) |>
  ggml_layer_dense(3L,  activation = "softmax")
model <- ggml_compile(model, optimizer = "adam",
                      loss = "categorical_crossentropy")

blob <- ggml_marshal_model(model)
blob

model2 <- ggml_unmarshal_model(blob)

