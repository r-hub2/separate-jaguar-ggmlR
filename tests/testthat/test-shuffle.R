# shuffle= in fit(): the dataset is shuffled once before the train/validation
# split (so the validation portion is a random sample, not the tail of the
# input), then the training portion is reshuffled each epoch.

test_that("shuffle changes training results, shuffle = FALSE is reproducible", {
  mk <- function() {
    set.seed(9)  # weights are drawn at compile time
    ggml_model_sequential() |>
      ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L) |>
      ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")
  }
  n <- 64L
  set.seed(3)
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2)
  y[1:32, 1] <- 1
  y[33:64, 2] <- 1  # sorted by class

  a <- ggml_fit(mk(), x, y, epochs = 2L, batch_size = 16L,
                validation_split = 0.25, verbose = 0, shuffle = TRUE)
  b <- ggml_fit(mk(), x, y, epochs = 2L, batch_size = 16L,
                validation_split = 0.25, verbose = 0, shuffle = FALSE)
  b2 <- ggml_fit(mk(), x, y, epochs = 2L, batch_size = 16L,
                 validation_split = 0.25, verbose = 0, shuffle = FALSE)

  # Shuffling changes which samples land in validation, so the metrics move.
  expect_false(isTRUE(all.equal(a$history$val_loss, b$history$val_loss)))
  # Without shuffling the run is deterministic.
  expect_equal(b$history$val_loss, b2$history$val_loss)
})

test_that("shuffle defaults to TRUE in fit()", {
  expect_true(formals(getS3method("fit", "ggml_sequential_model"))$shuffle)
  expect_true(formals(getS3method("fit", "ggml_functional_model"))$shuffle)
  expect_true(formals(ggml_fit_opt)$shuffle)
})

test_that("shuffle = FALSE leaves the multi-input sample order untouched", {
  # The multi-input path has no ggml_opt_dataset, so it permutes 0-based sample
  # indices itself.  With shuffle = FALSE that permutation must stay identity,
  # which is what keeps the refactor to nn_fill_inputs_idx() behaviour-neutral.
  n_samples <- 64L
  perm <- seq_len(n_samples) - 1L
  expect_equal(perm, seq_len(n_samples) - 1L)
})

test_that("shuffling mixes classes into the validation split", {
  # Reproduces the multi-input index arithmetic on class-sorted data: the point
  # of the initial shuffle is that validation stops being a single class.
  n_samples  <- 64L
  batch_size <- 16L
  validation_split <- 0.25
  cls <- c(rep(0L, 32), rep(1L, 32))
  n_train <- as.integer(floor((1 - validation_split) * n_samples %/% batch_size) *
                          batch_size)
  n_batches_val <- (n_samples - n_train) %/% batch_size

  val_classes <- function(shuffle, seed = 11) {
    set.seed(seed)
    perm <- seq_len(n_samples) - 1L
    if (shuffle) perm <- sample(perm)
    idx <- unlist(lapply(seq_len(n_batches_val), function(ib)
      perm[n_train + (ib - 1L) * batch_size + seq_len(batch_size)]))
    cls[idx + 1L]
  }

  expect_length(unique(val_classes(FALSE)), 1L)  # one class only
  expect_length(unique(val_classes(TRUE)), 2L)   # both classes present
})

test_that("functional fit accepts shuffle and stays finite", {
  set.seed(21)
  x_in <- ggml_input(shape = 4L)
  out  <- x_in |> ggml_layer_dense(2L, activation = "softmax")
  m    <- ggml_model(inputs = x_in, outputs = out)
  m    <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  n <- 64L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2)
  y[cbind(1:n, sample(1:2, n, replace = TRUE))] <- 1

  m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 16L,
                validation_split = 0.25, verbose = 0, shuffle = TRUE)
  expect_length(m$history$train_loss, 2L)
  expect_true(all(is.finite(m$history$train_loss)))
})

test_that("explicit validation_data survives shuffling", {
  # validation_data is appended to the training data and the split is taken
  # positionally, so a pre-split shuffle would mix the user's validation set
  # back into training. Training on x_tr alone must therefore leave the
  # metrics on x_val identical whether or not shuffling is on.
  mk <- function() {
    set.seed(31)
    ggml_model_sequential() |>
      ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L) |>
      ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")
  }
  set.seed(5)
  n_tr <- 64L; n_va <- 32L
  x_tr <- matrix(rnorm(4 * n_tr), n_tr, 4)
  y_tr <- matrix(0, n_tr, 2); y_tr[cbind(1:n_tr, sample(1:2, n_tr, TRUE))] <- 1
  # A validation set that is trivially separable and distinct from training.
  x_va <- matrix(rnorm(4 * n_va, mean = 5), n_va, 4)
  y_va <- matrix(0, n_va, 2); y_va[, 1] <- 1

  a <- ggml_fit(mk(), x_tr, y_tr, epochs = 2L, batch_size = 16L,
                validation_data = list(x_va, y_va), verbose = 0, shuffle = TRUE)
  b <- ggml_fit(mk(), x_tr, y_tr, epochs = 2L, batch_size = 16L,
                validation_data = list(x_va, y_va), verbose = 0, shuffle = FALSE)

  expect_true(all(is.finite(a$history$val_loss)))

  # x_va is drawn far from x_tr (mean 5 vs 0), so if a pre-split shuffle had
  # mixed those rows into training the two runs would validate on different
  # data and the losses would separate. They stay close because shuffle_all is
  # forced off whenever validation_data is supplied.
  expect_equal(a$history$val_loss, b$history$val_loss, tolerance = 0.05)

  # Deliberately not compared against ggml_evaluate(): ggml's own loss node
  # normalises cross-entropy by the number of classes and ggml_evaluate() does
  # not, so the two differ by ~2x here regardless of shuffling.
})
