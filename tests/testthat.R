
library(testthat)
library(ggmlR)

# Limit threads to avoid CRAN NOTE about CPU time >> elapsed time
ggml_set_n_threads(2L)

test_check("ggmlR")
