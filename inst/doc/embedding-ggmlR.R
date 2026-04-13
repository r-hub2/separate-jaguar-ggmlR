## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(eval = TRUE)

## -----------------------------------------------------------------------------
my_inference <- function(input) {
  .Call("R_my_inference", as.numeric(input))
}

