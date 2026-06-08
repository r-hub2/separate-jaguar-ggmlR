## ----setup, include=FALSE-----------------------------------------------------
# Vignette code is executed locally (NOT_CRAN=true) but not on CRAN, where
# the CPU fallback would multi-thread and trip the "CPU time > elapsed" NOTE.
knitr::opts_chunk$set(eval = identical(Sys.getenv("NOT_CRAN"), "true"))

## -----------------------------------------------------------------------------
# my_inference <- function(input) {
#   .Call("R_my_inference", as.numeric(input))
# }

