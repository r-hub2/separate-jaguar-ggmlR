#' Compare Backend Operations Against the CPU
#'
#' Runs each supported operation on \code{backend} and on the CPU with identical
#' inputs, and reports the normalized mean squared error between the two. This is
#' the differential test used to find operations whose accelerated implementation
#' disagrees with the reference one.
#'
#' Each case is executed three times against the same allocated graph and the
#' final run is the one compared, so faults that only appear when a graph is
#' re-executed are visible. That matters: both backend defects found in this
#' package so far were of that kind -- the first run was exact and later ones
#' were not.
#'
#' @param backend Backend pointer, e.g. from \code{ggml_vulkan_init(0)}. Compared
#'   against a freshly created CPU backend.
#' @param filter Optional substring; only operations whose name contains it are
#'   run. \code{NULL} (default) runs all of them.
#' @return A data frame with columns \code{op}, \code{nmse} and \code{note}.
#'   \code{nmse} is \code{NA} when the case could not be run, and \code{note}
#'   then says why.
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available()) {
#'   be <- ggml_vulkan_init(0)
#'   res <- ggml_test_backend_ops(be)
#'   res[res$nmse > 1e-6, ]        # operations that disagree
#'   ggml_backend_free(be)
#' }
#' }
ggml_test_backend_ops <- function(backend, filter = NULL) {
  if (is.null(backend)) {
    stop("backend must be a backend pointer")
  }
  res <- .Call("R_ggml_test_backend_ops", backend,
               if (is.null(filter)) NULL else as.character(filter),
               PACKAGE = "ggmlR")
  data.frame(
    op   = res$op,
    nmse = res$nmse,
    note = res$note,
    stringsAsFactors = FALSE
  )
}

#' Compare the AdamW Optimizer Step Across Backends
#'
#' Runs \code{ggml_opt_step_adamw()} for several steps on \code{backend} and on
#' the CPU from the same starting weights and the same gradients, and reports the
#' weight checksum after each step.
#'
#' This needs its own function rather than a case in
#' \code{\link{ggml_test_backend_ops}}: the op updates the weights and both
#' moment buffers in place, so step N depends on every step before it, and the
#' bias-correction terms it receives change every step. Comparing a single
#' execution proves nothing here -- the first Adam step agrees bit-for-bit on
#' both backends and the disagreement only starts at the second.
#'
#' @param backend Backend pointer to compare against the CPU.
#' @return A data frame with \code{step}, \code{w_cpu}, \code{w_backend} and
#'   \code{abs_diff}, or a one-column data frame with \code{note} when the op
#'   could not be run.
#' @export
#' @seealso \code{\link{ggml_test_backend_ops}}
#' @examples
#' \donttest{
#' if (ggml_vulkan_available()) {
#'   be <- ggml_vulkan_init(0)
#'   print(ggml_test_adamw_steps(be))
#'   ggml_backend_free(be)
#' }
#' }
ggml_test_adamw_steps <- function(backend) {
  if (is.null(backend)) {
    stop("backend must be a backend pointer")
  }
  res <- .Call("R_ggml_test_adamw_steps", backend, PACKAGE = "ggmlR")
  if (!is.null(res$note)) {
    return(data.frame(note = res$note, stringsAsFactors = FALSE))
  }
  data.frame(
    step      = res$step,
    w_cpu     = res$w_cpu,
    w_backend = res$w_backend,
    abs_diff  = res$abs_diff,
    stringsAsFactors = FALSE
  )
}
