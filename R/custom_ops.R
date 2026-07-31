# Custom operations: graph nodes computed by a C kernel registered by name.
#
# The kernel itself lives in C, in this or another package, and is addressed by
# name rather than by a raw function pointer. See inst/include/ggmlR.h for the
# registration side.

#' Maximum Task Count for Custom Ops
#'
#' Sentinel for the \code{n_tasks} argument of \code{\link{ggml_custom}} and
#' \code{\link{ggml_custom_inplace}}, meaning "use as many threads as the
#' backend allows" (mirrors \code{GGML_N_TASKS_MAX} in ggml.h).
#'
#' @export
GGML_N_TASKS_MAX <- -1L

#' Custom Operation Node
#'
#' Creates a graph node whose value is computed by a C kernel registered under
#' \code{name}. The kernel is looked up in ggmlR's custom-op registry; supplying
#' an unregistered name is an error, listing the names that are available.
#'
#' Kernels are registered from C, not from R. A package that links against
#' ggmlR registers its kernels at load time via the \code{ggmlR_register_custom_op}
#' callable; see \code{system.file("include", "ggmlR.h", package = "ggmlR")} for
#' the C-side contract and a worked example. R callbacks are not supported: the
#' kernel may run off the main thread, where calling into R is unsafe.
#'
#' The output tensor is newly allocated with the given type and shape. Input
#' tensors passed in \code{args} become the node's sources and are visible to
#' the kernel as \code{dst->src[0]}, \code{dst->src[1]}, and so on.
#'
#' @section Backend:
#' Custom ops run on the CPU backend only, because the kernel is a host function
#' pointer that a GPU backend cannot execute. Computing a graph that contains a
#' custom node raises an error when the target backend is not the CPU, or when a
#' scheduler has no CPU backend to fall back to. Keep custom nodes out of graphs
#' you intend to run entirely on Vulkan.
#'
#' @param ctx GGML context
#' @param name Name of a registered custom op (character scalar)
#' @param args List of input tensors, or NULL for none. At most 9 tensors.
#' @param ne Output shape as a numeric vector of up to 4 dimensions, innermost
#'   first (ggml column-major order). Missing dimensions default to 1.
#' @param type Output data type (default \code{GGML_TYPE_F32})
#' @param n_tasks Number of threads the kernel may be split across, or
#'   \code{GGML_N_TASKS_MAX} (-1) to use as many as the backend allows. Use 1
#'   for a kernel that is not thread-safe.
#' @return Tensor pointer for the custom node
#' @seealso \code{\link{ggml_custom_inplace}}, \code{\link{ggml_custom_ops}}
#' @export
#' @examples
#' # "row_median" is one of ggmlR's built-in kernels; see ggml_custom_ops().
#' ggml_set_n_threads(1L)
#' ctx <- ggml_init(16 * 1024 * 1024)
#' x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 2)
#' ggml_set_f32(x, c(5, 1, 3, 2, 4,
#'                   9, 7, 8, 6, 10))
#'
#' # One median per row: the output collapses ne[1] to 1.
#' med <- ggml_custom(ctx, "row_median", args = list(x), ne = c(1, 2))
#'
#' graph <- ggml_build_forward_expand(ctx, med)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(med)   # 3, 8
#' ggml_free(ctx)
ggml_custom <- function(ctx, name, args = NULL, ne, type = GGML_TYPE_F32,
                        n_tasks = GGML_N_TASKS_MAX) {
  if (!is.character(name) || length(name) != 1L) {
    stop("'name' must be a single character string naming a registered custom op")
  }
  if (!is.null(args) && !is.list(args)) {
    stop("'args' must be a list of tensors or NULL")
  }

  ne <- as.numeric(ne)
  if (length(ne) < 1L || length(ne) > 4L) {
    stop("'ne' must have between 1 and 4 dimensions")
  }
  if (anyNA(ne) || any(ne < 1)) {
    stop("'ne' must contain positive dimensions")
  }
  # Pad to 4D: unused trailing dimensions are 1, as elsewhere in ggml.
  ne <- c(ne, rep(1, 4L - length(ne)))

  .Call("R_ggml_custom_4d", ctx, as.integer(type),
        ne[1], ne[2], ne[3], ne[4],
        args, name, as.integer(n_tasks), PACKAGE = "ggmlR")
}

#' In-Place Custom Operation Node
#'
#' Creates a custom-op node that writes into a view of \code{a} rather than into
#' a freshly allocated tensor. The output has the shape and type of \code{a},
#' and \code{a} itself is the node's first source; tensors in \code{args} follow
#' it, so the kernel sees \code{a} as \code{dst->src[0]} and the first extra
#' argument as \code{dst->src[1]}.
#'
#' See \code{\link{ggml_custom}} for how kernels are registered and for the
#' CPU-only backend restriction, which applies here too.
#'
#' @param ctx GGML context
#' @param a Tensor to write into
#' @param name Name of a registered custom op (character scalar)
#' @param args List of additional input tensors, or NULL for none. At most 8
#'   tensors, one fewer than \code{\link{ggml_custom}} because \code{a} occupies
#'   the first source slot.
#' @param n_tasks Number of threads the kernel may be split across, or
#'   \code{GGML_N_TASKS_MAX} (-1). Use 1 for a kernel that is not thread-safe.
#' @return Tensor pointer for the custom node (a view of \code{a})
#' @seealso \code{\link{ggml_custom}}, \code{\link{ggml_custom_ops}}
#' @export
#' @examples
#' # "clip_inplace" takes its bounds as a 2-element tensor, since a custom node
#' # carries no scalar parameters of its own.
#' ggml_set_n_threads(1L)
#' ctx <- ggml_init(16 * 1024 * 1024)
#' x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(x, c(-3, -1, 0, 1, 3))
#'
#' bounds <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
#' ggml_set_f32(bounds, c(-1, 1))
#'
#' y <- ggml_custom_inplace(ctx, x, "clip_inplace", args = list(bounds))
#' graph <- ggml_build_forward_expand(ctx, y)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(y)   # -1, -1, 0, 1, 1
#' ggml_free(ctx)
ggml_custom_inplace <- function(ctx, a, name, args = NULL,
                                n_tasks = GGML_N_TASKS_MAX) {
  if (!is.character(name) || length(name) != 1L) {
    stop("'name' must be a single character string naming a registered custom op")
  }
  if (!is.null(args) && !is.list(args)) {
    stop("'args' must be a list of tensors or NULL")
  }

  .Call("R_ggml_custom_inplace", ctx, a, args, name,
        as.integer(n_tasks), PACKAGE = "ggmlR")
}

#' List Registered Custom Ops
#'
#' Returns the names of the C kernels currently registered with ggmlR, in
#' registration order. Useful for checking that a downstream package registered
#' its kernels as expected.
#'
#' @section Built-in kernels:
#' ggmlR registers three kernels at load time. Each one exercises a different
#' path through the custom-op API, and they double as worked examples:
#'
#' \describe{
#'   \item{\code{"row_median"}}{Median of each row. One input; the output shape
#'     is chosen by the caller as \code{ne[1] = 1}, collapsing the row. Use
#'     \code{ne = c(1, nrow)} for an \code{n x nrow} input.}
#'   \item{\code{"row_permute"}}{\code{out[i] = x[perm[i]]} within each row, with
#'     \code{perm} 0-based. Two inputs: F32 data and an I32 index tensor.
#'     Indices outside the row yield 0. This is not \code{\link{ggml_get_rows}},
#'     which permutes whole rows rather than elements inside one.}
#'   \item{\code{"clip_inplace"}}{Clamps values into \code{[lo, hi]}, writing
#'     into the input. For use with \code{\link{ggml_custom_inplace}}; the
#'     bounds are passed as a 2-element F32 tensor, since a custom node carries
#'     no scalar parameters of its own. \code{\link{ggml_clamp}} does the same
#'     thing as a graph op -- this kernel exists as the in-place example.}
#' }
#'
#' @return Character vector of registered custom op names, empty if none
#' @seealso \code{\link{ggml_custom}}, \code{\link{ggml_custom_inplace}}
#' @export
#' @examples
#' ggml_custom_ops()
ggml_custom_ops <- function() {
  .Call("R_ggml_custom_ops", PACKAGE = "ggmlR")
}
