# piece_transformer.R — a transformer over the pieces of a puzzle
#
# The model reads a puzzle state as a sequence of PIECES rather than of
# stickers. A 4x4x4 cube has 96 stickers but only 56 pieces -- 8 corners of
# three stickers, 24 wings of two, 24 centres of one -- and a piece is the
# thing that actually moves, so that is what the attention runs over.
#
# Each piece becomes one token: its stickers' colours are embedded, the empty
# sticker slots masked off, the lot projected to d_model, and a position and a
# type embedding added. A CLS token is prepended, four pre-norm blocks run, and
# the CLS row is read out to a value per action.
#
# The weights come from a PyTorch checkpoint via pth_load(). What differs
# between the two libraries is handled at load time, in pt_transformer_load():
#
#   - torch stores Q, K, V as one in_proj_weight [3d, d]; ag_multihead_
#     attention keeps them apart, so it is split in three
#   - ag_multihead_attention has no input biases, so in_proj_bias is applied
#     by hand around the layer
#   - pth_load() returns arrays with the dimensions reversed (row-major bytes
#     read column-major), which suits ag_embedding but not ag_linear
#
# Minimal API:
#   pt_cube4_layout()            — the 56 pieces of a 4x4x4
#   pt_transformer(layout, ...)  — build the model
#   pt_transformer_load(path, …) — build it and fill it from a .pth
#   pt_forward(model, states)    — run it

# ---- layout ----------------------------------------------------------------

#' The piece layout of a 4x4x4 cube
#'
#' Which stickers make up each piece, for a cube written as 96 stickers in face
#' order. Pieces come in three kinds and the model is told which is which: 8
#' corners of three stickers, 24 wings of two, 24 centres of one. The short
#' pieces are padded to three slots and the padding masked off.
#'
#' Sticker numbers are 0-based, as the checkpoint's own layout table has them.
#'
#' @return A list with \code{positions} (a \code{56 x 3} integer matrix of
#'   sticker indices), \code{mask} (\code{56 x 3} logical, which slots are
#'   real), \code{types} (length-56 integer, 0 corner / 1 wing / 2 centre),
#'   \code{state_size} and \code{num_classes}.
#' @export
#' @seealso \code{\link{pt_transformer}}
#' @examples
#' lay <- pt_cube4_layout()
#' dim(lay$positions)
#' table(lay$types)
pt_cube4_layout <- function() {
  corners <- matrix(c(
    0, 51, 64,   3, 35, 48,  12, 16, 67,  15, 19, 32,
    28, 79, 80,  31, 44, 83,  47, 60, 95,  63, 76, 92
  ), ncol = 3L, byrow = TRUE)

  wings <- matrix(c(
    1, 50,   2, 49,   4, 65,   7, 34,   8, 66,  11, 33,
    13, 17,  14, 18,  20, 71,  23, 36,  24, 75,  27, 40,
    29, 81,  30, 82,  39, 52,  43, 56,  45, 87,  46, 91,
    55, 68,  59, 72,  61, 94,  62, 93,  77, 88,  78, 84
  ), ncol = 2L, byrow = TRUE)

  # The four centre stickers of each face, in face order.
  centres <- as.vector(outer(c(5L, 6L, 9L, 10L), 0:5 * 16L, "+"))
  centres <- sort(centres)

  positions <- rbind(
    corners,
    cbind(wings, 0L),
    cbind(centres, 0L, 0L)
  )
  storage.mode(positions) <- "integer"

  mask <- rbind(
    matrix(TRUE, 8L, 3L),
    matrix(c(TRUE, TRUE, FALSE), 24L, 3L, byrow = TRUE),
    matrix(c(TRUE, FALSE, FALSE), 24L, 3L, byrow = TRUE)
  )

  types <- c(rep(0L, 8L), rep(1L, 24L), rep(2L, 24L))

  list(positions = positions, mask = mask, types = types,
       state_size = 96L, num_classes = 6L)
}

# ---- the model -------------------------------------------------------------

#' Build a piece transformer
#'
#' Creates the model with its weights at their initial values. To run a trained
#' one, use \code{\link{pt_transformer_load}}, which builds this and then fills
#' it from a checkpoint.
#'
#' @param layout A layout list, as \code{\link{pt_cube4_layout}} returns.
#' @param d_model Width of the model (default 256).
#' @param n_heads Attention heads (default 8).
#' @param n_layers Encoder blocks (default 4).
#' @param ff_dim Width of the feed-forward hidden layer (default 1024).
#' @param n_actions Size of the output --- one value per action (default 24).
#' @param activation Feed-forward activation, \code{"relu"} (default) or
#'   \code{"gelu"} / \code{"silu"}.
#' @return A \code{pt_transformer} object.
#' @export
#' @seealso \code{\link{pt_transformer_load}}, \code{\link{pt_forward}}
pt_transformer <- function(layout, d_model = 256L, n_heads = 8L, n_layers = 4L,
                           ff_dim = 1024L, n_actions = 24L,
                           activation = "relu") {
  d_model <- as.integer(d_model); n_heads <- as.integer(n_heads)
  n_layers <- as.integer(n_layers); ff_dim <- as.integer(ff_dim)
  n_actions <- as.integer(n_actions)

  n_pieces <- nrow(layout$positions)
  max_slot <- ncol(layout$positions)

  # A block's weights, at zero until a checkpoint fills them. Attention keeps
  # Q, K and V apart, the way ag_multihead_attention does and torch does not.
  blocks <- lapply(seq_len(n_layers), function(i) {
    list(norm1 = list(g = rep(1, d_model), b = numeric(d_model)),
         norm2 = list(g = rep(1, d_model), b = numeric(d_model)),
         attn  = list(w_q = matrix(0.0, d_model, d_model),
                      w_k = matrix(0.0, d_model, d_model),
                      w_v = matrix(0.0, d_model, d_model),
                      b_q = numeric(d_model), b_k = numeric(d_model),
                      b_v = numeric(d_model),
                      w_o = matrix(0.0, d_model, d_model),
                      b_o = numeric(d_model)),
         ff1   = list(w = matrix(0.0, d_model, ff_dim), b = numeric(ff_dim)),
         ff2   = list(w = matrix(0.0, ff_dim, d_model), b = numeric(d_model)))
  })

  structure(list(
    layout      = layout,
    d_model     = d_model,
    n_heads     = n_heads,
    n_layers    = n_layers,
    ff_dim      = ff_dim,
    n_actions   = n_actions,
    activation  = activation,
    n_pieces    = n_pieces,
    max_slot    = max_slot,
    # [d_model, max_slot * num_classes]: one row of the table per (slot,
    # colour) pair, which is how the checkpoint indexes it.
    value_embed = matrix(0.0, d_model, max_slot * layout$num_classes),
    proj_w      = matrix(0.0, max_slot * d_model, d_model),
    proj_b      = numeric(d_model),
    pos_embed   = matrix(0.0, d_model, n_pieces),
    type_embed  = matrix(0.0, d_model, max(layout$types) + 1L),
    cls_token   = numeric(d_model),
    # Filled in properly by pt_transformer_load(); enough here for the shapes
    # to line up before a checkpoint arrives.
    piece_bias  = matrix(0.0, n_pieces, d_model),
    mask_batch  = function(n) array(layout$mask[rep(seq_len(n_pieces), n), ,
                                                drop = FALSE],
                                    dim = c(n * n_pieces, max_slot, d_model)),
    input_norm  = list(g = rep(1, d_model), b = numeric(d_model)),
    blocks      = blocks,
    output_norm = list(g = rep(1, d_model), b = numeric(d_model)),
    out_w       = matrix(0.0, d_model, n_actions),
    out_b       = numeric(n_actions)
  ), class = "pt_transformer")
}

#' @export
print.pt_transformer <- function(x, ...) {
  cat("Piece transformer\n")
  cat("  pieces  :", x$n_pieces, "of up to", x$max_slot, "stickers\n")
  cat("  d_model :", x$d_model, "/ heads:", x$n_heads, "/ layers:", x$n_layers,
      "/ ff:", x$ff_dim, "\n")
  cat("  actions :", x$n_actions, "\n")
  invisible(x)
}

# ---- loading a checkpoint --------------------------------------------------

# A linear layer's weight, ready to be multiplied on the right: x %*% w.
#
# pth_load() returns torch's [out, in] reversed to [in, out], which is exactly
# that form -- so this is the identity, and the point of it is what does NOT
# happen. Transposing at load time rather than in the forward pass takes a
# fifth of the pass off: the weights never change, and t() on every call was
# recomputing the same matrices millions of times over a run.
#
# `dim` says what the weight is expected to be, and is checked rather than
# trusted: a shape mistake here would otherwise surface as wrong numbers.
.pt_w <- function(w, dim = NULL) {
  if (!is.null(dim) && !identical(base::dim(w), as.integer(dim)))
    stop("pt: weight is ", paste(base::dim(w), collapse = "x"),
         ", expected ", paste(dim, collapse = "x"), call. = FALSE)
  w
}

#' Build a piece transformer and load a checkpoint into it
#'
#' Reads a \code{.pth} saved by \code{torch.save()} and returns the model ready
#' to run. Everything that differs between torch's conventions and this
#' package's is resolved here rather than at inference time.
#'
#' @param path Path to the \code{.pth} checkpoint.
#' @param layout A layout list; defaults to \code{\link{pt_cube4_layout}}.
#' @param ... Passed to \code{\link{pt_transformer}} --- \code{d_model},
#'   \code{n_heads} and the rest, when they differ from the defaults.
#' @return A \code{pt_transformer} with the checkpoint's weights.
#' @export
#' @seealso \code{\link{pth_catalogue}} to see what a checkpoint holds.
#' @examples
#' \dontrun{
#' model <- pt_transformer_load("model.pth")
#' q <- pt_forward(model, rep(0:5, each = 16))
#' which.max(q)
#' }
pt_transformer_load <- function(path, layout = pt_cube4_layout(), ...) {
  w <- pth_load(path)
  model <- pt_transformer(layout, ...)

  need <- function(nm) {
    if (is.null(w[[nm]]))
      stop("pth: checkpoint has no tensor '", nm, "'", call. = FALSE)
    w[[nm]]
  }

  # Embeddings are stored [vocab, dim] by torch and read back [dim, vocab],
  # which is the layout wanted here -- no transpose.
  model$value_embed <- need("local_value_embedding.weight")
  model$pos_embed   <- need("piece_position_embedding.weight")
  model$type_embed  <- need("piece_type_embedding.weight")
  model$cls_token   <- as.vector(need("cls_token"))

  model$proj_w <- .pt_w(need("piece_projection.weight"),
                        c(model$max_slot * model$d_model, model$d_model))
  model$proj_b <- as.vector(need("piece_projection.bias"))

  model$input_norm  <- list(g = as.vector(need("input_norm.weight")),
                            b = as.vector(need("input_norm.bias")))
  model$output_norm <- list(g = as.vector(need("output_norm.weight")),
                            b = as.vector(need("output_norm.bias")))

  model$out_w <- .pt_w(need("output_layer.weight"),
                       c(model$d_model, model$n_actions))
  model$out_b <- as.vector(need("output_layer.bias"))

  d <- model$d_model
  for (i in seq_len(model$n_layers)) {
    p <- paste0("blocks.", i - 1L, ".")

    model$blocks[[i]]$norm1 <- list(g = as.vector(need(paste0(p, "norm1.weight"))),
                                    b = as.vector(need(paste0(p, "norm1.bias"))))
    model$blocks[[i]]$norm2 <- list(g = as.vector(need(paste0(p, "norm2.weight"))),
                                    b = as.vector(need(paste0(p, "norm2.bias"))))

    # Q, K and V share one tensor in torch, stacked in that order. Held here
    # ready to multiply on the right, the three sit side by side in COLUMNS
    # rather than stacked in rows -- torch's [3d, d] reversed is [d, 3d].
    inw <- .pt_w(need(paste0(p, "attn.in_proj_weight")), c(d, 3L * d))
    inb <- as.vector(need(paste0(p, "attn.in_proj_bias")))          # [3d]
    model$blocks[[i]]$attn <- list(
      w_q = inw[, seq_len(d), drop = FALSE],
      w_k = inw[, seq.int(d + 1L, 2L * d), drop = FALSE],
      w_v = inw[, seq.int(2L * d + 1L, 3L * d), drop = FALSE],
      b_q = inb[seq_len(d)],
      b_k = inb[seq.int(d + 1L, 2L * d)],
      b_v = inb[seq.int(2L * d + 1L, 3L * d)],
      w_o = .pt_w(need(paste0(p, "attn.out_proj.weight")), c(d, d)),
      b_o = as.vector(need(paste0(p, "attn.out_proj.bias")))
    )

    model$blocks[[i]]$ff1 <- list(
      w = .pt_w(need(paste0(p, "ff.0.weight")), c(d, model$ff_dim)),
      b = as.vector(need(paste0(p, "ff.0.bias"))))
    model$blocks[[i]]$ff2 <- list(
      w = .pt_w(need(paste0(p, "ff.3.weight")), c(model$ff_dim, d)),
      b = as.vector(need(paste0(p, "ff.3.bias"))))
  }

  # What the forward pass would otherwise rebuild every time from constants:
  # the position and type embeddings a piece carries, added together once, and
  # the mask laid out to match the gathered embeddings.
  model$piece_bias <- t(model$pos_embed) +
    t(model$type_embed[, layout$types + 1L, drop = FALSE])

  # The mask for a batch of n states, kept for the last n asked for. A beam
  # runs at one width for a whole solve, so the array is built once and handed
  # back on every step after.
  model$mask_batch <- local({
    mask <- layout$mask
    np <- model$n_pieces; ms <- model$max_slot; dm <- model$d_model
    cached_n <- 0L
    cached <- NULL
    function(n) {
      if (!identical(n, cached_n)) {
        cached <<- array(mask[rep(seq_len(np), n), , drop = FALSE],
                         dim = c(n * np, ms, dm))
        cached_n <<- n
      }
      cached
    }
  })

  model$loaded <- TRUE
  model
}

# ---- forward ---------------------------------------------------------------

.pt_layer_norm <- function(x, g, b, eps = 1e-5) {
  # x is [tokens, d_model]; normalise each token over d_model. Recycling does
  # the column scaling that two sweep() calls did, at a fraction of the cost:
  # a matrix is stored by column, so `x - mu` runs down the rows and
  # `rep(g, each = nrow)` across them.
  xc <- x - rowMeans(x)
  xc <- xc / sqrt(rowMeans(xc * xc) + eps)
  xc * matrix(g, nrow(x), ncol(x), byrow = TRUE) +
    matrix(b, nrow(x), ncol(x), byrow = TRUE)
}

.pt_act <- function(x, kind) {
  switch(kind,
    relu = pmax(x, 0),
    gelu = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))),
    silu = x / (1 + exp(-x)),
    stop("pt: unknown activation ", kind, call. = FALSE))
}

# A row bias added to every row of a matrix. sweep() does this too, and does
# it through match.fun and an aperm; on matrices this size the wrapper costs
# more than the addition. Of the ways to write it without sweep, building the
# bias as a byrow matrix measured fastest -- ahead of rep(b, each = nrow),
# t(t(x) + b) and a column loop.
.pt_add_bias <- function(x, b)
  x + matrix(b, nrow(x), ncol(x), byrow = TRUE)

.pt_softmax_rows <- function(x) {
  # apply() over the rows, measured against building the maximum column by
  # column with pmax: at this size apply() is the faster of the two.
  m <- apply(x, 1L, max)
  e <- exp(x - m)
  e / rowSums(e)
}

.pt_attention <- function(h, a, n_heads, n_tok = nrow(h)) {
  # h is [batch * tokens, d_model], the states stacked one under another. The
  # projections and the output are the same arithmetic whatever the batch --
  # every row goes through the same weights -- so they run over the whole
  # stack at once. Attention itself does not: a token may only attend within
  # its own state, so the scores are worked out per state, per head.
  d  <- ncol(h)
  dk <- d %/% n_heads
  n_batch <- nrow(h) %/% n_tok

  q <- .pt_add_bias(h %*% a$w_q, a$b_q)
  k <- .pt_add_bias(h %*% a$w_k, a$b_k)
  v <- .pt_add_bias(h %*% a$w_v, a$b_v)
  kt <- t(k)

  out <- matrix(0.0, nrow(h), d)
  for (bi in seq_len(n_batch)) {
    rows <- seq.int((bi - 1L) * n_tok + 1L, bi * n_tok)
    for (i in seq_len(n_heads)) {
      cols <- seq.int((i - 1L) * dk + 1L, i * dk)
      scores <- (q[rows, cols, drop = FALSE] %*%
                 kt[cols, rows, drop = FALSE]) / sqrt(dk)
      out[rows, cols] <- .pt_softmax_rows(scores) %*%
        v[rows, cols, drop = FALSE]
    }
  }
  .pt_add_bias(out %*% a$w_o, a$b_o)
}

#' Run a piece transformer
#'
#' Reads a state, or a batch of them, and returns a value per action.
#'
#' @section One state or many:
#' A batch is not a convenience here, it is the way to get the work done
#' quickly. Every layer but the attention treats a row on its own, so a batch
#' of states is the same arithmetic on taller matrices --- and a tall matmul is
#' what BLAS is good at, where a short one spends its time in the call rather
#' than the multiply. Scoring the ten states of a beam in one call is several
#' times faster than scoring them one after another.
#'
#' @param model A \code{pt_transformer}, normally from
#'   \code{\link{pt_transformer_load}}.
#' @param state Colours in \code{0..num_classes-1}: an integer vector of
#'   \code{state_size} for one state, or a matrix with \code{state_size}
#'   columns --- one state per row --- for a batch. A solved 4x4x4 is
#'   \code{rep(0:5, each = 16)}.
#' @return For one state, a numeric vector of \code{n_actions} values, one per
#'   move. For a batch, a matrix with one row per state and one column per
#'   action.
#' @export
#' @seealso \code{\link{pt_transformer_load}}
#' @examples
#' \dontrun{
#' model <- pt_transformer_load("model.pth")
#' solved <- rep(0:5, each = 16)
#'
#' pt_forward(model, solved)                       # 24 values
#' pt_forward(model, rbind(solved, solved))        # 2 x 24
#' }
pt_forward <- function(model, state) {
  stopifnot(inherits(model, "pt_transformer"))
  lay <- model$layout

  one <- is.null(dim(state))
  states <- if (one) matrix(as.integer(state), 1L)
            else matrix(as.integer(state), nrow(state))

  if (ncol(states) != lay$state_size)
    stop("pt: state has ", ncol(states), " entries, expected ", lay$state_size,
         call. = FALSE)
  if (any(states < 0L | states >= lay$num_classes))
    stop("pt: colours must be 0..", lay$num_classes - 1L, call. = FALSE)

  n_batch  <- nrow(states)
  n_pieces <- model$n_pieces
  max_slot <- model$max_slot
  d        <- model$d_model
  n_tok    <- n_pieces + 1L

  # Every state's pieces, as colours, stacked with a whole state's pieces
  # together: rows 1..n_pieces are state 1, and so on. That is the order the
  # mask, the piece bias and the token stack below are all laid out in.
  #
  # states[, positions] alone gives the transpose of what is wanted -- it
  # varies the state fastest, not the piece -- so the gather runs down the
  # transposed states and the result is read back a state at a time.
  st_t <- t(states)                                    # [state_size, n_batch]
  values <- matrix(st_t[as.vector(t(lay$positions)) + 1L, , drop = FALSE],
                   max_slot, n_pieces * n_batch)
  values <- t(values)                                  # [n_batch*n_pieces, max_slot]

  # The embedding table is indexed by (slot, colour) together, so a colour in
  # slot 2 is a different row from the same colour in slot 1.
  idx <- values + rep(seq_len(max_slot) - 1L, each = n_batch * n_pieces) *
    lay$num_classes

  emb <- array(t(model$value_embed[, as.vector(idx) + 1L, drop = FALSE]),
               dim = c(n_batch * n_pieces, max_slot, d))
  emb <- emb * model$mask_batch(n_batch)
  flat <- matrix(aperm(emb, c(1L, 3L, 2L)), n_batch * n_pieces, max_slot * d)

  h <- .pt_add_bias(flat %*% model$proj_w, model$proj_b) +
    model$piece_bias[rep(seq_len(n_pieces), n_batch), , drop = FALSE]

  # The CLS token leads each state, so the stack is interleaved rather than
  # simply bound: state 1's tokens, then state 2's, and so on.
  hb <- matrix(0.0, n_batch * n_tok, d)
  cls_rows <- (seq_len(n_batch) - 1L) * n_tok + 1L
  hb[cls_rows, ] <- rep(model$cls_token, each = n_batch)
  hb[-cls_rows, ] <- h

  hb <- .pt_layer_norm(hb, model$input_norm$g, model$input_norm$b)

  for (blk in model$blocks) {
    hb <- hb + .pt_attention(.pt_layer_norm(hb, blk$norm1$g, blk$norm1$b),
                             blk$attn, model$n_heads, n_tok)
    x <- .pt_layer_norm(hb, blk$norm2$g, blk$norm2$b)
    x <- .pt_act(.pt_add_bias(x %*% blk$ff1$w, blk$ff1$b), model$activation)
    hb <- hb + .pt_add_bias(x %*% blk$ff2$w, blk$ff2$b)
  }

  pooled <- .pt_layer_norm(hb, model$output_norm$g,
                           model$output_norm$b)[cls_rows, , drop = FALSE]
  out <- .pt_add_bias(pooled %*% model$out_w, model$out_b)

  if (one) as.vector(out) else out
}
