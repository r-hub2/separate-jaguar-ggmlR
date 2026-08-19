# pth.R — read the catalogue of a PyTorch .pth checkpoint
#
# A .pth file is a zip holding one pickle of metadata (data.pkl) and one entry
# per tensor under data/, each entry the tensor's bytes and nothing else. The
# pickle says which entry a tensor lives in, what shape it has and how it is
# strided; the bytes carry no header of their own.
#
# This file reads the pickle far enough to list the tensors -- name, shape,
# dtype, element count, and the zip entry holding the data. It does not read
# the tensors themselves.
#
# Minimal API:
#   pth_catalogue(path)  — data.frame of tensors in a checkpoint

# ---- pickle ---------------------------------------------------------------
#
# Only the opcodes a torch.save() checkpoint actually emits are handled. A
# checkpoint is a narrow use of the format: it builds an OrderedDict of
# name -> rebuilt tensor, and the tensors are built by one function,
# torch._utils._rebuild_tensor_v2, called with a fixed argument shape. So the
# machine below is a real stack machine, but it needs a fraction of the
# opcodes a general unpickler would.

.pth_op <- c(
  MARK = 0x28, EMPTY_TUPLE = 0x29, STOP = 0x2e, BINPUT = 0x71, LONG_BINPUT = 0x72,
  EMPTY_LIST = 0x5d, BINGET = 0x68, LONG_BINGET = 0x6a, EMPTY_DICT = 0x7d,
  APPENDS = 0x65, BINUNICODE = 0x58, SHORT_BINUNICODE = 0x8c, BININT = 0x4a,
  BININT1 = 0x4b, BININT2 = 0x4d, LONG1 = 0x8a, TUPLE = 0x74, TUPLE1 = 0x85,
  TUPLE2 = 0x86, TUPLE3 = 0x87, NEWTRUE = 0x88, NEWFALSE = 0x89, NONE = 0x4e,
  REDUCE = 0x52, BUILD = 0x62, GLOBAL = 0x63, STACK_GLOBAL = 0x93, NEWOBJ = 0x81,
  PROTO = 0x80, FRAME = 0x95, SETITEM = 0x73, SETITEMS = 0x75, BINPERSID = 0x51
)

# torch storage classes, and the R type each one reads back as. F64/F32 read as
# doubles, the integer kinds as integers; the byte width is what the catalogue
# reports and what a reader would need to seek by.
.pth_dtypes <- list(
  DoubleStorage = list(dtype = "f64", size = 8L),
  FloatStorage  = list(dtype = "f32", size = 4L),
  HalfStorage   = list(dtype = "f16", size = 2L),
  BFloat16Storage = list(dtype = "bf16", size = 2L),
  LongStorage   = list(dtype = "i64", size = 8L),
  IntStorage    = list(dtype = "i32", size = 4L),
  ShortStorage  = list(dtype = "i16", size = 2L),
  CharStorage   = list(dtype = "i8",  size = 1L),
  ByteStorage   = list(dtype = "u8",  size = 1L),
  BoolStorage   = list(dtype = "bool", size = 1L)
)

# A persistent id in a torch checkpoint: the tuple
# ("storage", <storage class>, <key>, <device>, <numel>). Everything the
# catalogue needs about where a tensor's bytes are comes from here.
.pth_persid <- function(args) {
  if (length(args) < 5L || !identical(args[[1L]], "storage")) return(NULL)
  cls <- args[[2L]]
  if (is.list(cls)) cls <- cls$name           # a GLOBAL lands as a named list
  list(storage = as.character(cls), key = as.character(args[[3L]]),
       numel = as.numeric(args[[5L]]))
}

# _rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks)
.pth_rebuild <- function(args) {
  if (length(args) < 4L) return(NULL)
  st <- args[[1L]]
  if (is.null(st) || is.null(st$storage)) return(NULL)
  list(kind = "tensor", storage = st$storage, key = st$key, numel = st$numel,
       offset = as.numeric(args[[2L]]),
       shape = as.numeric(unlist(args[[3L]])),
       stride = as.numeric(unlist(args[[4L]])))
}

.pth_unpickle <- function(r) {
  n <- length(r)
  i <- 1L
  stack <- list()
  memo <- list()
  marks <- integer(0)

  push <- function(v) stack[[length(stack) + 1L]] <<- v
  # A mark is a stack depth; everything above it is one group.
  popmark <- function() {
    at <- marks[length(marks)]; marks <<- marks[-length(marks)]
    out <- if (length(stack) > at) stack[seq.int(at + 1L, length(stack))] else list()
    stack <<- if (at > 0L) stack[seq_len(at)] else list()
    out
  }
  # Lengths and memo indices, read as signed: a checkpoint never carries a
  # string or a memo index anywhere near 2^31, and R has no unsigned 4-byte
  # integer to read them into.
  u32 <- function(at) readBin(r[at:(at + 3L)], "integer", 1L, 4L, endian = "little")
  i32 <- u32

  op <- .pth_op
  while (i <= n) {
    o <- as.integer(r[i]); i <- i + 1L

    if (o == op[["PROTO"]]) { i <- i + 1L
    } else if (o == op[["FRAME"]]) { i <- i + 8L
    } else if (o == op[["STOP"]]) { break

    } else if (o == op[["BINUNICODE"]]) {
      len <- u32(i); i <- i + 4L
      push(rawToChar(r[seq.int(i, length.out = len)])); i <- i + len
    } else if (o == op[["SHORT_BINUNICODE"]]) {
      len <- as.integer(r[i]); i <- i + 1L
      push(rawToChar(r[seq.int(i, length.out = len)])); i <- i + len

    } else if (o == op[["BININT"]])  { push(i32(i)); i <- i + 4L
    } else if (o == op[["BININT1"]]) { push(as.integer(r[i])); i <- i + 1L
    } else if (o == op[["BININT2"]]) {
      push(readBin(r[i:(i + 1L)], "integer", 1L, 2L, signed = FALSE,
                   endian = "little")); i <- i + 2L
    } else if (o == op[["LONG1"]]) {
      len <- as.integer(r[i]); i <- i + 1L
      v <- 0
      if (len > 0L) {
        b <- as.numeric(r[seq.int(i, length.out = len)])
        v <- sum(b * 256^(seq_along(b) - 1L))
        if (b[len] >= 128) v <- v - 256^len        # two's complement
      }
      push(v); i <- i + len

    } else if (o == op[["NEWTRUE"]])  { push(TRUE)
    } else if (o == op[["NEWFALSE"]]) { push(FALSE)
    } else if (o == op[["NONE"]])     { push(NULL)

    } else if (o == op[["MARK"]]) { marks <- c(marks, length(stack))
    } else if (o == op[["EMPTY_TUPLE"]]) { push(list())
    } else if (o == op[["EMPTY_LIST"]])  { push(list())
    } else if (o == op[["EMPTY_DICT"]])  { push(list())

    } else if (o == op[["TUPLE"]])  { push(popmark())
    } else if (o == op[["TUPLE1"]]) {
      k <- length(stack)
      t1 <- list(stack[[k]])
      stack <- stack[seq_len(k - 1L)]; push(t1)
    } else if (o == op[["TUPLE2"]]) {
      k <- length(stack)
      t2 <- list(stack[[k - 1L]], stack[[k]])
      stack <- stack[seq_len(k - 2L)]; push(t2)
    } else if (o == op[["TUPLE3"]]) {
      k <- length(stack)
      t3 <- list(stack[[k - 2L]], stack[[k - 1L]], stack[[k]])
      stack <- stack[seq_len(k - 3L)]; push(t3)

    } else if (o == op[["BINPUT"]]) {
      memo[[as.character(as.integer(r[i]))]] <- stack[[length(stack)]]; i <- i + 1L
    } else if (o == op[["LONG_BINPUT"]]) {
      memo[[as.character(u32(i))]] <- stack[[length(stack)]]; i <- i + 4L
    } else if (o == op[["BINGET"]]) {
      push(memo[[as.character(as.integer(r[i]))]]); i <- i + 1L
    } else if (o == op[["LONG_BINGET"]]) {
      push(memo[[as.character(u32(i))]]); i <- i + 4L

    } else if (o == op[["GLOBAL"]]) {
      # module \n name \n
      e1 <- which(r[i:n] == as.raw(0x0a))[1L]; mod <- rawToChar(r[seq.int(i, length.out = e1 - 1L)])
      i <- i + e1
      e2 <- which(r[i:n] == as.raw(0x0a))[1L]; nmv <- rawToChar(r[seq.int(i, length.out = e2 - 1L)])
      i <- i + e2
      push(list(kind = "global", module = mod, name = nmv))
    } else if (o == op[["STACK_GLOBAL"]]) {
      k <- length(stack)
      push2 <- list(kind = "global", module = as.character(stack[[k - 1L]]),
                    name = as.character(stack[[k]]))
      stack <- stack[seq_len(k - 2L)]; push(push2)

    } else if (o == op[["BINPERSID"]]) {
      a <- stack[[length(stack)]]; stack <- stack[-length(stack)]
      push(.pth_persid(a))

    } else if (o == op[["REDUCE"]]) {
      k <- length(stack)
      args <- stack[[k]]; fn <- stack[[k - 1L]]
      stack <- stack[seq_len(k - 2L)]
      nmv <- if (is.list(fn) && !is.null(fn$name)) fn$name else ""
      push(if (identical(nmv, "_rebuild_tensor_v2")) .pth_rebuild(args) else list())

    } else if (o == op[["NEWOBJ"]]) {
      k <- length(stack); stack <- stack[seq_len(k - 2L)]; push(list())
    } else if (o == op[["BUILD"]]) {
      # state applied to an object; the catalogue does not need it
      stack <- stack[-length(stack)]

    } else if (o == op[["SETITEM"]]) {
      k <- length(stack)
      key <- stack[[k - 1L]]; val <- stack[[k]]
      stack <- stack[seq_len(k - 2L)]
      d <- stack[[length(stack)]]; d[[as.character(key)]] <- val
      stack[[length(stack)]] <- d
    } else if (o == op[["SETITEMS"]]) {
      items <- popmark()
      d <- stack[[length(stack)]]
      if (length(items) >= 2L)
        for (j in seq(1L, length(items) - 1L, by = 2L))
          d[[as.character(items[[j]])]] <- items[[j + 1L]]
      stack[[length(stack)]] <- d
    } else if (o == op[["APPENDS"]]) {
      items <- popmark()
      d <- stack[[length(stack)]]
      stack[[length(stack)]] <- c(d, items)

    } else {
      stop("pth: unsupported pickle opcode 0x", format(as.hexmode(o), width = 2),
           " at byte ", i - 1L, call. = FALSE)
    }
  }
  if (length(stack)) stack[[length(stack)]] else list()
}

# ---- the tensor table both readers start from ------------------------------

# Everything the pickle says about a checkpoint's tensors, in the checkpoint's
# own order: name, storage class, zip entry, offset and shape. pth_catalogue()
# formats this; pth_load() reads the bytes it points at.
.pth_tensors <- function(path) {
  entries <- utils::unzip(path, list = TRUE)
  pkl <- grep("(^|/)data\\.pkl$", entries$Name, value = TRUE)
  if (!length(pkl))
    stop("pth: no data.pkl in ", basename(path),
         " -- is this a torch.save() checkpoint?", call. = FALSE)
  prefix <- sub("data\\.pkl$", "", pkl[1L])

  con <- unz(path, pkl[1L], open = "rb")
  on.exit(close(con), add = TRUE)
  raw <- readBin(con, "raw", entries$Length[match(pkl[1L], entries$Name)])

  obj <- .pth_unpickle(raw)

  # A checkpoint is usually the state_dict itself, but torch.save() of a
  # training run often wraps it: {"model_state_dict": ..., "optimizer": ...}.
  # Take the outer dict when its values are tensors, and go one level in when
  # they are not.
  is_tensor <- function(x) is.list(x) && identical(x$kind, "tensor")
  if (!any(vapply(obj, is_tensor, logical(1)))) {
    for (key in c("model_state_dict", "state_dict", "model")) {
      if (!is.null(obj[[key]]) && any(vapply(obj[[key]], is_tensor, logical(1)))) {
        obj <- obj[[key]]
        break
      }
    }
  }

  keep <- vapply(obj, is_tensor, logical(1))
  if (!any(keep))
    stop("pth: no tensors found in ", basename(path), call. = FALSE)
  obj <- obj[keep]

  for (nm in names(obj)) {
    d <- .pth_dtypes[[obj[[nm]]$storage]]
    if (is.null(d))
      stop("pth: unsupported storage ", obj[[nm]]$storage, " for tensor '", nm,
           "'", call. = FALSE)
    obj[[nm]]$dtype <- d$dtype
    obj[[nm]]$size <- d$size
    obj[[nm]]$entry <- paste0(prefix, "data/", obj[[nm]]$key)
  }
  obj
}

# ---- catalogue -------------------------------------------------------------

#' List the tensors in a PyTorch checkpoint
#'
#' Reads the metadata of a \code{.pth} file --- the tensor names, shapes and
#' dtypes --- without reading the tensors. A checkpoint is a zip holding one
#' pickle of metadata and one entry per tensor, so the catalogue is cheap
#' whatever the file weighs.
#'
#' What comes back is the checkpoint's own \code{state_dict} order, which is
#' the order the model's modules were built in.
#'
#' @param path Path to a \code{.pth} file saved by \code{torch.save()}.
#' @return A data.frame with one row per tensor and the columns \code{name},
#'   \code{shape} (as \code{"a x b"}), \code{dtype}, \code{n} (element count),
#'   \code{bytes}, and \code{entry} (the zip entry holding the data).
#'   The attribute \code{"file"} carries the path.
#' @export
#' @examples
#' \dontrun{
#' cat <- pth_catalogue("model.pth")
#' head(cat)
#' sum(cat$n)          # total parameters
#' }
pth_catalogue <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  obj <- .pth_tensors(path)

  info <- lapply(names(obj), function(nm) {
    t <- obj[[nm]]
    n <- if (length(t$shape)) prod(t$shape) else 1
    data.frame(name = nm,
               shape = if (length(t$shape)) paste(t$shape, collapse = " x ")
                       else "scalar",
               dtype = t$dtype,
               n = n,
               bytes = n * t$size,
               entry = t$entry,
               stringsAsFactors = FALSE)
  })

  out <- do.call(rbind, info)
  rownames(out) <- NULL
  attr(out, "file") <- path
  out
}

# ---- reading the tensors ---------------------------------------------------

# One storage, read whole. A zip entry is one storage and holds nothing but its
# bytes, so the read needs no header -- only how many elements of what kind.
.pth_read_storage <- function(path, entry, dtype, numel) {
  con <- unz(path, entry, open = "rb")
  on.exit(close(con), add = TRUE)

  switch(dtype,
    f32  = readBin(con, "double",  numel, 4L, endian = "little"),
    f64  = readBin(con, "double",  numel, 8L, endian = "little"),
    i64  = readBin(con, "integer", numel, 8L, endian = "little"),
    i32  = readBin(con, "integer", numel, 4L, endian = "little"),
    i16  = readBin(con, "integer", numel, 2L, endian = "little"),
    i8   = readBin(con, "integer", numel, 1L, endian = "little"),
    u8   = readBin(con, "integer", numel, 1L, signed = FALSE, endian = "little"),
    bool = readBin(con, "integer", numel, 1L, signed = FALSE,
                   endian = "little") != 0L,
    # Half precision has no readBin type: take the bits and widen them.
    f16  = .pth_half(readBin(con, "raw", numel * 2L)),
    bf16 = .pth_bfloat(readBin(con, "raw", numel * 2L)),
    stop("pth: cannot read dtype ", dtype, call. = FALSE)
  )
}

# IEEE half -> double. Sign, 5-bit exponent, 10-bit mantissa; subnormals and
# the infinities are the two cases worth spelling out.
.pth_half <- function(bytes) {
  lo <- as.integer(bytes[c(TRUE, FALSE)])
  hi <- as.integer(bytes[c(FALSE, TRUE)])
  bits <- hi * 256L + lo
  sign <- ifelse(bits >= 32768L, -1, 1)
  expo <- bitwAnd(bits %/% 1024L, 31L)
  mant <- bitwAnd(bits, 1023L)
  out <- ifelse(
    expo == 0L, sign * 2^-14 * (mant / 1024),
    ifelse(expo == 31L, sign * ifelse(mant == 0L, Inf, NaN),
           sign * 2^(expo - 15L) * (1 + mant / 1024)))
  out
}

# bfloat16 is the top half of an f32, so widening is a shift and nothing else.
.pth_bfloat <- function(bytes) {
  full <- raw(length(bytes) * 2L)
  full[c(FALSE, FALSE, TRUE, TRUE)] <- bytes
  readBin(full, "double", length(bytes) %/% 2L, 4L, endian = "little")
}

#' Read the tensors of a PyTorch checkpoint
#'
#' Reads a \code{.pth} file saved by \code{torch.save()} into plain R arrays.
#' No Python and no torch: a checkpoint is a zip of one metadata pickle and one
#' entry per tensor, and both are read here directly.
#'
#' @section Row-major, and what that means for shapes:
#' PyTorch stores a tensor row-major and R stores an array column-major, so the
#' same bytes describe transposed arrays in the two languages. The tensors come
#' back with their dimensions \emph{reversed} --- a PyTorch \code{[768, 256]}
#' weight arrives as a \code{256 x 768} R matrix --- which keeps the elements
#' where they belong without moving a single one. For a linear layer that is
#' the happy case: \code{x \%*\% w} is then the layer's own arithmetic, no
#' transpose needed.
#'
#' @param path Path to a \code{.pth} file saved by \code{torch.save()}.
#' @param names Optional character vector of tensor names to read. The default
#'   reads all of them. Use \code{\link{pth_catalogue}} to see what is there.
#' @return A named list of numeric arrays, in the checkpoint's own order.
#'   Scalars come back as length-1 vectors; everything else carries a
#'   \code{dim}. The attribute \code{"file"} carries the path.
#' @export
#' @seealso \code{\link{pth_catalogue}} to list a checkpoint without reading it.
#' @examples
#' \dontrun{
#' w <- pth_load("model.pth")
#' dim(w$output_layer.weight)
#'
#' # just the one tensor
#' pth_load("model.pth", names = "output_layer.bias")
#' }
pth_load <- function(path, names = NULL) {
  path <- normalizePath(path, mustWork = TRUE)
  obj <- .pth_tensors(path)

  if (!is.null(names)) {
    missing <- setdiff(names, base::names(obj))
    if (length(missing))
      stop("pth: no such tensor(s): ", paste(missing, collapse = ", "),
           call. = FALSE)
    obj <- obj[names]
  }

  out <- lapply(base::names(obj), function(nm) {
    t <- obj[[nm]]
    n <- if (length(t$shape)) prod(t$shape) else 1

    # The storage may be longer than the tensor and shared with others: read it
    # whole, then take the tensor's own window of it.
    raw <- .pth_read_storage(path, t$entry, t$dtype, t$numel)
    if (length(raw) < t$offset + n)
      stop("pth: tensor '", nm, "' runs past the end of its storage",
           call. = FALSE)
    v <- raw[seq.int(t$offset + 1, length.out = n)]

    # Row-major bytes read into a column-major array: reversing the dimensions
    # is what makes the two agree, and it costs no rearranging.
    if (length(t$shape) > 1L) dim(v) <- rev(t$shape)
    v
  })
  base::names(out) <- base::names(obj)
  attr(out, "file") <- path
  out
}
