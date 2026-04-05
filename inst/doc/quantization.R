## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(eval = FALSE)

## -----------------------------------------------------------------------------
# library(ggmlR)

## -----------------------------------------------------------------------------
# # Original float weights (must be a multiple of block size, typically 32)
# weights <- rnorm(256L)
# 
# # Quantize to Q4_0
# raw_q4 <- quantize_q4_0(weights)
# cat("Original size: ", length(weights) * 4L, "bytes\n")
# cat("Q4_0 size:     ", length(raw_q4), "bytes\n")
# cat("Compression:   ", round(length(weights) * 4L / length(raw_q4), 1), "x\n")
# 
# # Dequantize back to float
# recovered <- dequantize_row_q4_0(raw_q4, length(weights))
# cat("Max abs error: ", max(abs(recovered - weights)), "\n")

## -----------------------------------------------------------------------------
# weights <- rnorm(512L)
# 
# # Q4_K — 4-bit K-quant
# raw_q4k <- quantize_q4_K(weights)
# rec_q4k <- dequantize_row_q4_K(raw_q4k, length(weights))
# cat("Q4_K max error:", max(abs(rec_q4k - weights)), "\n")
# 
# # Q8_K — 8-bit K-quant (near-lossless)
# raw_q8k <- quantize_q8_K(weights)
# rec_q8k <- dequantize_row_q8_K(raw_q8k, length(weights))
# cat("Q8_K max error:", max(abs(rec_q8k - weights)), "\n")

## -----------------------------------------------------------------------------
# weights    <- rnorm(512L)
# importance <- abs(weights)^2          # example: weight magnitude as importance
# 
# # IQ4_XS — 4-bit with importance
# raw_iq4 <- quantize_iq4_xs(weights, importance = importance)
# rec_iq4 <- dequantize_row_iq4_xs(raw_iq4, length(weights))
# cat("IQ4_XS max error:", max(abs(rec_iq4 - weights)), "\n")

## -----------------------------------------------------------------------------
# weights <- rnorm(512L)
# n_bytes_f32 <- length(weights) * 4L
# 
# formats <- list(
#   Q4_0 = list(q = quantize_q4_0,  dq = dequantize_row_q4_0),
#   Q8_0 = list(q = quantize_q8_0,  dq = dequantize_row_q8_0),
#   Q4_K = list(q = quantize_q4_K,  dq = dequantize_row_q4_K),
#   Q6_K = list(q = quantize_q6_K,  dq = dequantize_row_q6_K),
#   Q8_K = list(q = quantize_q8_K,  dq = dequantize_row_q8_K)
# )
# 
# cat(sprintf("%-8s  %6s  %8s  %10s\n", "Format", "Bytes", "Ratio", "MaxError"))
# cat(strrep("-", 40), "\n")
# for (nm in names(formats)) {
#   raw <- formats[[nm]]$q(weights)
#   rec <- formats[[nm]]$dq(raw, length(weights))
#   cat(sprintf("%-8s  %6d  %8.2fx  %10.6f\n",
#               nm, length(raw),
#               n_bytes_f32 / length(raw),
#               max(abs(rec - weights))))
# }

## -----------------------------------------------------------------------------
# row <- rnorm(32L)   # exactly one Q4_0 block
# 
# raw_row <- quantize_row_q4_0_ref(row, length(row))
# rec_row <- dequantize_row_q4_0(raw_row, length(row))

