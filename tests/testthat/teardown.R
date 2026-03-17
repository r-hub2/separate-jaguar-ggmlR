# Restore original thread count after tests
if (exists(".ggmlr_old_threads")) {
  ggml_set_n_threads(.ggmlr_old_threads)
}
