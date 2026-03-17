# Limit threads to avoid CRAN NOTE about CPU time >> elapsed time
.ggmlr_old_threads <- ggml_get_n_threads()
ggml_set_n_threads(2L)
