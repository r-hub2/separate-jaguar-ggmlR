# Limit threads during CRAN checks to avoid NOTE about CPU time >> elapsed time
if (identical(Sys.getenv("_R_CHECK_LIMIT_CORES_"), "TRUE")) {
  .ggmlr_old_threads <- ggml_get_n_threads()
  ggml_set_omp_threads(2L)
}
