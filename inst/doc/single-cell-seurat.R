## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment  = "#>",
  # Executed locally (NOT_CRAN=true) only when Seurat is available; skipped on
  # CRAN to avoid the "CPU time > elapsed" vignette NOTE from the CPU fallback.
  eval     = identical(Sys.getenv("NOT_CRAN"), "true") &&
             requireNamespace("Seurat", quietly = TRUE) &&
             requireNamespace("SeuratObject", quietly = TRUE)
)
library(ggmlR)
if (requireNamespace("Seurat", quietly = TRUE)) {
  suppressMessages(library(Seurat))
  suppressMessages(library(SeuratObject))
}

## -----------------------------------------------------------------------------
# set.seed(1)
# ng <- 400L; nc <- 200L
# counts <- matrix(rpois(ng * nc, lambda = 5), nrow = ng, ncol = nc)
# rownames(counts) <- paste0("gene", seq_len(ng))
# colnames(counts) <- paste0("cell", seq_len(nc))
# counts <- methods::as(counts, "dgCMatrix")
# 
# pbmc <- CreateSeuratObject(counts = counts)
# pbmc

## -----------------------------------------------------------------------------
# pbmc <- RunGGML(pbmc, op = "normalize")   # -> assay "data" layer
# pbmc <- RunGGML(pbmc, op = "scale")       # -> assay "scale.data" layer

## -----------------------------------------------------------------------------
# dim(LayerData(pbmc, layer = "data"))
# dim(LayerData(pbmc, layer = "scale.data"))

## -----------------------------------------------------------------------------
# gpu_data <- as.matrix(LayerData(pbmc, layer = "data"))
# ref_data <- as.matrix(LayerData(
#   NormalizeData(pbmc, verbose = FALSE), layer = "data"))
# max(abs(gpu_data - ref_data))

## -----------------------------------------------------------------------------
# pbmc <- RunGGML(pbmc, op = "embed", n_components = 20, reduction_name = "ggml")
# Embeddings(pbmc, "ggml")[1:3, 1:4]

## -----------------------------------------------------------------------------
# pbmc <- RunGGML(pbmc, op = "umap", reduction = "ggml", dims = 1:20,
#                 reduction_name = "umap")
# Embeddings(pbmc, "umap")[1:3, ]

## -----------------------------------------------------------------------------
# pbmc <- RunGGML(pbmc, op = "neighbors", reduction = "ggml", dims = 1:20)
# Graphs(pbmc)                                   # <assay>_nn and <assay>_snn
# 
# pbmc <- FindClusters(pbmc, graph.name = paste0(DefaultAssay(pbmc), "_snn"),
#                      verbose = FALSE)
# table(pbmc$seurat_clusters)

## ----eval=FALSE---------------------------------------------------------------
# DimPlot(pbmc, reduction = "umap", group.by = "seurat_clusters", label = TRUE)

## -----------------------------------------------------------------------------
# Misc(pbmc, "data_ggml")$backend          # normalize
# Misc(pbmc, "scale.data_ggml")$backend    # scale
# Misc(pbmc, "ggml_ggml")$backend          # embed (and neighbors)
# Misc(pbmc, "umap_ggml")$backend_sgd      # umap: layout phase backend

## -----------------------------------------------------------------------------
# # What can the adapter do?
# names(ggml_ops_registry())
# ggml_ops_registry("embed")
# 
# # Compose the layers manually on a plain matrix:
# mat  <- ggml_extract(gpu_data)                       # genes x cells, dense
# task <- ggml_task("embed", mat, params = list(n_components = 10))
# res  <- ggml_run(task)                               # ggml_result
# dim(res$embedding)                                   # cells x components

## ----eval=identical(Sys.getenv("NOT_CRAN"), "true") && requireNamespace("SingleCellExperiment", quietly = TRUE) && requireNamespace("S4Vectors", quietly = TRUE)----
# library(SingleCellExperiment)
# 
# # a self-contained SCE (genes x cells), so this section does not depend on the
# # Seurat object built earlier
# set.seed(1)
# ng <- 200L; nc <- 120L
# sce_counts <- matrix(stats::rpois(ng * nc, lambda = 5), ng, nc)
# rownames(sce_counts) <- paste0("gene", seq_len(ng))
# colnames(sce_counts) <- paste0("cell", seq_len(nc))
# sce <- SingleCellExperiment(assays = list(
#   counts    = sce_counts,
#   logcounts = log1p(sce_counts)))
# 
# sce <- RunGGML(sce, op = "embed", n_components = 20)         # -> reducedDim "ggml"
# sce <- RunGGML(sce, op = "neighbors", reduction = "ggml", dims = 1:20)
# 
# reducedDimNames(sce)                                         # "ggml"
# names(S4Vectors::metadata(sce))                              # ggml_nn / ggml_snn / ggml_ggml

