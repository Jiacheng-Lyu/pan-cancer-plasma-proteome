clusterkegg <- function(gene, universe = NULL, ...) {
  library(org.Hs.eg.db)
  library(clusterProfiler)
  library(DOSE)
  gene_id <- bitr(gene, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
  if (length(universe) > 0) {
    universe <- bitr(universe, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)$ENTREZID
  }
  kk <- enrichKEGG(
    organism = "hsa",
    gene = gene_id$ENTREZID,
    pvalueCutoff = 0.05,
    qvalueCutoff = 0.05,
    universe = universe,
    ...
  )

  if (is.null(kk)) {
    y <- data.frame()
  } else {
    y <- setReadable(
      kk,
      OrgDb = org.Hs.eg.db,
      keyType = "ENTREZID"
    )
  }
  return(as.data.frame(y))
}

clustergo <- function(gene, universe = NULL, ont = "BP", pvalue = 0.05, qvalue = 1) {
  library(org.Hs.eg.db)
  library(clusterProfiler)
  library(DOSE)

  gene_id <- bitr(gene, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
  if (length(universe) > 0) {
    universe <- bitr(universe, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)$ENTREZID
  }

  kk <- enrichGO(
    gene = gene_id$ENTREZID,
    OrgDb = org.Hs.eg.db,
    ont = ont,
    readable = TRUE,
    pvalueCutoff = pvalue,
    qvalueCutoff = qvalue,
    universe = universe
  )
  return(as.data.frame(kk))
}

clustermsigdb <- function(gene, universe = NULL) {
  library(org.Hs.eg.db)
  library(clusterProfiler)
  library(msigdbr)
  library(dplyr)

  gene_id <- bitr(gene, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
  if (length(universe) > 0) {
    universe <- bitr(universe, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)$ENTREZID
  }

  m_df <- msigdbr(species = "Homo sapiens", category = "H") %>%
    dplyr::select(gs_name, entrez_gene)
  em <- enricher(
    gene = gene_id$ENTREZID,
    TERM2GENE = m_df,
    pvalueCutoff = 0.05,
    universe = universe
  )
  if (is.null(em)) {
    y <- data.frame()
  } else {
    y <- setReadable(
      em,
      OrgDb = org.Hs.eg.db,
      keyType = "ENTREZID"
    )
  }
  return(as.data.frame(y))
}

clusterreactome <- function(gene, universe = NULL) {
  library(org.Hs.eg.db)
  library(ReactomePA)
  library(clusterProfiler)

  gene_id <- bitr(gene, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
  if (length(universe) > 0) {
    universe <- bitr(universe, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)$ENTREZID
  }

  over <- enrichPathway(
    gene = gene_id$ENTREZID,
    pvalueCutoff = 0.05,
    readable = T,
    universe = universe
  )
  return(as.data.frame(over))
}

ssgsea <- function(data, gmtname = "KEGG_HIF", minsz = 10, maxsz = 200, parallelsz = 1, method = "gsva", ...) {
  library(GSVA)
  library(GSEABase)
  library(tidyverse)

  gmtpath <- paste("/Users/lvjiacheng/Fudan/research/Individual/database/gmt/", gmtname, ".gmt", sep = "")

  geneSets <- getGmt(gmtpath)
  mydata <- as.matrix(data)
  ssgsea_re <- gsva(mydata, geneSets, min.sz = minsz, max.sz = maxsz, verbose = TRUE, parallel.sz = parallelsz, method = method, ...)
  ssgsea_re <- rownames_to_column(as.data.frame(ssgsea_re), "term")

  return(ssgsea_re)
}

sumer_usage <- function(path, file_name) {
  outpath <- paste(path, file_name, sep = "\\")
  library(sumer)
  setwd(path)
  sumer("config.json", outpath)
}

combat_batch <- function(data, batch = NULL, mod = NULL, par.prior = T, prior.plots = F) {
  # '''
  # remove batch effect;
  # parameters: data: expression matrix, batch: category
  # '''
  library(sva)
  combat_data <- ComBat(
    dat = data,
    batch = batch,
    mod = mod,
    par.prior = par.prior,
    prior.plots = prior.plots
  )
  return(as.data.frame(combat_data))
}

ccp <- function(matrix, distance = "euclidean", clusterAlg = "km", reps = 200, maxK = 5, verbose=T, ...) {
  matrix <- as.matrix(matrix)
  library(ConsensusClusterPlus)
  rcc <- ConsensusClusterPlus(
    matrix,
    maxK = maxK,
    reps = reps,
    pItem = 0.8,
    pFeature = 1,
    distance = distance,
    clusterAlg = clusterAlg,
    title = paste(distance, clusterAlg, sep = "_"),
    writeTable = T,
    verbose = verbose,
    plot = "pdf"
  )
}

dip_test <- function(vector, full.result = FALSE, min.is.0 = FALSE, debug = FALSE) {
  library(diptest)
  return(dip(vector))
}

impute <- function(data, k = 10, rowmax = 0.5, colmax = 1, log = 10, var = "Symbol", ...) {
  library(impute)
  library(tidyverse)

  imputed <- impute.knn(as.matrix(data), k = k, rowmax = rowmax, colmax = colmax, rng.seed = 362436069, ...)
  if (log == "no") {
    print("success")
    out <- as.data.frame(imputed$data)
  } else {
    out <- as.data.frame(log**imputed$data)
  }

  out <- out %>% rownames_to_column(., var = var)
  return(out)
}

viper <- function(data, tf_filter_condition=c("A", "B", "C"), minsz=10, eset.filter=FALSE, cores=1, verbose=FALSE, ...){
  library(dorothea)
  library(dplyr)
  library(viper)

  regulons <- dorothea_hs %>%
  filter(confidence %in% tf_filter_condition)

  data = as.matrix(data)
  tf_activities <- run_viper(data, regulons, tidy=TRUE,
  options = list( # method = "scale",
    minsize = minsz,
    eset.filter = eset.filter, cores = cores,
    verbose = verbose, ...
    )
  )
  return(tf_activities)
}
