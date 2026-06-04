## Preprocessing examples for ecountgmifs.
## These examples are intentionally small so that they can be run quickly.

library(ecountgmifs)

## RNA-seq style preprocessing
data(mrna97_rnaseq)

X_cpm <- normalize_counts(
  mrna97_rnaseq$X,
  method = "cpm"
)

X_pre <- transform_predictors(
  X_cpm,
  method = "standardize_log1p"
)

j <- 1
fit <- ecountgmifs(
  X = X_pre,
  y = mrna97_rnaseq$Y[, j],
  weight.vec = ifelse(mrna97_rnaseq$prior[j, ], 0.1, 1),
  family = "negative.binomial",
  enet.alpha = 0.75
)

## Airport style temporal preprocessing
data(airport_t100)

X_air <- transform_time_series(
  airport_t100$X,
  method = "log1p_month_demean",
  month = airport_t100$sample$month
)

i <- 1
fit_air <- ecountgmifs(
  X = X_air[, -i, drop = FALSE],
  y = airport_t100$X[, i],
  family = "negative.binomial",
  enet.alpha = 0.75
)
