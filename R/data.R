#' RNA-seq miRNA-mRNA example dataset with 97 mRNA targets
#'
#' @description
#' RNA-seq dataset for demonstrating extended count GMIFS models with prior
#' network information. The dataset contains expression profiles for 434
#' biological samples, with 97 mRNA response variables and 2636 predictor
#' variables. It also includes a binary prior-knowledge matrix, a binary
#' ground-truth interaction matrix, and sample-level tissue annotations.
#'
#' @format A list with five elements:
#' \describe{
#'   \item{\code{Y}}{
#'     Numeric response matrix with 434 rows and 97 columns. Rows correspond to
#'     biological samples and columns correspond to mRNA target variables.
#'   }
#'   \item{\code{X}}{
#'     Numeric predictor matrix with 434 rows and 2636 columns. Rows correspond
#'     to the same biological samples as in \code{Y}; columns correspond to
#'     candidate predictor variables.
#'   }
#'   \item{\code{prior}}{
#'     Logical or binary prior-knowledge matrix with 97 rows and 2636 columns.
#'     Rows correspond to mRNA target variables in \code{Y}; columns correspond
#'     to predictor variables in \code{X}. A value of \code{TRUE} or \code{1}
#'     indicates that the corresponding predictor-target interaction is present
#'     in the prior interaction network.
#'   }
#'   \item{\code{truth}}{
#'     Logical or binary ground-truth matrix with 97 rows and 2636 columns.
#'     Rows correspond to mRNA target variables in \code{Y}; columns correspond
#'     to predictor variables in \code{X}. A value of \code{TRUE} or \code{1}
#'     indicates that the corresponding predictor-target interaction is present
#'     in the reference interaction network.
#'   }
#'  \item{\code{organ}}{RNAAtlas-based organ-type annotation for each of 434 samples.}
#'  \item{\code{organ7cluster}}{organ-type metadata clustered into 7 groups for each of 434 samples.}
#' }
#'
#' @details
#' The dataset is derived from the RNA-seq regulatory-network setting described
#' by Anuarbekov and Kléma (2025). In that work, RNA-seq profiles are used
#' together with prior interaction information to improve inference of
#' RNA-RNA regulatory interactions. The learning task is formulated as a
#' feature-selection problem: for each target transcript, the goal is to infer
#' a parent set of regulatory predictors from RNA-seq data and prior knowledge.
#'
#' This package dataset contains the miRNA-mRNA-style design used for examples:
#' \code{Y} stores the 97 mRNA target variables, \code{X} stores 2636 candidate
#' predictors, \code{prior} stores the initial prior interaction network, and
#' \code{truth} stores the reference binary interaction network. The original
#' test set used in the paper is not included in this package dataset.
#'
#' The matrices are stored in sample-by-variable orientation. Thus,
#' \code{nrow(Y) == nrow(X) == nrow(sample)}, while \code{prior} and
#' \code{truth} are stored in target-by-predictor orientation:
#' \code{nrow(prior) == ncol(Y)} and \code{ncol(prior) == ncol(X)}.
#'
#' @source
#' Adapted from the RNA-seq and prior-knowledge setup described in:
#' Anuarbekov, A. and Kléma, J. (2025). Utilizing RNA-seq data in monotone
#' iterative generalized linear model to elevate prior knowledge quality of the
#' circRNA-miRNA-mRNA regulatory axis. \emph{BMC Bioinformatics}, 26, 139.
#' \doi{10.1186/s12859-025-06161-w}.
#'
#' @references
#' Anuarbekov, A. and Kléma, J. (2025). Utilizing RNA-seq data in monotone
#' iterative generalized linear model to elevate prior knowledge quality of the
#' circRNA-miRNA-mRNA regulatory axis. \emph{BMC Bioinformatics}, 26, 139.
#' \doi{10.1186/s12859-025-06161-w}.
#'
#' @examples
#' data(mrna97_rnaseq)
#'
#' names(mrna97_rnaseq)
#'
#' dim(mrna97_rnaseq$Y)
#' dim(mrna97_rnaseq$X)
#' dim(mrna97_rnaseq$prior)
#' dim(mrna97_rnaseq$truth)
#'
#' table(mrna97_rnaseq$organ)
#' table(mrna97_rnaseq$organ7cluster)
#'
#' stopifnot(nrow(mrna97_rnaseq$Y) == nrow(mrna97_rnaseq$X))
#' stopifnot(nrow(mrna97_rnaseq$Y) == length(mrna97_rnaseq$organ))
#' stopifnot(nrow(mrna97_rnaseq$prior) == ncol(mrna97_rnaseq$Y))
#' stopifnot(ncol(mrna97_rnaseq$prior) == ncol(mrna97_rnaseq$X))
#' stopifnot(identical(dim(mrna97_rnaseq$prior), dim(mrna97_rnaseq$truth)))
"mrna97_rnaseq"





#' T-100 airport passenger-flow network dataset
#'
#' @description
#' Monthly airport passenger-flow dataset for demonstrating network recovery
#' with extended count GMIFS. The dataset contains a monthly airport-level count
#' matrix and multiple binary ground-truth route networks constructed using
#' different temporal persistence thresholds.
#'
#' @format A list with eight elements:
#' \describe{
#'   \item{\code{X}}{
#'     Numeric matrix with rows corresponding to monthly observations and
#'     columns corresponding to airports. Each column represents an airport-level
#'     passenger-count time series used as a response or predictor depending on
#'     the target airport.
#'   }
#'   \item{\code{GT1}}{
#'     Logical or binary adjacency matrix for routes active in at least 1 month.
#'   }
#'   \item{\code{GT12}}{
#'     Logical or binary adjacency matrix for routes active in at least 12 months.
#'   }
#'   \item{\code{GT120}}{
#'     Logical or binary adjacency matrix for routes active in at least 120 months.
#'   }
#'   \item{\code{GT216}}{
#'     Logical or binary adjacency matrix for routes active in at least 216 months.
#'   }
#'   \item{\code{GT432}}{
#'     Logical or binary adjacency matrix for routes active in all 432 months.
#'   }
#'   \item{\code{sample}}{
#'     Data frame with one row per monthly observation, containing time metadata
#'     such as year, month, and date.
#'   }
#'   \item{\code{airport}}{
#'     Data frame with one row per airport. The rows correspond to the columns of
#'     \code{X} and to the rows and columns of the ground-truth matrices.
#'   }
#' }
#'
#' @details
#' The airport benchmark is formulated as a directed network-recovery task.
#' For each destination airport, the response is its monthly inbound passenger
#' count and the candidate predictors are monthly outbound passenger counts from
#' the other airports. A selected predictor is interpreted as a directed route
#' from the predictor/origin airport to the response/destination airport.
#'
#' All ground-truth matrices use destination-by-origin orientation. Rows
#' correspond to destination airports and columns correspond to origin airports.
#' For a temporal persistence threshold \code{q}, an edge is present if the
#' corresponding origin-destination route is active in at least \code{q} months.
#'
#' The included thresholds are:
#' \describe{
#'   \item{\code{GT1}}{Route observed in at least 1 month.}
#'   \item{\code{GT12}}{Route observed in at least 12 months.}
#'   \item{\code{GT120}}{Route observed in at least 120 months.}
#'   \item{\code{GT216}}{Route observed in at least 216 months.}
#'   \item{\code{GT432}}{Route observed in all 432 months.}
#' }
#'
#' @examples
#' data(airport_t100)
#'
#' names(airport_t100)
#' dim(airport_t100$X)
#' dim(airport_t100$GT12)
#'
#' i <- 1
#' y <- airport_t100$X[, i]
#' X <- airport_t100$X[, -i, drop = FALSE]
#' truth_i <- airport_t100$GT12[i, -i]
#'
#' stopifnot(nrow(airport_t100$X) == nrow(airport_t100$sample))
#' stopifnot(ncol(airport_t100$X) == nrow(airport_t100$airport))
#' stopifnot(identical(dim(airport_t100$GT1), dim(airport_t100$GT12)))
#' stopifnot(identical(dim(airport_t100$GT12), dim(airport_t100$GT120)))
#' stopifnot(identical(dim(airport_t100$GT120), dim(airport_t100$GT216)))
#' stopifnot(identical(dim(airport_t100$GT216), dim(airport_t100$GT432)))
#'
#' @source
#' Adapted from the T-100 Domestic Segment passenger-flow benchmark described in
#' Anuarbekov and Kléma, using data from the U.S. Bureau of Transportation
#' Statistics.
#'
#' @references
#' Anuarbekov, A. and Kléma, J. Extended generalized monotone incremental
#' forward stagewise regression for penalized negative binomial path-following
#' modeling of high-dimensional count data.
"airport_t100"
