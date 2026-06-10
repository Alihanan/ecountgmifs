.ecountgmifs.criterion.cache <- new.env(parent = emptyenv())

.ecountgmifs.criterion.cache.key <- function(code, symbol) {
  paste0(symbol, "::", digest::digest(code, algo = "xxhash64"))
}

#' Compile a custom C++ selection criterion from code
#'
#' @description
#' Compiles user-supplied C++ code defining a criterion function and returns a
#' criterion object that can be passed to ecountgmifs selection routines.
#'
#' Compiled criteria are cached by default. Repeated calls with the same
#' \code{code} and \code{symbol} return the cached criterion object rather than
#' recompiling the C++ source.
#'
#' The C++ code should define a C-compatible function with signature
#' \preformatted{
#' extern "C" double criterion_name(const EcountgmifsContext* ctx);
#' }
#'
#' The C++ code should include the public API header:
#' \preformatted{
#' #include <Rcpp.h>
#' // [[Rcpp::depends(ecountgmifs)]]
#'
#' #include <ecountgmifs/api.h>
#' }
#'
#' The context object provides read-only access to the model input, control
#' settings, and current state:
#' \preformatted{
#' ctx->input.X
#' ctx->input.y
#' ctx->state.beta
#' ctx->state.loglik
#' ctx->state.iteration
#' }
#'
#' @param code character. C++ source code defining the criterion function.
#' @param symbol character. Name of the C symbol to look up after compilation.
#' @param name character. Friendly name stored in the returned criterion object.
#' @param use.cache logical. If \code{TRUE}, reuse a previously compiled
#'   criterion with the same \code{code} and \code{symbol}.
#'
#' @return An object of class \code{"ecountgmifs.criterion"}.
#'
#' @examples
#' code <- '
#' #include <Rcpp.h>
#' // [[Rcpp::depends(ecountgmifs)]]
#'
#' #include <ecountgmifs/api.h>
#' #include <cmath>
#'
#' extern "C" double cpp_score_my_bic(const EcountgmifsContext* ctx)
#' {
#'   double negloglik = -1.0 * ctx->state.loglik;
#'   double n = static_cast<double>(ctx->input.X.n_rows);
#'   double nnz = arma::accu(ctx->state.beta != 0.0);
#'
#'   return 2.0 * negloglik + std::log(n) * nnz;
#' }
#'
#' // [[Rcpp::export]]
#' int cpp_anchor_my_bic()
#' {
#'   return 0;
#' }
#' '
#'
#' criterion <- ecountgmifs.compile.criterion(
#'   code = code,
#'   symbol = "cpp_score_my_bic",
#'   name = "my_bic"
#' )
#'
#' ecountgmifs.test.criterion(
#'   criterion,
#'   loglik = -200,
#'   n = 100,
#'   p = 20,
#'   nnz = 3
#' )
#'
#' @export
ecountgmifs.compile.criterion <- function(
    code,
    symbol,
    name = symbol,
    use.cache = TRUE
) {
  if (!is.character(code) || length(code) != 1L || is.na(code)) {
    stop("value of 'code' must be a single character string")
  }

  if (!is.character(symbol) || length(symbol) != 1L || is.na(symbol)) {
    stop("value of 'symbol' must be a single character string")
  }

  if (!is.character(name) || length(name) != 1L || is.na(name)) {
    stop("value of 'name' must be a single character string")
  }

  if (!is.logical(use.cache) || length(use.cache) != 1L || is.na(use.cache)) {
    stop("value of 'use.cache' must be TRUE or FALSE")
  }

  cache.key <- .ecountgmifs.criterion.cache.key(code, symbol)

  if (use.cache &&
      exists(cache.key, envir = .ecountgmifs.criterion.cache, inherits = FALSE)) {
    criterion <- get(cache.key, envir = .ecountgmifs.criterion.cache, inherits = FALSE)
    criterion$name <- name
    criterion$cached <- TRUE
    return(criterion)
  }

  before <- names(getLoadedDLLs())

  Rcpp::sourceCpp(code = code, rebuild = TRUE)

  after <- names(getLoadedDLLs())
  new.dlls <- setdiff(after, before)

  dlls <- getLoadedDLLs()

  for (nm in c(new.dlls, setdiff(names(dlls), new.dlls))) {
    out <- try(getNativeSymbolInfo(symbol, dlls[[nm]]), silent = TRUE)

    if (!inherits(out, "try-error")) {
      criterion <- structure(
        list(
          name = name,
          symbol = symbol,
          pointer = out$address,
          dll = dlls[[nm]],
          code = code,
          cache.key = cache.key,
          cached = FALSE
        ),
        class = "ecountgmifs.criterion"
      )

      if (use.cache) {
        assign(
          cache.key,
          criterion,
          envir = .ecountgmifs.criterion.cache
        )
      }

      return(criterion)
    }
  }

  stop("compiled criterion symbol was not found: ", symbol, call. = FALSE)
}


#' Compile a custom C++ selection criterion from a file
#'
#' @description
#' Reads a C++ source file, prints its contents, and compiles it using
#' \code{ecountgmifs.compile.criterion()}.
#'
#' Compiled criteria are cached by default using the file contents and symbol.
#'
#' @param file character. Path to a C++ source file.
#' @param symbol character. Name of the C symbol to look up after compilation.
#' @param name character. Friendly name stored in the returned criterion object.
#' @param use.cache logical. If \code{TRUE}, reuse a previously compiled
#'   criterion with the same source code and \code{symbol}.
#'
#' @return An object of class \code{"ecountgmifs.criterion"}.
#'
#' @export
ecountgmifs.compile.criterion.file <- function(
    file,
    symbol,
    name = symbol,
    use.cache = TRUE
) {
  if (!is.character(file) || length(file) != 1L || is.na(file)) {
    stop("value of 'file' must be a single character string")
  }

  if (!is.character(symbol) || length(symbol) != 1L || is.na(symbol)) {
    stop("value of 'symbol' must be a single character string")
  }

  if (!is.character(name) || length(name) != 1L || is.na(name)) {
    stop("value of 'name' must be a single character string")
  }

  if (!file.exists(file)) {
    stop("file does not exist: ", file, call. = FALSE)
  }

  code <- paste(readLines(file, warn = FALSE), collapse = "\n")

  cat(code)
  cat("\n")

  ecountgmifs.compile.criterion(
    code = code,
    symbol = symbol,
    name = name,
    use.cache = use.cache
  )
}


#' Clear compiled criterion cache
#'
#' @description
#' Clears the in-memory cache used by \code{ecountgmifs.compile.criterion()}.
#' This does not unload shared libraries already loaded by R.
#'
#' @return Invisibly returns \code{TRUE}.
#'
#' @export
ecountgmifs.clear.criterion.cache <- function() {
  rm(list = ls(envir = .ecountgmifs.criterion.cache, all.names = TRUE),
     envir = .ecountgmifs.criterion.cache)
  invisible(TRUE)
}


#' List cached compiled criteria
#'
#' @description
#' Lists cache keys for compiled criteria stored in the current R session.
#'
#' @return Character vector of cache keys.
#'
#' @export
ecountgmifs.list.criterion.cache <- function() {
  ls(envir = .ecountgmifs.criterion.cache, all.names = TRUE)
}


#' Test a compiled criterion
#'
#' @description
#' Calls a compiled criterion object on a manually constructed context. This is
#' useful for checking that custom C++ criteria compile correctly and return the
#' expected value before using them inside \code{ecountgmifs()}.
#'
#' @param criterion object of class \code{"ecountgmifs.criterion"}.
#' @param ... Context fields passed to the internal criterion-calling routine,
#'   such as \code{loglik}, \code{iteration}, \code{epsilon}, \code{dispersion},
#'   and small test matrices/vectors when supported by the test helper.
#'
#' @return Numeric value returned by the compiled criterion.
#'
#' @export
ecountgmifs.test.criterion <- function(criterion, ...) {
  if (!inherits(criterion, "ecountgmifs.criterion")) {
    stop("value of 'criterion' must be an object of class 'ecountgmifs.criterion'")
  }

  ecountgmifs_call_score_criterion(
    criterion = criterion$pointer,
    ...
  )
}


#' Print a compiled criterion
#'
#' @param x object of class \code{"ecountgmifs.criterion"}.
#' @param ... unused.
#'
#' @return Invisibly returns \code{x}.
#'
#' @export
print.ecountgmifs.criterion <- function(x, ...) {
  cat("<ecountgmifs criterion>\n")
  cat("  name:    ", x$name, "\n", sep = "")
  cat("  symbol:  ", x$symbol, "\n", sep = "")
  cat("  cached:  ", isTRUE(x$cached), "\n", sep = "")

  if (inherits(x$pointer, "NativeSymbol")) {
    cat("  pointer: <NativeSymbol>\n")
  } else if (inherits(x$pointer, "externalptr")) {
    cat("  pointer: <externalptr>\n")
  } else {
    cat("  pointer: <unknown>\n")
  }

  invisible(x)
}
