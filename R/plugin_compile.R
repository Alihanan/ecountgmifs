.plugin.cache <- new.env(parent = emptyenv())

.plugin.cache.key <- function(code, factory, arguments, type, name) {
  digest::digest(
    list(
      code = code,
      factory = factory,
      arguments = arguments,
      type = type,
      name = name,
      package.version = tryCatch(
        as.character(utils::packageVersion("ecountgmifs")),
        error = function(...) "development"
      ),
      R.version = as.character(getRversion()),
      platform = R.version$platform
    ),
    algo = "xxhash64"
  )
}

.validate.compiled.plugin.pointer <- function(pointer, type) {
  if (!identical(typeof(pointer), "externalptr")) {
    stop(
      "The compiled factory must return an external pointer for a ",
      type,
      " plugin.",
      call. = FALSE
    )
  }

  validator <- switch(
    type,
    link = inspect_link_plugin,
    family = inspect_family_plugin,
    family.link = inspect_family_link_plugin,
    criterion = inspect_criterion_plugin,
    stop("Unknown plugin type.", call. = FALSE)
  )

  validator(pointer)
}

.decorate.compiled.plugin <- function(
    pointer,
    type,
    name,
    factory,
    code,
    arguments,
    cache.key,
    cached,
    compile.environment,
    validation
) {
  attr(pointer, "plugin.type") <- type
  attr(pointer, "plugin.name") <- name
  attr(pointer, "plugin.factory") <- factory
  attr(pointer, "plugin.code") <- code
  attr(pointer, "plugin.arguments") <- arguments
  attr(pointer, "plugin.cache.key") <- cache.key
  attr(pointer, "plugin.cached") <- cached
  attr(pointer, "plugin.compile.environment") <- compile.environment
  attr(pointer, "plugin.validation") <- validation

  if (identical(type, "criterion")) {
    attr(pointer, "criterion.name") <- name
    attr(pointer, "criterion.factory") <- factory
    attr(pointer, "criterion.code") <- code
    attr(pointer, "criterion.arguments") <- arguments
    attr(pointer, "criterion.cache.key") <- cache.key
    attr(pointer, "criterion.cached") <- cached
    attr(pointer, "criterion.compile.environment") <- compile.environment
  }

  class(pointer) <- unique(
    c("compiled.plugin", paste0("compiled.", type), class(pointer))
  )
  pointer
}

.compile.plugin <- function(
    code,
    factory,
    type,
    name = factory,
    arguments = list(),
    cache = TRUE
) {
  type <- match.arg(type, c("link", "family", "family.link", "criterion"))

  if (!is.character(code) || length(code) != 1L || is.na(code)) {
    stop("`code` must be one character string.", call. = FALSE)
  }
  if (!is.character(factory) || length(factory) != 1L || is.na(factory) ||
      !nzchar(factory)) {
    stop("`factory` must be one non-empty character string.", call. = FALSE)
  }
  if (!is.character(name) || length(name) != 1L || is.na(name) || !nzchar(name)) {
    stop("`name` must be one non-empty character string.", call. = FALSE)
  }
  if (!is.list(arguments)) {
    stop("`arguments` must be a list.", call. = FALSE)
  }
  if (!is.logical(cache) || length(cache) != 1L || is.na(cache)) {
    stop("`cache` must be TRUE or FALSE.", call. = FALSE)
  }

  key <- .plugin.cache.key(code, factory, arguments, type, name)

  if (cache && exists(key, envir = .plugin.cache, inherits = FALSE)) {
    pointer <- get(key, envir = .plugin.cache, inherits = FALSE)
    attr(pointer, "plugin.cached") <- TRUE
    if (identical(type, "criterion")) {
      attr(pointer, "criterion.cached") <- TRUE
    }
    return(pointer)
  }

  compile.environment <- new.env(parent = baseenv())
  Rcpp::sourceCpp(code = code, env = compile.environment, rebuild = TRUE)

  if (!exists(factory, envir = compile.environment, mode = "function", inherits = FALSE)) {
    stop("Compiled source did not export factory `", factory, "`.", call. = FALSE)
  }

  pointer <- do.call(
    get(factory, envir = compile.environment, inherits = FALSE),
    arguments
  )

  validation <- tryCatch(
    .validate.compiled.plugin.pointer(pointer, type),
    error = function(error) {
      stop(
        "Factory `", factory, "` returned an invalid ", type,
        " plugin: ", conditionMessage(error),
        call. = FALSE
      )
    }
  )

  pointer <- .decorate.compiled.plugin(
    pointer = pointer,
    type = type,
    name = name,
    factory = factory,
    code = code,
    arguments = arguments,
    cache.key = key,
    cached = FALSE,
    compile.environment = compile.environment,
    validation = validation
  )

  if (cache) {
    assign(key, pointer, envir = .plugin.cache)
  }

  pointer
}

#' Live-compile a C++ link plugin
#'
#' @param code One character string containing complete C++ source.
#' @param factory Name of the exported factory defined by `code`.
#' @param name Friendly name retained as R metadata.
#' @param arguments Named list passed to the factory.
#' @param cache Reuse an identical compiled plugin in the current session.
#' @return An owning external pointer returned by the factory.
#' @export
#' @family live-compiled plugins
compile.link <- function(code, factory, name = factory, arguments = list(), cache = TRUE) {
  .compile.plugin(code, factory, "link", name, arguments, cache)
}

#' Live-compile a C++ family plugin
#' @inheritParams compile.link
#' @return An owning external pointer returned by the factory.
#' @export
#' @family live-compiled plugins
compile.family <- function(code, factory, name = factory, arguments = list(), cache = TRUE) {
  .compile.plugin(code, factory, "family", name, arguments, cache)
}

#' Live-compile a C++ fused family-link plugin
#' @inheritParams compile.link
#' @return An owning external pointer returned by the factory.
#' @export
#' @family live-compiled plugins
compile.family.link <- function(
    code,
    factory,
    name = factory,
    arguments = list(),
    cache = TRUE
) {
  .compile.plugin(code, factory, "family.link", name, arguments, cache)
}

#' Live-compile a C++ criterion plugin
#' @inheritParams compile.link
#' @return An owning external pointer returned by the factory.
#' @export
#' @family live-compiled plugins
compile.criterion <- function(
    code,
    factory,
    name = factory,
    arguments = list(),
    cache = TRUE
) {
  .compile.plugin(code, factory, "criterion", name, arguments, cache)
}

.compile.plugin.file <- function(
    file,
    factory,
    type,
    name = factory,
    arguments = list(),
    cache = TRUE
) {
  if (!is.character(file) || length(file) != 1L || is.na(file)) {
    stop("`file` must be one character string.", call. = FALSE)
  }
  if (!file.exists(file)) {
    stop("File does not exist: ", file, call. = FALSE)
  }

  .compile.plugin(
    code = paste(readLines(file, warn = FALSE), collapse = "\n"),
    factory = factory,
    type = type,
    name = name,
    arguments = arguments,
    cache = cache
  )
}

#' @rdname compile.link
#' @param file Path to a complete C++ source file.
#' @export
compile.link.file <- function(
    file,
    factory,
    name = factory,
    arguments = list(),
    cache = TRUE
) {
  .compile.plugin.file(file, factory, "link", name, arguments, cache)
}

#' @rdname compile.family
#' @param file Path to a complete C++ source file.
#' @export
compile.family.file <- function(
    file,
    factory,
    name = factory,
    arguments = list(),
    cache = TRUE
) {
  .compile.plugin.file(file, factory, "family", name, arguments, cache)
}

#' @rdname compile.family.link
#' @param file Path to a complete C++ source file.
#' @export
compile.family.link.file <- function(
    file,
    factory,
    name = factory,
    arguments = list(),
    cache = TRUE
) {
  .compile.plugin.file(file, factory, "family.link", name, arguments, cache)
}

#' @rdname compile.criterion
#' @param file Path to a complete C++ source file.
#' @export
compile.criterion.file <- function(
    file,
    factory,
    name = factory,
    arguments = list(),
    cache = TRUE
) {
  .compile.plugin.file(file, factory, "criterion", name, arguments, cache)
}

.plugin.cache.keys <- function(type = NULL) {
  keys <- ls(envir = .plugin.cache, all.names = TRUE)
  if (is.null(type) || length(keys) == 0L) {
    return(keys)
  }

  keep <- vapply(
    keys,
    function(key) {
      identical(
        attr(get(key, envir = .plugin.cache, inherits = FALSE),
             "plugin.type", exact = TRUE),
        type
      )
    },
    logical(1)
  )
  keys[keep]
}

#' Clear the live-compiled plugin cache
#' @return Invisibly returns `TRUE`.
#' @export
#' @family live-compiled plugins
clear.plugin.cache <- function() {
  keys <- .plugin.cache.keys()
  if (length(keys) > 0L) {
    rm(list = keys, envir = .plugin.cache)
  }
  invisible(TRUE)
}

#' List live-compiled plugin cache keys
#' @return Character vector of cache keys.
#' @export
#' @family live-compiled plugins
list.plugin.cache <- function() {
  .plugin.cache.keys()
}

#' @export
print.compiled.plugin <- function(x, ...) {
  cat("<live-compiled ", attr(x, "plugin.type", exact = TRUE), ">\n", sep = "")
  cat("  name:    ", attr(x, "plugin.name", exact = TRUE), "\n", sep = "")
  cat("  factory: ", attr(x, "plugin.factory", exact = TRUE), "\n", sep = "")
  cat("  cached:  ", isTRUE(attr(x, "plugin.cached", exact = TRUE)), "\n", sep = "")
  invisible(x)
}

#' @rdname clear.plugin.cache
#' @export
clear.criterion.cache <- function() {
  keys <- .plugin.cache.keys("criterion")
  if (length(keys) > 0L) {
    rm(list = keys, envir = .plugin.cache)
  }
  invisible(TRUE)
}

#' @rdname list.plugin.cache
#' @export
list.criterion.cache <- function() {
  .plugin.cache.keys("criterion")
}

# Compatibility names from the earlier criterion-only compiler.
ecountgmifs.compile.criterion <- compile.criterion
ecountgmifs.compile.criterion.file <- compile.criterion.file
ecountgmifs.clear.criterion.cache <- clear.criterion.cache
ecountgmifs.list.criterion.cache <- list.criterion.cache
