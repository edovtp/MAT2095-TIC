source('pruebas/00-helpers.R')
set.seed(219)


# Korwar and Hollander (1973) -------------------------------------------------------
## As n -> infty then k grows as M log(n)

n_unique_fun <- function(n, M){
  n <- floor(n)
  sim_data <- rdp_data(n, M, 'rnorm', list(mean = 0, sd = 1))
  n_unique <- length(unique(sim_data))
}

n_samples <- seq(100, 1000000, length.out = 10)

## M = 1
k <- sapply(n_samples, n_unique_fun, M = 1)
curve(log(x), from = 1, to = 1000000, ylim = c(0, max(k) + 1))
points(n_samples, k, pch = 16, col = 'red')

## M = 10
k <- sapply(n_samples, n_unique_fun, M = 10)
curve(10 * log(x), from = 1, to = 1000000)
points(n_samples, k, pch = 16, col = 'red')

## M = 100
k <- sapply(n_samples, n_unique_fun, M = 100)
curve(100 * log(x), from = 1, to = 1000000, ylim = c(0, max(k) + 1))
points(n_samples, k, pch = 16, col = 'red')

## M = 1000
k <- sapply(n_samples, n_unique_fun, M = 1000)
curve(1000 * log(x), from = 1, to = 1000000)
points(n_samples, k, pch = 16, col = 'red')

## log10?
## M = 1
k <- sapply(n_samples, n_unique_fun, M = 1)
curve(log10(x), from = 1, to = 100000, ylim = c(0, max(k) + 1))
points(n_samples, k, pch = 16, col = 'red')

## M = 10
k <- sapply(n_samples, n_unique_fun, M = 10)
curve(10 * log10(x), from = 1, to = 100000, ylim = c(0, max(k) + 1))
points(n_samples, k, pch = 16, col = 'red')

## M = 100
k <- sapply(n_samples, n_unique_fun, M = 100)
curve(100 * log10(x), from = 1, to = 100000, ylim = c(0, max(k) + 1))
points(n_samples, k, pch = 16, col = 'red')

## M = 1000
k <- sapply(n_samples, n_unique_fun, M = 1000)
curve(1000 * log10(x), from = 1, to = 100000, ylim = c(0, max(k) + 1))
points(n_samples, k, pch = 16, col = 'red')
