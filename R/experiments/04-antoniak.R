set.seed(219)
source('experiments/00-helpers.R')


# Empirical evaluation of Antoniak (1974) -------------------------------------------
# Result: E(k|M, n) approx M * ln(1 + n/M) for n moderately large

# Function to simulate a sample of size n from a DP with normal centering measure
n_unique_fun <- function(n, M){
  n <- floor(n)
  sim_data <- rdp_data(n, M, 'rnorm', list(mean = 0, sd = 1))
  n_unique <- length(unique(sim_data))
}

## M = 1
M <- 1

n <- 100
k_100 <- replicate(100, n_unique_fun(n, M), simplify = 'vector')
ek_approx_100 <- mean(k_100)

n <- 300
k_300 <- replicate(100, n_unique_fun(n, M), simplify = 'vector')
ek_approx_300 <- mean(k_300)

n <- 500
k_500 <- replicate(100, n_unique_fun(n, M), simplify = 'vector')
ek_approx_500 <- mean(k_500)

n <- 700
k_700 <- replicate(100, n_unique_fun(n, M), simplify = 'vector')
ek_approx_700 <- mean(k_700)

n <- 900
k_900 <- replicate(100, n_unique_fun(n, M), simplify = 'vector')
ek_approx_900 <- mean(k_900)

n <- 1100
k_1100 <- replicate(100, n_unique_fun(n, M), simplify = 'vector')
ek_approx_1100 <- mean(k_1100)

n <- 1300
k_1300 <- replicate(100, n_unique_fun(n, M), simplify = 'vector')
ek_approx_1300 <- mean(k_1300)

n <- 1500
k_1500 <- replicate(100, n_unique_fun(n, M), simplify = 'vector')
ek_approx_1500 <- mean(k_1500)

n <- 2000
k_2000 <- replicate(100, n_unique_fun(n, M), simplify = 'vector')
ek_approx_2000 <- mean(k_2000)

n <- 3000
k_3000 <- replicate(100, n_unique_fun(n, M), simplify = 'vector')
ek_approx_3000 <- mean(k_3000)

n <- 5000
k_5000 <- replicate(100, n_unique_fun(n, M), simplify = 'vector')
ek_approx_5000 <- mean(k_1500)

n_values <- c(100, 300, 500, 700, 900, 1100, 1300, 1500, 2000, 3000, 5000)
k_values <- c(ek_approx_100, ek_approx_300, ek_approx_500, ek_approx_700,
              ek_approx_900, ek_approx_1100, ek_approx_1300, ek_approx_1500,
              ek_approx_2000, ek_approx_3000, ek_approx_5000)

curve(M * log(1 + x/M), from = 1, to = 5000, ylim = c(0, max(k_values) + 1))
points(n_values, k_values, pch = 16, col = 'red')


# Experiment ------------------------------------------------------------------------
n_unique_fun <- function(n, M){
  n <- floor(n)
  sim_data <- rdp_data(n, M, 'rnorm', list(mean = 0, sd = 1))
  n_unique <- length(unique(sim_data))
}

antoniak_fun <- function(n_values, M, plot = TRUE){
  # Vector for the approximate expectations of k
  ek_values <- vector(mode = 'numeric', length = length(n_values))
  
  for (i in seq_along(n_values)){
    k_sim <- replicate(100, n_unique_fun(n_values[i], M), simplify = 'vector')
    ek_values[i] <- mean(k_sim)
  }
  
  if (plot){
    curve(M * log(x), from = 1, to = max(n_values) + 100,
          col = 'purple', lwd = 2)
    curve(M * log(1 + x/M), from = 1, to = max(n_values) + 100,
          col = 'blue', lwd = 2, add = TRUE)
    points(n_values, ek_values, pch = 16, col = 'red')
    legend(x = 'bottomright', legend = c('Korwar & Hollander', 'Antoniak'),
           col = c('purple', 'blue'), lwd = c(2, 2))
  }
  
  return(list(n_values = n_values, ek_values = ek_values))
}

# Different values of M
n_values <- c(100, 500, seq(1000, 21000, by = 2000))

antoniak_fun(n_values, 1)
antoniak_fun(n_values, 50)
antoniak_fun(n_values, 100)
antoniak_fun(n_values, 1000)
