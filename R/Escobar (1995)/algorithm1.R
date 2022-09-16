source("experiments/00-helpers.R")


set.seed(219)

# Data simulation -------------------------------------------------------------------
# We take a sample of size n from the Dirichlet Process, then we simulate data from each
# sample using the normal distribution
n <- 30
M <- 1 # Use large M for higher values of k

m <- 0
tau <- 100

s <- 10
S <- 2

## Means and variances
pi_vector <- rdp_data(
  n, M, rnormigamma,
  G0_params = list(
    mu = m, lambda = 1 / tau,
    alpha = s / 2, beta = 2 / S
  )
)
pi_vector

## We can retrieve the number of components
k <- length(unique(pi_vector[, 1]))
k

## Data
y <- apply(
  pi_vector, 1,
  function(params) rnorm(1, mean = params[1], sd = sqrt(params[2]))
)

## We can plot the data, visualizing the latent components
unique_mu <- unique(pi_vector[, 1])
plot(density(y),
  main = "Simulated data from a Dirichlet Process Mixture",
  xlab = "y"
)
abline(v = unique_mu, col = "red", lwd = 2, lty = "dashed")


# Algorithm -------------------------------------------------------------------------
rm(list = ls())

# Load helper functions and create a function for easier use of DPM
source("experiments/00-helpers.R")
dpm_data <- function(n, M, m, tau, s, S) {
  # Means and variances
  pi_vector <- rdp_data(
    n, M, rnormigamma,
    G0_params = list(
      mu = m, lambda = 1 / tau, alpha = s / 2, beta = 2 / S
    )
  )

  # Number of components
  k <- length(unique(pi_vector[, 1]))

  # Data
  y <- apply(
    pi_vector, 1, function(par) rnorm(1, mean = par[1], sd = sqrt(par[2]))
  )

  # We return a list
  return(list(y = y, params = pi_vector, k = k))
}

# We first simulate our values
set.seed(219)
n <- 10000
a1_data <- dpm_data(
  n,
  M = 1,
  m = 0, tau = 100,
  s = 10, S = 2
)

# Number of distinct components
a1_data$k

# Plot of the data
plot(density(a1_data$y),
  main = "Simulated data from a Dirichlet Process Mixture",
  xlab = "y"
)
abline(v = unique(a1_data$params[, 1]), col = "red", lwd = 2, lty = "dashed")

## Same parameters ------------------------------------------------------------------
## Set variables from the data
list2env(a1_data, envir = globalenv())

## Parameters for the prior distributions
alpha <- 1
m <- 0
tau <- 100
s <- 10
S <- 2

## Samples
n_samples <- 1000
samples <- array(numeric(n * (n_samples + 1) * 2), dim = c(n_samples + 1, n, 2))
seq_id <- seq.int(1, n)

## TODO: Starting values...
for (i in 1:n) {
  samples[1, i, ] <- rnormigamma(
    n = 1,
    mu = (m + tau*y[i])/(1 + tau),
    lambda = (1 + tau)/tau,
    alpha = (1 + s)/2,
    beta = 2/(S + (y[i] - m)^2 / (1 + tau))
  )
}

#### Algorithm
for (n_sample in 2:(n_samples + 1)) {
  prev_sample <- samples[(n_sample - 1), , ]
  
  for (i in 1:n) {
    # Weights
    q_weights <- vector(mode = "numeric", length = n)
    q_weights[i] <- alpha * dstudent(y[i], nu = s, mu = m,
                                     sigma = sqrt((1 + tau) * S / s))
    
    for (k in seq_id[-i]) {
      prev_sample_k <- prev_sample[k, ]
      q_weights[k] <- dnorm(
        y[i],
        mean = prev_sample_k[1],
        sd = sqrt(prev_sample_k[2])
      )
    }
    
    q_weights <- q_weights / sum(q_weights)
    
    idx_new <- sample(n, size = 1, prob = q_weights)
    if (idx_new == 1) {
      samples[n_sample, i, ] <- rnormigamma(
        n = 1,
        mu = (m + tau*y[i])/(1+ tau),
        lambda = (1 + tau)/tau,
        alpha = (1 + s)/2,
        beta = 2/(S + (y[i] - m)^2 / (1 + tau))
      )
    } else {
      samples[n_sample, i, ] <- prev_sample[idx_new, ]
    }
  }
}
