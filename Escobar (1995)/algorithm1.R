library(tidyverse)
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
