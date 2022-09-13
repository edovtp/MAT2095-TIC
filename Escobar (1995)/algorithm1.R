library(tidyverse)
source("experiments/00-helpers.R")


set.seed(219)

# Data simulation -------------------------------------------------------------------
# We take a sample of size n from the Dirichlet Process, then we simulate data from each
# sample using the normal distribution

n <- 100
M <- 10
m <- 1
tau <- 1
s <- 1
S <- 1


pi_vector <- rdp_data(n, M, rnormigamma,
                      G0_params = list(mu = 1, lambda = 1, alpha = 100, beta = 1))
pi_vector
