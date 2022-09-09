set.seed(219)
source('experiments/00-helpers.R')


# Data simulation - Simulated DP ----------------------------------------------------
## M = 1
sim_dp_1 <- rdp(1, 'rnorm', list(mean = 0, sd = 1), tol = 1e-10)
length(sim_dp_1$locations)

sim_data_1 <- sample(
  x = sim_dp_1$locations,
  size = 100,
  replace = TRUE,
  prob = sim_dp_1$probs
)

plot(density(sim_data_1), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from a DP with M = 1')
curve(dnorm, add = TRUE, lwd = 2)

## M = 10
sim_dp_2 <- rdp(10, 'rnorm', list(mean = 0, sd = 1), tol = 1e-10)
length(sim_dp_2$locations)

sim_data_2 <- sample(
  x = sim_dp_2$locations,
  size = 100,
  replace = TRUE,
  prob = sim_dp_2$probs
)

plot(density(sim_data_2), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from a DP with M = 10')
curve(dnorm, add = TRUE, lwd = 2)

## M = 50
sim_dp_3 <- rdp(50, 'rnorm', list(mean = 0, sd = 1), tol = 1e-10)
length(sim_dp_3$locations)

sim_data_3 <- sample(
  x = sim_dp_3$locations,
  size = 100,
  replace = TRUE,
  prob = sim_dp_3$probs
)

plot(density(sim_data_3), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from a DP with M = 50')
curve(dnorm, add = TRUE, lwd = 2)

## M = 100
sim_dp_4 <- rdp(100, 'rnorm', list(mean = 0, sd = 1), tol = 1e-10)
length(sim_dp_4$locations)

sim_data_4 <- sample(
  x = sim_dp_4$locations,
  size = 300,
  replace = TRUE,
  prob = sim_dp_4$probs
)

plot(density(sim_data_4), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from a DP with M = 100')
curve(dnorm, add = TRUE, lwd = 2)

## M = 1000
sim_dp_5 <- rdp(1000, 'rnorm', list(mean = 0, sd = 1), tol = 1e-10)
sim_data_5 <- sample(
  x = sim_dp_5$locations,
  size = 100,
  replace = TRUE,
  prob = sim_dp_5$probs
)

plot(density(sim_data_5), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from a DP with M = 1000')
curve(dnorm, add = TRUE, lwd = 2)

#### All plots together
plot(density(sim_data_1), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from a simulated DP')
lines(density(sim_data_2), xlim = c(-3, 3), col = 'blue', lwd = 2)
lines(density(sim_data_3), xlim = c(-3, 3), col = 'green', lwd = 2)
lines(density(sim_data_4), xlim = c(-3, 3), col = 'purple', lwd = 2)
lines(density(sim_data_5), xlim = c(-3, 3), col = 'brown', lwd = 2)
curve(dnorm, add = TRUE, lwd = 2)

# Data simulation - Marginal distribution -------------------------------------------
## DP parameters
M <- 100
G0 <- 'rnorm'

## Simulation parameters
set.seed(219)
n <- 100
aux <- rep(NA_real_, n)

## Simulation
counter_env <- new.env(hash = TRUE)

for (i in 1:n) {
  norm_term <- 1/(M + i - 1)
  new_value <- do.call(G0, list(n = 1, mean = 0, sd = 1))
  
  values <- c(aux[0:(i-1)], new_value)
  probs <- c(rep(norm_term, i - 1), M * norm_term)
  aux[i] <- sample(x = values, size = 1, prob = probs)
}

## We can check the value of k
k <- length(unique(aux))
k

## Function for data simulation
rdp_data <- function(n, M, G0, G0_params){
  # n        : length of the sample 
  # M        : precision parameter
  # G0       : sampling function of the desired centering measure
  # G0_params: parameters for G0
  
  # We add the number of samples
  G0_params <- c(G0_params, n = 1)
  
  # Sample vector
  dp_sample <- vector(mode = 'numeric', length = n)
  
  # Frequency counter
  counter <- dict()
  
  # aux function - See https://stackoverflow.com/questions/13990125/
  # sample-from-vector-of-varying-length-including-1
  sample.vec <- function(x, ...){x[sample(length(x), ...)]}
  
  for (i in 1:n){
    norm_term <- 1/(M + i - 1)
    new_value <- do.call(G0, G0_params)
    
    old_values <- unlist(counter$keys())
    old_freq <- unlist(counter$values())
    
    values <- c(old_values, new_value)
    probs <- c(old_freq * norm_term, M * norm_term)
    
    new_value <- sample.vec(x = values, size = 1, prob = probs)
    dp_sample[i] <- new_value
    
    # Update frequencies
    freq <- counter$get(new_value, 0)
    counter$set(new_value, freq + 1)
  }
  
  return(dp_sample)
}

## Replicate simulation
sim_data_1_n <- rdp_data(500, 1, 'rnorm', list(mean = 0, sd = 1))
sim_data_2_n <- rdp_data(500, 10, 'rnorm', list(mean = 0, sd = 1))
sim_data_3_n <- rdp_data(500, 50, 'rnorm', list(mean = 0, sd = 1))
sim_data_4_n <- rdp_data(500, 100, 'rnorm', list(mean = 0, sd = 1))
sim_data_5_n <- rdp_data(500, 1000, 'rnorm', list(mean = 0, sd = 1))
sim_data_6_n <- rdp_data(500, 10000, 'rnorm', list(mean = 0, sd = 1))

par(mfrow = c(2, 3))
plot(density(sim_data_1_n), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 1')
curve(dnorm, add = TRUE, lwd = 2)

plot(density(sim_data_2_n), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 10')
curve(dnorm, add = TRUE, lwd = 2)

plot(density(sim_data_3_n), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 50')
curve(dnorm, add = TRUE, lwd = 2)

plot(density(sim_data_4_n), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 100')
curve(dnorm, add = TRUE, lwd = 2)

plot(density(sim_data_5_n), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 1000')
curve(dnorm, add = TRUE, lwd = 2)

plot(density(sim_data_6_n), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 10000')
curve(dnorm, add = TRUE, lwd = 2)

## Comparison
par(mfrow = c(1, 2))
plot(density(sim_data_1), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from a simulated DP')
lines(density(sim_data_2), xlim = c(-3, 3), col = 'blue', lwd = 2)
lines(density(sim_data_3), xlim = c(-3, 3), col = 'green', lwd = 2)
lines(density(sim_data_4), xlim = c(-3, 3), col = 'purple', lwd = 2)
lines(density(sim_data_5), xlim = c(-3, 3), col = 'brown', lwd = 2)
curve(dnorm, add = TRUE, lwd = 2)

plot(density(sim_data_1_n), xlim = c(-3, 3), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution')
lines(density(sim_data_2_n), xlim = c(-3, 3), col = 'blue', lwd = 2)
lines(density(sim_data_3_n), xlim = c(-3, 3), col = 'green', lwd = 2)
lines(density(sim_data_4_n), xlim = c(-3, 3), col = 'purple', lwd = 2)
lines(density(sim_data_5_n), xlim = c(-3, 3), col = 'brown', lwd = 2)
curve(dnorm, add = TRUE, lwd = 2)

## Replicate simulation - Gamma centering measure
set.seed(219)

sim_data_1_n <- rdp_data(500, 1, 'rgamma', list(shape = 3, rate = 10))
sim_data_2_n <- rdp_data(500, 10, 'rgamma', list(shape = 3, rate = 10))
sim_data_3_n <- rdp_data(500, 50, 'rgamma', list(shape = 3, rate = 10))
sim_data_4_n <- rdp_data(500, 100, 'rgamma', list(shape = 3, rate = 10))
sim_data_5_n <- rdp_data(500, 1000, 'rgamma', list(shape = 3, rate = 10))
sim_data_6_n <- rdp_data(500, 10000, 'rgamma', list(shape = 3, rate = 10))

par(mfrow = c(2, 3))
plot(density(sim_data_1_n), xlim = c(0, 1), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 1')
curve(dgamma(x, shape = 3, rate = 10), add = TRUE, lwd = 2)

plot(density(sim_data_2_n), xlim = c(0, 1), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 10',
     ylim = c(0, 2.8))
curve(dgamma(x, shape = 3, rate = 10), add = TRUE, lwd = 2)

plot(density(sim_data_3_n), xlim = c(0, 1), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 50',
     ylim = c(0, 2.8))
curve(dgamma(x, shape = 3, rate = 10), add = TRUE, lwd = 2)

plot(density(sim_data_4_n), xlim = c(0, 1), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 100',
     ylim = c(0, 2.8))
curve(dgamma(x, shape = 3, rate = 10), add = TRUE, lwd = 2)

plot(density(sim_data_5_n), xlim = c(0, 1), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 1000',
     ylim = c(0, 2.8))
curve(dgamma(x, shape = 3, rate = 10), add = TRUE, lwd = 2)

plot(density(sim_data_6_n), xlim = c(0, 1), col = 'red', lwd = 2,
     main = 'Simulated data from marginal distribution - M = 10000',
     ylim = c(0, 2.8))
curve(dgamma(x, shape = 3, rate = 10), add = TRUE, lwd = 2)
