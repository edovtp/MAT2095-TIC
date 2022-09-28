# EW (1995) - Algorithm 1 -----------------------------------------------------------
ew_algorithm1 <- function(y, prior_par, iter, warmup = floor(iter / 2)) {
  # prior_par should contain: alpha (precision parameter), m, tau, s and S
  list2env(prior_par, envir = environment())
  n <- length(y)

  # Array in which we'll save the samples (plus one for the initial values)
  samples <- array(numeric((iter + 1) * n * 2), dim = c(iter + 1, n, 2))

  # Initial values
  for (i in 1:n) {
    samples[1, i, ] <- rnormigamma(
      n      = 1,
      mu     = (m + tau * y[i]) / (1 + tau),
      lambda = (1 + tau) / tau,
      alpha  = (1 + s) / 2,
      beta   = (S + (y[i] - m)^2 / (1 + tau)) / 2
    )
  }
  
  # Start of the algorithm
  for (n_sample in 2:(iter + 1)) {
    prev_sample <- samples[(n_sample - 1), , ]

    # Update of each component
    for (i in 1:n) {
      # Weights
      q_weights <- vector(mode = "numeric", length = n)
      q_weights[i] <- alpha * dstudent(y[i],
        nu = s, mu = m,
        sigma = sqrt((1 + tau) * S / s)
      )
      q_weights[-i] <- dnorm(y[i], prev_sample[-i, 1], sqrt(prev_sample[-i, 2]))

      # We normalize the vector
      q_weights <- q_weights / sum(q_weights)

      # New sample
      idx_new <- sample(n, size = 1, prob = q_weights)
      if (idx_new == i) {
        samples[n_sample, i, ] <- rnormigamma(
          n      = 1,
          mu     = (m + tau * y[i]) / (1 + tau),
          lambda = (1 + tau) / tau,
          alpha  = (1 + s) / 2,
          beta   = (S + (y[i] - m)^2 / (1 + tau)) / 2
        )
      } else {
        samples[n_sample, i, ] <- prev_sample[idx_new, ]
      }
      
      prev_sample[i, ] <- samples[n_sample, i, ]
    }
  }

  # Sum 2 because of the initial values
  return(samples[(warmup + 2):(iter + 1), , ])
}

# EW (1995) - Algorithm 2 -----------------------------------------------------------
ew_algorithm2 <- function(y, prior_par, iter, warmup = floor(iter / 2)) {
  # prior_par should contain: alpha (precision parameter), a, A, w, W, s and S
  list2env(prior_par, envir = environment())
  n <- length(y)

  # Arrays in which we'll save the samples (plus one for the initial values)
  mt_samples <- array(numeric((iter + 1) * 2), dim = c(iter + 1, 2))
  pi_samples <- array(numeric((iter + 1) * n * 2), dim = c(iter + 1, n, 2))

  # Initial values
  ## Choose values of m and tau from the priors
  m <- rnorm(1, a, sqrt(A))
  tau <- 1 / rgamma(1, w / 2, rate = W / 2)
  mt_samples[1, ] <- c(m, tau)

  ## Initial values of pi
  for (i in 1:n) {
    pi_samples[1, i, ] <- rnormigamma(
      n      = 1,
      mu     = (m + tau * y[i]) / (1 + tau),
      lambda = (1 + tau) / tau,
      alpha  = (1 + s) / 2,
      beta   = (S + (y[i] - m)^2 / (1 + tau)) / 2
    )
  }

  # Start of the algorithm
  for (n_sample in 2:(iter + 1)) {
    prev_mt <- mt_samples[(n_sample - 1), ]
    prev_pi <- pi_samples[(n_sample - 1), , ]
    unique_pi <- unique(prev_pi)

    ## Update m
    Vbar <- 1 / sum(1 / unique_pi[, 2])
    x <- A / (A + prev_mt[2] * Vbar)
    m <- rnorm(
      n    = 1,
      mean = (1 - x) * a + x * Vbar * sum(unique_pi[, 1] / unique_pi[, 2]),
      sd   = sqrt(x * prev_mt[2] * Vbar)
    )

    ## Update tau
    k <- nrow(unique_pi)
    K <- sum((unique_pi[, 1] - m)^2 / unique_pi[, 2])
    tau <- 1 / rgamma(1, (w + k) / 2, rate = (W + K) / 2)

    mt_samples[n_sample, ] <- c(m, tau)
    
    # Update pi for each observation
    for (i in 1:n) {
      # Weights
      q_weights <- vector(mode = "numeric", length = n)
      q_weights[i] <- alpha * dstudent(y[i],
        nu = s, mu = m,
        sigma = sqrt((1 + tau) * S / s)
      )
      q_weights[-i] <- dnorm(y[i], prev_pi[-i, 1], sqrt(prev_pi[-i, 2]))

      # We normalize the vector
      q_weights <- q_weights / sum(q_weights)

      # New sample
      idx_new <- sample(n, size = 1, prob = q_weights)
      if (idx_new == i) {
        pi_samples[n_sample, i, ] <- rnormigamma(
          n      = 1,
          mu     = (m + tau * y[i]) / (1 + tau),
          lambda = (1 + tau) / tau,
          alpha  = (1 + s) / 2,
          beta   = (S + (y[i] - m)^2 / (1 + tau)) / 2
        )
      } else {
        pi_samples[n_sample, i, ] <- prev_pi[idx_new, ]
      }
      
      prev_pi[i, ] <- pi_samples[n_sample, i, ]
    }
  }

  # Sum 2 because of the initial values
  return(list(
    pi = pi_samples[(warmup + 2):(iter + 1), , ],
    mt = mt_samples[(warmup + 2):(iter + 1), ]
  ))
}

# EW (1995) - Algorithm 3 -----------------------------------------------------------
ew_algorithm <- function(y, prior_par, iter, warmup = floor(iter / 2)) {
  # prior_par should contain: a, b, A, w, W, s and S
  list2env(prior_par, envir = environment())
  n <- length(y)
  
  # Arrays in which we'll save the samples (plus one for the initial values)
  et_samples <- vector(mode = "numeric", length = iter)
  al_samples <- vector(mode = "numeric", length = iter + 1)
  mt_samples <- array(numeric((iter + 1) * 2), dim = c(iter + 1, 2))
  pi_samples <- array(numeric((iter + 1) * n * 2), dim = c(iter + 1, n, 2))
  
  # Initial values
  ## Choose a value of alpha from its prior
  alpha <- rgamma(1, a, rate = b)
  al_samples[1] <- alpha
  
  ## Choose values of m and tau from its priors
  tau <- 1 / rgamma(1, w/2, rate = W/2)
  m <- rnorm(1, 0, 1)
  mt_samples[1, ] <- c(m, tau)
  
  ## Initial values of pi
  for (i in 1:n) {
    pi_samples[1, i, ] <- rnormigamma(
      n      = 1,
      mu     = (m + tau * y[i]) / (1 + tau),
      lambda = (1 + tau) / tau,
      alpha  = (1 + s) / 2,
      beta   = (S + (y[i] - m)^2 / (1 + tau)) / 2
    )
  }
  
  # Start of the algorithm
  for (n_sample in 2:(iter + 1)) {
    prev_al <- al_samples[n_sample - 1]
    prev_mt <- mt_samples[(n_sample - 1), ]
    prev_pi <- pi_samples[(n_sample - 1), , ]
    
    unique_pi <- unique(prev_pi)
    k         <- nrow(unique_pi)
    
    ## Update alpha
    eta <- rbeta(1, prev_al + 1, n)
    et_samples[n_sample] <- eta
    odds_weight <- (a + k - 1)/(n*(b - log(eta)))
    weight <- odds_weight/(1 + odds_weight)
    component <- sample(c(0, 1), 1, prob = c(weight, 1 - weight))
    alpha <- rgamma(1, a + k - 1 * component, rate = b - log(eta))
    al_samples[n_sample] <- alpha
    
    ## Update m and tau
    # m
    Vbar <- 1 / sum(1 / unique_pi[, 2])
    x <- A / (A + prev_mt[2] * Vbar)
    m <- rnorm(
      n    = 1,
      mean = x * Vbar * sum(unique_pi[, 1] / unique_pi[, 2]),
      sd   = sqrt(x * prev_mt[2] * Vbar)
    )
    
    # tau
    K <- sum((unique_pi[, 1] - m)^2 / unique_pi[, 2])
    tau <- 1 / rgamma(1, (w + k)/2, rate = (W + K)/2)
    
    mt_samples[n_sample, ] <- c(m, tau)
    
    ## Update pi for each observation
    for (i in 1:n) {
      # Weights
      q_weights <- vector(mode = "numeric", length = n)
      q_weights[i] <- alpha * dstudent(y[i],
                                       nu = s, mu = m,
                                       sigma = sqrt((1 + tau) * S / s)
      )
      q_weights[-i] <- dnorm(y[i], prev_pi[-i, 1], sqrt(prev_pi[-i, 2]))
      
      # New sample
      idx_new <- sample(n, size = 1, prob = q_weights)
      if (idx_new == i) {
        pi_samples[n_sample, i, ] <- rnormigamma(
          n      = 1,
          mu     = (m + tau * y[i]) / (1 + tau),
          lambda = (1 + tau) / tau,
          alpha  = (1 + s) / 2,
          beta   = (S + (y[i] - m)^2 / (1 + tau)) / 2
        )
      } else {
        pi_samples[n_sample, i, ] <- prev_pi[idx_new, ]
      }
      
      # Update previous pi
      prev_pi[i, ] <- pi_samples[n_sample, i, ]
    }
  }
  
  # Sum 2 because of the initial values
  return(list(
    eta = et_samples[(warmup + 2):(iter + 1)],
    alpha = al_samples[(warmup + 2):(iter + 1)],
    pi = pi_samples[(warmup + 2):(iter + 1), , ],
    mt = mt_samples[(warmup + 2):(iter + 1), ]
  ))
}


# ew_algorithm1 <- function(y, prior_par, iter, warmup = floor(iter/2)){
#   # prior_par should contain: alpha (precision parameter), m, tau, s and S
#   list2env(prior_par, envir = environment())
#
#   n <- length(y)
#   seq_id  <- seq.int(1, n)
#
#   # Array in which we'll save the samples (plus one for the initial values)
#   samples <- array(numeric(n * (iter + 1) * 2), dim = c(iter + 1, n, 2))
#
#   # Initial values
#   for (i in 1:n){
#     samples[1, i, ] <- rnormigamma(
#       n      = 1,
#       mu     = (m + tau * y[i])/(1 + tau),
#       lambda = (1 + tau)/tau,
#       alpha  = (1 + s)/2,
#       beta   = (S + (y[i] - m)^2 / (1 + tau))/2
#     )
#   }
#
#   # Start of the algorithm
#   for (n_sample in 2:(iter + 1)){
#     prev_sample <- samples[(n_sample - 1), , ]
#
#     # Update of each component
#     for (i in 1:n){
#       # Weights
#       simplified_sample <- as_tibble(prev_sample[-i, ]) %>%
#         dplyr::count(V1, V2)
#
#       q_weights <- c(
#         alpha * dstudent(y[i], nu = s, mu = m,
#                          sigma = sqrt((1 + tau) * S / s)),
#         simplified_sample$n * dnorm(y[i],
#                                     simplified_sample$V1,
#                                     sqrt(simplified_sample$V2))
#       )
#
#       # We normalize the vector
#       q_weights <- q_weights / sum(q_weights)
#
#       # New sample
#       idx_new <- sample(length(q_weights), size = 1, prob = q_weights)
#       if (idx_new == 1){
#         samples[n_sample, i, ] <- rnormigamma(
#           n      = 1,
#           mu     = (m + tau * y[i])/(1 + tau),
#           lambda = (1 + tau)/tau,
#           alpha  = (1 + s)/2,
#           beta   = (S + (y[i] - m)^2 / (1 + tau))/2
#         )
#       } else {
#         samples[n_sample, i, ] <- as.numeric(simplified_sample[idx_new - 1, 1:2])
#       }
#     }
#   }
#
#   # Sum 2 because of the initial values
#   return(samples[(warmup + 2):(iter + 1), , ])
# }
