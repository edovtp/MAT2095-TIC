# EW (1995) - Algorithm 1 -----------------------------------------------------------
ew_algorithm1 <- function(y, prior_par, iter, warmup = floor(iter/2)){
  # prior_par should contain: alpha (precision parameter), m, tau, s and S
  list2env(prior_par, envir = environment())
  
  n <- length(y)
  seq_id  <- seq.int(1, n)
  
  # Array in which we'll save the samples (plus one for the initial values)
  samples <- array(numeric(n * (iter + 1) * 2), dim = c(iter + 1, n, 2))
  
  # Initial values
  for (i in 1:n){
    samples[1, i, ] <- rnormigamma(
      n      = 1,
      mu     = (m + tau * y[i])/(1 + tau),
      lambda = (1 + tau)/tau,
      alpha  = (1 + s)/2,
      beta   = (S + (y[i] - m)^2 / (1 + tau))/2
    )
  }
  
  # Start of the algorithm
  for (n_sample in 2:(iter + 1)){
    prev_sample <- samples[(n_sample - 1), , ]
    
    # Update of each component
    for (i in 1:n){
      # Weights
      q_weights <- vector(mode = 'numeric', length = n)
      q_weights[i] <- alpha * dstudent(y[i], nu = s, mu = m, 
                                       sigma = sqrt((1 + tau) * S / s))
      q_weights[-i] <- dnorm(y[i], prev_sample[-i, 1], sqrt(prev_sample[-i, 2]))
      
      # We normalize the vector
      q_weights <- q_weights / sum(q_weights)
      
      # New sample
      idx_new <- sample(n, size = 1, prob = q_weights)
      if (idx_new == i){
        samples[n_sample, i, ] <- rnormigamma(
          n      = 1,
          mu     = (m + tau * y[i])/(1 + tau),
          lambda = (1 + tau)/tau,
          alpha  = (1 + s)/2,
          beta   = (S + (y[i] - m)^2 / (1 + tau))/2
        )
      } else {
        samples[n_sample, i, ] <- prev_sample[idx_new, ]
      }
    }
  }
  
  # Sum 2 because of the initial values
  return(samples[(warmup + 2):(iter + 1), , ])
}

# EW (1995) - Algorithm 2 -----------------------------------------------------------
ew_algorithm2 <- function(data, prior_par, iter, warmup = floor(iter/2)){
  1
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