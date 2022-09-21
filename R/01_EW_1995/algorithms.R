library(here)
source(here('code', '00_extras.R'))


# EW (1995) - Algorithm 1 -----------------------------------------------------------
ew_algorithm1 <- function(data, prior_par, n_samples, warmup = floor(n_samples/2)){
  # data should contain: params and y
  list2env(data, envir = environment())
  
  # prior_par should contain: M, m, tau, s and S
  list2env(prior_par, envir = environment())
  
  # Array in which we'll save the samples
  samples <- array(numeric(n * (n_samples + 1) * 2), dim = c(n_samples + 1, n, 2))
  seq_id  <- seq.int(1, n)
  
  # Initial values
  for (i in 1:n){
    samples[1, i, ] <- rnormigamma(
      n      = 1,
      mu     = (m + tau * y[i])/(1 + tau),
      lambda = (1 + tau)/tau,
      alpha  = (1 + s)/2,
      beta   = 2/(S + (y[i] - m)^2 / (1 + tau))
    )
  }
  
  # Start of the algorithm
  for (n_sample in 2:(n_samples + 1)){
    prev_sample <- samples[(n_sample - 1), , ]
    
    # Update of each component
    for (i in 1:n){
      # Weights
      q_weights    <- vector(mode = 'numeric', length = n)
      q_weights[i] <- alpha * dstudent(y[i], nu = s, mu = m,
                                       sigma = sqrt((1 + tau) * S / s))
      
      for (k in seq_id[-i]){
        prev_sample_k <- prev_sample[k, ]
        q_weights[k] <- dnorm(
          y[i], mean = prev_sample_k[1], sd = sqrt(prev_sample_k[2])
        )
      }
      
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
          beta   = 2/(S + (y[i] - m)^2 / (1 + tau))
        )
      } else {
        samples[n_sample, i, ] <- prev_sample[idx_new, ]
      }
    }
  }
  
  return(samples[warmup:n_samples, , ])
}

# EW (1995) - Algorithm 2 -----------------------------------------------------------
