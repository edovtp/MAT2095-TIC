# Dirichlet process simulation ------------------------------------------------------
tic_rdp <- function(M, G0, G0_params, tol = 1e-6) {
  
  # TODO: Multivariate G0
  
  # M        : precision parameter
  # G0       : sampling function of the desired centering measure (list)
  # G0_params: parameters for G0
  
  # We add the number of samples
  G0_params <- c(G0_params, n = 1)
  
  ## Components of the distribution
  ups_aux <- c(0)
  probs <- c()
  locations <- c()
  
  # We get probabilities that adds to a certain point close to 1
  while (sum(probs) < 1 - tol) {
    # Auxiliar upsilon ~ Beta(1, M)
    ups <- rbeta(1, shape1 = 1, shape2 = M)
    
    # Probability (or weight)
    prob <- ups * prod(1 - ups_aux)
    probs <- c(probs, prob)
    ups_aux <- c(ups_aux, ups)
    
    # Location
    location <- do.call(G0, args = G0_params)
    locations <- c(locations, location)
  }
  
  # Return ordered values of the (discrete) distribution
  ord_locations <- sort(locations)
  ord_probs <- probs[order(locations)]
  return(list(locations = ord_locations, probs = ord_probs))
}

# Data simulation from a Dirichlet Process ------------------------------------------
tic_rdp_data <- function(n, M, G0, G0_params) {
  # n        : length of the sample
  # M        : precision parameter
  # G0       : sampling function of the desired centering measure
  # G0_params: parameters for G0
  
  # We add the number of samples
  G0_params <- c(G0_params, n = 1)
  
  # Sample vector
  dp_sample <- vector(mode = "list", length = n)
  
  # Frequency counter
  counter <- dict()
  
  for (i in 1:n) {
    # All values
    candidate <- do.call(G0, G0_params)
    all_values <- counter$keys()
    k <- length(all_values)
    all_values[[k + 1]] <- candidate
    
    # Probabilities
    norm_term <- 1 / (M + i - 1)
    old_freq <- unlist(counter$values())
    probs <- c(old_freq * norm_term, M * norm_term)
    
    # Select value
    value_index <- sample(k + 1, size = 1, prob = probs)
    new_value <- all_values[[value_index]]
    dp_sample[[i]] <- new_value
    
    # Update frequencies
    freq <- counter$get(new_value, 0)
    counter$set(new_value, freq + 1)
  }
  
  dp_sample <- simplify2array(dp_sample)
  
  if (length(new_value) != 1) {
    dp_sample <- t(dp_sample)
  }
  
  return(dp_sample)
}
