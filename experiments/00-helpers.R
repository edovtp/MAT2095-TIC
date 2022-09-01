library(collections)


# Dirichlet process simulation ------------------------------------------------------
rdp <- function(M, G0, G0_params, tol=1e-6){
  # M        : precision parameter
  # G0       : sampling function of the desired base measure
  # G0_params: parameters for G0
  
  # We add the number of samples
  G0_params <- c(G0_params, n = 1)
  
  ## Components of the distribution
  ups_aux <- c(0)
  probs <- c()
  locations <- c()
  
  while (sum(probs) < 1 - tol) {
    # Auxiliar eta ~ Beta(1, M)
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
rdp_data <- function(n, M, G0, G0_params){
  # n        : length of the sample 
  # M        : precision parameter
  # G0       : sampling function of the desired base measure
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
