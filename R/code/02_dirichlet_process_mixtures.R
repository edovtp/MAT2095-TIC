source('code/00_extras.R')
source('code/01_dirichlet_process.R')


# Data simulation from a DPM with Normal model for the data --------------------------
tic_rdpm_data <- function(n, M, m, tau, s, S){
  # TODO: Use any model for the data
  # TODO: Use any centering measure
  
  # Parameter simulation
  pi_vector <- tic_rdp_data(
    n, M, rnormigamma,
    G0_params = list(
      mu = m, lambda = 1 / tau,
      alpha = s / 2, beta = 2 / S
    )
  )
  
  # Number of distinct components
  k <- length(unique(pi_vector[, 1]))
  
  # Data
  y <- apply(
    pi_vector, 1, function(params) rnorm(1, mean = params[1], sd = sqrt(params[2]))
  )
  
  # List with all the values
  return(list(y = y, params = pi_vector, k = k))
}
