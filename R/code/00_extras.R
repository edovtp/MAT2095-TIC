# Normal-inverse-gamma simulation function ------------------------------------------
rnormigamma <- function(n, mu, lambda, alpha, beta) {
  # We save the values
  samples_s2 <- vector(mode = "numeric", length = n)
  samples_mu <- vector(mode = "numeric", length = n)
  
  # We take the samples
  for (i in 1:n) {
    inv_sigma2 <- rgamma(1, shape = alpha, rate = beta)
    sigma2_sampled <- 1 / inv_sigma2
    mu_sampled <- rnorm(1, mean = mu, sd = sqrt(sigma2_sampled / lambda))
    
    samples_s2[i] <- sigma2_sampled
    samples_mu[i] <- mu_sampled
  }
  
  if (n == 1) {
    samples <- c(samples_mu, samples_s2)
  } else {
    samples <- cbind(samples_mu, samples_s2)
  }
  return(samples)
}

# Non-standardize t-distribution ----------------------------------------------------
## Credits: Richard McElreath, rethinking package
dstudent <- function(x, nu = 2, mu = 0, sigma = 1, log = FALSE) {
  y <- lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(pi * nu) - log(sigma) +
    (-(nu + 1) / 2) * log(1 + (1 / nu) * ((x - mu) / sigma)^2)
  if (log == FALSE) y <- exp(y)
  
  return(y)
}

rstudent <- function(n, nu=2, mu=0, sigma=1) {
  y <- rt(n, df=nu)
  y <- y*sigma + mu
  return(y)
}
