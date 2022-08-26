# Stick Breaking construction of a DP (Sethuraman, 1994)
set.seed(219)


# Simulation ------------------------------------------------------------------------
## DP precision parameter (G0 = normal dist (0, 1))
M <- 100

## Tolerance for total probability
eps <- 1e-6

## Components of the distribution
ups_aux <- c(0)
probs <- c()
locations <- c()

while (sum(probs) < 1 - eps) {
  # Auxiliar upsilon ~ Beta(1, M)
  ups <- rbeta(1, shape1 = 1, shape2 = M)
  
  # Probability
  prob <- ups * prod(1 - ups_aux)
  probs <- c(probs, prob)
  ups_aux <- c(ups_aux, ups)
  
  # Location
  location <- rnorm(1, mean = 0, sd = 1)
  locations <- c(locations, location)
}

## We can check the total probability
sum(probs)

## We can check how the probabilities change with M
par(lwd = 2)
curve(dbeta(x, shape1 = 1, shape2 = 0.1), ylim = c(0, 7),
      ylab = 'Density', main = 'Comparison of Beta(1, M) distributions')
curve(dbeta(x, shape1 = 1, shape2 = 1), col = 'red', add = TRUE)
curve(dbeta(x, shape1 = 1, shape2 = 10), col = 'blue', add = TRUE)
curve(dbeta(x, shape1 = 1, shape2 = 20), col = 'purple', add = TRUE)
curve(dbeta(x, shape1 = 1, shape2 = 40), col = 'green', add = TRUE)      
legend(x = 'topright', legend = c('M = 0.1', 'M = 1', 'M = 10', 'M = 20', 'M = 40'),
       col = c('black', 'red', 'blue', 'purple', 'green'), lwd = 2)

## Cumulative distribution function
ord_locations <- sort(locations)
ord_probs <- probs[order(locations)]

cdf_G <- stepfun(ord_locations, c(0, cumsum(ord_probs)))
plot(cdf_G, col = 'red', lwd = 2, do.points = FALSE, ylim = c(0, 1),
     main = 'Sample from a Dirichlet Process', ylab = 'F(x)')
curve(pnorm, add = TRUE, lwd = 1.5)


# Multiple samples ------------------------------------------------------------------
rdp <- function(M, tol){
  ## Components of the distribution
  ups_aux <- c(0)
  probs <- c()
  locations <- c()
  
  while (sum(probs) < 1 - eps) {
    # Auxiliar eta ~ Beta(1, M)
    ups <- rbeta(1, shape1 = 1, shape2 = M)
    
    # Probability
    prob <- ups * prod(1 - ups_aux)
    probs <- c(probs, prob)
    ups_aux <- c(ups_aux, ups)
    
    # Location
    location <- rnorm(1, mean = 0, sd = 1)
    locations <- c(locations, location)
  }
  
  ord_locations <- sort(locations)
  ord_probs <- probs[order(locations)]
  return(list(locations = ord_locations, probs = ord_probs))
}

dp_cdf <- function(results){
  dp_cdf_plot <- stepfun(results$locations, c(0, cumsum(results$probs)))
  plot(dp_cdf_plot, add = TRUE, col = 'red', lwd = 1.5, do.points = FALSE)
}

## Plot of samples
### M = 1
dp_samples_1 <- replicate(30, rdp(1, eps))
curve(pnorm, lwd = 1.5, xlim = c(-3, 3))
invisible(apply(dp_samples_1, 2, dp_cdf))

### M = 20
dp_samples_20 <- replicate(30, rdp(20, eps))
curve(pnorm, lwd = 1.5, xlim = c(-3, 3))
invisible(apply(dp_samples_20, 2, dp_cdf))

### M = 500
dp_samples_500 <- replicate(15, rdp(500, eps))
curve(pnorm, lwd = 1.5, xlim = c(-3, 3))
invisible(apply(dp_samples_500, 2, dp_cdf))
