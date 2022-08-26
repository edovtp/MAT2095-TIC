library(tidyverse)


set.seed(219)

# Data simulation -------------------------------------------------------------------
## We consider k = 3 and n = 9
mu_1 <- 10; V_1 <- 1
mu_2 <- 2 ; V_2 <- 2
mu_3 <- 15; V_3 <- 0.5

## We simulate 10, 20 and 15 values from each component
y_1 <- rnorm(10, mu_1, sqrt(V_1))
y_2 <- rnorm(20, mu_2, sqrt(V_2))
y_3 <- rnorm(15, mu_3, sqrt(V_3))
y <- c(y_1, y_2, y_3)

## We observe our data
hist(y, col='turquoise', main='Histogram of simulated data',
     freq = FALSE, xlim = c(-2, 20), breaks=20)
# curve(dnorm(x, mean=mu_1, sd=sqrt(V_1)), add=TRUE, col='red', lwd=2)
# curve(dnorm(x, mean=mu_2, sd=sqrt(V_2)), add=TRUE, col='blue', lwd=2)
# curve(dnorm(x, mean=mu_3, sd=sqrt(V_3)), add=TRUE, col='green', lwd=2)



