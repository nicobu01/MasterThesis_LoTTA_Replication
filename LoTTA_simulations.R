# =======================================================================================================================
#
#    MODIFIED REPLICATION CODE
#
#    Original Work:
#    Paper:          "Bayesian Regression Discontinuity Design with Unknown Cutoff"
#    Authors:        Kowalska, van de Wiel, and van der Pas (2025)
#    Source:         https://github.com/JuliaMKowalska/RDD_unknown_cutoff
#
#    Purpose:        Master Thesis: "Prior Sensitivity and Robustness of Bayesian Methods 
#                    in Regression Discontinuity Designs with Unknown Cutoff"
#
#    Description:    This script performs the primary Bayesian estimation using the Local 
#                    Trimmed Taylor Approximation (LoTTA). It extends the original code 
#                    to support configurable prior specifications (informative, weak, diffuse),
#                    various sample sizes (N=200, 300, 500), and specific misspecification 
#                    scenarios. It uses JAGS for MCMC sampling and computes posterior 
#                    diagnostics and performance metrics.
# =======================================================================================================================

SIM_FUNCTION <- "A"              # Options: "A", "B", "C", "lee", "ludwig"
TREATMENT_PROB <- "0.55"          # Options: "0.55" or "0.3"
PRIOR_TYPE <- "baseline"              # Options: "compliance" or "cutoff" or "baseline"
PRIOR_STRENGTH <- "baseline"              # Options: "diffuse", "weak", or "strong", "baseline"
MISSPEC_LEVEL <- "baseline"             # 0 , 2, 4 or baseline
SAMPLE_SIZE <- "500"            # Options: "300", "200" or "500" 

# Prior bounds
JLB <- 0.2                     # Lower bound for compliance prior
CLB <- -0.8                    # Lower bound for cutoff prior
CUB <- 0.2                     # Upper bound for cutoff prior

# Simulation batches
N_SIMS_BATCH1 <- 1:500          # First batch
N_SIMS_BATCH2 <- 501:1000       # Second batch (change to 501:502 for testing)

################################################################################
#                     END OF CONFIGURATION SECTION
################################################################################

# Generate output names automatically
OUTPUT_SUFFIX <- paste0(PRIOR_TYPE , JLB, "_mis", MISSPEC_LEVEL, "_", SAMPLE_SIZE, "_", 
                        SIM_FUNCTION, "_", PRIOR_STRENGTH, "_", gsub("\\.", "", TREATMENT_PROB))
OUTPUT_DIR <- paste0("~/plots_", OUTPUT_SUFFIX)
PERFORMANCE_FILE <- paste0("performance_", OUTPUT_SUFFIX, ".csv")
CONVERGENCE_FILE <- paste0("convergence_", OUTPUT_SUFFIX, ".csv")

# Get true parameter values
TRUE_J <- ifelse(TREATMENT_PROB == "0.55", 0.55, 0.30)
TRUE_C <- 0
TRUE_TAU <- switch(SIM_FUNCTION,
                   "A" = ifelse(TREATMENT_PROB == "0.55", 0.31, 0.57),
                   "B" = ifelse(TREATMENT_PROB == "0.55", -0.36, -0.67),
                   "C" = 0,
                   "lee" = ifelse(TREATMENT_PROB == "0.55", 0.07, 0.13),
                   "ludwig" = ifelse(TREATMENT_PROB == "0.55", -6.24, -11.43))

# Print configuration
cat("\n===============================================================================\n")
cat("SIMULATION CONFIGURATION\n")
cat("===============================================================================\n")
cat("Function:           ", SIM_FUNCTION, "\n")
cat("Treatment prob:     ", TREATMENT_PROB, " (j = ", TRUE_J, ")\n")
cat("Prior type:         ", PRIOR_TYPE, "\n")
cat("Misspecification:   ", MISSPEC_LEVEL, "\n")
cat("jlb:                ", JLB, "\n")
cat("Cutoff bounds:      [", CLB, ", ", CUB, "]\n")
cat("True tau:           ", TRUE_TAU, "\n")
cat("Output directory:   ", OUTPUT_DIR, "\n")
cat("===============================================================================\n\n")

################################################################################
#                           MAIN CODE
################################################################################

library(rjags)
library(runjags)
library(coda)
library(ggplot2)
library(tidyverse)
library(bayestestR)
library(dplyr)
library(parallel)

# Helper Functions
logit <- function(x) { log(x/(1-x)) }
invlogit <- function(x) { 1/(1+exp(-x)) }

# Treatment Sampling Functions
sample_prob55 <- function(X){
    P <- rep(0, length(X))
    P[X>=0.] <- invlogit((8.5*X[X>=0.]-1.5))/10.5+0.65-0.0007072
    P[X<0.] <- (X[X<0.]+1)^4/15+0.05 
    T <- rep(0, length(X))
    for(j in 1:length(X)){
      T[j] <- sample(c(1,0), 1, prob=c(P[j], 1-P[j]))
    }
    return(T)
}

sample_prob30 <- function(X){
  P <- rep(0, length(X))
  P[X>=0.] <- invlogit((8.5*X[X>=0.]-1.5))/10.5+0.4-0.0007072
  P[X<0.] <- (X[X<0.]+1)^4/15 +0.05 
  T <- rep(0, length(X))
  for(j in 1:length(X)){
    T[j] <- sample(c(1,0), 1, prob=c(P[j], 1-P[j]))
  }
  return(T)
}

# Outcome Functions
funA_sample <- function(X){
  Y <- X
  X2 <- X[X>=0]
  X1 <- X[X<0]
  Y[X<0] <- (1.8*X1^3+2.*X1^2)+0.05
  Y[X>=0] <- 0.05*X2-0.1*X2^2+0.22
  Y <- Y+rnorm(length(X), 0, 0.1)
  return(Y)
}

ludwig_sample <- function(X){
  X1 <- X[X<0]
  X2 <- X[X>=0]
  Y1 <- 3.71+ 2.3*X1+ 3.28*X1^2 +1.45*X1^3 + 0.23*X1^4 + 0.03*X1^5
  Y2 <- 0.26+ 18.49*X2 - 54.81*X2^2+74.3*X2^3 -45.02*X2^4 + 9.83*X2^5
  Y <- append(Y1, Y2)+rnorm(length(X), 0, 0.1295)
  return(Y) 
}

lee_sample <- function(X){
  X1 <- X[X<0]
  X2 <- X[X>=0]
  Y1 <- 0.48+ 1.27*X1+ 7.18*X1^2 +20.21*X1^3 + 21.54*X1^4 + 7.33*X1^5
  Y2 <- 0.52+ 0.84*X2 - 3.00*X2^2+7.99*X2^3 -9.01*X2^4 + 3.56*X2^5
  Y <- append(Y1, Y2)+rnorm(length(X), 0, 0.1295)
  return(Y)
}

funB_sample <- function(X){
  Y <- X
  X2 <- X[X>=0]
  X1 <- X[X<0]
  Y[X<0] <- 1/(1+exp(-2*X1))-0.5+0.4
  Y[X>=0] <- (log(X2*2.+1)-0.15*X2^2)*0.6-0.20+0.4
  Y <- Y+rnorm(length(X), 0, 0.1)
  return(Y)
}

funC_sample <- function(X){
  Y <- 0.48- 2.7*X+ 1.18*X^2 +1.21*X^3 + 2.54*X^4 - 3*X^5-1.9*X^6-5/(1+exp(-10*(X+1)))-10+sin(5*X-2)
  Y <- Y/10+rnorm(length(X), 0, 0.1)
  return(Y)
}

# Generate JAGS Models
cat("model
    {
  
    for ( i in 1:N ) {
      
      t[i]~dbern(paramt[i])
      paramt[i] <- ifelse(x[i]<c,al,al+j) 
    }
    
    
    al~dunif(0,1-j)
    j~dunif(jlb,1)
    c~dunif(clb,cub)
    
    
    }", file="cutoff_initial_CONT.txt") 

cat("model
    {
  
    for ( i in 1:N ) {
      y[i]~dnorm(param[i],Tau[i])
      t[i]~dbern(paramt[i])
      param[i] <-ifelse(x[i]<c,a1l*(xc[i])+a0l+ilogit(100*(kl-x[i]))*((a3l)*(xc[i])^3+(a2l)*(xc[i])^2),a1r*(xc[i])+a0r+ilogit(100*(x[i]-kr))*((a3r)*(xc[i])^3+(a2r)*(xc[i])^2))
      Tau[i]<-ifelse(x[i]<c,tau1l+tau2l*ilogit(100*(kl-x[i])),tau1r+tau2r*ilogit(100*(x[i]-kr)))
      paramt[i] <- ifelse(x[i]<c,ifelse(x[i]>=c-k1t,a1lt*x[i]+b1lt,a2lt*x[i]+b2lt),ifelse(x[i]<=c+k2t,a1rt*x[i]+b1rt,a2rt*x[i]+b2rt)) 
    }
    pr=0.0001
    MAX=max(x)
    MIN=min(x)
    
    
    ### Define the priors
    c~dunif(clb,cub)
    xc=x-c
    j~dunif(jlb,1)
    k1t~dunif(lb,c-ublt)
    k2t~dunif(lb,ubrt-c)
    a2lt~dunif(0,(1-j)/(c-k1t-MIN))
    b2lt~dunif(-a2lt*MIN,1-j-a2lt*(c-k1t))
    a1lt~dunif(0,(1-j-a2lt*(c-k1t)-b2lt)/k1t)
    b1lt=(c-k1t)*(a2lt-a1lt)+b2lt
    a1rt~dunif(0,(1-a1lt*c-b1lt-j)/k2t)
    b1rt=a1lt*c+b1lt+j-a1rt*c
    a2rt~dunif(0,(1-b1rt-(c+k2t)*a1rt)/(MAX-c-k2t))
    b2rt=(c+k2t)*(a1rt-a2rt)+b1rt
    
    tau1l~dgamma(0.01,0.01)
    tau2pl~dbeta(1,1)
    tau2l=-tau2pl*tau1l
    klt~dbeta(1,1)
    kl=klt*(c-lb-ubl)+ubl
    a0l~dnorm(0,pr)
    a1l~dnorm(0,pr)
    a2l~dnorm(0,pr*(c-kl))
    a3l~dnorm(0,pr*(c-kl))
    
    tau1r~dgamma(0.01,0.01)
    tau2pr~dbeta(1,1)
    tau2r=-tau2pr*tau1r
    krt~dbeta(1,1)
    kr=krt*(ubr-c-lb)+c+lb
    a0r~dnorm(0,pr)
    a1r~dnorm(0,pr)
    a2r~dnorm(0,pr*(kr-c))
    a3r~dnorm(0,pr*(kr-c))
    
    
    
    eff=(a0r-a0l)/j
    
    }", file="LoTTA_CONT_CONT.txt") 


Initial_CONT_CONT <- function(X, T, Y, C_start, lb, ubr, ubl, jlb, s){
  set.seed(s)
  pr <- sd(Y)
  MIN <- min(X)
  MAX <- max(X)
  
  c <- sample(C_start, 1)
  tl <- mean(T[X<c])
  tr <- mean(T[X>=c])
  yl <- mean(Y[X<c])
  yr <- mean(Y[X>=c])
  
  klt <- rbeta(1, 1, 1)
  kl <- klt*(c-lb-ubl)+ubl
  krt <- rbeta(1, 1, 1)
  kr <- krt*(ubr-c-lb)+c+lb
  kld <- kl-c
  krd <- kr-c
  a0l <- rnorm(1, yl, pr)
  a1l <- rnorm(1, 0, pr)
  a2l <- rnorm(1, 0, pr)
  a3l <- 0
  a0r <- rnorm(1, yr, pr)
  a1r <- rnorm(1, 0, pr)
  a2r <- rnorm(1, 0, pr)
  a3r <- 0
  tau1r <- rchisq(1, 7)
  tau2pr <- rbeta(1, 1, 1)
  tau1l <- rchisq(1, 7)
  tau2pl <- rbeta(1, 1, 1)
  j <- max(tr-tl, jlb+0.0001)
  k1t <- runif(1, lb+0.1*max(c-ubl-lb, 0), max(c-ubl-0.1*(c-ubl-lb), lb+0.1*max(c-ubl-lb, 0)+0.0001))
  k2t <- runif(1, lb+0.1*max(ubr-c-lb, 0), max(ubr-c-0.1*(ubr-c-lb), lb+0.1*max(ubr-c-lb, 0)+0.0001))
  a2lt <- 0
  b2lt <- tl
  a1lt <- 0
  b1lt <- (c-k1t)*(a2lt-a1lt)+b2lt
  a1rt <- 0
  b1rt <- a1lt*c+b1lt+j-a1rt*c
  a2rt <- 0
  b2rt <- (c+k2t)*(a1rt-a2rt)+b1rt
  
  return(list(c=c, j=j, a0l=a0l, a1l=a1l, a2l=a2l, a3l=a3l, a0r=a0r, a1r=a1r, a2r=a2r, a3r=a3r, 
              tau1r=tau1r, tau2pr=tau2pr, tau1l=tau1l, tau2pl=tau2pl, klt=klt, krt=krt, 
              k1t=k1t, k2t=k2t, a1lt=a1lt, a2lt=a2lt, b2lt=b2lt, a1rt=a1rt, a2rt=a2rt, 
              .RNG.seed=s)) 
}

bounds <- function(X, ns=25){
  Xu <- sort(unique(X))
  ql <- ns/length(X)
  q <- as.numeric(quantile(X, c(ql, 1-ql)))
  ubr <- q[2]
  ubl <- q[1]
  N <- length(Xu)
  diff <- Xu[2:N]-Xu[1:(N-1)]
  diff1 <- diff[1:(N-2)]
  diff2 <- diff[2:(N-1)]
  Diff <- ifelse(diff1>diff2, diff1, diff2)
  qd <- quantile(Diff, c(0.75))
  lb <- qd[[1]]
  return(list(ubl=ubl, ubr=ubr, lb=lb))
}

performance_sample_FUZZY <- function(post_sample, name, jump){
  c <- 0
  functions  <- list("A"=funA_sample, "B"=funB_sample, "C"=funC_sample, "lee"=lee_sample, "ludwig"=ludwig_sample)
  effects    <- list("A"=0.17, "B"=-0.2, "C"=0, "lee"=0.04, "ludwig"=-3.45)
  compliance <- list("0.55"=0.55, "0.3"=0.30)

  j      <- compliance[[jump]]
  tr_eff <- effects[[name]] / compliance[[jump]]

  Samples <- combine.mcmc(post_sample)
  C   <- as.numeric(Samples[,1])
  J   <- as.numeric(Samples[,2])
  Eff <- as.numeric(Samples[,7])

  # Treatment effect 

  qeff   <- as.numeric(quantile(Eff, c(0.025, 0.975)))
  meff   <- median(Eff)
  mapeff <- as.numeric(map_estimate(Eff))
  hdieff <- as.numeric(ci(Eff, method = "HDI"))[2:3]

  coveff    <- ifelse(tr_eff <= qeff[2] & tr_eff >= qeff[1], 1, 0)
  hdicoveff <- ifelse(tr_eff <= hdieff[2] & tr_eff >= hdieff[1], 1, 0)

  if(tr_eff < 0){
    signeff    <- ifelse(0 > qeff[2], 1, 0)
    hdisigneff <- ifelse(0 > hdieff[2], 1, 0)
  } else {
    signeff    <- ifelse(0 < qeff[1], 1, 0)
    hdisigneff <- ifelse(0 < hdieff[1], 1, 0)
  }

  merreff      <- abs(tr_eff - meff)
  maperreff    <- abs(tr_eff - mapeff)
  qcieff_len   <- as.numeric(qeff[2] - qeff[1])
  hdicieff_len <- as.numeric(hdieff[2] - hdieff[1])

  # Cutoff c 

  qc    <- as.numeric(quantile(C, c(0.025, 0.975)))
  mc    <- median(C)
  mapc  <- as.numeric(map_estimate(C))
  hdic  <- as.numeric(ci(C, method = "HDI"))[2:3]

  covc_sym <- ifelse(c <= qc[2] & c >= qc[1], 1, 0)
  covc_hdi <- ifelse(c <= hdic[2] & c >= hdic[1], 1, 0)

  cerr_med <- abs(c - mc)
  cerr_map <- abs(c - mapc)

  c_bias_med <- -(c - mc)
  c_bias_map <- -(c - mapc)

  c_ci_len_sym <- as.numeric(qc[2] - qc[1])
  c_ci_len_hdi <- as.numeric(hdic[2] - hdic[1])

 # Compliance j

  qj    <- as.numeric(quantile(J, c(0.025, 0.975)))
  mj    <- median(J)
  mapj  <- as.numeric(map_estimate(J))
  hdij  <- as.numeric(ci(J, method = "HDI"))[2:3]

  covj_sym <- ifelse(j <= qj[2] & j >= qj[1], 1, 0)
  covj_hdi <- ifelse(j <= hdij[2] & j >= hdij[1], 1, 0)

  jerr_med <- abs(j - mj)
  jerr_map <- abs(j - mapj)

  j_bias_med <- -(j - mj)
  j_bias_map <- -(j - mapj)

  j_ci_len_sym <- as.numeric(qj[2] - qj[1])
  j_ci_len_hdi <- as.numeric(hdij[2] - hdij[1])

  return(list(
    abs_err_med    = merreff,
    abs_err_map    = maperreff,
    ci_length_sym  = qcieff_len,
    ci_length_hdi  = hdicieff_len,
    bias_med       = -(tr_eff - meff),
    bias_map       = -(tr_eff - mapeff),
    cov_med        = coveff,
    cov_hdi        = hdicoveff,
    sign_med       = signeff,
    sign_hdi       = hdisigneff,
    c_abs_err_med   = cerr_med,
    c_abs_err_map   = cerr_map,
    c_ci_length_sym = c_ci_len_sym,
    c_ci_length_hdi = c_ci_len_hdi,
    c_bias_med      = c_bias_med,
    c_bias_map      = c_bias_map,
    c_cov_sym       = covc_sym,
    c_cov_hdi       = covc_hdi,
    j_abs_err_med   = jerr_med,
    j_abs_err_map   = jerr_map,
    j_ci_length_sym = j_ci_len_sym,
    j_ci_length_hdi = j_ci_len_hdi,
    j_bias_med      = j_bias_med,
    j_bias_map      = j_bias_map,
    j_cov_sym       = covj_sym,
    j_cov_hdi       = covj_hdi
  ))
}

# Simulation Function

simulation_FUZZY <- function(i, name, jump){                      
  functions <- list("A"=funA_sample, "B"=funB_sample, "C"=funC_sample, "lee"=lee_sample, "ludwig"=ludwig_sample)
  probs <- list("0.55"=sample_prob55, "0.3"=sample_prob30)
  fun <- functions[[name]]
  prob <- probs[[jump]]
  print(i)
  set.seed(i)
  X <- sort(2*rbeta(1000, 2, 4)-1)
  Y <- fun(X)
  T <- prob(X)
  (b_f1 <- bounds(X, 25))
  b_f1t <- bounds(X, 25)
  ubr <- b_f1$ubr
  ubl <- b_f1$ubl
  ubrt <- b_f1t$ubr
  ublt <- b_f1t$ubl
  lb <- b_f1$lb
  (b_s <- bounds(X, 50))
  ubrs <- b_s$ubr
  ubls <- b_s$ubl
  nc <- 1
  dat1T <- list(N=length(X), x=X, t=T, jlb=JLB, clb=CLB, cub=CUB)
  param_full <- c('c', 'j', 'kl', 'kr', 'klt', 'krt', 'eff', 'a0l', 'a1l', 'a2l', 'a3l', 'a0r', 'a1r', 'a2r', 'a3r', 'b1lt', 'a1lt', 'a2lt', 'b2lt', 'b1rt', 'a1rt', 'a2rt', 'b2rt', 'k1t', 'k2t', 'tau1r', 'tau2r', 'tau1l', 'tau2l')
  param_c <- c('c')
  initc1 <- list(c=-0.3, al=0.5, j=0.3, .RNG.seed=1, .RNG.name="base::Mersenne-Twister")
  dat1_c <- run.jags('cutoff_initial_CONT.txt', inits = list(initc1), data=dat1T, monitor=param_c, burnin = 900, sample=2000, adapt = 100, n.chains = 1, method = 'simple')
  C_start <- as.numeric(combine.mcmc(dat1_c$mcmc))
  init1 <- Initial_CONT_CONT(X, T, Y, C_start, lb, ubrs, ubls, JLB, 1)
  init2 <- Initial_CONT_CONT(X, T, Y, C_start, lb, ubrs, ubls, JLB, 2)
  init3 <- Initial_CONT_CONT(X, T, Y, C_start, lb, ubrs, ubls, JLB, 3)
  init4 <- Initial_CONT_CONT(X, T, Y, C_start, lb, ubrs, ubls, JLB, 4)
  dat <- list(N=length(X), x=X, t=T, y=Y, ubr=ubr, ubl=ubl, ubrt=ubrt, ublt=ublt, lb=lb, nc=nc, jlb=JLB, clb=CLB, cub=CUB, seed=i)
  posterior <- run.jags('LoTTA_CONT_CONT.txt', data=dat, inits = list(init1, init2, init3, init4), monitor=param_full, burnin = 5000, sample=25000, adapt = 1000, n.chains = 4, method = 'parallel')
  return(posterior)
}

# Simulations

n_cores <- detectCores()

system.time(res1 <- mclapply(N_SIMS_BATCH1, simulation_FUZZY, SIM_FUNCTION, TREATMENT_PROB, mc.cores = n_cores))

system.time(res2 <- mclapply(N_SIMS_BATCH2, simulation_FUZZY, SIM_FUNCTION, TREATMENT_PROB, mc.cores = n_cores))

# Performance Calculation

is_valid_result <- function(x) {
  inherits(x, "runjags") && !inherits(x, "try-error")
}

res1_valid <- Filter(is_valid_result, res1)
res2_valid <- Filter(is_valid_result, res2)

all_results <- c(res1_valid, res2_valid)
cat("Valid simulations:", length(all_results), "out of", length(N_SIMS_BATCH1) + length(N_SIMS_BATCH2), "\n\n")

Results1 <- mclapply(res1_valid, performance_sample_FUZZY, SIM_FUNCTION, TREATMENT_PROB, mc.cores = n_cores)

Results2 <- mclapply(res2_valid, performance_sample_FUZZY, SIM_FUNCTION, TREATMENT_PROB, mc.cores = n_cores)

Results1 <- do.call(rbind.data.frame, Results1)
Results2 <- do.call(rbind.data.frame, Results2)

Results <- rbind(Results1, Results2) 

write.csv(Results, PERFORMANCE_FILE, row.names = FALSE)

# Convergence Diagnostics

gel <- mclapply(all_results, gelman.diag, mc.cores = n_cores)

Convergence <- do.call(rbind.data.frame, lapply(1:length(all_results), function(i) {
  gd <- gel[[i]]
  summ <- summary(all_results[[i]])
  psrf_vals <- gd$psrf[, "Point est."]
  c_sseff <- summ["c", "SSeff"]
  j_sseff <- summ["j", "SSeff"]
  eff_sseff <- summ["eff", "SSeff"]
  df <- as.data.frame(t(psrf_vals))
  df$c_sseff <- c_sseff
  df$j_sseff <- j_sseff
  df$eff_sseff <- eff_sseff
  df$mpsrf <- gd$mpsrf
  return(df)
}))

Convergence$sim <- 1:length(all_results)

write.csv(Convergence, CONVERGENCE_FILE, row.names = FALSE)

### Plots ###

dir.create(OUTPUT_DIR, showWarnings = FALSE)

all_c <- c()
all_j <- c()
all_eff <- c()

samples_list <- mclapply(1:length(all_results), function(i) {
  mcmc_combined <- do.call(rbind, all_results[[i]]$mcmc)
  list(
    c = mcmc_combined[, "c"],
    j = mcmc_combined[, "j"],
    eff = mcmc_combined[, "eff"]
  )
}, mc.cores = n_cores)

all_c <- unlist(lapply(samples_list, function(x) x$c))
all_j <- unlist(lapply(samples_list, function(x) x$j))
all_eff <- unlist(lapply(samples_list, function(x) x$eff))

# Posterior Plots

pdf(paste0(OUTPUT_DIR, "/posterior_densities_combined.pdf"), width=12, height=4, family="serif")
par(mfrow=c(1,3), family="serif")
plot(density(all_c), main="Cutoff", xlab=expression(italic(c)), ylab="Density",
     lwd=2, col="#0072B2", cex.main=1.2, cex.lab=1.0, cex.axis=0.9)
polygon(density(all_c), col=adjustcolor("#0072B2", alpha.f=0.3), border=NA)
abline(v=TRUE_C, col="gray", lty=2, lwd=2)

plot(density(all_j), main="Compliance Jump", xlab=expression(italic(j)), ylab="Density",
     lwd=2, col="#009E73", cex.main=1.2, cex.lab=1.0, cex.axis=0.9)
polygon(density(all_j), col=adjustcolor("#009E73", alpha.f=0.3), border=NA)
abline(v=TRUE_J, col="gray", lty=2, lwd=2)

plot(density(all_eff), main="Treatment Effect", xlab=expression(tau), ylab="Density",
     lwd=2, col="#D55E00", cex.main=1.2, cex.lab=1.0, cex.axis=0.9)
polygon(density(all_eff), col=adjustcolor("#D55E00", alpha.f=0.3), border=NA)
abline(v=TRUE_TAU, col="gray", lty=2, lwd=2)
dev.off()

# Convergence diagnostics
pdf(paste0(OUTPUT_DIR, "/traceplot_sim1.pdf"), width=10, height=8, family="serif")
par(mfrow=c(3,1), family="serif")
traceplot(all_results[[1]]$mcmc[, "c"], main="Trace Plot: Cutoff", 
          ylab=expression(italic(c)), cex.main=1.2, cex.lab=1.0)
traceplot(all_results[[1]]$mcmc[, "j"], main="Trace Plot: Compliance Jump", 
          ylab=expression(italic(j)), cex.main=1.2, cex.lab=1.0)
traceplot(all_results[[1]]$mcmc[, "eff"], main="Trace Plot: Treatment Effect", 
          ylab=expression(tau), cex.main=1.2, cex.lab=1.0)
dev.off()

pdf(paste0(OUTPUT_DIR, "/densityplot_sim1.pdf"), width=10, height=8, family="serif")
par(mfrow=c(3,1), family="serif")
densplot(all_results[[1]]$mcmc[, "c"], main="Density Plot: Cutoff", 
         xlab=expression(italic(c)), cex.main=1.2, cex.lab=1.0)
densplot(all_results[[1]]$mcmc[, "j"], main="Density Plot: Compliance Jump", 
         xlab=expression(italic(j)), cex.main=1.2, cex.lab=1.0)
densplot(all_results[[1]]$mcmc[, "eff"], main="Density Plot: Treatment Effect", 
         xlab=expression(tau), cex.main=1.2, cex.lab=1.0)
dev.off()

pdf(paste0(OUTPUT_DIR, "/autocorr_sim1.pdf"), width=10, height=8, family="serif")
par(mfrow=c(3,1), family="serif")
autocorr.plot(all_results[[1]]$mcmc[, "c"], main="Cutoff", 
              ylab=expression(paste("ACF of ", italic(c))), cex.main=1.2, cex.lab=1.0)
autocorr.plot(all_results[[1]]$mcmc[, "j"], main="Compliance Jump", 
              ylab=expression(paste("ACF of ", italic(j))), cex.main=1.2, cex.lab=1.0)
autocorr.plot(all_results[[1]]$mcmc[, "eff"], main="Treatment Effect", 
              ylab=expression(paste("ACF of ", tau)), cex.main=1.2, cex.lab=1.0)
dev.off()

pdf(paste0(OUTPUT_DIR, "/gelman_rubin_across_sims.pdf"), width=12, height=4, family="serif")
par(mfrow=c(1,3), family="serif")
plot(1:length(all_results), Convergence$c, type="l", 
     main="Cutoff", xlab="Simulation", ylab=expression(hat(R)),
     cex.main=1.2, cex.lab=1.0, cex.axis=0.9)
abline(h=1.1, col="gray", lty=2, lwd=2)
plot(1:length(all_results), Convergence$j, type="l", 
     main="Compliance Jump", xlab="Simulation", ylab=expression(hat(R)),
     cex.main=1.2, cex.lab=1.0, cex.axis=0.9)
abline(h=1.1, col="gray", lty=2, lwd=2)
plot(1:length(all_results), Convergence$eff, type="l", 
     main="Treatment Effect", xlab="Simulation", ylab=expression(hat(R)),
     cex.main=1.2, cex.lab=1.0, cex.axis=0.9)
abline(h=1.1, col="gray", lty=2, lwd=2)
dev.off()
