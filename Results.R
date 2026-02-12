# =======================================================================================================================
#
#    MODIFIED REPLICATION CODE
#
#    
#    Based on:       Kowalska, van de Wiel, and van der Pas (2025)
#    Source:         https://github.com/JuliaMKowalska/RDD_unknown_cutoff
#
#    Purpose:        Master Thesis: "Prior Sensitivity and Robustness of Bayesian Methods 
#                    in Regression Discontinuity Designs with Unknown Cutoff"
#
#    Description:    TThis script imports the raw posterior samples from the LoTTA simulations,
#                    cleans the data, and calculates key performance metrics. 
#
#
# =======================================================================================================================




library(coda)
library(ggplot2)
library(tidyverse)
library(bayestestR)
library(dplyr)
library(parallel)
library(ggplot2)

########################
### Helper Functions ###
########################

funA_sample<-function(X){
  Y=X
  X2=X[X>=0]
  X1=X[X<0]
  Y[X<0]=(1.8*X1^3+2.*X1^2)+0.05
  Y[X>=0]=0.05*X2-0.1*X2^2+0.22
  Y=Y+rnorm(length(X),0,0.1)
  return(Y)
}

funB_sample<-function(X){
  Y=X
  X2=X[X>=0]
  X1=X[X<0]
  Y[X<0]=1/(1+exp(-2*X1))-0.5+0.4
  Y[X>=0]=(log(X2*2.+1)-0.15*X2^2)*0.6-0.20+0.4
  Y=Y+rnorm(length(X),0,0.1)
  return(Y)
}

funC_sample<-function(X){
  Y=0.48- 2.7*X+ 1.18*X^2 +1.21*X^3 + 2.54*X^4 - 3*X^5-1.9*X^6-5/(1+exp(-10*(X+1)))-10+sin(5*X-2)
  Y=Y/10+rnorm(length(X),0,0.1)
  return(Y)
}

lee_sample<-function(X){
  X1=X[X<0]
  X2=X[X>=0]
  Y1=0.48+ 1.27*X1+ 7.18*X1^2 +20.21*X1^3 + 21.54*X1^4 + 7.33*X1^5
  Y2=0.52+ 0.84*X2 - 3.00*X2^2+7.99*X2^3 -9.01*X2^4 + 3.56*X2^5
  Y=append(Y1,Y2)+rnorm(length(X),0,0.1295)
  return(Y)
}

ludwig_sample<-function(X){
  X1=X[X<0]
  X2=X[X>=0]
  Y1=3.71+ 2.3*X1+ 3.28*X1^2 +1.45*X1^3 + 0.23*X1^4 + 0.03*X1^5
  Y2=0.26+ 18.49*X2 - 54.81*X2^2+74.3*X2^3 -45.02*X2^4 + 9.83*X2^5
  Y=append(Y1,Y2)+rnorm(length(X),0,0.1295) 
  return(Y) 
}

sample_prob55<-function(X){
    P=rep(0,length(X))
    P[X>=0.]=invlogit((8.5*X[X>=0.]-1.5))/10.5+0.65-0.0007072
    P[X<0.]=(X[X<0.]+1)^4/15+0.05 
    T=rep(0,length(X))
    for(j in 1:length(X)){
      T[j]=sample(c(1,0),1,prob=c(P[j],1-P[j]))
    }
    return(T)
}

sample_prob30<-function(X){
  P=rep(0,length(X))
  P[X>=0.]=invlogit((8.5*X[X>=0.]-1.5))/10.5+0.4-0.0007072
  P[X<0.]=(X[X<0.]+1)^4/15 +0.05 
  T=rep(0,length(X)) 
  for(j in 1:length(X)){
    T[j]=sample(c(1,0),1,prob=c(P[j],1-P[j]))
  }
  return(T)
}

#######################
### Result Function ### 
#######################

performance_sample_FUZZY<-function(post_sample,name,jump){
  c=0
  functions=list("A"=funA_sample,"B"=funB_sample, "C"=funC_sample, "lee"=lee_sample, "ludwig"=ludwig_sample)
  probs=list("0.55"=sample_prob55,"0.3"=sample_prob30)
  fun=functions[[name]]
  prob=probs[[jump]]
  effects=list("A"=0.17,"B"=-0.2, "C"=0, "lee"=0.04, "ludwig"=-3.45)
  compliance=list("0.55"=0.55,"0.3"=0.30)
  j=compliance[[jump]]
  tr_eff=effects[[name]]/compliance[[jump]]
  
  Samples=combine.mcmc(post_sample)
  C=as.numeric(Samples[,1])
  J=as.numeric(Samples[,2])
  Eff=as.numeric(Samples[,7])
  
  qeff=as.numeric(quantile(Eff,c(0.025,0.975)))
  meff=median(Eff)
  mapeff=as.numeric(map_estimate(Eff))
  hdieff=as.numeric(ci(Eff,method='HDI'))[2:3]
  coveff=ifelse(tr_eff<=qeff[2]&tr_eff>=qeff[1],1,0)
  hdicoveff=ifelse(tr_eff<=hdieff[2]&tr_eff>=hdieff[1],1,0)
  
  if(tr_eff<0){
    signeff=ifelse(0>qeff[2],1,0)
    hdisigneff=ifelse(0>hdieff[2],1,0)
  }
  else{
    signeff=ifelse(0<qeff[1],1,0)
    hdisigneff=ifelse(0<hdieff[1],1,0)
  }
  
  merreff=abs(tr_eff-meff)
  maperreff=abs(tr_eff-mapeff)
  sample_err=mean(abs(tr_eff-Eff))
  qcieff_len=as.numeric(qeff[2]-qeff[1])
  hdicieff_len=as.numeric(hdieff[2]-hdieff[1])
  jes=as.numeric(map_estimate(J))
  jabs=abs(j-jes)
  ces=as.numeric(map_estimate(C))
  cabs=abs(c-ces)
  mjes=as.numeric(median(J))
  mjabs=abs(j-mjes)
  mces=as.numeric(median(C))
  mcabs=abs(c-mces)
  return(list(abs_err_med=merreff,abs_err_map=maperreff,ci_length_sym=qcieff_len,ci_length_hdi=hdicieff_len,bias_med=-(tr_eff-meff),bias_map=-(tr_eff-mapeff),cov_med=coveff,cov_hdi=hdicoveff,sign_med=signeff,sign_hdi=hdisigneff,c_abs_med=mcabs,c_abs_map=cabs,j_abs_med=mjabs,j_abs_map=jabs))
}

######################################
### Actual Performance Calculation ###
######################################

### Import Performance Results from previous runs and evaluate performance ###

A55 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/baseline prior/55/performance_base_A55_1000.csv')
B55 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/baseline prior/55/performance_base_B55_1000.csv')
C55 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/baseline prior/55/performance_base_C55_1000.csv')
lee55 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/baseline prior/55/performance_base_lee55_1000.csv')
ludwig55 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/baseline prior/55/performance_base_ludwig55_1000.csv')

A30 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/baseline prior/30/performance_base_A30_1000.csv')
B30 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/baseline prior/30/performance_base_B30_1000.csv')
C30 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/baseline prior/30/performance_base_C30_1000.csv')
lee30 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/baseline prior/30/performance_base_lee30_1000.csv')
ludwig30 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/baseline prior/30/performance_base_ludwig30_1000.csv')



numeric_cols <- c("abs_err_med", "abs_err_map", "ci_length_sym", "ci_length_hdi",
                  "bias_med", "bias_map", "cov_med", "cov_hdi", "sign_med", "sign_hdi",
                  "c_abs_err_med", "c_abs_err_map", "c_ci_length_sym", "c_ci_length_hdi",
                  "c_bias_med", "c_bias_map", "c_cov_sym", "c_cov_hdi",
                  "j_abs_err_med", "j_abs_err_map", "j_ci_length_sym", "j_ci_length_hdi",
                  "j_bias_med", "j_bias_map", "j_cov_sym", "j_cov_hdi")



#A55

A55_clean <- A55[!grepl("Error", A55$abs_err_map), ]

#View(A55_clean)

for(col in numeric_cols) {
  A55_clean[[col]] <- as.numeric(A55_clean[[col]])
}

apply(A55_clean, 2,mean)
apply(A55_clean, 2,median)          

sqrt(mean(A55_clean[['abs_err_map']]^2)) # RMSE tr.eff. (MAP estimate)
sqrt(mean(A55_clean[['c_abs_err_map']]^2)) # RMSE cutoff location (MAP estimate)
sqrt(mean(A55_clean[['j_abs_err_map']]^2)) # RMSE compliance rate (MAP estimate)

# B55 

B55_clean <- B55[!grepl("Error", B55$abs_err_map), ]    

for(col in numeric_cols) {
  B55_clean[[col]] <- as.numeric(B55_clean[[col]])
}

apply(B55_clean, 2,mean)
apply(B55_clean, 2,median)
sqrt(mean(B55_clean[['abs_err_map']]^2)) # RMSE tr.eff. (MAP estimate)
sqrt(mean(B55_clean[['c_abs_err_map']]^2)) # RMSE cutoff location (MAP estimate)
sqrt(mean(B55_clean[['j_abs_err_map']]^2)) # RMSE compliance rate (MAP estimate)

# C55

C55_clean <- C55[!grepl("Error", C55$abs_err_map), ]
for(col in numeric_cols) {
  C55_clean[[col]] <- as.numeric(C55_clean[[col]])
}
apply(C55_clean, 2,mean)
apply(C55_clean, 2,median)

sqrt(mean(C55_clean[['abs_err_map']]^2)) # RMSE tr.eff. (MAP estimate)
sqrt(mean(C55_clean[['c_abs_err_map']]^2)) # RMSE cutoff location
sqrt(mean(C55_clean[['j_abs_err_map']]^2)) # RMSE compliance rate (MAP estimate)

# lee55 
lee55_clean <- lee55[!grepl("Error", lee55$abs_err_map), ]
for(col in numeric_cols) {
  lee55_clean[[col]] <- as.numeric(lee55_clean[[col]])
}
apply(lee55_clean, 2,mean)
apply(lee55_clean, 2,median)

sqrt(mean(lee55_clean[['abs_err_map']]^2)) # RMSE tr.eff. (MAP estimate)
sqrt(mean(lee55_clean[['c_abs_err_map']]^2)) # RMSE cutoff
sqrt(mean(lee55_clean[['j_abs_err_map']]^2)) # RMSE compliance rate (MAP estimate)


# ludwig55
ludwig55_clean <- ludwig55[!grepl("Error", ludwig55$abs_err_map), ]
for(col in numeric_cols) {
  ludwig55_clean[[col]] <- as.numeric(ludwig55_clean[[col]])
}
apply(ludwig55_clean, 2,mean)
apply(ludwig55_clean, 2,median)   

sqrt(mean(ludwig55_clean[['abs_err_map']]^2)) # RMSE tr.eff. (MAP estimate)
sqrt(mean(ludwig55_clean[['c_abs_err_map']]^2)) # RMSE
sqrt(mean(ludwig55_clean[['j_abs_err_map']]^2)) # RMSE compliance rate (MAP estimate)


# compliance tests

c0 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/test_simulations/performance_compS2_B55_1000.csv')

c2 <- read.csv('/Users/nicobuschhorn/Desktop/Master Thesis/R/MarvRes/test_simulations/performance_compS_B55_1000.csv')


c0 <- c0[!grepl("Error", c0$abs_err_map), ]    

for(col in numeric_cols) {
  c0[[col]] <- as.numeric(c0[[col]])
}

apply(c0, 2,mean)
apply(c0, 2,median)
sqrt(mean(c0[['abs_err_map']]^2)) # RMSE tr.eff. (MAP estimate)
sqrt(mean(c0[['c_abs_err_map']]^2)) # RMSE cutoff location (MAP estimate)
sqrt(mean(c0[['j_abs_err_map']]^2)) # RMSE compliance rate (MAP estimate)


c2 <- c2[!grepl("Error", c2$abs_err_map), ]

for(col in numeric_cols) {
  c2[[col]] <- as.numeric(c2[[col]])
}

apply(c2, 2,mean)
apply(c2, 2,median)

sqrt(mean(c2[['abs_err_map']]^2)) # RMSE tr.eff. (MAP estimate)
sqrt(mean(c2[['c_abs_err_map']]^2)) # RMSE cutoff location (
sqrt(mean(c2[['j_abs_err_map']]^2)) # RMSE compliance rate (MAP estimate)

# ============================================================================
# Metrics Explained
# ============================================================================

# tr_eff    : True treatment effect (known from simulation)

# j         : True compliance (known from simulation)

# C         : Estimated cutoff from posterior samples (all chains)

# J         : Estimated compliance from posterior samples (all chains)

# Eff       : Estimated treatment effect from posterior samples (all chains)

# meff      : Median of treatment effect samples (point estimate)

# mapeff    : Maximum a posteriori (MAP: the most probable value - highest density) of treatment effect samples (point estimate)

# mces      : Median of cutoff estimates (point estimate)

# ces       : MAP of cutoff estimates (point estimate)

# mjes      : Median of compliance estimates (point estimate)

# jes       : MAP of compliance estimates (point estimate)

# qeff      : Symmetric 95% Credible Interval for treatment effect

# hdieff    : Highest Density Interval (HDI) 95% Credible Interval for treatment effect (contains most probable values, so shortest 95% interval, usually narrower than symmetric CI)

# merreff   : Median absolute error of treatment effect estimate (|tr_eff - meff|)

# maperreff : MAP absolute error of treatment effect estimate (|tr_eff - mapeff|)

# bias_med : Bias of median estimate -(tr_eff - meff) (positive means overestimation, negative means underestimation, ideally close to 0)

# bias_map : Bias of MAP estimate -(tr_eff - mapeff)

# qcieff_len : Length of symmetric 95% credible interval (precision of estimate, the narrower the better)

# hdicieff_len : Length of HDI 95% credible interval (precision of estimate, the narrower the better)

# coveff    : Coverage indicator for symmetric CI (1 if tr_eff in qeff, else 0; ideally close to 0.95 over many simulations)

# hdicoveff : Coverage indicator for HDI (1 if tr_eff in hdieff, else 0; ideally close to 0.95 over many simulations)

# signeff   : Sign correctness indicator for median estimate (1 if sign of meff matches sign of tr_eff, else 0; ideally close to 1), Is ENTIRE symmetric CI above/below 0? (depending on sign of tr_eff)

# hdisigneff: Sign correctness indicator for HDI (1 if sign of hdieff matches sign of tr_eff, else 0; ideally close to 1), Is ENTIRE symmetric CI above/below 0? (depending on sign of tr_eff)

# cabs      : Absolute error of cutoff MAP estimate (|c - ces|)

# mcabs     : Absolute error of cutoff median estimate (|c - mces|)

# jabs      : Absolute error of compliance MAP estimate (|j - jes|)

# mjabs     : Absolute error of compliance median estimate (|j - mjes|)

