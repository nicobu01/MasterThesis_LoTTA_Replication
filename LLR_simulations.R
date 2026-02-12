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
#    Description:    This script executes the frequentist benchmark simulations. It applies 
#                    standard LLR (with automatic bandwidth selection) to the same DGP 
#                    used for the LoTTA analysis to facilitate direct performance comparison
#                    across sample sizes and compliance scenarios.
#
# =======================================================================================================================


library(bayestestR)
library(dplyr)
library(rdrobust)

logit<-function(x){
  return(log(x/(1-x)))
}
invlogit<-function(x){
  return(1/(1+exp(-x)))
}


# Sampling outcome

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

# Sampling

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


## LLR simulations results - sharp design 
# N - number of samples,
# name - function name: "A", "B", "C", "lee", "ludwig", 
# jump - string, jumpsize: "0.55" or "0.3", indicates treatment probablity function
# st - first seed - 1 (0 corresponds to starting simulations with seed 1)
# Returns data frame with the following parameters for each simulation:
# absolute error, 95% CI length, bias,
# binary indicator if CI contains tr.eff. value, binary indicator if CI correctly identifies sign of tr.eff,
#  absolute error of compliance rate estimate #

LLR_performance_FUZZY<-function(N,name,jump,st=0){
  functions=list("A"=funA_sample,"B"=funB_sample, "C"=funC_sample, "lee"=lee_sample, "ludwig"=ludwig_sample)
  probs=list("0.55"=sample_prob55,"0.3"=sample_prob30)
  fun=functions[[name]]
  prob=probs[[jump]]
  effects=list("A"=0.17,"B"=-0.2, "C"=0, "lee"=0.04, "ludwig"=-3.45)
  compliance=list("0.55"=0.55,"0.3"=0.30)
  tr_eff=effects[[name]]/compliance[[jump]]
  rddat=data.frame(abs_err=0,ci_length=0,bias=0,cov=0,sign=0,j_abs=0)
  for(i in 1:N){
    set.seed(st+i)
    X=sort(2*rbeta(1000,2,4)-1)
    Y=fun(X)
    T=prob(X)
    r=rdrobust(Y,X,0,fuzzy = T)
    meff=as.numeric(r$Estimate[2])
    merreff=abs(tr_eff-meff)
    qeff=as.numeric(r$ci[3,])
    coveff=ifelse(tr_eff<=qeff[2]&tr_eff>=qeff[1],1,0)
    if(tr_eff<0){
      signeff=ifelse(0>qeff[2],1,0)
    }
    else{
      signeff=ifelse(0<qeff[1],1,0)
    }
    qcieff_len=as.numeric(qeff[2]-qeff[1])
    j=r$tau_T[2]
    out=meff*j
    biases=-(tr_eff-meff)
    l=list(abs_err=merreff,ci_length=qcieff_len,bias=biases,cov=coveff,sign=signeff,j_abs_err=abs(compliance[[jump]]-j))
    rddat[i,]=l
  }
  return(rddat)
}

# LLR simulations ###

# 0.55 jump

LLR1=LLR_performance_FUZZY(1000,'A','0.55',1)

LLR2=LLR_performance_FUZZY(1000,'B','0.55',2)

LLR3=LLR_performance_FUZZY(1000,'C','0.55',3)

LLR4=LLR_performance_FUZZY(1000,'lee','0.55',4)

LLR5=LLR_performance_FUZZY(1000,'ludwig','0.55',5)

# 0.3 jump

LLR6=LLR_performance_FUZZY(1000,'A','0.3',6)

LLR7=LLR_performance_FUZZY(1000,'B','0.3',7)

LLR8=LLR_performance_FUZZY(1000,'C','0.3',8)

LLR9=LLR_performance_FUZZY(1000,'lee','0.3',9)

LLR10=LLR_performance_FUZZY(1000,'ludwig','0.3',10)

# Results

apply(LLR1, 2,mean)
apply(LLR1[1:3], 2,median)
sqrt(mean(LLR1[['abs_err']]^2)) # RMSE tr.eff. 
sqrt(mean(LLR1[['j_abs']]^2)) # RMSE compliance rate 


apply(LLR2, 2,mean)
apply(LLR2[1:3], 2,median)
sqrt(mean(LLR2[['abs_err']]^2)) # RMSE tr.eff. 
sqrt(mean(LLR2[['j_abs']]^2)) # RMSE compliance rate 

apply(LLR3, 2,mean)
apply(LLR3[1:3], 2,median)
sqrt(mean(LLR3[['abs_err']]^2)) # RMSE tr.e
sqrt(mean(LLR3[['j_abs']]^2)) # RMSE compliance rate

apply(LLR4, 2,mean)
apply(LLR4[1:3], 2,median)
sqrt(mean(LLR4[['abs_err']]^2)) # RMSE tr.e
sqrt(mean(LLR4[['j_abs']]^2)) # RMSE compliance rate

apply(LLR5, 2,mean)
apply(LLR5[1:3], 2,median)
sqrt(mean(LLR5[['abs_err']]^2)) # RMSE tr.eff
sqrt(mean(LLR5[['j_abs']]^2)) # RMSE compliance rate

apply(LLR6, 2,mean)
apply(LLR6[1:3], 2,median)
sqrt(mean(LLR6[['abs_err']]^2)) # RMSE tr.e
sqrt(mean(LLR6[['j_abs']]^2)) # RMSE compliance rate

apply(LLR7, 2,mean)
apply(LLR7[1:3], 2,median)
sqrt(mean(LLR7[['abs_err']]^2)) # RMSE tr.e
sqrt(mean(LLR7[['j_abs']]^2)) # RMSE compliance rate

apply(LLR8, 2,mean)
apply(LLR8[1:3], 2,median)
sqrt(mean(LLR8[['abs_err']]^2)) # RMSE tr.e
sqrt(mean(LLR8[['j_abs']]^2)) # RMSE compliance rate

apply(LLR9, 2,mean)
apply(LLR9[1:3], 2,median)
sqrt(mean(LLR9[['abs_err']]^2)) # RMSE tr.eff
sqrt(mean(LLR9[['j_abs']]^2)) # RMSE compliance rate

apply(LLR10, 2,mean)
apply(LLR10[1:3], 2,median)
sqrt(mean(LLR10[['abs_err']]^2)) # RMSE tr.eff
sqrt(mean(LLR10[['j_abs']]^2)) # RMSE compliance rate



