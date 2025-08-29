#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:13:23 2025

@author: taei7404


"""

### IMPORT MODULES
import numpy as np
from scipy.stats import norm
from scipy.stats import beta

#%%  example values- these would be extracted from each trial's data


true_Haz=0.2 
ev=[-1,-1,1,1,1,1,1,1] #evidence

mu1=-1 #mean location 1 

mu2=1 #mean lcoation 2

sigma=0.1 #sigma for normal distributions

hs=np.arange(0, 1, 0.05) #array of possible hazard rates

#%%

# first let's create a function that can be our Bayesian ideal observer
def BayesianObserver(ev,mu1,mu2,sigma,hs):
    # this function will computer the responses from a bayesian observer (with or without bias or mismatched environmental knowledge)
    #ev= evidence (the 20 observations)
    # mu 1 and mu2= location 1 and 2, the means of the Gaussian distributions
    # sigma= standard deviation of the Gaussian distributions
    #hs= parameterized values of hazard rate (an array)
    # nEvidence= the number of piece
    
    nEvidence=len(ev) #number of pieces of evidence
    bias=0 #how much prior bias the model has. you can ignore this for now. 
    
    ### beta distribution prior - FOR NOW, this will just come out to be a uniform (flat) prior. The bias term controls these changes. 
      # Initialize the arrays
    L_n = [np.zeros((len(hs), nEvidence + 1)) for _ in range(2)]  # L_n{1} and L_n{2}
    
    # Set initial alpha and beta for Beta distribution
    alpha = 1
    beta_param = 1
    
    # Adjust alpha and beta based on the bias
    if bias > 0:
        alpha += bias
    elif bias < 0:
        beta_param -= bias
    
    # Calculate the Beta distribution prior
    beta_prior = beta.pdf(hs, alpha, beta_param)
    
    # Marginalization factor
    marg = 2 * len(hs)
    
    # Initialize the first column of L_n{1} and L_n{2}
    L_n[0][:, 0] = beta_prior / marg
    L_n[1][:, 0] = L_n[0][:, 0]
    
    # Check if the sum of probabilities is 1, and adjust if necessary
    P_check = np.sum(L_n[0][:, 0]) + np.sum(L_n[1][:, 0])
    P_diff = 1 - P_check  # The difference from 1 due to discretization
    
    # If the sum is less than 1, distribute the difference equally
    P_marg = P_diff / (2 * len(hs))
    L_n[0][:, 0] += P_marg
    L_n[1][:, 0] += P_marg
        
    # Compute the normal pdf for all evidence once
    if sigma != 0:
        norm_P_S1 = norm.pdf(ev, mu1, sigma)
        norm_P_S2 = norm.pdf(ev, mu2, sigma)
    
        # Main loop this computes the likelihoods
    for n in range(nEvidence):  # Iterate over each piece of evidence
    
        for s in range(2):  # Two states
        
            for h in range(len(hs)):  # Iterate over possible hs values
            
                if s == 0:  # State 1
                
                    # set the probability of that evidence in state 1
                    if sigma==0: #in the perfect evidence case
                      if int(ev[n]) == mu1:
                          P_S1 = 1
                      elif int(ev[n]) == mu2:
                          P_S1 = 0
                    else:
                        P_S1=norm_P_S1[n]
                        
                    #now compute the likelihood based on the hazard rates for state 1
                    L_n[0][h, n + 1] = P_S1 * ((1 - hs[h]) * L_n[0][h, n] + hs[h] * L_n[1][h, n])
                    
                elif s == 1:  # State 2
                 
                     # set the probability of that evidence in state 1
                     if sigma==0: #in the perfect evidence case
                       if int(ev[n]) == mu2:
                           P_S2 = 1
                       elif int(ev[n]) == mu1:
                           P_S2 = 0
                     else:
                         P_S2=norm_P_S2[n]
                         
                     #now compute the likelihood based on the hazard rates for state 1
                     L_n[1][h, n + 1] = P_S2 * ((1 - hs[h]) * L_n[1][h, n] + hs[h] * L_n[0][h, n])

        # Renormalization
        T = np.sum(L_n[0][:, n + 1]) + np.sum(L_n[1][:, n + 1])
        
        # Avoid division by zero
        if T > 0:
            L_n[0][:, n + 1] /= T
            L_n[1][:, n + 1] /= T
            

    ### now let's determine responses for either state (for report) or hazard for predict


    ## for hazard rate
    L_haz = np.zeros((len(hs), nEvidence + 1))
         # let's marginalize across both states for the likelihoods
    for n in range(nEvidence+1):  # Iterate over each piece of evidence     
        for h in range(len(hs)):  # Iterate over possible hs values
           L_haz[h,n] = L_n[0][h, n] + L_n[1][h, n] #NOTE!! This is where you can read out the belief on every piece of evidence. For each column, the largest value is the hazard belief for that piece of evidence
           
       
    ## for state
    L_state = np.zeros((2, nEvidence + 1))
         # let's marginalize across both states
    for n in range(nEvidence+1):  # Iterate over each piece of evidence     
        for s in range(2):  # Iterate over states
           L_state[s,n] = np.sum(L_n[s][:, n] )
           

    
    ## for report AFTER ALL PIECES OF EVIDENCE (State)
    P_s1=L_state[0,-1]
    P_s2=L_state[1,-1]
    resp_Rep=0
   
    if P_s1>P_s2:
        resp_Rep = -1  # State 1 logged as -1
    elif P_s1 <P_s2:
        resp_Rep = 1  # State 2 logged as 1
    elif P_s1==P_s2:
        resp_Rep = np.random.choice([-1, 1])  # No preference, random choice if there is no specific state

    
    # for predict AFTER ALL PIECES OF EVIDENCE (hazard)
    P_haz_switch=hs*L_haz[:,-1] # this becomes a weighted sum because of the probabilistic nature
    P_haz_stay=(1-hs)*L_haz[:,-1]
    P_stay=np.sum(P_haz_stay)    
    P_switch=np.sum(P_haz_switch)
    resp_Pred=0
    
    if P_stay == 0.5 and P_switch == 0.5:
        resp_Pred = np.random.choice([-1, 1])  # No preference, random choice, hazard belief=0.5
    elif P_stay>P_switch:
        resp_Pred = -1  # Stay
    elif P_stay<P_switch:
        resp_Pred = 1  # Stay

    return L_haz, L_state, resp_Rep, resp_Pred