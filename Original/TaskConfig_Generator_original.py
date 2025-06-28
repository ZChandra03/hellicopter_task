#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:27:00 2025

@author: tahraeissa
"""

 ### IMPORT MODULES
import numpy as np
import pandas as pd
import os
import scipy.stats as spst

### SET CONFIGURATION PARAMETERS
params = {}

# trial parameters
params['nTrials'] = 100 #number of trials in each block

# State Parameters
params['Mu'] = 1 #mean for normal distribution
params['HazRes']=0.05
params['Hazards'] =np.arange(0, 1, params['HazRes']) 
params['nEvidence']=20
params['xLim']=5
params['testSigmas']=[0,0.1,0.5,1] #variance for the normal distribution
params['nBlocks']=len(params['testSigmas'])

# responseTime limit 
params['responseTimeLimit_s']=5

# block list
params['block_list'] = ['preTest','easy','medium','hard'] 
params['block_type']=['report','predict']


# trial dictionary fields (rt is also in sec)
params['trial_fields'] = ['blockNum','blockDifficulty','sigma','blockType','trialInBlock','trueHazard','evidence','states','trueVal']
params['variants']=50

#  Save directory
params['saveDir'] = os.getcwd()+'/variants'
# make the save folder if it doesnt exist
if os.path.exists(params['saveDir'])==False:
    os.mkdir(params['saveDir'])


#%% --- generate and display evidence
def genEvidence(hz,sigma,params):
    
     # parse inputs
    r=np.random.rand()
    if r<=0.5:
        mu=-params['Mu']
    else:
        mu=params['Mu']


    # create a set of evidence stimulus
    state=[]
    
    ev=[]
    for x in range(0,params['nEvidence']):
        lw=-params['xLim']
        hg=params['xLim']
        if sigma>0:
            xtrunc=spst.truncnorm((lw-mu)/sigma,(hg-mu)/sigma,mu,sigma)
            e=xtrunc.rvs()
            ev.append((e))
        elif sigma==0:
            e=np.random.normal(mu,sigma,1)
            ev.append(e[0])
        state.append(mu)
        r=np.random.rand()
        if r<hz: #switch states
            mu=-mu
    if sigma==0:
      ev=np.round(ev).tolist()
    return ev, state


#%% --- generate block sequence and parameterizations
def genBlockParam(params):
   
    hzTrial=int(params['nTrials']/len(params['Hazards']))
    
    #create list of hazards
    trialHaz=[]
    
    for x in range(0,len(params['Hazards'])):
        for y in range(hzTrial):  
            trialHaz.append(params['Hazards'][x])
    
    np.random.shuffle(trialHaz)
    
    return trialHaz


#%% --- Generate a trial list within a block we will copy this over twice
def makeBlockTrials(params):

    # create a blocklist
    trialDict_list = []
    
    bl=np.arange(params['nBlocks'])
    np.random.shuffle(bl)
    r=np.random.rand()
    if r<=0.5:
        first_set='report'
    else:
        first_set='predict'
    
    # loop over blocks and trials
    for b in range(params['nBlocks']):
        blockNum=bl[b]
        blockName=params['block_list'][bl[b]]
        trialHaz=genBlockParam(params)
        sigma=params['testSigmas'][bl[b]]       
         # trial counter (within block)
        trialInBlock = 0

        for t in np.arange(0,params['nTrials']):
                # update within block counter
                trialInBlock+=1
                [ev,state]=genEvidence(trialHaz[t],sigma,params)
                evFloat = [float(x) for x in ev]

                # make an empty trial dictionary
                trialDict = {}
                for tf in params['trial_fields']:
                    trialDict[tf] = np.nan
                    
                # populate trialDict with basic trial features
                trialDict['blockNum']=blockNum 
                trialDict['sigma']=sigma
                trialDict['blockType'] =first_set
                trialDict['blockDifficulty'] = blockName
                trialDict['trueHazard']=trialHaz[t]
                trialDict['evidence'] = evFloat
                trialDict['states'] = state
                trialDict['trialInBlock'] = trialInBlock
                if trialDict['blockType']=='report':
                    trialDict['trueVal']=state[-1]
                elif trialDict['blockType']=='predict':
                    if trialDict['trueHazard']<0.5:
                        trialDict['trueVal']=-1
                    elif trialDict['trueHazard']>0.5:
                        trialDict['trueVal']=1
                # append to list
                trialDict_list.append(trialDict)
                
                
    if first_set=='report':
        second_set='predict'
    else:
        second_set='report'
        
    second_list=[]
    for t in trialDict_list:
        trial_copy=t.copy()
        trial_copy['blockType']=second_set
        if trial_copy['blockType']=='report':
            trial_copy['trueVal']=trial_copy['states'][-1]
        elif trial_copy['blockType']=='predict':
            if trial_copy['trueHazard']<0.5:
                trial_copy['trueVal']=-1
            elif trial_copy['trueHazard']>0.5:
                trial_copy['trueVal']=1
        second_list.append(trial_copy)
    
    trialDict_list.extend(second_list)
        
        

    # return block list
    return trialDict_list

#%% run configuration
for k in range(0, params['variants']):
    var_id = "var" + str(k)

    train_trials = makeBlockTrials(params) 
    test_trials = makeBlockTrials(params)

    # Save train and test data separately
    train_df = pd.DataFrame(train_trials)
    train_df.to_csv(path_or_buf=params['saveDir'] + '/trainConfig_' + var_id + '.csv')

    test_df = pd.DataFrame(test_trials)
    test_df.to_csv(path_or_buf=params['saveDir'] + '/testConfig_' + var_id + '.csv')

config_df = pd.Series(params)
config_df.to_csv(path_or_buf=params['saveDir'] + '/TaskConfig.csv')