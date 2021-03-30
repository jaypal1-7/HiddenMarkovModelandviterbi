#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np


class HMM_Viterbi(object):
    #HMM Viterbi class holds initial, transition and Emmission probabality data
    
    def __init__(self,States=[],Observations=[], InitialProb = {},TransProb ={},EmissionProb={},TransMat=[]):
        self.States = States
        self.Observations=Observations
        self.InitialProb=InitialProb
        self.TransProb=TransProb
        self.EmissionProb=EmissionProb
        self.TransMat = TransMat
        self.ForwardValues =[]
        self.BackwardValues = []
        self.ProbForward = []
        self.PosteriorValues =[]


    #Set Parameter Values 
    def Set_Parameters(self,encodedTextList):
        self.States = (list(range(max(encodedTextList))))
        self.Observations = (list(range(max(encodedTextList))))
        self.InitialProb = Initial_prob(encodedTextList)
        self.TransMat = Trans_Mat(encodedTextList)
        Transmat = Trans_Mat(encodedTextList)
        self.TransProb = Convert_to_dict(Transmat)
        self.EmissionProb = Convert_to_dict(Transmat.transpose(1,0))
        
    #Initial Probability
    def Initial_prob(encodedTextList):
        dictnry = {}
        tranDist = sorted(set(encodedTextList))
        tranProb = []
        length= len(encodedTextList)
        
        for x in tranDist:
            tranProb.append(len(np.where(encodedTextList==x)[0])/length)
        arr = np.array(tranProb)
        for i,prob in zip(list(range(len(arr))),arr):
            dictnry[i]=prob
        return dictnry

    
    #Transition Matrix calculation for m states
    def Trans_Mat(encodedTextList):
        m_states = 1+max(encodedTextList)
        TranMatrix = np.zeros(shape=(m_states,m_states))
        
        for i in range(0,m_states):
            EL = list(np.where(encodedTextList==i)[0])
            for el in EL:
                if(el +1 != len(encodedTextList)):
                    TranMatrix[i,encodedTextList[el+1]] += 1/len(EL)
        TranMatrix[TranMatrix!=0]= TranMatrix[TranMatrix!=0] + (1/m_states)
        TranMatrix[TranMatrix==0]=1/m_states
        return TranMatrix


    def Convert_to_dict(TranMatrix):
        dict1 = {}
        for i,row in zip(list(range(np.size(matrix,axis=0))),matrix):
            temp = {}
            for j,val in zip(list(range(np.size(row,axis=0))),row):
                temp[j]=val
            dict1[i]=temp
        return dict1
    


    #Viterbi Algorithm Implementation
    def Viterbi_Algo(TranMatrix, EmissMat, observations):
        num_obs = len(observations)
        num_states = TranMatrix.shape[0]
        log_probs = np.zeros(num_states)
        paths = np.zeros( (num_states, num_obs+1 ))
        paths[:, 0] = np.arange(num_states)
        for obs_ind, obs_val in enumerate(observations):
            for state_ind in range(num_states):
                val = 0
                if obs_val< np.size(EmissMat,1):
                    val = np.log(EmissMat[state_ind, obs_val])
                temp_probs = log_probs +                               val +                              np.log(TranMatrix[:, state_ind])
                best_temp_ind = np.argmax(temp_probs)
                paths[state_ind,:] = paths[best_temp_ind,:]
                paths[state_ind,(obs_ind+1)] = state_ind
                log_probs[state_ind] = temp_probs[best_temp_ind]
        best_path_ind = np.argmax(log_probs)
    
        return (paths[best_path_ind], log_probs[best_path_ind])


    def Posterior(self):
        posterior = []
        for i in range(len(self.Observations)):
            posterior.append({st: self.ForwardValues[i][st] * self.BackwardValues[i][st] / self.ProbForward for st in self.States})

        return posterior

    #Forward Algorithm
    def ForwardAlgo(self, EndState):
        fwd = []
        f_prev = {}
        for i, observation_i in enumerate(self.Observations):
            f_curr = {}
            for st in self.States:
                if i == 0:
                    prev_f_sum = self.InitialProb[st]
                else:
                    prev_f_sum = sum(f_prev[k]*self.TransProb[k][st] for k in self.States)

                f_curr[st] = self.EmissionProb[st][observation_i] * prev_f_sum

            fwd.append(f_curr)
            f_prev = f_curr

        p_fwd = sum(f_curr[k] * self.TransProb[k][EndState] for k in self.States)


        return fwd,p_fwd

    #Backward Algorithm
    def BackwardAlgo(self, EndState = 'PERIOD'):
        backw = []
        b_prev = {}
        for i, observation_i_plus in enumerate(reversed(self.Observations[1:]+[None,])):
            b_curr = {}
            for st in self.States:
                if i == 0:
                    # base case for backward part
                    b_curr[st] = self.TransProb[st][EndState]
                else:
                    b_curr[st] = sum(self.TransProb[st][l] * self.EmissionProb[l][observation_i_plus] * b_prev[l] for l in self.States)

            backw.insert(0,b_curr)
            b_prev = b_curr

        p_bkw = sum(self.InitialProb[l] * self.EmissionProb[l][self.Observations[0]] * b_curr[l] for l in self.States)

        return backw

    

