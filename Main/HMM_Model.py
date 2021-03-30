#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class HMM_Model(object):
    
    def __init__(self,txtformat=[],HMM=[]):
        self.Text_format = txtformat
        self.HMM_Viterbi = HMM

