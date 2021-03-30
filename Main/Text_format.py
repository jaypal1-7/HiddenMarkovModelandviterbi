#!/usr/bin/env python
# coding: utf-8

# In[30]:


from sklearn.preprocessing import LabelEncoder
import string

class Text_format(object):
    """Text formatting Class - data related to text to be fed"""

    def __init__(self,Encoder = None):
        self.Encoder = preprocessing.LabelEncoder()
        self.EncodedTxtList=[]
    
    def Text_Process(self,text):
        text = text.replace(',', ' COMMA')
        text = text.replace('.', ' PERIOD')
        text = text.lower()
        remPunct = str.maketrans('', '', string.punctuation)
        text = text.translate(remPunct)
        return text

    def TextEncoding(self,text):
        text = self.Text_Process(text)
        TxtList = list(filter(None,text.split()))
        txtList_NoDuplicate = list(set(TxtList))
        self.EncodedTxtList = self.Encoder.fit_transform(txtList_NoDuplicate)

        




    

