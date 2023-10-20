#!/usr/bin/env python
# coding: utf-8

# In[30]:


from sklearn.preprocessing import LabelEncoder
import string

class TextFormatter(object):
    """Text Formatting Class - data related to text to be fed"""

    def __init__(self):
        self.Encoder = LabelEncoder()
        self.EncodedTxtList = []

    def text_process(self, text):
        text = text.replace(',', ' COMMA')
        text = text.replace('.', ' PERIOD')
        text = text.lower()
        rem_punct = str.maketrans('', '', string.punctuation)
        text = text.translate(rem_punct)
        return text

    def text_encoding(self, text):
        text = self.text_process(text)
        txt_list = list(filter(None, text.split()))
        txt_list_no_duplicates = list(set(txt_list))
        self.EncodedTxtList = self.Encoder.fit_transform(txt_list_no_duplicates)
        




    

