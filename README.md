# HiddenMarkovModelandviterbi

It is an implementation of Hidden Markov Model and Viterbi alogorithm using python.

# Overview
This model is based on concept of hidden markov model which uses forward and backword algorithm for text generation and viterbi algorithm for text prediction. Shakespeare-play data is used to train the model. Initially, the text was tokenized and encoded for further calculation of transition probability and emission probability. 

# Dataset
Shakespeare Plays : https://www.kaggle.com/kingburrito666/shakespeare-plays
AllLines.txt - This file contains the text corpus for model building

# Functioning
 1)Text processing
   Drop all punctuation except comma and period. Tokenize and encode all the words in Text_format.
 
 2)HMM parameters
  a)Initial probability: It contains probability of each word in given dataset.
  b)Transition Probability Matrix:It contains the probability of transition from one state to other state. Here each word corresponds to unique state.
  c)Emmision Probability Matrix: It contains the Probability of emmision of observations to the states. For ex, Given a state, It will give the next word.
  
 3)Text Generation
  Using forward and backward alogorithm, probability of each state(word) is calculated. Text is generated using the maximum probability of each word.
  
 4)Text Prediction
  Based on the sequence of words given by user and HMM parameters, Viterbi algorithm returns the sequence of most likely words following the given sequence.
  
# Structure

1)TextGenerate.py
  This is the startup file to call all the other funtions.
2)Main 
  a)HMM_Viterbi: A Class object which holds HMM Parameters and functions related to algorithms.
  b)Text_format: A Class object which holds functions for text processing of dataset
  c)HMM_Model: A class object to hold HMM_Viterbi and Text_format.

