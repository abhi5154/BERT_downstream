import json
import requests
import random
import string
import secrets
import time
import re
import collections
import os
import numpy as np
import pandas as pd


def build_dictionary(dictionary_file_location):
    text_file = open(dictionary_file_location,"r")
    full_dictionary = text_file.read().splitlines()
    text_file.close()
    return full_dictionary

base_folder = "C:\\Users\\Abhishek\\Downloads\\Compressed\\trex"
os.chdir(base_folder)
VOWEL_FACTOR = 1000
full_dictionary_location = "words_250000_train.txt"
full_dictionary = build_dictionary(full_dictionary_location)        

def feature_maker(curr_word):
    
    tot_len          = len(curr_word)
    filled_len       = len([char for char in curr_word if char != '_'])
    empty_len        = tot_len - filled_len

    char_array = np.zeros(26)
    
    for letter in curr_word:
        if letter!= '_':
            char_array[ord(letter) - ord('a')] += 1
    
    feature_array       = np.zeros(29)
    feature_array[0]   = tot_len
    feature_array[1]   = filled_len
    feature_array[2]   = empty_len
    feature_array[3:29]   = char_array

    return feature_array

def label_maker(curr_word ,full_word):
        
    char_array = np.zeros(26)
    
    for i,letter in enumerate(curr_word):
        if letter == '_':
            char_array[ord(full_word[i]) - ord('a')] = 1
    
    return char_array

def word_splitter(full_word):
    
    tot_len = len(full_word)
    remaining_letters = random.randint(0, tot_len - 1)
    chosen_letters_indices = random.sample(range(tot_len), remaining_letters)
    curr_word = [full_word[i] if i in chosen_letters_indices else '_' for i in range(tot_len)]
    curr_word = ''.join(curr_word)
        
    return curr_word , full_word
    

def feature_label_maker():
    TOTX = 100000
    
    feature_array = np.zeros(shape = (TOTX ,29))
    label_array   = np.zeros(shape = (TOTX ,26))
    
    for i in np.arange(TOTX):
        full_word             = random.choice(full_dictionary)
        curr_word , full_word =  word_splitter(full_word)
        features              = feature_maker(curr_word)
        labels                = label_maker(curr_word ,full_word)
        feature_array[i,:]    = features
        label_array[i,:]      = labels
        
    return feature_array , label_array

feature_array , label_array =  feature_label_maker()
    
    




