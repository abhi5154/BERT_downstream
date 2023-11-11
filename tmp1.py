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

def fill_dict():
    
    
    
    for word in full_dictionary:
        
        word = word.lower()
        
        for i in range(len(word)):
            char     = word[i]
            curr_len = len(word)
            
            dict_1d[ord(char) - ord('a')] += 1.0#/curr_len
            
            if i > 0:
                prev_char = word[i - 1]
                dict_2d[ord(prev_char) - ord('a')][ord(char) - ord('a')] += 1.0#/(curr_len-1)
            
            if i > 0 and i < len(word) - 1:
                prev_char = word[i - 1]                
                next_char = word[i + 1]
                dict_3d[ord(prev_char) - ord('a')][ord(char) - ord('a')][ord(next_char) - ord('a')] += 1.0#/(curr_len-1)

            prev_prev_char = word[i - 2] if i >= 2 else None        
            if prev_prev_char is not None:
                dict_4d[ord(prev_prev_char) - ord('a')][ord(prev_char) - ord('a')][ord(char) - ord('a')][ord(next_char) - ord('a')] += 1.0


base_folder = "C:\\Users\\Abhishek\\Downloads\\Compressed\\trex"
os.chdir(base_folder)
full_dictionary_location = "words_250000_train.txt"
full_dictionary = build_dictionary(full_dictionary_location)        
full_dictionary_common_letter_sorted = collections.Counter("".join(full_dictionary)).most_common()



def get_best_match(curr_dict , filled_alphabets):
    
    relative_freq = np.zeros(26)
    dict_alphas  = set([chr(word + ord('a')) for word in np.arange(26) ])
    valid_alphas = list(dict_alphas.difference(filled_alphabets))
    valid_idxs   = [ord(x) - ord('a') for x in valid_alphas]
    
    for word in curr_dict:
        
        lenx = len(word)
        
        if lenx == 2:
            
            if word[0] == '_':
                y_axis       = ord(word[1]) - ord('a')
                max_idx      = valid_idxs[np.argmax(dict_2d[valid_idxs,y_axis])]
                relative_freq[max_idx] += 1
                
            if word[1] == '_':
                x_axis       = ord(word[0]) - ord('a')
                max_idx      = valid_idxs[np.argmax(dict_2d[x_axis,valid_idxs])]
                relative_freq[max_idx] += 1
                
        if lenx == 3:
            
            if word[0] == '_':
                y_axis       = ord(word[1]) - ord('a')
                z_axis       = ord(word[2]) - ord('a')
                max_idx      = valid_idxs[np.argmax(dict_3d[valid_idxs,y_axis,z_axis])]
                relative_freq[max_idx] += 1
                
            if word[2] == '_':
                x_axis       = ord(word[0]) - ord('a')
                y_axis       = ord(word[1]) - ord('a')
                max_idx      = valid_idxs[np.argmax(dict_3d[x_axis,y_axis,valid_idxs])]
                relative_freq[max_idx] += 1
                
            
    max_idx = np.argmax(relative_freq)
    max_val = chr(max_idx + ord('a'))
            
    
    return max_val

def get_best_match2(curr_dict , filled_alphabets):
    
    relative_freq = np.zeros(26)
    dict_alphas  = set([chr(word + ord('a')) for word in np.arange(26) ])
    valid_alphas = list(dict_alphas.difference(filled_alphabets))
    valid_idxs   = [ord(x) - ord('a') for x in valid_alphas]
    
    for word in curr_dict:
        
        lenx = len(word)
        
        if lenx == 2:
            
            if word[0] == '_':
                y_axis       = ord(word[1]) - ord('a')
                max_idx      = dict_2d[valid_idxs,y_axis]
                relative_freq[valid_idxs] += max_idx
                
            if word[1] == '_':
                x_axis       = ord(word[0]) - ord('a')
                max_idx      = dict_2d[x_axis,valid_idxs]
                relative_freq[valid_idxs] += max_idx
                
        if lenx == 3:
            
            if word[0] == '_':
                y_axis       = ord(word[1]) - ord('a')
                z_axis       = ord(word[2]) - ord('a')
                max_idx      = dict_3d[valid_idxs,y_axis,z_axis]
                relative_freq[valid_idxs] += max_idx
                
            if word[2] == '_':
                x_axis       = ord(word[0]) - ord('a')
                y_axis       = ord(word[1]) - ord('a')
                max_idx      = dict_3d[x_axis,y_axis,valid_idxs]
                relative_freq[valid_idxs] += max_idx
                
            
    max_idx = np.argmax(relative_freq)
    max_val = chr(max_idx + ord('a'))
            
    
    return max_val


def guess(curr_word ,filled_alphabets):
    
    ##
    tot_len          = len(curr_word)
    filled_len       = len([char for char in curr_word if char != '_'])
    empty_len        = tot_len - filled_len
    
    if tot_len == empty_len:
        
        dict_alphas  = set([chr(word + ord('a')) for word in np.arange(26) ])
        valid_alphas = dict_alphas.difference(filled_alphabets)
        valid_idxs   = [ord(x) - ord('a') for x in valid_alphas]
        # valid_probs  = dict_1d[valid_idxs]
        # valid_probs[list(vowels.keys())] = valid_probs[list(vowels.keys())]*VOWEL_FACTOR
        # valid_probs  = valid_probs/np.sum(valid_probs)
        # random_sample = np.random.choice(valid_idxs, size=1, p=valid_probs)
        # random_char   = list(valid_alphas)[random_sample[0]]
        
        random_char = list(valid_alphas)[np.argmax(dict_1d[valid_idxs])]

        return random_char
        

    if tot_len == 1:
        
        dict_alphas  = set([chr(word + ord('a')) for word in np.arange(26) ])
        valid_alphas = dict_alphas.difference(filled_alphabets)
        valid_idxs   = [ord(x) - ord('a') for x in valid_alphas]
        valid_probs  = dict_1d[valid_idxs]
        valid_probs  = valid_probs/np.sum(valid_probs)

        random_sample = np.random.choice(valid_idxs, size=1, p=valid_probs)
        random_char   = list(valid_alphas)[random_sample[0]]

        return random_char

    curr_dict = []

    if tot_len == 2:
        
        if curr_word[tot_len - 1] == '_' and curr_word[tot_len - 2]!= '_':
            curr_dict.append(curr_word[(tot_len-2):tot_len])
            
        if curr_word[0] == '_' and curr_word[1]!= '_':
            curr_dict.append(curr_word[0:2])
            
        return get_best_match(curr_dict , filled_alphabets)
        
        
    for i in range(1,len(curr_word)-1):
        
        curr_alphabet       = curr_word[i]
        prev_alphabet       = curr_word[i-1]
        next_alphabet       = curr_word[i+1]
        prev_prev_alphabet  = curr_word[i - 2] if i >= 2 else None        
        next_next_alphabet  = curr_word[i + 2] if i <= len(curr_word) - 3 else None        

        if  prev_alphabet != '_' and curr_alphabet == '_' :
            curr_dict.append(curr_word[(i-1):(i+1)])
            
        if  curr_alphabet == '_' and next_alphabet != '_' :
            curr_dict.append(curr_word[(i):(i+2)])

        if  prev_alphabet != '_' and curr_alphabet != '_' and  next_alphabet == '_':
            curr_dict.append(curr_word[(i-1):(i+2)])
            
        if  prev_alphabet == '_' and curr_alphabet != '_' and  next_alphabet != '_':
            curr_dict.append(curr_word[(i-1):(i+2)])

        if prev_prev_alphabet is not None:
            if  prev_prev_alphabet == '_' and prev_alphabet != '_' and curr_alphabet != '_' and  next_alphabet != '_':
                curr_dict.append(curr_word[(i-2):(i+2)])
            
        if next_next_alphabet is not None:
            if  prev_alphabet != '_' and curr_alphabet != '_' and  next_alphabet != '_' and next_next_alphabet == '_':
                curr_dict.append(curr_word[(i-1):(i+3)])

            
    if curr_word[tot_len - 1] == '_' and curr_word[tot_len - 2]!= '_':
        curr_dict.append(curr_word[(tot_len-2):tot_len])
        
    if curr_word[0] == '_' and curr_word[1]!= '_':
        curr_dict.append(curr_word[0:2])
    
    curr_dict = list(set(curr_dict))
    #print(curr_dict)
    
    outp = get_best_match(curr_dict , filled_alphabets)
    # outp = get_best_match2(curr_dict , filled_alphabets)
    
    return outp

def guess2(word,filled_alphabets,full_dictionary):

    # clean the word so that we strip away the space characters
    # replace "_" with "." as "." indicates any character in regular expressions
    clean_word = word[::2].replace("_",".")
    
    # find length of passed word
    len_word = len(clean_word)
    
    # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
    current_dictionary = full_dictionary
    new_dictionary = []
    
    # iterate through all of the words in the old plausible dictionary
    for dict_word in current_dictionary:
        # continue if the word is not of the appropriate length
        if len(dict_word) != len_word:
            continue
            
        # if dictionary word is a possible match then add it to the current dictionary
        if re.match(clean_word,dict_word):
            new_dictionary.append(dict_word)
    
    # overwrite old possible words dictionary with updated version
    # full_dictionary = new_dictionary
    
    
    # count occurrence of all characters in possible word matches
    full_dict_string = "".join(new_dictionary)
    
    c = collections.Counter(full_dict_string)
    sorted_letter_count = c.most_common()                   
    
    guess_letter = '!'
    
    # return most frequently occurring letter in all possible words that hasn't been guessed yet
    for letter,instance_count in sorted_letter_count:
        if letter not in filled_alphabets:
            guess_letter = letter
            break
        
    # if no word matches in training dictionary, default back to ordering of full dictionary
    if guess_letter == '!':
        sorted_letter_count = full_dictionary_common_letter_sorted
        for letter,instance_count in sorted_letter_count:
            if letter not in filled_alphabets:
                guess_letter = letter
                break            
    
    return guess_letter

def guess3(curr_word ,filled_alphabets):
    
    ##
    tot_len          = len(curr_word)
    filled_len       = len([char for char in curr_word if char != '_'])
    empty_len        = tot_len - filled_len
    
    if tot_len == empty_len:
        
        dict_alphas  = set([chr(word + ord('a')) for word in np.arange(26) ])
        valid_alphas = dict_alphas.difference(filled_alphabets)
        valid_idxs   = [ord(x) - ord('a') for x in valid_alphas]
        # valid_probs  = dict_1d[valid_idxs]
        # valid_probs[list(vowels.keys())] = valid_probs[list(vowels.keys())]*VOWEL_FACTOR
        # valid_probs  = valid_probs/np.sum(valid_probs)
        # random_sample = np.random.choice(valid_idxs, size=1, p=valid_probs)
        # random_char   = list(valid_alphas)[random_sample[0]]
        
        random_char = list(valid_alphas)[np.argmax(dict_1d[valid_idxs])]

        return random_char
        

    if tot_len == 1:
        
        dict_alphas  = set([chr(word + ord('a')) for word in np.arange(26) ])
        valid_alphas = dict_alphas.difference(filled_alphabets)
        valid_idxs   = [ord(x) - ord('a') for x in valid_alphas]
        valid_probs  = dict_1d[valid_idxs]
        valid_probs  = valid_probs/np.sum(valid_probs)

        random_sample = np.random.choice(valid_idxs, size=1, p=valid_probs)
        random_char   = list(valid_alphas)[random_sample[0]]

        return random_char

    curr_dict = []

    if tot_len == 2:
        
        if curr_word[tot_len - 1] == '_' and curr_word[tot_len - 2]!= '_':
            curr_dict.append(curr_word[(tot_len-2):tot_len])
            
        if curr_word[0] == '_' and curr_word[1]!= '_':
            curr_dict.append(curr_word[0:2])
            
        return get_best_match(curr_dict , filled_alphabets)
        
        
    for i in range(1,len(curr_word)-1):
        
        curr_alphabet       = curr_word[i]
        prev_alphabet       = curr_word[i-1]
        next_alphabet       = curr_word[i+1]
        prev_prev_alphabet  = curr_word[i - 2] if i >= 2 else None        
        next_next_alphabet  = curr_word[i + 2] if i <= len(curr_word) - 3 else None        

        if  prev_alphabet != '_' and curr_alphabet == '_' :
            curr_dict.append(curr_word[(i-1):(i+1)])
            
        if  curr_alphabet == '_' and next_alphabet != '_' :
            curr_dict.append(curr_word[(i):(i+2)])

        if  prev_alphabet != '_' and curr_alphabet != '_' and  next_alphabet == '_':
            curr_dict.append(curr_word[(i-1):(i+2)])
            
        if  prev_alphabet == '_' and curr_alphabet != '_' and  next_alphabet != '_':
            curr_dict.append(curr_word[(i-1):(i+2)])

        if prev_prev_alphabet is not None:
            if  prev_prev_alphabet == '_' and prev_alphabet != '_' and curr_alphabet != '_' and  next_alphabet != '_':
                curr_dict.append(curr_word[(i-2):(i+2)])
            
        if next_next_alphabet is not None:
            if  prev_alphabet != '_' and curr_alphabet != '_' and  next_alphabet != '_' and next_next_alphabet == '_':
                curr_dict.append(curr_word[(i-1):(i+3)])

            
    if curr_word[tot_len - 1] == '_' and curr_word[tot_len - 2]!= '_':
        curr_dict.append(curr_word[(tot_len-2):tot_len])
        
    if curr_word[0] == '_' and curr_word[1]!= '_':
        curr_dict.append(curr_word[0:2])
    
    curr_dict = list(set(curr_dict))
    #print(curr_dict)
    # outp = get_best_match(curr_dict , filled_alphabets)
    outp = get_best_match2(curr_dict , filled_alphabets)
    
    return outp

def update_word(curr_word, full_word, alphabet):
    
    updated_word = list(curr_word)
    
    for i, char in enumerate(curr_word):
        if curr_word[i] == '_' and full_word[i] == alphabet:
            updated_word[i] = alphabet
            
    return ''.join(updated_word)

def check_strategy(full_word , MAX_TRIES = 6):

    tot_len          = len(full_word)

    updated_word = list()
    for i, char in enumerate(full_word):
        updated_word.append('_')

    curr_word =  ''.join(updated_word)  # Convert the list back to a string    
    
    filled_len       = len([char for char in curr_word if char != '_'])
    empty_len        = tot_len - filled_len
    filled_alphabets = {char for char in curr_word if char.isalpha()}
    
    while MAX_TRIES > 0:
        
        # alphabet = guess(curr_word ,filled_alphabets)
        # alphabet = guess2(curr_word ,filled_alphabets,full_dictionary)
        alphabet = guess3(curr_word ,filled_alphabets)
        
        updated_word = list(curr_word)
        check = 0
        for i, char in enumerate(curr_word):
            if curr_word[i] == '_' and full_word[i] == alphabet:
                updated_word[i] = alphabet
                check = 1
                
        curr_word =  ''.join(updated_word)
        if check == 0:
            MAX_TRIES = MAX_TRIES - 1

        filled_alphabets.add(alphabet)
        print(MAX_TRIES," ",alphabet, " ", curr_word," ", filled_alphabets)
    
        filled_len       = len([char for char in curr_word if char != '_'])
        if filled_len == tot_len:
            return 1
        
    return 0



vowels  = {0:'a' ,4:'e' ,8:'i',14:'o',20:'u'}
dict_1d = np.zeros(shape = 26)
dict_2d = np.zeros(shape = (26,26))
dict_3d = np.zeros(shape = (26,26,26))
dict_4d = np.zeros(shape = (26,26,26,26))

# Call the fill_dict method to populate the dictionaries
fill_dict()
# dict_1d = dict_1d/len(full_dictionary)
# dict_2d = dict_2d/len(full_dictionary)
# dict_3d = dict_3d/len(full_dictionary)

ans = 0
NUM_WORDS  = 1000
for i in np.arange(NUM_WORDS):

    full_word = random.choice(full_dictionary)
    # full_word = 'worldspoiled'

    updated_word = list()
    for i, char in enumerate(full_word):
        updated_word.append('_')

    curr_word =  ''.join(updated_word)  # Convert the list back to a string    
    
    
    print(full_word)
    ans += check_strategy(full_word)
    
print(ans/NUM_WORDS)


