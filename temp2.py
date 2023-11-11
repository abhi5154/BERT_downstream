import numpy as np
import pandas as pd

subsequence_to_count = 'ba'

def allx(word_dict, subsequence_to_count):

    substring_count = 0
    
    # Iterate through the words in the dictionary
    for word in word_dict:
        # Count the occurrences of 'ab' in each word
        substring_count += word.count(subsequence_to_count)

    return substring_count

print(allx(full_dictionary ,subsequence_to_count))
