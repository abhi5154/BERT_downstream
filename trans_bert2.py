from transformers import BertTokenizer, BertForMaskedLM
import torch
import string ,random
import numpy as np
import torch.nn.functional as F
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



base_folder = "C:\\Users\\Abhishek\\Downloads\\Compressed\\trex\\"

def build_dictionary(dictionary_file_location):
    text_file = open(dictionary_file_location,"r")
    full_dictionary = text_file.read().splitlines()
    text_file.close()
    return full_dictionary

def adjust_temperature(logits, temperature=1.0):
    # Apply temperature scaling to logits
    scaled_logits = logits / temperature

    # Apply softmax to obtain probabilities
    probabilities = F.softmax(scaled_logits, dim=-1)

    return probabilities

def bert_guess_func(curr_word, guessed_letters ,tokenizer, model):
    
    curr_word2 = " ".join(list(curr_word))
    tokenized_sequence = tokenizer.tokenize(curr_word2.replace('_', '[MASK]'))
    # print(tokenized_sequence)

    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sequence)
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_ids)
    predictions = outputs.logits[0, -1, :]
    if len(guessed_letters) < 2:
        temperature   = 1.0
        probabilities = adjust_temperature(predictions, temperature)
    else:
        temperature   = 1.0
        probabilities = adjust_temperature(predictions, temperature)
    
    alphabet_characters = set(string.ascii_lowercase)
    filtered_probs = [prob.item() if token in alphabet_characters and token not in guessed_letters and token not in curr_word else float('-inf') for prob, token in zip(probabilities, tokenizer.convert_ids_to_tokens(range(len(probabilities))))]
    predicted_index = filtered_probs.index(max(filtered_probs))
    predicted_letter = tokenizer.convert_ids_to_tokens([predicted_index])

    return predicted_letter

def bert_guess_func2(curr_word, guessed_letters ,tokenizer, model):
    
    # curr_word2 = " ".join(list(curr_word))
    tokenized_sequence = tokenizer.tokenize(curr_word.replace('_', '[MASK]'))
    # print(tokenized_sequence)

    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sequence)
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    probabilities = F.softmax(logits, dim=-1)
    
    filtered_probs = [
        prob.item() if token.isalpha() and token.isascii() else 0
        for prob, token in zip(probabilities, tokenizer.convert_ids_to_tokens(range(len(probabilities))))
    ]

    non_zero_tokens = [token for token, prob in zip(tokenizer.convert_ids_to_tokens(range(len(probabilities))), filtered_probs) if prob != 0]
    all_text = ' '.join(non_zero_tokens)
    alphabet_counts = Counter(char.lower() for char in all_text if char.isalpha())

    for letter in guessed_letters:
        del alphabet_counts[letter]

    most_common_alphabet, most_common_count = alphabet_counts.most_common(1)[0]

    return most_common_alphabet

def bert_guess_func_train(curr_word, filled_alphabets ,tokenizer, model):
    
    curr_word2 = " ".join(list(curr_word))
    tokenized_sequence = tokenizer.tokenize(curr_word2.replace('_', '[MASK]'))
    tokenized_sequence = tokenized_sequence[:max_sequence_length] + ['[PAD]'] * (max_sequence_length - len(tokenized_sequence))
    
    # print(tokenized_sequence)

    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sequence)
    
    input_ids = torch.tensor([input_ids])
    
    with torch.no_grad():
        outputs = model(input_ids)
        
    predictions = outputs.logits[0, :]
    probabilities = F.softmax(predictions, dim=-1)
    
    # if len(guessed_letters) < 2:
    #     temperature   = 1.0
    #     probabilities = adjust_temperature(predictions, temperature)
    # else:
    #     temperature   = 1.0
    #     probabilities = adjust_temperature(predictions, temperature)

    dict_alphas  = set([chr(word + ord('a')) for word in np.arange(26) ])
    valid_alphas = list(dict_alphas.difference(filled_alphabets))
    valid_idxs   = [ord(x) - ord('a') for x in valid_alphas]
    max_idx      = valid_idxs[np.argmax(probabilities[valid_idxs])]

    alphabet      = chr(max_idx + ord('a'))
    

    return alphabet



class data_maker:
    
    def __init__(self,full_dictionary ,TOTX , feature_type ,NUM_FEATURES):
        
        self.full_dictionary = full_dictionary
        self.TOTX  = TOTX        
        self.NUM_FEATURES = NUM_FEATURES
        self.feature_type = feature_type

    def label_maker(self,curr_word ,full_word):
            
        char_array = np.zeros(26)
        
        for i,letter in enumerate(curr_word):
            if letter == '_':
                char_array[ord(full_word[i]) - ord('a')] = 1
        
        return char_array

                    
    def word_splitter(self,full_word):
        
        tot_len = len(full_word)
        remaining_letters = random.randint(0, tot_len - 1)
        chosen_letters_indices = random.sample(range(tot_len), remaining_letters)
        curr_word = [full_word[i] if i in chosen_letters_indices else '_' for i in range(tot_len)]
        curr_word = ''.join(curr_word)
            
        return curr_word , full_word

    def word_splitter2(self,full_word):
        
        tot_len = len(full_word)
        remaining_letters = random.randint(0, tot_len - 1)
        chosen_letters_indices = random.sample(range(tot_len), remaining_letters)
        curr_word = [full_word[i] if i in chosen_letters_indices else '_' for i in range(tot_len)]
        curr_word = ''.join(curr_word)

        non_curr_word = [full_word[i] if i not in chosen_letters_indices else '' for i in range(tot_len)]
        non_curr_word = ''.join(non_curr_word)
            
        return curr_word , full_word ,non_curr_word

        
    def update_word(self,curr_word, full_word, alphabet):
        
        updated_word = list(curr_word)
        
        for i, char in enumerate(curr_word):
            if curr_word[i] == '_' and full_word[i] == alphabet:
                updated_word[i] = alphabet
                
        return ''.join(updated_word)
            
    def check_strategy(self ,curr_word ,full_word,NUM_TRIES ,tokenizer,model):
        
        tot_len          = len(curr_word)        
        filled_len       = len([char for char in curr_word if char != '_'])
        empty_len        = tot_len - filled_len
        filled_alphabets = {char for char in curr_word if char.isalpha()}
        
        while NUM_TRIES>0:
            
            # alphabet = bert_guess_func(curr_word ,filled_alphabets ,tokenizer,model)[0]
            # alphabet = bert_guess_func2(curr_word ,filled_alphabets ,tokenizer,model)[0]
            alphabet = bert_guess_func_train(curr_word ,filled_alphabets ,tokenizer,model)[0]
            
            updated_word = list(curr_word)
            check = 0
            for i, char in enumerate(curr_word):
                if curr_word[i] == '_' and full_word[i] == alphabet:
                    check = 1
                    updated_word[i] = alphabet
    
            curr_word =  ''.join(updated_word)
            if check == 0:
                NUM_TRIES = NUM_TRIES - 1
                        
            print(NUM_TRIES," " ,alphabet, " " , curr_word, " " ,filled_alphabets)
            filled_alphabets.add(alphabet)
            # print(filled_alphabets)
        
            filled_len       = len([char for char in curr_word if char != '_'])
            if filled_len == tot_len:
                return 1
            
        return 0

    def test(self ,NUM_TRIES , NUM_WORDS ,tokenizer,model):
        
        tot_ans = 0
        for i in np.arange(NUM_WORDS):
            
            NUM_TRIES = 6        
            full_word = random.choice(self.full_dictionary)
            # full_word = 'naieve'
        
            updated_word = list()
            for i, char in enumerate(full_word):
                updated_word.append('_')
        
            curr_word =  ''.join(updated_word)  # Convert the list back to a string    
                        
            print(full_word)
            ans = self.check_strategy(curr_word,full_word,NUM_TRIES ,tokenizer,model)
            print(ans)
            tot_ans += ans
        print("ratio " ,tot_ans/NUM_WORDS)
                
    def get_masked_words(self,NUM_WORDS):

        masked_words = []
        full_words   = []
        non_curr_words   = []
        
        label_array   = np.zeros(shape = (NUM_WORDS ,26))
            
        for i in np.arange(NUM_WORDS):
            full_word             = random.choice(self.full_dictionary)
            curr_word , full_word ,non_curr_word = self.word_splitter2(full_word)

            masked_words.append(curr_word)
            full_words.append(full_word)
            non_curr_words.append(non_curr_word)
            labels                = self.label_maker(curr_word ,full_word)
            label_array[i,:]      = labels

        return masked_words , label_array ,full_words


full_dictionary = build_dictionary(base_folder + "words_250000_train.txt")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')

DICTIONARY_SIZE = 1000
FEATURE_TYPE    = 0
NUM_FEATURES    = 30 # 706
NUM_TRIES       = 6
NUM_WORDS       = 50
num_classes     = 26

curr_word = "rock__ar"

dm = data_maker(full_dictionary ,DICTIONARY_SIZE , FEATURE_TYPE , NUM_FEATURES)



train_words ,label_words ,full_words = dm.get_masked_words(1000)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Tokenize and encode the data
tokenized_inputs = []
labels = []

max_sequence_length = 27

for ii in np.arange(len(full_words)):
    
    partial_word        = train_words[ii]

    partial_word = " ".join(list(partial_word))

    # Tokenize the partial word
    tokenized_sequence = tokenizer.tokenize(partial_word.replace('_', '[MASK]'))
    tokenized_sequence = tokenized_sequence[:max_sequence_length] + ['[PAD]'] * (max_sequence_length - len(tokenized_sequence))

    # tokenized_labels = tokenizer.encode(removed_alphabets, add_special_tokens=False)
    # tokenized_labels = tokenized_labels[:max_sequence_length] + [tokenizer.pad_token_id] * (max_sequence_length - len(tokenized_labels))


    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sequence)
    # label_ids = tokenizer.convert_tokens_to_ids(tokenized_labels)

    tokenized_inputs.append(input_ids)
    # labels.append(label_ids)

# Convert to PyTorch tensors
input_ids_tensor = torch.tensor(tokenized_inputs)
labels_tensor = torch.tensor(label_words)

# Create a DataLoader
train_dataset = TensorDataset(input_ids_tensor, labels_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define loss function and optimizer
criterion = BCEWithLogitsLoss()

optimizer = Adam(model.parameters(), lr=2e-5)  # You may need to adjust the learning rate

# Training loop
num_epochs = 5  # Adjust as needed

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_dataloader):
        inputs, labels = batch

        # Forward pass
        outputs = model(inputs)
        logits = outputs.logits

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print loss at each step
        print(f'Epoch {epoch + 1}/{num_epochs}, Step {step + 1}/{len(train_dataloader)}, Loss: {loss.item()}')


    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')

# Save the trained model
model.save_pretrained('fine_tuned_bert_model')

dm.test(NUM_TRIES,NUM_WORDS,tokenizer,model)



