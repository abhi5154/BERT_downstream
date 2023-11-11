from transformers import BertTokenizer, BertForMaskedLM
import torch
import string


def bert_predict(tokenizer, model, tokenized_sequence):
    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sequence)
    
    # Convert to PyTorch tensor
    input_ids = torch.tensor([input_ids])

    # Get BERT predictions
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Extract predicted probabilities for the masked token
    predictions = outputs.logits[0, -1, :]
    probabilities = torch.nn.functional.softmax(predictions, dim=0)

    return probabilities

def bert_guess_func(tokenizer, model, hangman_state, guessed_letters):
    # Tokenize the current state of the word
    tokenized_sequence = tokenizer.tokenize(hangman_state.replace('_', '[MASK]'))

    # Get BERT predictions
    probabilities = bert_predict(tokenizer, model, tokenized_sequence)

    # Filter out non-alphabetic characters and guessed letters
    alphabet_characters = set(string.ascii_lowercase)
    filtered_probs = [prob.item() if token in alphabet_characters and token not in guessed_letters and token not in hangman_state else float('-inf') for prob, token in zip(probabilities, tokenizer.convert_ids_to_tokens(range(len(probabilities))))]

    # Choose the letter with the highest probability among alphabetic characters
    predicted_index = filtered_probs.index(max(filtered_probs))
    predicted_letter = tokenizer.convert_ids_to_tokens([predicted_index])

    return predicted_letter


# Example usage:
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

full_word = ""
guessed_letters = "sitd"

hangman_state = "_a___"
bert_guess = bert_guess_func(tokenizer, model, hangman_state,guessed_letters)
print("BERT's guess:", bert_guess)


