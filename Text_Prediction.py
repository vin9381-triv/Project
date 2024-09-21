import numpy as np
from keras.models import load_model
import pickle

# Load the saved model and the word mappings
model = load_model('keras_next_word_model.h5')

# Load the saved word-to-index and index-to-word mappings
with open('unique_word_index.pkl', 'rb') as f:
    unique_word_index = pickle.load(f)

indices_char = {i: c for c, i in unique_word_index.items()}  # Reverse mapping

# The length of word sequences (number of words in each sequence)
max_sequence_length = 5  # This is based on your model's input shape

# Function to prepare input text (word-level) for prediction
def prepare_input(text, max_sequence_length):
    # Tokenize the input text into words
    words = text.lower().split()
    
    # Truncate or pad the input sequence to fit the model's required length
    if len(words) > max_sequence_length:
        words = words[-max_sequence_length:]
    
    x = np.zeros((1, max_sequence_length, len(unique_word_index)))  # Shape: (1, max_sequence_length, vocab_size)
    
    # One-hot encode the words in the input sequence
    for t, word in enumerate(words):
        if word in unique_word_index:
            x[0, t, unique_word_index[word]] = 1  # One-hot encode the word at position t
    
    return x

# Function to predict the next word in the sequence
def predict_next_word(text):
    # Ensure the input is trimmed to the maximum sequence length
    if len(text.split()) > max_sequence_length:
        text = ' '.join(text.split()[-max_sequence_length:])
    
    # Prepare the input and predict the next word
    x = prepare_input(text, max_sequence_length)
    predictions = model.predict(x, verbose=0)[0]  # Predict probabilities
    
    # Get the index of the next word (argmax of predicted probabilities)
    next_index = np.argmax(predictions)
    next_word = indices_char[next_index]  # Map the index back to the word
    
    return next_word

# Function to generate a sequence of predicted words
def predict_sequence(seed_text, num_words=10):
    generated_text = seed_text
    for _ in range(num_words):
        next_word = predict_next_word(generated_text)
        generated_text += ' ' + next_word
        
        # Optionally stop generation if end punctuation is predicted
        if next_word in ['.', '\n']:
            break
    
    return generated_text

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a sentence: ")
    predicted_text = predict_sequence(user_input)
    print(f"Predicted continuation: {predicted_text}")
