import numpy as np
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer

# Load the saved model
model = load_model('keras_next_word_model.h5')

# Define the path to your text data (corpus)
path = '1661-0.txt'  # Make sure this file exists in your working directory

# Read the corpus (text file) and convert it to lowercase
text = open(path, encoding='utf-8').read().lower()

# Tokenize the text into words using RegexpTokenizer (removes punctuation)
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

# Get unique words from the corpus
unique_words = np.unique(words)

# Word to index mapping (for preparing input for the model)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))  # Word-to-index dictionary
indices_char = dict((i, c) for c, i in unique_word_index.items())  # Index-to-word dictionary (reverse mapping)

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
