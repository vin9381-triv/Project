import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

# Define the path to the text file you want to read
path = '1661-0.txt'

# Open the file at the specified path, read its contents as a string, and convert it to lowercase
# 'encoding="utf-8"' ensures the file is read with UTF-8 encoding, which supports a wide range of characters
text = open(path, encoding='utf-8').read().lower()

# Print the length (number of characters) of the text in the file
print('corpus length:', len(text))

# Create a tokenizer that splits the text into words
# The RegexpTokenizer uses the regular expression r'\w+' to match sequences of word characters (letters, numbers, and underscores)
# This ensures that punctuation and special characters are ignored
tokenizer = RegexpTokenizer(r'\w+')

# Tokenize the text using the tokenizer created above
# This splits the text into a list of individual words based on the regular expression
words = tokenizer.tokenize(text)

# Find the unique words from the tokenized word list
# np.unique() returns a sorted array of the unique elements from the 'words' list (eliminating duplicates)
unique_words = np.unique(words)

# Create a dictionary that maps each unique word to an index
# enumerate(unique_words) assigns a unique index to each word, and dict() creates a dictionary with word-index pairs
# 'c' represents each word, and 'i' is the corresponding index
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

# Define the length of word sequences (number of words in each sequence)
WORD_LENGTH = 5

# Initialize two empty lists to store sequences of previous words and their corresponding next word
prev_words = []
next_words = []

# Loop through the 'words' list, but stop before the last 'WORD_LENGTH' words to avoid index out of range
# For each position in the 'words' list, create a sequence of 'WORD_LENGTH' words and store it in 'prev_words'
# The word that follows each sequence is stored in 'next_words'
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])  # Append a list of 'WORD_LENGTH' words
    next_words.append(words[i + WORD_LENGTH])    # Append the word that comes after the 'WORD_LENGTH' words

# Print the first sequence of previous words
print(prev_words[0])

# Print the first word that comes after the sequence
print(next_words[0])

# Initialize a 3D array 'X' of shape (number of sequences, WORD_LENGTH, number of unique words), filled with False (or 0)
# Each element in 'X' represents whether a specific word appears at a specific position in a sequence
X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)

# Initialize a 2D array 'Y' of shape (number of sequences, number of unique words), filled with False (or 0)
# 'Y' will store the one-hot encoded representation of the next word for each sequence
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)

# Loop over each sequence of previous words (prev_words) and their corresponding index
for i, each_words in enumerate(prev_words):
    # Loop over each word in the sequence and its position (index) within the sequence
    for j, each_word in enumerate(each_words):
        # Set the position in X to True (1) where the word appears in the sequence
        # This encodes the word using its index from unique_word_index
        X[i, j, unique_word_index[each_word]] = 1

    # Set the position in Y to True (1) for the next word that follows the current sequence
    # This represents the target word (the word to predict)
    Y[i, unique_word_index[next_words[i]]] = 1

# Initialize a Sequential model, which allows us to add layers one by one in a linear stack
model = Sequential()

# Add an LSTM layer with 128 units
# 'input_shape=(WORD_LENGTH, len(unique_words))' specifies the input shape of the data
# 'WORD_LENGTH' is the number of words in each sequence, and 'len(unique_words)' is the size of the vocabulary (one-hot encoded input)
model.add(LSTM(128, input_shape=(WORD_LENGTH, len(unique_words))))

# Add a Dense (fully connected) layer with 'len(unique_words)' units
# This layer will output a vector of probabilities, one for each unique word
model.add(Dense(len(unique_words)))

# Add a softmax activation function to the output layer
# Softmax converts the output vector into a probability distribution over all unique words
model.add(Activation('softmax'))

# Compile the model using RMSprop optimizer and categorical crossentropy as the loss function
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# Train the model with the input data (X) and the target data (Y)
# Use 5% of the data for validation, batch size of 128, and train for 2 epochs
history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=6, shuffle=True).history

# Save the trained model and the training history to disk
model.save('keras_next_word_model.h5')
pickle.dump(history, open("history.p", "wb"))

# Load the saved model and history for future predictions or further training
model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p", "rb"))

# Prepare the input by one-hot encoding a given sequence of words
def prepare_input(text):
    # Split the input text into words
    words = text.split()
    
    # Truncate the text to the last WORD_LENGTH words if it's longer than WORD_LENGTH
    if len(words) > WORD_LENGTH:
        words = words[-WORD_LENGTH:]
    
    # Pad the input with empty words if it's shorter than WORD_LENGTH
    x = np.zeros((1, WORD_LENGTH, len(unique_words)))
    
    for t, word in enumerate(words):
        if word in unique_word_index:  # Check if the word is in the dictionary
            x[0, t, unique_word_index[word]] = 1
            
    return x


# Sample the next word prediction based on the model's output
def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)

# Predict the completion of a given sequence of text
def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        
        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion
# Create reverse mapping for indices to words
indices_char = dict((i, c) for c, i in unique_word_index.items())

# Function to predict word completions
# Modify the predict_completion function to limit recursion
def predict_completions(text, max_len=20):
    if len(text) >= max_len:
        return ''
    
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 1)[0]
    next_char = indices_char[next_index]
    
    return next_char + predict_completion(text[1:] + next_char, max_len)



# Sample quotes to test the next word prediction
quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]

# Predict and print word completions for each sample quote
for q in quotes:
    seq = q[:40].lower()  # Use the first 40 characters of the quote
    print(seq)
    print(predict_completions(seq, 5))
    print()
