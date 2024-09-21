### Next-Word Prediction using LSTM


🎉 Welcome to the world of **Next-Word Prediction**! 🚀

Imagine a smart assistant that can finish your sentences for you—pretty cool, right? This project showcases a deep learning model powered by an LSTM (Long Short-Term Memory) network, a brainy type of neural network designed to handle sequences like a pro! 

In this adventure, our model dives into a treasure trove of text data, learning to predict the next word based on what it just read. Picture it like a game of word association, where it picks up clues from the previous words to craft the perfect next word. 

The magic happens in its architecture: an LSTM layer that’s fantastic at remembering context, followed by a dense layer that uses a softmax activation function to dish out probabilities for potential next words. It’s like a little word chef, whipping up a delicious selection of choices! 

To train this genius, we use one-hot encoding—think of it as giving each word a unique badge so the model knows exactly who’s who in the lineup. 

So, buckle up as we explore the fascinating journey of next-word prediction, where technology meets language in the most exciting ways! 🌟✍️


### **Files in the repo:**
- **1661-0.txt**: The text corpus used for training the next-word prediction model.
- **Model_Training.py**: The script that trains the LSTM model on the text data.
- **README.md**: A documentation file providing an overview of the project and instructions for use.
- **Text_Prediction.py**: The script that loads the trained model and generates next-word predictions based on user input.
- **history.p**: A pickle file storing the training history of the model, including metrics like loss and accuracy.
- **keras_next_word_model.h5**: The saved LSTM model file containing the learned weights and architecture for next-word prediction.

  
### **Technologies Used:**
- Keras for neural network implementation.
- NLTK for text tokenization.
- Python libraries such as NumPy and Pickle for data handling.

## Code Explanation (Model_Training.py)

### 1. Importing Libraries
- Import necessary libraries (`NumPy`, `nltk`, `Keras`, `pickle`, `heapq`).

### 2. Loading and Preprocessing Text Data
- Load text file (`1661-0.txt`) and convert it to lowercase.
- Tokenize text using `RegexpTokenizer`.
- Extract unique words and create word-to-index mapping (`unique_word_index`).
- Save the `unique_word_index` to a file for later use.

### 3. Creating Word Sequences for Training
- Extract sequences of `WORD_LENGTH` words and the subsequent target word.

### 4. One-Hot Encoding of Sequences
- One-hot encode input (`X`) and output (`Y`) data.

### 5. Building the LSTM Model
- Initialize a `Sequential` model with an `LSTM` layer.
- Add a `Dense` layer with `softmax` activation.

### 6. Compiling and Training the Model
- Compile the model using `RMSprop` and `categorical_crossentropy`.
- Train the model for 6 epochs with a batch size of 128.

### 7. Saving and Loading the Model
- Save the model and training history using `pickle`.
- Reload the model and the `unique_word_index` mapping for future use.

### 8. Preparing Input for Prediction
- Process input text by one-hot encoding and trimming to the last `WORD_LENGTH` words.

### 9. Sampling Predictions
- Use the `sample()` function to select the top `n` predictions.

### 10. Predicting Word Completion
- Generate predicted word completions using `predict_completion()` and `predict_completions()` functions.

### 11. Testing the Model with Sample Quotes
- Test the model with sample quotes and print predictions.


## Code Explanation (Text_Prediction.py)

### 1. Importing Libraries
- Import necessary libraries (`NumPy`, `Keras`, `pickle`).

### 2. Loading the Model and Word Mappings
- Load the pre-trained model (`keras_next_word_model.h5`).
- Load the saved word-to-index mapping (`unique_word_index.pkl`) and create reverse index mapping (`indices_char`).

### 3. Setting Sequence Length
- Define the sequence length (`max_sequence_length`) for the model.

### 4. Preparing Input for Prediction
- Tokenize the input text into words.
- Truncate or pad the sequence to fit the required length (`max_sequence_length`).
- One-hot encode the input text sequence using the `unique_word_index`.

### 5. Predicting the Next Word
- Predict the next word by feeding the input sequence into the model.
- Use `argmax` to get the index of the word with the highest probability.
- Convert the predicted index back to the word using the reverse mapping (`indices_char`).

### 6. Generating a Sequence of Predicted Words
- Generate a sequence of words starting with the input text.
- Continuously predict and append words until the desired number of words is generated or an end punctuation is reached.

### 7. Example Usage
- Accept user input and predict the next word(s) using the pre-trained model.

Feel free to clone the repo and try the prediction on different text inputs!
