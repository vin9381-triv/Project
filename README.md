# Next Word Prediction Model using RNN

Hello readers, in this project we use a Recurrent Neural Network (RNN) combined with Long Short-Term Memory (LSTM) to predict the next word in a text sequence. RNNs are adept at handling sequential data, and when combined with LSTM, their capabilities increase significantly. LSTM adds memory to the RNN, allowing the model to "remember" context and retain important information over longer sequences.

We also use a Dense layer with a SoftMax activation function, which outputs probabilities for the next word. For training the data, we apply **One-Hot encoding**, a technique that assigns each unique word a unique index in a binary format.

With the basics covered, let's dive into the project by reviewing the files in our repository.

## Files in the Repo:

- **`1661-0.txt`**: The text corpus used for training the next-word prediction model.
- **`Model_Training.py`**: The script that trains the LSTM model on the text data.
- **`README.md`**: A documentation file providing an overview of the project and instructions for use.
- **`Text_Prediction.py`**: The script that loads the trained model and generates next-word predictions based on user input.
- **`history.p`**: A pickle file storing the training history of the model, including metrics like loss and accuracy.
- **`keras_next_word_model.h5`**: The saved LSTM model file containing the learned weights and architecture for next-word prediction.

## Packages and Tools Used:

- **Keras**: For building and training the neural network.
- **NLTK**: For tokenizing the text into words.
- **Numpy**: For numerical operations.
- **Pickle**: For saving and loading Python objects (like the model and word index).
- **Heapq**: For efficiently retrieving the top N predictions.

---

## Model Training Flow (`Model_Training.py`):

### 1. Importing the Necessary Libraries:
- **numpy**: For numerical operations.
- **nltk.tokenize**: For tokenizing the text into words.
- **keras**: For building and training the neural network.
- **pickle**: For saving and loading Python objects.
- **heapq**: For retrieving the top N predictions.

### 2. Loading and Pre-processing the Data:
- **a.** Load the text file and convert the entire dataset into lowercase for consistent processing.
- **b.** Tokenize the text using `RegexpTokenizer` to split sentences into individual words.
- **c.** Create a dictionary that stores all unique words encountered in the dataset and map them to unique indices.
- **d.** Use `pickle` to store the unique words and their indices for later use.

### 3. Preparing Input and Output Data:
- **a.** We set the sequence length to 5, meaning the model will use the last 5 words to predict the next word.
- **b.** Create data pairs `prev_words` and `next_words`. `prev_words` contains sequences of 5 words, and `next_words` contains the word following each sequence.

### 4. One-Hot Encoding:
- **a.** Create two arrays: 
  - **`X`** for input data (number of sequences, length of each sequence, number of unique words).
  - **`Y`** for output data (number of sequences, number of unique words).
- **b.** Populate `X` with sequences of 5 words represented as binary vectors using one-hot encoding, and `Y` with the next word in each sequence as a one-hot encoded binary vector.

### 5. Defining the Training Model:
- **a.** Initialize a Keras Sequential model.
- **b.** Add an LSTM layer with 128 units.
- **c.** Add a Dense output layer with a size equal to the number of unique words.
- **d.** Use the **SoftMax** activation function to output a probability distribution over all possible words.

### 6. Compiling and Training the Model:
- **a.** Use RMSprop as the optimizer with a learning rate of 0.01.
- **b.** Define the loss function as **categorical crossentropy** and track the **accuracy** metric.
- **c.** Fit the model on `X` and `Y`, using 5% of the data for validation.
- **d.** Save the trained model to reuse in other projects.

---

## Using the Trained Model for Text Prediction

### 1. Loading the Model and Unique Word Index:
- **a.** Load the trained LSTM model and the unique word index dictionary for text prediction.

### 2. Reverse Mapping:
- **a.** Perform reverse mapping on the unique word index (`word: index`), creating a new dictionary (`index: word`).
- **b.** This is necessary because the model predicts indices, not words. The reverse mapping makes it easy to retrieve the corresponding word from the predicted index.

### 3. Creating Prediction Functions:

 - ### a. Prepare Input Function: 
    This function takes a string as input, splits it into the last 5 words, and converts them into a one-hot encoded format for the model.

- ### b. Sample Function:
    This function converts the model’s output (a probability distribution over possible next words) and retrieves the top N predictions.

- ### c. Prediction Completion Function:
    Recursively predicts the next word based on the input text until the specified maximum length is reached.

- ### d. Prediction Completions Function:
    Similar to the `Prediction Completion` function, but this function outputs the top N most likely word completions and returns them.

---

## Sample Quotes and Prediction:
- **a.** A list of sample quotes is created.
- **b.** Loop through each quote, take the first 40 characters, and print the top N most likely word completions for each quote.

---
## Original Article Link:
This project was inspired by and referenced from the article below:
[https://thecleverprogrammer.com/2020/07/20/next-word-prediction-model/](https://thecleverprogrammer.com/2020/07/20/next-word-prediction-model/)

---
## Text Prediction From User Input:
In the `Text_Prediction.py` file, we use the trained model to create a script that takes user input and predicts the next words in the sequence. This script loads the trained model, prepares the input text, and predicts the next word using the LSTM model.

1. **Loading the Model and Mappings:**
   - The pre-trained LSTM model is loaded using `load_model('keras_next_word_model.h5')`.
   - The word-to-index and index-to-word mappings, which were saved during training, are loaded from the `unique_word_index.pkl` file. This allows the model to retrive words and their corresponding numeric indices.

2. **Preparing the Input:**
   - The user's input sentence is tokenized into words and then padded to match the model's input sequence length (in our case, 5 words).
   - 0ne-hot encoding is performed on the input sequence, where each word in the sequence is converted into a binary vector based on its index in the vocabulary.

3. **Predicting the Next Word:**
   - The one-hot encoded input is fed into the trained LSTM model, which predicts the probabilities for the next word in the sequence.
   - The word with the highest probability is chosen as the next word, we use the `np.argmax()` function to retrieve its index, which is then mapped back to the corresponding word.

4. **Generating a Sequence of Words:**
   - After predicting the next word, it is appended to the input sequence, and the process is repeated for a predefined number of words (e.g., 10 words) or until a punctuation mark (like `.`) is predicted.
   - The script continues to generate words recursively based on our input, creating a meaningful sentence or sequence of words.

5. **User Interaction:**
   - The script prompts the user for a sentence, which serves as the input text for the word prediction model.
   - After processing the input, the script outputs the predicted continuation of the sentence based on the learned patterns in the training data.

Thank you for following through! This project demonstrates how RNNs and LSTMs can be used for next-word prediction tasks by leveraging sequential data.

