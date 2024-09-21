### ***Next-Word Prediction using LSTM***

### **Description:**
This project demonstrates a deep learning model for next-word prediction using an LSTM (Long Short-Term Memory) network. The model is trained on text data, where it learns to predict the next word in a sequence based on the previous words. The architecture includes an LSTM layer followed by a dense layer with a softmax activation function to output probabilities for the next word. The training process utilizes one-hot encoding to represent words and sequences.

### **Features:**
- Text preprocessing with tokenization.
- Sequence generation for training the model.
- LSTM-based architecture for sequential data prediction.
- One-hot encoding of words for input and output layers.
- Word completion and prediction functions.
  
### **Technologies Used:**
- Keras for neural network implementation.
- NLTK for text tokenization.
- Python libraries such as NumPy and Pickle for data handling.

### **Project.py - Explanation:**
The code in `project.py` demonstrates the process of building and training an LSTM-based next-word prediction model. Below is a step-by-step explanation of each section of the code:

### 1. Importing Libraries
The code begins by importing the necessary libraries:
- `NumPy` for numerical computations.
- `nltk` for tokenizing the text.
- `Keras` for building the LSTM model.
- `pickle` for saving the model's training history.
- `heapq` for selecting the top predictions.

### 2. Loading and Preprocessing Text Data
- The text file (`1661-0.txt`) is loaded, and its content is converted to lowercase to ensure uniformity.
- Tokenization is performed using `RegexpTokenizer` to split the text into words, ignoring punctuation.
- Unique words from the text are extracted using `np.unique()`, and a dictionary (`unique_word_index`) is created to map each word to a unique index.

### 3. Creating Word Sequences for Training
- The training data consists of sequences of `WORD_LENGTH` words (`prev_words`), where the model learns to predict the next word (`next_words`).
- For every position in the text, a sequence of `WORD_LENGTH` words is extracted, along with the word that follows the sequence.

### 4. One-Hot Encoding of Sequences
- The input (`X`) and output (`Y`) data are one-hot encoded:
  - `X` is a 3D array where each sequence of words is represented by a binary matrix indicating whether a word is present at a specific position in the sequence.
  - `Y` is a 2D array where each row represents the one-hot encoded target word (the word to predict).

### 5. Building the LSTM Model
- A `Sequential` model is initialized with one `LSTM` layer containing 128 units, which processes the sequences of words.
- A `Dense` layer with the size of the vocabulary (number of unique words) is added, followed by a `softmax` activation function. This layer outputs a probability distribution for each word in the vocabulary.

### 6. Compiling and Training the Model
- The model is compiled using the `RMSprop` optimizer with a learning rate of 0.01 and `categorical_crossentropy` as the loss function, which is ideal for multi-class classification problems like word prediction.
- The model is trained on the input data (`X` and `Y`), using 5% of the data for validation. The training runs for 6 epochs with a batch size of 128. The `history` object stores the model's performance during training.

### 7. Saving and Loading the Model
- After training, the model is saved to a file (`keras_next_word_model.h5`) for future use. The training history is also saved using `pickle`.
- The saved model and history can be reloaded later using `load_model` and `pickle.load`, allowing for predictions or further training without retraining the model from scratch.

### 8. Preparing Input for Prediction
- The `prepare_input()` function processes an input text sequence by one-hot encoding it into a format compatible with the model.
- The function trims the input to the last `WORD_LENGTH` words and converts each word into its one-hot encoded form.

### 9. Sampling Predictions
- The `sample()` function selects the most probable next words from the model's output. It uses the softmax probabilities and returns the top `n` predictions, with `n` specified as a parameter (`top_n=3` by default).

### 10. Predicting Word Completion
- The `predict_completion()` function generates the predicted completion for a given text sequence. It iteratively predicts the next word based on the previous sequence and updates the sequence by appending the predicted word.
- The `predict_completions()` function further refines this, predicting multiple completions for a given sequence, up to a defined maximum length.

### 11. Testing the Model with Sample Quotes
- Several sample quotes are used to test the model’s next-word prediction. The first 40 characters of each quote are passed to the model, which predicts the subsequent words.
- The predictions are printed for evaluation.

**Usage:**
After training the model, users can input a sequence of text, and the model will predict the next word based on the learned patterns from the training data.

Feel free to clone the repo and try the prediction on different text inputs!
