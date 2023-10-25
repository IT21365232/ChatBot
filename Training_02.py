import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()  # Initialize a lemmatizer object

intents = json.loads(open('intents.json').read())  # Load the intents from a JSON file

words = []     # Create empty lists to store words, classes, and documents
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']  # Characters to ignore

# Iterate through the intents in the JSON data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)  # Tokenize the patterns into words
        words.extend(wordList)  # Add the words to the words list
        documents.append((wordList, intent['tag']))  # Create a document with words and its associated tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Add the tag to the classes list

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]  # Lemmatize words and remove ignored characters
words = sorted(set(words))  # Sort and make a set to remove duplicates

classes = sorted(set(classes))  # Sort classes and make a set to remove duplicates

# Save the processed words and classes to files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []  # Create an empty list to store training data
outputEmpty = [0] * len(classes)  # Create a list of zeros for the output

# Iterate through the documents to create training data
for document in documents:
    bag = []  # Create a bag to represent the presence of words
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]  # Lemmatize and lowercase words
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)  # Append 1 if word is in the patterns, else 0

    outputRow = list(outputEmpty)  # Create an output row with zeros
    outputRow[classes.index(document[1])] = 1  # Set the appropriate class to 1 in the output row
    training.append(bag + outputRow)  # Add the bag and output row to training data

random.shuffle(training)  # Shuffle the training data
training = np.array(training)  # Convert the training data to a NumPy array

trainX = training[:, :len(words)]  # Split the training data into input (X) and output (Y)
trainY = training[:, len(words):]

model = tf.keras.Sequential()  # Create a sequential neural network model

# Add layers to the model
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # Define the SGD optimizer

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # Compile the model

# Train the model using the training data
hist = model.fit(trainX, trainY, epochs=300, batch_size=5, verbose=1)

# Save the trained model to a file
model.save('chatbot_model.h5', hist)
print('Done')  # Print a message to indicate that the training is complete
