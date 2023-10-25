import random
import json
import pickle
import numpy as np
import tensorflow as tf
import mysql.connector  # Import the database connector library
from PIL import Image

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()  # Initialize a WordNet lemmatizer

intents = json.loads(open('intents.json').read())  # Load chatbot intents from a JSON file

words = pickle.load(open('words.pkl', 'rb'))  # Load processed words from a pickle file
classes = pickle.load(open('classes.pkl', 'rb'))  # Load classes from a pickle file
model = load_model('chatbot_model.h5')  # Load the pre-trained chatbot model

bot_name = "Sam"  # Set the chatbot's name

# Define functions for processing user input

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize the user's sentence
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lemmatize words
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1  # Create a bag of words representation
    return np.array(bag)


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']  # Get the intent tag from the user's input
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])  # Choose a random response from the matched intent
            break
    return result


def predict_class(sentence):
    intents = json.loads(open('intents.json').read())
    bow = bag_of_words(sentence)  # Create a bag of words for the user's input
    rest = model.predict(np.array([bow]))[0]  # Use the pre-trained model to predict the intent
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(rest) if r > ERROR_THRESHOLD]  # Filter predictions based on a threshold

    result.sort(key=lambda x: x[1], reverse=True)  # Sort results by probability
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Create a list of predicted intents

    return return_list

# Define functions for handling database queries

def get_order_by_id(order_id):
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="irwa"
    )

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM orders WHERE OrderID = %s", (order_id,))
    order_data = cursor.fetchone()  # Fetch order data from the database

    return order_data


def handle_product_inquiry(ItemName):
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="irwa"
    )

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM menuitems WHERE ItemName = %s", (ItemName,))
    Items = cursor.fetchone()  # Fetch product details from the database

    return Items

# Define a response function for user input


def response(msg):

    global availability
    ints = predict_class(msg)  # Get predicted intents

    # Handle different intents with specific responses

    if ints[0]['intent'] == 'product_info':
        product_name = msg  # Assuming that the product name is the same as the user's input
        stopwords = ['laptop', 'can', 'you', 'tell', 'me', 'about', 'menu', 'information', 'details', 'I', 'wanna',
                     'know', 'about', 'info', 'give', 'please', 'foods', 'food', 'the']
        querywords = msg.split()
        resultwords = [word for word in querywords if word.lower() not in stopwords]
        itemName = ' '.join(resultwords)
        menudata = handle_product_inquiry(itemName)  # Query product details
        if menudata:
            if ( menudata[7] == 1) :
                res = f" Food : {menudata[1]}<br> Price: {menudata[4]}<br> Description: {menudata[2]}<br> Ingredients : {menudata[5]}<br> Available"
            else:
                res = f" Food : {menudata[1]}<br>, Price: {menudata[4]}<br>, Description: {menudata[2]}<br>, Ingredients : {menudata[5]}<br>, Not Available {menudata[7]}"
        else:
            res = "Order not found."

        return res

    if ints[0]['intent'] == 'track.order':
        product_name = msg  # Assuming that the product name is the same as the user's input
        stopwords = ['check', 'the', 'status', 'of', 'my', 'order', 'track', 'existing', 'order', 'my', 'is', 'id', 'want', 'to', ]

        querywords = msg.split()
        resultwords = [word for word in querywords if word.lower() not in stopwords]
        order = ' '.join(resultwords)
        orderdata = get_order_by_id(order)  # Query order details
        if orderdata:
            res = f"Order ID: {orderdata[0]}<br> Customer name : {orderdata[1]}<br> Orders: {orderdata[2]}<br> Delivaery Status : {orderdata[5]}\n"
        else:
            res = "Order not found. Please enter order ID"

        return res

    if ints[0]['intent'] == 'menu_image':
        img = Image.open("static/images/Menu Image.png")  # Open an image (assuming it's a PNG)
        return img.show() + get_response(ints, intents)  # Display the image and provide further responses

    if ints[0]['intent'] == 'Vegi':
        result_list = []

        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="irwa"
        )

        cursor = connection.cursor()
        cursor.execute("SELECT ItemName FROM menuItems WHERE SpecialFeatures = 'Vegetarian';")  # Query vegetarian items
        result = cursor.fetchall()

        for row in result:
            result_list.append(row[0])  # Create a list of vegetarian items

        return result_list

    else:
        return get_response(ints, intents)  # Return a response based on the predicted intents

print("Bot is running")