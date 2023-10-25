# Import necessary Flask modules
from flask import Flask, render_template, request, jsonify

# Import JSON and the chatbot's response function
import json
from chatbot import response

# Import the Flask CORS extension for handling Cross-Origin Resource Sharing (CORS)
from flask_cors import CORS

# Load intents from the JSON file
intents = json.loads(open('intents.json').read())

# Create a Flask application
app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)

# Define a route for the root URL (GET request)
@app.route("/")
def index_get():
    return render_template("base.html")

# Define a route for the "/predict" endpoint (POST request)
@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    res = response(text)
    message = {"answer": res}
    return jsonify(message)



if __name__ == "__main__":
    app.run(debug=True)