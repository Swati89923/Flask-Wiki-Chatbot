from flask import Flask, render_template, request, jsonify
import torch
import random
import json
import wikipedia
import re
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from textblob import TextBlob   # ✅ Added for spell correction

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.json["message"]
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).to(device)
    X = X.reshape(1, X.shape[0])

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return jsonify({"response": random.choice(intent['responses'])})
    else:
        # ✅ Improved Wikipedia fallback with spell correction
        try:
            # Clean up query (remove filler words)
            query = re.sub(r"(tell me about|who is|what is|something about)", "", msg, flags=re.IGNORECASE).strip()

            if not query:
                return jsonify({"response": "Sorry, I couldn’t understand your question."})

            # Spell correction
            corrected_query = str(TextBlob(query).correct())

            # If correction was made, tell the user
            if corrected_query.lower() != query.lower():
                response_prefix = f"Showing results for '{corrected_query}':\n\n"
            else:
                response_prefix = ""

            summary = wikipedia.summary(corrected_query, sentences=2)
            return jsonify({"response": response_prefix + summary})

        except wikipedia.DisambiguationError as e:
            options = e.options[:5]  # Show only top 5 options
            return jsonify({"response": f"Your query is too broad. Did you mean: {', '.join(options)}?"})

        except wikipedia.PageError:
            suggestions = wikipedia.search(query)
            if suggestions:
                return jsonify({"response": f"Sorry, I couldn’t find an exact match. Did you mean: {', '.join(suggestions[:5])}?"})
            else:
                return jsonify({"response": "Sorry, I could not find anything about that."})

        except Exception:
            return jsonify({"response": "Sorry, something went wrong while searching Wikipedia."})


if __name__ == "__main__":
    app.run(debug=True)
