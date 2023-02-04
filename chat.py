import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r") as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
input_s = data["input"]
hidden = data["hidden"]
output = data["output"]
all_words = data["words"]
tags = data["tags"]
modelst = data["model_state"]

model = NeuralNet(input_s, hidden, output).to(device)
model.load_state_dict(modelst)
model.eval()

bot_name ="Buddy"

def response(message):
    sent = tokenize(message)
    X = bag_of_words(sent, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim = 1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "I don't get it : ( Why don't you tell me one more time, please."