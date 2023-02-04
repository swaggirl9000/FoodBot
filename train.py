import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open("intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []     #patterns and tags

for words in intents["intents"]:
    tag = words["tag"]
    tags.append(tag)
    for pattern in words["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", "!", ".", "-", ","]
all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

batch_size = 8
input_s = len(all_words)
hidden = 8
output = len(tags)
learning_rate = 0.001
num_epochs = 10000

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_s, hidden, output).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')
print(f'final loss = {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input": input_s,
    "output": output,
    "hidden": hidden, 
    "words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"training complete. file saved to {FILE}")
