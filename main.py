#!/usr/bin/python3
from model import Model, relu, sigmoid
import numpy as np
import sys

IMAGE_MAGIC = 0x803
LABEL_MAGIC = 0x801

def parse_data(data_dir):
    with open(f"{data_dir}/train-images", "br") as file: train_images = file.read()
    with open(f"{data_dir}/train-labels", "br") as file: train_labels = file.read()
    with open(f"{data_dir}/test-images", "br")  as file: test_images  = file.read()
    with open(f"{data_dir}/test-labels", "br")  as file: test_labels  = file.read()

    if int.from_bytes(train_images[:4], "big") != IMAGE_MAGIC or \
       int.from_bytes(train_labels[:4], "big") != LABEL_MAGIC or \
       int.from_bytes(test_images[:4], "big")  != IMAGE_MAGIC or \
       int.from_bytes(test_labels[:4], "big")  != LABEL_MAGIC:
        sys.exit("invalid data")

    train_size = int.from_bytes(train_images[4:8], "big")
    test_size  = int.from_bytes(test_images[4:8], "big")

    print("formatting train images...")
    train_images = group_data(train_images[16:], train_size)
    print("formatting train labels...")
    train_labels = onehot_data(train_labels[8:])
    print("formatting test images...")
    test_images =  group_data(test_images[16:],  test_size)
    print("formatting test labels...")
    test_labels =  onehot_data(test_labels[8:])

    print("normalizing data...")
    return ((normalize(train_images),
             train_labels),

            (normalize(test_images),
             test_labels))

def group_data(data, group_count):
    grouped = []
    group_size = int(len(data) / group_count)

    for group_num in range(int(group_count)):
        start = group_num * group_size
        end = (group_num + 1) * group_size
        group = list(data[start:end])
        grouped.append(group)

    return grouped

# too lazy to make this readable
# takes the data 0 -> 255 and makes it -1 -> 1
# should make it play nicer w/ neural nets
def normalize(data):
    return [list(map(lambda x: (x / 255 * 2) - 1, group)) for group in data]

def onehot_data(data):
    onehots = []

    for item in data:
        onehot = [0] * 10
        onehot[item] = 1
        onehots.append(onehot)

    return onehots

def un_onehot(onehot):
    most = -1
    unencoded = -1
    for idx, value in enumerate(onehot):
        if value > most:
            most = value
            unencoded = idx
    return unencoded

train_data, test_data = parse_data("./data")
print("DONE!")

network = Model([784, 16, 16, 10], (relu, relu, sigmoid))
network.fit(train_data[0], train_data[1], batch_size=10, epochs=1, learn_rate=0.3)

correct = 0
wrong = 0

for image, label in zip(test_data[0], test_data[1]):
    prediction = un_onehot(network.predict(image))
    correct_answer = un_onehot(label)

    if prediction == correct_answer: correct += 1
    else: wrong += 1

accuracy = correct / len(test_data[0])
print(f"the model got {correct} right, {wrong} wrong, with an accuracy of {accuracy*100:3f}%")
