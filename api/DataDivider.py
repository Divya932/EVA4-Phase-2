import os
import random

def train_test_divide(data_folder):
    data = []
    labels = []

    for folder in os.listdir(data_folder):
        for files in os.listdir(data_folder + folder):
            data.append(data_folder + folder +"/" + files)
            labels.append(folder)


    #random shuffling data and labels together
    temp = list(zip(data, labels))
    random.shuffle(temp)
    data, labels = zip(*temp)

    #dividing both lists into train:test, i.e., 70:30
    train_len = int(0.70 * len(data))
    test_len = len(data) - train_len

    train_data = data[:train_len]
    train_label = labels[:train_len]

    test_data = data[-test_len:]
    test_label = data[-test_len:]

    print("Total data:", len(data))
    print("Total labels:", len(labels))

    return list(train_data), list(train_label), list(test_data), list(test_label)

        