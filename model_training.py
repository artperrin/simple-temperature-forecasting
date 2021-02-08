import matplotlib.pyplot as plt
import numpy as np
import logging as lg
import tensorflow as tf
import pandas as pd
import time
import argparse
import config
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# general setup
lg.getLogger().setLevel(lg.INFO)
lg.getLogger("matplotlib.blocking_input").setLevel(lg.WARNING)

# tensorflow setup to work with GPU
gpus = tf.config.experimental.list_physical_devices("GPU")  # to work with GPU
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",type=str,required=True,help="path to the .csv file training dataset",
)
ap.add_argument("-p","--plot",type=str,default="./plot",help="path to folder to plot the model loss/accuracy and test graph",
)
args = vars(ap.parse_args())

# starting of the program
lg.info("Program starting :")
start = time.time()

# initialize the number of epochs to train for and batch size
EPOCHS = config.N_EPOCHS
BS = config.BS

# reading data
lg.info("Reading the data...")

# it is assumed that the data are written in the second column, separated by commas, with no header
dataset = pd.read_csv(args["dataset"], header=None, sep=',')
# dataset = pd.read_csv('data/weather_processed.csv', header=None, sep=',')
data = dataset[1]

# partition the data into training and testing splits
# 5-day prediction using 30 days data
lg.info("Preparing dataset for training...")

train_split = config.TRAIN_SPLIT
split_idx = int(len(data) * train_split)
training_set = data[:split_idx]

n_future = config.N_FUTUR # Next days weather forecast
n_past = config.N_PAST # Past days to predict

x_train = []
y_train = []
for i in range(0, len(training_set) - n_past - n_future + 1):
    x_train.append(training_set[i : i + n_past])
    y_train.append(training_set[i + n_past : i + n_past + n_future])

# format the data
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# build the RNN model
lg.info("Building of the model...")

model = Sequential()
model.add(
    Bidirectional(
        LSTM(units=30, return_sequences=True, input_shape=(x_train.shape[1], 1))
    )
)
model.add(Dropout(0.2))
model.add(LSTM(units=30, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=30, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=30))
model.add(Dropout(0.2))
model.add(Dense(units=n_future, activation="linear"))

# Compiling the RNN
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

# train the model
lg.info("Start training...")

M = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BS, validation_split=0.1)

lg.info(f"End of the model training within {round(time.time()-start,2)} seconds.")

# serialize the model to disk
lg.info("Saving mask detector model...")
model.save("./model.h5", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), M.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"]+'/model_stat.png')

# test the model
lg.info('Preparing data for testing...')

test_set = data[split_idx:].to_numpy()

x_test = test_set[: n_past]
y_test = test_set[n_past : n_past + n_future]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (1, x_test.shape[0], 1))

lg.info("Evaluating network...")

pred = model.predict(x_test)

# visualize the predictions
plt.figure()
plt.plot(y_test, 'o--', color="red", label="Original data")
plt.plot(pred[0], 'o--', color="cyan", label="Predicted data")
plt.legend()
plt.savefig(args["plot"]+'/test_plot.png')

lg.info(f"End of the script within {round(time.time()-start,2)} seconds.")