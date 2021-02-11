import matplotlib.pyplot as plt
import numpy as np
import logging as lg
import tensorflow as tf
import pandas as pd
import time
import argparse
import config
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.models import load_model

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
ap.add_argument(
    "-d",
    "--dataset",
    type=str,
    required=True,
    help="path to the .csv file training dataset",
)
ap.add_argument(
    "-p",
    "--plot",
    type=str,
    default="./plot",
    help="path to folder to plot the model loss/accuracy and test graph",
)
ap.add_argument(
    "-t",
    "--test",
    type=str,
    default="n",
    help="y/[n] if the user wants to test the model",
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
dataset = pd.read_csv(args["dataset"], header=None, sep=",")
data = dataset[1]

# partition the data into training and testing splits
# 5-day prediction using 30 days data
lg.info("Preparing dataset for training...")

n_future = config.N_FUTUR  # Next days weather forecast
n_past = config.N_PAST  # Past days to predict

if args["test"]=='y': # if the user asks for a test process
    split_idx = int(len(data) - n_past - n_future)
else: # if not, take more data
    split_idx = int(len(data))

training_set = data[:split_idx]

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
model.add(LSTM(units=15, input_shape=(x_train.shape[1], 1), dropout=.1))
model.add(Dense(units=n_future))

# Compiling the RNN
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

# # train the model
lg.info("Start training...")

M = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BS, validation_split=.15)

lg.info(f"End of the model training within {round(time.time()-start,2)} seconds.")

# serialize the model to disk
lg.info("Saving model...")
model.save("./model.h5", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot()
plt.grid()
ax1.set_title("Training Loss/Accuracy")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss (mean square error)", color='red')
ax1.set_yscale("log", base=10)
ax1.plot(np.arange(0, N), M.history["loss"], '--', color="red", lw=1.5, label='training')
ax1.plot(np.arange(0, N), M.history["val_loss"], '-.', color="red", lw=1.5, label='validation')
ax1.tick_params(axis="y", color='red')
ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy (percent)", color="green")
ax2.plot(np.arange(0, N), M.history["accuracy"], '--', color="green", label='training')
ax2.plot(np.arange(0, N), M.history["val_accuracy"], '-.', color="green", label='validation')
ax2.tick_params(axis="y", color="green")

fig.legend()
fig.tight_layout()
plt.savefig(args["plot"] + "/model_stat.png")

if args["test"]=='y':
    # test the model
    lg.info("Testing the model...")

    model = load_model("model.h5")

    test_set = data[split_idx:].to_numpy()
    test_set = np.array(test_set)
    test_x = test_set[:-n_future]
    test_y = test_set[-n_future:]
    to_predict = np.reshape(test_x, (1, test_x.shape[0], 1))
    pred = model.predict(to_predict)

    # visualize the predictions
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot()
    plt.grid()
    ax1.set_title(f"Original vs predicted data for the {n_past} past times of the test")
    ax1.set_xlabel("time")
    ax1.set_ylabel("temperature")
    ax1.plot(test_y, "o--", color="red", label="Original data", lw=1.5)
    ax1.plot(pred[0], "o--", color="cyan", label="Predicted data", lw=1.5)
    ax1.tick_params(axis="y")
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel("temperature difference", color="green")
    ax2.bar(
        range(len(test_y)),
        np.abs(np.array(test_y) - np.array(pred[0])),
        color="green",
        width=0.5,
        alpha=0.2,
    )
    ax2.tick_params(axis="y", color="green")

    fig.tight_layout()
    plt.savefig(args["plot"] + "/test_plot.png")

lg.info(f"End of the script within {round(time.time()-start,2)} seconds.")