import matplotlib.pyplot as plt
import numpy as np
import logging as lg
import tensorflow as tf
import pandas as pd
import time
import argparse
import config
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
    help="path to the .csv file",
)
ap.add_argument(
    "-p",
    "--plot",
    type=str,
    default="./plot",
    help="path to folder to plot the prediction",
)
ap.add_argument(
    "-e",
    "--extended",
    type=int,
    default=config.N_FUTUR,
    help="mode of prediction : if <= N_FUTUR, the model will predict the <= N_FUTUR days temperatures ; if >0, the model will predict more days based on a sliding day-to-day prediction.",
)
args = vars(ap.parse_args())

# check the selected mode
extended = False
if args["extended"] > config.N_FUTUR:
    extended = True
    units_to_predict = args["extended"]
else:
    units_to_predict = -1 # to ignore "possibly unbound" warning

# beginning of the script
lg.info("Reading the data...")
start = time.time()
# it is assumed that the data are written in the second column, separated by commas, with no header
dataset = pd.read_csv(args["dataset"], header=None, sep=",")
data = dataset[1].to_numpy()

# preparing the data to run the model
lg.info("Preparing dataset for running...")

if len(data) > config.N_PAST:
    lg.info(
        f"Dataset has {len(data)} entries, the model will run with the {config.N_PAST} last ones."
    )
    data = data[-config.N_PAST :]
elif len(data) < config.N_PAST:
    lg.error(f"Not enough data to run the model, {config.N_PAST - len(data)} missing.")

# formatting the data
data = np.array(data)

# applying the model to the data
lg.info("Loading the model...")
model = load_model("model.h5")

lg.info("Model loaded, running the model...")

if extended:
    res = []
    for i in range(units_to_predict):
        to_predict = data[i : i + config.N_PAST]
        to_predict = np.reshape(to_predict, (1, to_predict.shape[0], 1))
        predict = model.predict(to_predict)
        res.append(predict[0][0])
        data = np.concatenate((data, [predict[0][0]]), axis=None)

        # visualize the progress
        progress = np.ceil(i / units_to_predict * 50)  # progress from 0 to 50
        progress_line = ""
        for k in range(51):
            if k <= progress:
                progress_line += "="
            else:
                progress_line += "."

        print(
            "Progress of the predictions : [" + progress_line + "]",
            end="\r",
            flush=True,
        )
    print("The end ")
else:
    to_predict = data
    to_predict = np.reshape(to_predict, (1, to_predict.shape[0], 1))
    predict = model.predict(to_predict)
    res = predict[0][: args["extended"]]

lg.info(f"End of running process within {round(time.time()-start,2)} seconds.")

# plotting the results
lg.info("Exporting the output...")
nbtime = args["extended"]
plt.figure(figsize=(20, 10))
plt.grid()
plt.plot(range(1, len(res) + 1), res, "o--")
plt.title(f"Predicted temperatures for the next {nbtime} time units")
plt.xlabel("time")
plt.ylabel("temperature")
plt.savefig(args["plot"] + "/predictions.png")

lg.info(f"End of the script within {round(time.time()-start,2)} seconds.")