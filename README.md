# Temperature forecasting

This program forecasts temperatures for a given period of time, based on data over a given period of time.

This repository contains:
* the model_training.py script to train one's own model,
* the model_apply.py script to use a .h5 model,
* a config.py file to set parameters for model training,
* a data_preprocessed.py script to help the user pre-process his data,
* a model.h5 file containing a model I trained myself.

## Train the model

With the model_training.py script, the user can train a machine learning model based on his own dataset, to be saved as model.h5.

### Data pre-processing :
Before training a model, the user needs to format his dataset --- the data_preprocess.py script is here to help. Very simple and documented, the script is self-explanatory.

The needed format of the dataset is a single .csv file, containing a single list of numbers, without any header.

### How to use the training script :
First of all, the user must install the needed packages:
```sh
$ pip install -r requirements.txt   
```
Then, in a python terminal, use the command line:
```sh
$ python model_training.py --dataset path/to/data
```

There are a few optionnal arguments: 
* `--plot path/to/folder/to/plot/visualization`
* `--test y/[n]`

and one can find their usage using the command line:
```sh
$ python model_training.py --help
```
Some parameters are stored in the config.py, where the user can set :
* the number of epochs,
* the batch size,

and the forecasting method parameters:
* the number of time units to forecast (`N_FUTUR`),
* the number of time units based on which the forecast will be computed (`N_PAST`).

For example, with `N_FUTUR = 4` and `N_PAST = 30`, the model will forecast the next 4 time units temperatures based on the last 30 time units.

Be careful that those two last parameters are used in the model_apply.py script, so if the user changes them *after having trained his model* and runs the model_apply.py script, he may encounter some errors. 

### Visualization

In addition to the trained model, the model_training.py outputs some useful plots (by default in a `./plot` folder) :
* the training data history (loss and accuracy),

![output loss-accuracy](https://raw.githubusercontent.com/artperrin/simple-temperature-forecasting/master/readme_figures/ex_model_stat.png)

* the comparison between original and predicted data.

![output test plot](https://raw.githubusercontent.com/artperrin/simple-temperature-forecasting/master/readme_figures/ex_test_plot.png)


## Use the model

The model_apply.py script provides a temperature forecast based on past temperatures. The behaviour of this script is linked to the parameters the model has been trained with (stored in the file config.py).

If the required packages are not yet installed, the user must use the following command line :
```sh
$ pip install -r requirements.txt   
```
Then, in a python terminal, use the command line:
```sh
$ python model_apply.py --dataset path/to/data
```
There are a few optionnal arguments: 
* `--plot path/to/folder/to/plot/visualization`
* `--extended int`

and one can find their usage using the command line:
```sh
$ python model_training.py --help
```

Some precisions about the `--extended` argument : it is directly linked to the `N_FUTUR` parameter from config.py. It represents the number of time units to be predicted *by the trained model*. However, the user can choose to predict more or less time units with the model_training.py script :
* predict less : the results are truncated to match the user's expectations,

![output less predicted](https://raw.githubusercontent.com/artperrin/simple-temperature-forecasting/master/readme_figures/example_less.png)

* predict more : day-to-day predictions are made by predicting the next temperature, then taking it into account to predict the next one, and so on.

![output less predicted](https://raw.githubusercontent.com/artperrin/simple-temperature-forecasting/master/readme_figures/example_more.png)

The size of the input dataset is not important *as long as there are enough entries* (the minimum number of entries needed is the `N_PAST` parameter from config.py). If the size is too big, *only the last `N_PAST` entries will be taken into account*.

## Data credits

The default model in this repository (model.h5) has been trained with data from Berkeley Earth, found [here](https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data).