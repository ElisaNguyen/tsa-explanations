# Temporal Spike Attribution - A local feature-based explanation for Spiking Neural Networks

This repository contains the Python 3 code for Temporal Spike Attribution (TSA), a method to extract 
feature attribution explanations for temporally coded Spiking Neural Network. 
TSA is demonstrated on a time-series classification use case with SNN models built as recurrent networks in discrete 
time. [^1]

TSA generates class-wise feature attribution maps that can be overlayed with the spiking data to explain a 
certain prediction of a SNN, based on model internal variables (i.e., spike trains, learned weights). The intensity corresponds
to the attribution value and the color corresponds to the class that the input is attributing to. Additionally, 
the classification confidence is visualised to give context about the model behavior. 

![Example image of a TSA-S explanation](images/explanation_one.png)

### Prerequisites
* numpy
* pandas
* scikit-learn
* matplotlib
* pytorch
* pickle
* tqdm

### Data
The use case dataset is the activities of daily living (ADL) from binary sensors [^2] dataset which is openly 
available in the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Activities+of+Daily+Living+%28ADLs%29+Recognition+Using+Binary+Sensors).

In the frame of this work, the data has been interpreted as one long time series per subject ranging across all the days
of data collection. For this, the data has been preprocessed and transformed to build one long time series.
The code for this is in the `preprocessing` and folders. A constant bias sensor activation is added. Additionally, the classification task is broken down to each second, 
meaning that the models predict the subject's current activity at each second.

##### Instructions
1. Download the data from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Activities+of+Daily+Living+%28ADLs%29+Recognition+Using+Binary+Sensors).
2. Store the csv files in `data/raw/UCI ADL Binary Dataset`
3. Run `python preprocessing/adl_data_writing.py` to generate the "long time series" data.

### SNN model building
For model training, the dataset was then split into non-overlapping windows of 900 seconds. For the explanation experiments,
the whole time series per subject is maintained. 

[^1]: Per tutorial of Friedemann Zenke (https://github.com/fzenke/spytorch, License: http://creativecommons.org/licenses/by/4.0/)

[^2]: Ordonez, F.J.; de Toledo, P.; Sanchis, A. Activity Recognition Using Hybrid Generative/Discriminative Models on Home Environments Using Binary Sensors. Sensors 2013, 13, 5460-5477.