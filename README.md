# Temporal Spike Attribution

This repository contains the Python 3 code for Temporal Spike Attribution (TSA), a method to extract 
feature attribution explanations for spiking neural networks (SNN). 
TSA is demonstrated on time-series data with SNN models built as recurrent networks in discrete 
time.[^1]

TSA generates class-wise feature attribution maps that can be overlaid with the spiking data to explain a 
certain prediction of a SNN, based on model internal variables (i.e., spike trains, learned weights). The intensity corresponds
to the attribution value and the color corresponds to positive (red) or negative (blue) attribution.

Below is an example of a TSA explanation of a person's activity across time (x-axis) that the SNN correctly predicts as *eating breakfast*.

![TSA-NS](https://github.com/ElisaNguyen/tsa-explanations/blob/master/images/one_ns_177081_breakfast.png)

### Required Python packages
* numpy
* pandas
* scikit-learn
* matplotlib
* pytorch
* pickle
* tqdm

### Data
We use an artificial data set and a real-world data set of time series classification: 
1. Synthetic data set. The dataset is available as a `.csv` and `.pkl` file in `data\synthetic`. That folder also contains the Jupyter Notebook for generating the data set.
2. Activities of daily living (ADL) from binary sensors [^2] data set which is openly 
available in the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Activities+of+Daily+Living+%28ADLs%29+Recognition+Using+Binary+Sensors). This data set is preprocessed, the corresponding scripts can be found in `data\adl\preprocessing`. A constant bias sensor activation is added. Additionally, the classification task is broken down to each second, 
meaning that the models predict the subject's current activity at each second.

#### Instructions for preprocessing the ADL data set
1. Download the data from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Activities+of+Daily+Living+%28ADLs%29+Recognition+Using+Binary+Sensors).
2. Store the csv files in `data/raw/UCI ADL Binary Dataset`
3. Run `python preprocessing/adl_data_writing.py` to generate the "long time series" data, saved in `.csv` files.

### SNN model building
The SNN models use LIF neurons and were implemented as recurrent networks in discrete time trained 
with a fast sigmoid surrogate gradient as per tutorial of Dr. F. Zenke[^1]. There are some changes to Dr. Zenke's original
code as the membrane potential is retained in between epochs and batches to approximate sequential processing of the time series.
Purely sequential processing requires too much computation time, which is the reason for the approximation. 

To build the models, the data has to be transformed into a *spike time, spike unit* format. 
The dataset was split into non-overlapping windows of 900 seconds for tuning and training. 
For the explanation experiments, the whole time series per subject is maintained. 

Three SNNs with one, two, three layers respectively are tuned and trained on the ADL task, and their predictions are explained by TSA.

The following hyperparameters were tuned based on the NLL loss on the validation set after 20 epochs training: $\Delta t, \tau_{syn}, \tau_{mem}$, learning rate, batch size, hidden layer sizes. 
Then, the models are trained with early stopping and a patience of 20 epochs. For the synthetic task, the hyperparameter tuning is skipped and fixed hyperparameters are specified. The scripts to rerun the tuning and training 
are located in `models`, but the tuned hyperparameters and trained model weights are also available. 

#### Instructions
* Run `python data/adl/adl_data_generation.py` to generate and save the ADL data sets in the *times, units* format. `data/adl/dataset900.pkl` then corresponds to the dataset used for tuning and training, while `data/adl/dataset_max.pkl` will be used for the TSA experiments. For the synthetic data set, the data is directly generated in the right format.
* Run `python models/adl/tuning_1L.py`, `python models/adl/tuning_1L.py`, `python models/adl/tuning_3L.py` to tune the hyperparameters of the different SNN models respectively.
* Alternatively, the chosen hyperparameters can be inspected as `models/best_params_1L.pkl, models/best_params_2L.pkl, models/best_params_3L.pkl` respectively.
* Run `python models/training_1L.py`, `python models/training_2L.py`, `python models/training_3L.py` to train the SNN models.
* Alternatively, the trained weights are available at `models/weights_1L_epoch4.pt, models/weights_2L_epoch63.pt, models/weights_3L_epoch48.pt`.

### Explanation extraction with TSA

TSA builds on SNN model internal variables to build a feature attribution explanation. 
Namely, these are the spiking behavior of the neurons when presented an input, the learned weights and the state variables of the output layer which the model prediction is based on.
In the case of the SNN models presented earlier, this is the output layer membrane potential. 
The explanation is generated by a matrix multiplication approach of these elements per time step of the input, which is concatenated to make up the final explanation. 
The result is a 3 dimensional attribution map, specifying the input's attribution to the output classes at input dimension and time step granularity.

2 variants of TSA are distinguished by the way the attribution of the spike behavior component is interpreted: 
The first interpretation assumes the absence of a spike to mean no contribution of a neuron to its downstream neurons. 
This translates to a contribution of 0, so that only spikes are considered in these explanations. Therefore, this variant is called TSA-S (only spikes). 

The second interpretation assumes the absence of a spike to have an effect on the downstream neurons. 
The effect is the natural decay of the membrane potential. Then, there are also non-spikes considered in the explanation as negative contribution. 
As non-spikes are assumed to not be carrying as much information as spikes, they are scaled down. This approach is called TSA-NS (non-spikes).

The provided code enables the extraction of local explanations with either method. We also implement a version of SAM[^3] as a baseline, where weights are not considered. 
For the experiments, a subset of time stamps are defined to extract explanations for (code available in `data/adl/testset_extraction_quantitative_analysis.ipynb` for the ADL data set, for the synthetic data it is generated together with the data). 

#### Instructions
* Create subfolders `s`, `ns` and `sam` in both `experiments/adl` and `experiments/synthetic`.
* Run the `explanation_extraction_xx.py` scripts (replace xx with the dataset) to generate TSA-S, TSA-NS and baseline SAM[^3] explanations.

### Evaluation of TSA 
TSA's explanation performance is evaluated according to applicable Co-12 properties[^6]: in correctness (explanation selectivity[^4]), output-completeness, continuity (max-sensitivity[^5]
using the Frobenius norm) and compactness (size of the explanation).

The scripts to run this evaluation are provided in the `experiments/adl` and `experiments/synthetic` folders.  

#### Instructions
* Create folders `correctness`, `output_completeness`, `continuity` with the subfolders `s`, `ns` and `sam` each in `experiments/adl` and `experiments/synthetic`.
* Run `correctness_xx.py` (long run-time), `output_completeness_xx.py`, `continuity_xx.py` where `xx` is to be replaced by *adl* or *syn* depending on the script. 
* The results will be in the subfolders created before.
* View the results by running the notebook `experiments/adl/quantitative_analysis_adl.ipynb` and `experiments/synthetic/quantitative_analysis_syn.ipynb`. The evaluation of compactness is also in these notebooks.


[^1]: Per tutorial of Friedemann Zenke (https://github.com/fzenke/spytorch, License: http://creativecommons.org/licenses/by/4.0/)

[^2]: Ordonez, F.J.; de Toledo, P.; Sanchis, A. Activity Recognition Using Hybrid Generative/Discriminative Models on Home Environments Using Binary Sensors. Sensors 2013, 13, 5460-5477.

[^3]: Youngeun Kim and Priyadarshini Panda. Visual explanations from spiking neural networks using inter-spike intervals.Scientific Reports, 11(1), 2021.

[^4]: Grégoire Montavon, W. Samek, and K. Müller. Methods for interpreting and understanding deep neural networks.Digit. Signal Process., 73:1–15, 2018.

[^5]: Chih-Kuan Yeh, Cheng.Yu Hsieh, Arun Suggala, David I Inouye, and Pradeep K Ravikumar. On the (in)fidelity and sensitivity of explanations. In H. Wallach, H. Larochelle,A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors,Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[^6]: Nauta, Meike, et al. From Anecdotal Evidence to Quantitative Evaluation Methods: A Systematic Review on Evaluating Explainable AI. arXiv preprint arXiv:2201.08164 (2022).
