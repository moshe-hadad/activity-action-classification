# Introduction

This project is part of
["From Network Traffic Data to a Business-Level Event Log"](https://www.researchgate.net/publication/371324630_From_Network_Traffic_Data_to_a_Business-Level_Event_Log)
research
conducted at the university of Haifa by Moshe Hadad and Gal Engelberg under the supervision of prof. Pnina Soffer. <br>
The general goal is to identify an event logs from network traffic data.<br>
In computer network , computers communicates via sending data over the network as a stream of small
building blocks. In general each network layer has its own building blocks name i.e. bits, frames, packets and
segments.<br>
For simplicity, we can call them network packets.<br>

The objective of this project is to present the method for identifying an activity action in network traffic.<br>
Activity Action is defined as a START or an END of a business processes activity. <br>
The figure below depicts a sequence of packets resulted of running 4 activities A, B, C and D.
The activities were performed in parallel so the packets in the sequence are interleaved.<br>

The method presented in the project is able to receive such
sequence packets and identify in this sequence in which packet an activity start and in which packet an activity ends.

![forming_sequences.png](img%2Fforming_sequences.png)

# Files and folders in this project

[data](data) the data folder contains all training, ground truth and testing data for HR business process and purchase
to pay (PTP) business process. Contains the following files :

* [hr_extended_features.csv](data%2Fhr_extended_features.csv) - training data with extended features for the HR
  business process.
* [hr_ground_truth.csv](data%2Fhr_ground_truth.csv) - ground truth for the HR interleaved data. Marks in which packet
  an activity start and in which it ends (identified by frame.number).
* [hr_interleaved_for_classification.csv](data%2Fhr_interleaved_for_classification.csv) - the HR interleaved data which
  serves as a test data set for the model.
* [ptp_extended_features.csv](data%2Fptp_extended_features.csv) - raining data with extended features for the purchase
  to pay (PTP) business process.
* [ptp_ground_truth.csv](data%2Fptp_ground_truth.csv) - round truth for the purchase to pay business process.
* [ptp_interleaved_for_classification.csv](data%2Fptp_interleaved_for_classification.csv) - the purchase to pay
  business process interleaved data which serves as a test data set for the model.

[img](img) - images folder

* [forming_sequences.png](img%2Fforming_sequences.png) - the image for the illustration of packets in an interleaved
  network data recording

[models](models) - the models folder, contains the CRF folder

[crf](models%2Fcrf) - the crf folder which contains the CRF trained models :

* [hr-activity-action.model](models%2Fcrf%2Fhr-activity-action.model) - trained HR activity action model, this file
  is overwritten every run of the training for the HR business process.
* [hr-activity-action-soa.model](models%2Fcrf%2Fhr-activity-action-soa.model) - trained state-of-the-art HR
  activity action model.
* [ptp-activity-action.model](models%2Fcrf%2Fptp-activity-action.model) - trained PTP activity action model, this file
  is overwritten every run of the training for the PTP business process.
* [ptp-activity-action-soa.model](models%2Fcrf%2Fptp-activity-action-soa.model) - trained state-of-the-art PTP
  activity action model.

[src](src) - the source folder

[actionclassification](src%2Factionclassification) - the namespace folder, contains the following modules :

* [main.py](src%2Factionclassification%2Fmain.py) - the main module, run the project from this module.
* [conditional_random_fields.py](src%2Factionclassification%2Fconditional_random_fields.py) - a modules to hold all CRF
  related functions.
* [features_strategies.py](src%2Factionclassification%2Ffeatures_strategies.py) - a module to hold feature strategy
  classes, classes to help extract features from sequences.
* [train_activity_action.py](src%2Factionclassification%2Ftrain_activity_action.py) - a module to hold the activity
  action functions
* [train_sequence_model.py](src%2Factionclassification%2Ftrain_sequence_model.py) - a module to hold basic functions
  for sequence modeling.
* [utilities.py](src%2Factionclassification%2Futilities.py) - a general module to hold utility functions

# Getting Started (Windows)

1. Download and install python 3.11.0 or above on your computer (tested on python 3.11.0)
2. Create a virtual environment for the project
   ```python -m venv venv```
3. Activate the virtual environment
   ```venv\Scripts\activate.bat```
4. install requirements ```pip install -r requirements.txt```
5. set PYTHONPATH to include the project src as working directory
   ```set PYTHONPATH=%PYTHONPATH%;C:[project-path]\src```
6. CD to src/actionclassification and run
   ```python -m main```

# Execution configuration in the main module

When running the main module, one can configure it to run only the train, the test or both.
To run only the train, set the TEST flag at the top of the file to FALSE:

```
TRAIN = True
TEST = False
```

To run only the test, set the TRAIN flag at the top of the file to FALSE:

NOTE : When running the test, the project will use the last trained model in models/crf folder

```
TRAIN = False
TEST = True
```

To run both, set both to True

```
TRAIN = True
TEST = True
```

By default, the project is training and testing the data for the HR data, to control the subject of the execution
change the ExecutionParams variables.

To run on the HR, use the HR_PARAMS

```
data_folder, file_name, interleaved_file_name, ground_truth_file_name, model_name_prefix, model_name = HR_PARAMS
```

To run on the Purchase to Pay (PTP), use the PTP_PARAMS

```
data_folder, file_name, interleaved_file_name, ground_truth_file_name, model_name_prefix, model_name = PTP_PARAMS
```
HR_PARAMS and PTP_PARAMS are set at the top of the main module (ExecutionParams is a namedtuple)
```
HR_PARAMS = ExecutionParams(data_folder='../../data', training_file_name='hr_extended_features.csv',
                            interleaved_file_name='hr_interleaved_for_classification.csv',
                            ground_truth_file_name='hr_ground_truth.csv',
                            model_name_prefix='hr',
                            model_name='hr-activity-action.model')

PTP_PARAMS = ExecutionParams(data_folder='../../data', training_file_name='ptp_extended_features.csv',
                             interleaved_file_name='ptp_interleaved_for_classification.csv',
                             ground_truth_file_name='ptp_ground_truth.csv',
                             model_name_prefix='ptp',
                             model_name='ptp-activity-action.model')
```
