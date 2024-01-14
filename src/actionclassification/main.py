# This is the main module, it sets the basic parameters and calls the execution modules to perform training and testing
# To control the train or the test set the TRAIN, TEST flags.
# To control the subject of the execution HR or PTP, change the ExecutionParams
from collections import namedtuple

import actionclassification.train_activity_action as activity_action

TRAIN = True
TEST = True

ExecutionParams = namedtuple('ExecutionParams', ['data_folder',
                                                 'training_file_name',
                                                 'interleaved_file_name',
                                                 'ground_truth_file_name',
                                                 'model_name_prefix',
                                                 'model_name'])

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

if __name__ == '__main__':
    # To execute the train and test on the HR data, use HR_PARAMS, for the purchase to pay data use PTP_PARAMS
    data_folder, file_name, interleaved_file_name, ground_truth_file_name, model_name_prefix, model_name = HR_PARAMS

    if TRAIN:
        print(f'Perform training on the training  data set. training file name={file_name}')
        model_name = activity_action.train_activity_action(data_folder=data_folder, training_file_name=file_name,
                                                           model_name_prefix=model_name_prefix)

    # to use the state-of-the-art model set TRAIN to False and uncomment the   model_name bellow with the name of the
    # model to run
    # model_name = 'hr-activity-action-soa.model'

    # to control the sequence size, change the sequence_size bellow
    sequence_size = 1
    if TEST:
        print(
            f'Perform classification on the interleaved data set. interleaved file name={interleaved_file_name},'
            f' model_name={model_name}')
        activity_action.test_activity_action_classification(data_folder,
                                                            name_of_file_for_classification=interleaved_file_name,
                                                            ground_truth_file_name=ground_truth_file_name,
                                                            model_name=model_name, sequence_size=sequence_size)
    if not (TRAIN or TEST):
        print('Please set TRAIN and/or TEST flags in the main module for performing actions')
