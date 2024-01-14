import os.path

from sklearn.model_selection import train_test_split

import actionclassification.train_sequence_model as seq
import actionclassification.utilities as util

ACTIVITY_START_END_MODEL = '../../models/crf/activity_start_end.model'

START_ACTIVITY_COLUMN_NAME = 'IsStartActivity'
END_ACTIVITY_COLUMN_NAME = 'IsEndActivity'

FULL_BPS = [
    '29',
    '37',
    '50',
    '68',
    '70',
    '86',
    '87',
    '93',
    '94',
    '129',
    '132',
    '167',
    '174',
    '178'
]
USE_SPECIFIC_ACTIVITY = False


def creat_data_sets(activity_sequences, features, features_strategy, label_column):
    small_bp_sequences, remained_sequence = util.pop_sequences(activity_sequences, FULL_BPS)
    X, y = util.create_x_y_sequences(features, features_strategy, label_column, remained_sequence)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    small_X, small_y = util.create_x_y_sequences(features, features_strategy, label_column, small_bp_sequences)
    X_test.extend(small_X)
    y_test.extend(small_y)

    return X_train, X_test, y_train, y_test


def train_using_sequence_model(activity_sequences, target_column, features, features_strategy, labels_and_target_names,
                               model_name=ACTIVITY_START_END_MODEL, silent_mode=False):
    labels, target_names = labels_and_target_names()

    seq.train_and_test_crf(model_name=model_name,
                           activity_sequences=activity_sequences,
                           features=features, features_strategy=features_strategy, label=target_column, labels=labels,
                           target_names=target_names, create_data_sets_method=creat_data_sets, silent_mode=silent_mode)


def labels_and_names_for_activity_action(use_specific_activity=USE_SPECIFIC_ACTIVITY):
    labels = {"NoAction": 0,
              "GenerateJobApplicationActivity Start": 1,
              "GenerateJobApplicationActivity End": 2,
              "ResumeReviewActivity Start": 3,
              "ResumeReviewActivity End": 4,
              "ScheduleAnInterviewActivityCall Start": 5,
              "ScheduleAnInterviewActivityCall End": 6,
              "PerformAnInterviewCall Start": 7,
              "PerformAnInterviewCall End": 8,
              "ScheduleAnInterviewMeeting Start": 9,
              "ScheduleAnInterviewMeeting End": 10,
              "PerformAnInterviewMeeting Start": 11,
              "PerformAnInterviewMeeting End": 12,
              "ContractProposal Start": 13,
              "ContractProposal End": 14} if use_specific_activity else {"Activity Start": 0,
                                                                         "NoAction": 1,
                                                                         "Activity End": 2,
                                                                         }

    target_names = [
        "NoAction",
        "GenerateJobApplicationActivity Start",
        "GenerateJobApplicationActivity End",
        "ResumeReviewActivity Start",
        "ResumeReviewActivity End",
        "ScheduleAnInterviewActivityCall Start",
        "ScheduleAnInterviewActivityCall End",
        "PerformAnInterviewCall Start",
        "PerformAnInterviewCall End",
        "ScheduleAnInterviewMeeting Start",
        "ScheduleAnInterviewMeeting End",
        "PerformAnInterviewMeeting Start",
        "PerformAnInterviewMeeting End",
        "ContractProposal Start",
        "ContractProposal End"
    ] if use_specific_activity else ["Activity Start",
                                     "NoAction",
                                     "Activity End"]
    return labels, target_names


def load_and_prepare_validation_set(data_folder, file_name, ground_truth_file_name):
    data_set = util.load_bp_data_set(data_folder=data_folder, file_name=file_name)
    data_set = util.add_ground_truth_tagging(data_set, USE_SPECIFIC_ACTIVITY, ground_truth_file_name)
    # data_set = util.update_is_start_bp_from_activity_column(data_set, label='IsBPStart')
    # data_set = util.update_is_end_bp_from_domain_knowledge(data_set, label='ISBPEnd')

    # data_set = util.add_selective_filter_data(data_set)
    #
    # data_set = util.add_request_origin_data(data_set)
    return data_set


def create_activity_model_training_properties(data_set, label, features, features_strategy):
    labels, target_names = util.labels_and_target_names(data_set, label_column=label)
    return {
        'label': label,
        'labels': labels,
        'target_names': target_names,
        'features': features,
        'features_strategy': features_strategy,
    }


def train_action_model(data_set, models_folder, model_name, training_properties, window_size=20):
    activity_sequences = util.build_sequences(data_set, sequence_size=window_size)
    activity_sequences_with_bp = [(util.extract_instance_number(sequence), sequence) for sequence in
                                  activity_sequences]
    label, labels, target_names, features, features_strategy = util.get_properties_from(
        training_properties)

    def labels_and_target_names():
        return labels, target_names

    model_path = os.path.join(models_folder, model_name)
    train_using_sequence_model(activity_sequences_with_bp, target_column=label, features=features,
                               features_strategy=features_strategy,
                               labels_and_target_names=labels_and_target_names, model_name=model_path,
                               silent_mode=False)


def test_activity_action_classification(data_folder, name_of_file_for_classification, ground_truth_file_name,
                                        models_folder='../../models/crf',
                                        model_name='', sequence_size=1):
    data_set = load_and_prepare_validation_set(data_folder, file_name=name_of_file_for_classification,
                                               ground_truth_file_name=ground_truth_file_name)
    data_set = util.add_selective_filter_data(data_set)
    data_set = util.add_request_origin_data(data_set)

    activity_action_column = 'ActivityActionClassification'
    model_path = os.path.join(models_folder, model_name)
    features, features_strategy = get_features_and_strategy_for_activity_action_model()
    classified_data_set = seq.classify_data_set(model_name=model_path,
                                                data_set=data_set, features=features, features_strategy=
                                                features_strategy, sequence_size=sequence_size,
                                                classification_column=activity_action_column)

    labels, target_names = util.labels_and_target_names(classified_data_set, label_column=activity_action_column)
    util.report_classification(classified_data_set, truth_column='real_activity_action',
                               predicted_column=activity_action_column,
                               labels=labels, target_names=target_names)


def is_sequential(activity_sequence):
    for frame_1, frame_2 in zip(activity_sequence['frame.number'], activity_sequence['frame.number'].iloc[1:]):
        if frame_1 >= frame_2:
            return False
    return True


def validate_activity_sequence(activity_sequence, start_activity_label, end_activity_label):
    return len(activity_sequence[activity_sequence['ActivityAction'] == start_activity_label]) == 1 and len(
        activity_sequence[activity_sequence['ActivityAction'] == end_activity_label]) == 1 and \
        activity_sequence['ActivityAction'].iloc[0] == start_activity_label and \
        activity_sequence['ActivityAction'].iloc[
            -1] == end_activity_label


def validate_start_end_correctness(bp_data_set, start_activity_label='Start Activity',
                                   end_activity_label='End Activity'):
    validation_groups = bp_data_set.groupby(['InstanceNumber', 'BusinessActivity'])
    warnings = []
    for group_id, activity_sequence in validation_groups:
        activity = activity_sequence['BusinessActivity'].iloc[0]
        if not is_sequential(activity_sequence):
            warnings.append(f'Error validating group:{group_id} activity:{activity} - not sequence by frame.number')
        if not validate_activity_sequence(activity_sequence, start_activity_label, end_activity_label):
            warnings.append(f'Error validating group:{group_id} activity:{activity}')
    if warnings:
        print(warnings)
        return False
    return True


def get_features_and_strategy_for_activity_action_model():
    features = ['event_with_roles', 'request_method_call', 'selective_filter_data', 'origin_selective_filter_data']
    activity_features_strategy = util.create_features_strategy(features, window_backwards=5, window_forward=15)
    return features, activity_features_strategy


def train_activity_action(data_folder, training_file_name, models_folder='../../models/crf', model_name_prefix=''):
    """This function trains an activity action model. It saved the model in the models folder if given, if not, the
    default models folder name is  '../../models/crf'. The model is saved as '{model_name}-activity-action.model'

    :param data_folder: path to the name of the data folder containing the training data
    :param training_file_name: the training data filename
    :param models_folder: optional - a path to the models' folder, defaults to '../../models/crf'
    :param model_name_prefix: optional - a prefix name for the trained model, defaults to an empty string.
    :return: None
    """

    data_set = util.load_bp_data_set(data_folder=data_folder, file_name=training_file_name)
    data_set = data_set.sort_values(by=['sniff_time'])
    # data_set = util.add_selective_filter_data(data_set)
    # # data_set = util.add_request_origin_data(data_set)
    # data_set = util.add_activity_action_to(data_set,
    #                                        start_activity_column_name=START_ACTIVITY_COLUMN_NAME,
    #                                        end_activity_column_name=END_ACTIVITY_COLUMN_NAME,
    #                                        use_specific_activity=False, use_business_activity_column=True)
    # data_set.to_csv('../../data/testing.csv')
    if validate_start_end_correctness(data_set, start_activity_label='Activity Start',
                                      end_activity_label='Activity End'):
        print('Data Set for Start End Activity Was validated for correctness')
    activity_action_model_name = f'{model_name_prefix}-activity-action.model'
    activity_action_model_name = activity_action_model_name[1:] if activity_action_model_name[
                                                                       0] == '-' else activity_action_model_name

    features, features_strategy = get_features_and_strategy_for_activity_action_model()
    activity_action_training_properties = create_activity_model_training_properties(data_set, label='ActivityAction',
                                                                                    features=features,
                                                                                    features_strategy=features_strategy)
    train_action_model(data_set=data_set, models_folder=models_folder,
                       model_name=activity_action_model_name,
                       training_properties=activity_action_training_properties)

    return activity_action_model_name
