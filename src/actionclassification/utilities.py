# This module is a utility module, it holds several utility functions for loading, processing and saving data sets
# and files
import operator
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

from actionclassification.features_strategies import WindowFeatureStrategy

file_data_extractor = operator.itemgetter(4, 5)
file_data_extractor_with_sage_id = operator.itemgetter(4, 5, -2, -1)
file_data_extractor_with_summary = operator.itemgetter(-2, -1)
file_data_extractor_with_salary_proposed = operator.itemgetter(4, 5, -2)
file_data_extractor_with_search_read = operator.itemgetter(4, 5, -1)
file_data_extractor_with_salary_expected = operator.itemgetter(4, 5, 7)
file_data_extractor_mail_activity = operator.itemgetter(4, 5, -1)
file_data_extractor_mail_activity_search_read = operator.itemgetter(4, 5, 8)
first_item_in = operator.itemgetter(0)


def select_file_data_extractor(file_data):
    if 'salary_proposed' in file_data:
        return file_data_extractor_with_salary_proposed
    if 'search_read' in file_data:
        return file_data_extractor_with_search_read
    if 'salary_expected' in file_data:
        return file_data_extractor_with_salary_expected
    if 'mail.activity' in file_data:
        if 'action_done' in file_data:
            return file_data_extractor

        return file_data_extractor_mail_activity

    return file_data_extractor_with_sage_id if 'stage_id' in file_data else file_data_extractor


def load_business_process_data_set(file_path):
    """Load the data set created from running business processes and recording their network traffic"""
    dt_data = pd.read_csv(file_path)
    return dt_data.replace(' NULL', np.nan).replace('NULL', np.nan)


def strip_string(data_set):
    str_columns = data_set.select_dtypes(include=['object'])
    for column in str_columns.columns:
        try:
            data_set[column] = str_columns[column].apply(lambda x: x if pd.isnull(x) else x.strip())
        except AttributeError as ex:
            print(f'Error with column:{column} not an str column : {ex}')
    # data_set[str_columns.columns] = str_columns.apply(lambda x: x if pd.isnull(x) else x.strip())
    return data_set


def convert_to_str(value):
    if np.isnan(value):
        return value
    value = int(value)
    value = str(value)
    return value


def strip_null_strings(bp_data_set):
    """Replaces and NULL which is present in a string format into a numpy nan object"""
    bp_data_set = bp_data_set.replace(' NULL', np.nan).replace('NULL', np.nan)
    return bp_data_set


def convert_data_types(bp_data_set):
    if 'template_id' in bp_data_set.columns:
        bp_data_set['template_id'] = bp_data_set['template_id'].replace(np.nan, 0, regex=True)
        bp_data_set['template_id'] = pd.to_numeric(bp_data_set['template_id'], errors='coerce')
    columns_to_convert = {'parent_id',
                          'partner_id',
                          'commercial_partner_id',
                          'mail_followers_id',
                          'mail_message_id',
                          'mail_mail_id',
                          'wizard_id',
                          'hr_applicant_id',
                          'mail_compose_message_id',
                          'mail_message_res_partner_needaction_rel_id',
                          'mail_activity_id'}

    columns_to_convert = {'hr_applicant_id', 'mail_activity_id'}

    for column in columns_to_convert:
        try:
            if bp_data_set.dtypes[column] == float:
                bp_data_set[column] = bp_data_set[column].apply(convert_to_str)
        except KeyError:
            pass

    return bp_data_set


def labels_and_target_names(data_set, label_column):
    target_names = data_set[label_column].unique()
    labels = {activity: index for index, activity in enumerate(target_names)}
    return labels, target_names


def load_bp_data_set(data_folder=os.environ.get('DATA_FOLDER'), file_name=os.environ.get('BP_ATTRIBUTES'),
                     should_strip_string=True):
    bp_data_set_file_name = os.path.join(data_folder, file_name)
    bp_data_set = load_business_process_data_set(bp_data_set_file_name)
    if should_strip_string:
        bp_data_set = strip_string(bp_data_set)
    bp_data_set = strip_null_strings(bp_data_set)
    bp_data_set = convert_data_types(bp_data_set)
    return bp_data_set


def select_filter_data(row):
    file_data = row['file_data']
    file_data = str(file_data).replace('nan', '')
    file_data = eval(file_data) if file_data else ''
    if not file_data:
        return ''
    if isinstance(file_data, dict):
        return ''

    method: str = first_item_in(file_data)
    if method == 'version' or method == 'faultCode':
        return method

    if method.isnumeric():
        return 'IsNumber'

    if method == 'id' and 'name' in file_data:
        return ''

    if method == 'id' and 'salary_expected' in file_data:
        return 'salary_expected'

    if method == 'execute_kw' or method == 'id':

        extract_data = select_file_data_extractor(
            file_data) if method == 'execute_kw' else file_data_extractor_with_summary

        filtered_data = ''
        try:
            extracted_data = extract_data(file_data)
            filtered_data = '_'.join(extracted_data)
        except IndexError:
            pass

        return filtered_data

    return ''


def add_selective_filter_data(bp_data_set, extract_method=select_filter_data):
    selective_filter_data = bp_data_set.apply(extract_method, axis=1)
    bp_data_set['selective_filter_data'] = selective_filter_data
    return bp_data_set


def extract_origin_data(data_set, column):
    def origin_data(row):
        column_value = ''
        starting_frame_number = row['starting_frame_number']
        if isinstance(starting_frame_number, str):
            starting_frame_number = int(starting_frame_number) if starting_frame_number else np.nan
        if starting_frame_number and not np.isnan(starting_frame_number):
            row_origin = data_set[data_set['frame.number'] == int(starting_frame_number)]
            if not row_origin.empty:
                if len(row_origin) > 1:
                    # If the numbers of rows is bigger than 1 it means we are dealing with the training data
                    # Thus we need to filter also by instance number
                    row_origin = row_origin[row_origin['InstanceNumber'] == row['InstanceNumber']]
                if not row_origin.empty:
                    column_value = row_origin[column].iloc[0]
                else:
                    print(
                        f"Error, could not locate starting frame number:{starting_frame_number} and "
                        f"InstanceNumber:{row['InstanceNumber']}")

        return column_value

    return origin_data


def add_request_origin_data(data_set):
    origin_request_method = data_set.apply(extract_origin_data(data_set, 'request_method_call'), axis=1)
    origin_selective_filter_data = data_set.apply(extract_origin_data(data_set, 'selective_filter_data'), axis=1)
    data_set['origin_request_method'] = origin_request_method
    data_set['origin_selective_filter_data'] = origin_selective_filter_data
    return data_set


def extract_instance_number(sequence):
    instances = sequence['InstanceNumber'].unique()
    return first_item_in(instances)


def get_properties_from(training_properties):
    label, labels, target_names, features, features_strategy = training_properties.values()
    return label, labels, target_names, features, features_strategy


def convert_from_labels(flaten, labels, truth_y):
    truths = np.array([labels[tag] for row in truth_y for tag in row]) if flaten else np.array(
        [labels[tag] for tag in truth_y])
    return truths


def pop_sequences(activity_sequences, instances_to_pop):
    instance_map = set([int(instance_number) for instance_number in instances_to_pop])
    popped_sequence = [(instance_number, sequence) for instance_number, sequence in activity_sequences if
                       instance_number in instance_map]
    remained_sequence = [(instance_number, sequence) for instance_number, sequence in activity_sequences if
                         instance_number not in instance_map]
    return popped_sequence, remained_sequence


def extract_labels(sequence, label):
    return sequence[label].to_list()


def create_x_y_sequences(features, features_strategy, label_column, sequences):
    X = [extract_features(sequence, features, features_strategy) for _, sequence in
         tqdm(sequences)]
    y = [extract_labels(sequence, label_column) for _, sequence in tqdm(sequences)]
    return X, y


def print_classification_report(truth_y, predicted_y, labels, target_names, flaten=True, output_dict=False):
    # Convert the sequences of tags into a 1-dimensional array
    truths = convert_from_labels(flaten, labels, truth_y)
    predictions = convert_from_labels(flaten, labels, predicted_y)
    # Print out the classification report
    if output_dict:
        return classification_report(truths, predictions, target_names=target_names, output_dict=True)
    labels_for_report = list(labels.values())
    print(classification_report(truths, predictions, labels=labels_for_report, target_names=target_names))


def report_classification(data_set, truth_column, predicted_column, labels=None, target_names=None,
                          labels_and_target_name_func=None, silent_mode=False):
    if not labels and not target_names and not labels_and_target_name_func:
        raise ValueError('Please supply labels and target_names or labels_and_target_name_func')
    if labels_and_target_name_func:
        labels, target_names = labels_and_target_name_func()
    truth_y = data_set[truth_column]
    predicted_y = data_set[predicted_column]

    report = print_classification_report(truth_y, predicted_y, labels, target_names, flaten=False,
                                         output_dict=silent_mode)
    if silent_mode:
        return report


def create_features_strategy(features, window_backwards=5, window_forward=15):
    activity_window_strategy = WindowFeatureStrategy(window_backwards=window_backwards, window_forward=window_forward)
    activity_features_strategy = {feature: activity_window_strategy for feature in features}
    return activity_features_strategy


def update_values_in_indices(data_set, column, indices, values):
    data_set.iloc[indices, data_set.columns.get_loc(column)] = values
    return data_set


def get_activity(data_set, frame_number, use_specific_activity=True):
    single_activity = data_set[data_set['actual_end'] == frame_number][
        'activity_name'] if use_specific_activity else 'Activity'
    return single_activity + ' End'


def update_value_for_frame_number(data_set, column, frame_number, value):
    index = data_set[data_set['frame.number'] == frame_number].index
    value = value if isinstance(value, str) else str(value)
    data_set.iloc[index, data_set.columns.get_loc(column)] = value
    return data_set


def assign_tagging_to_frame_number(data_set, tagging_for_frame_number, column='real_activity_action'):
    if column not in data_set.columns:
        data_set[column] = 'NoAction'
    for frame_number, tagging in tagging_for_frame_number:
        data_set = update_value_for_frame_number(data_set, column, frame_number, tagging)
    return data_set


def add_ground_truth_tagging(data_set, use_specific_activity, ground_truth_file_name,
                             column='real_activity_action', convert=False):
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../data/{ground_truth_file_name}'))
    ground_truth_data = pd.read_csv(full_path)
    start_frame_number = ground_truth_data['start']
    end_frame_number = ground_truth_data['actual_end']
    data_set[column] = 'NoAction'

    activity_len = len(ground_truth_data['activity_name'])
    activity = ground_truth_data['activity_name'] if use_specific_activity else pd.Series(['Activity'] * activity_len)
    indices = data_set[data_set['frame.number'].isin(start_frame_number)].index
    data_set = update_values_in_indices(data_set, column, indices, activity + ' Start')

    # indices = data_set[data_set['frame.number'].isin(end_frame_number)].index
    tagging_for_frame_number = [(frame_number, get_activity(ground_truth_data, frame_number, use_specific_activity))
                                for frame_number in end_frame_number]
    if convert:
        tagging_for_frame_number = [(frame_number, item.iloc[0]) for frame_number, item in tagging_for_frame_number]
    data_set = assign_tagging_to_frame_number(data_set, tagging_for_frame_number, column=column)
    # data_set = update_values_in_indices(data_set, 'real_activity_action', indices, activity + ' End')

    return data_set


def tag_activity_action_with_columns(start_activity_column_name, end_activity_column_name, use_specific_activity=False):
    def tag_activity_action(row):
        start_value = row[start_activity_column_name]
        stop_value = row[end_activity_column_name]
        activity_name = row['BusinessActivity'] if use_specific_activity else 'Activity'
        if stop_value != 0:
            return f'{activity_name} End'
        if start_value != 0:
            return f'{activity_name} Start'

        return 'NoAction'

    return tag_activity_action


def add_activity_action_to(bp_data_set, start_activity_column_name, end_activity_column_name,
                           use_specific_activity=False, use_business_activity_column=False, label='ActivityAction'):
    bp_data_set = bp_data_set.sort_values(by=['sniff_time'])
    differentiation_column = 'BusinessActivity' if use_business_activity_column else 'InstanceNumber'
    if use_business_activity_column:
        bp_data_set[start_activity_column_name] = bp_data_set['BusinessActivity'].ne(
            bp_data_set['BusinessActivity'].shift())
    else:
        bp_data_set[start_activity_column_name] = bp_data_set[differentiation_column].diff()

    bp_data_set[end_activity_column_name] = bp_data_set[start_activity_column_name].shift(-1)
    activity_action = bp_data_set.apply(
        tag_activity_action_with_columns(start_activity_column_name, end_activity_column_name,
                                         use_specific_activity=use_specific_activity), axis=1)
    bp_data_set[label] = activity_action
    return bp_data_set


def extract_features_with_strategy(index, message, sequence, features_strategy):
    collected_features = []
    for feature, feature_extraction_strategy in features_strategy.items():
        collected_features.extend(feature_extraction_strategy(index, message, sequence, feature))
    return collected_features


def sequence2features(index, message, sequence, features: list, features_strategy: dict = None):
    features_strategy = {} if features_strategy is None else features_strategy
    features_with_no_strategy = features - features_strategy.keys()

    collected_features = ['bias']
    collected_features.extend([f'{feature}[0]={message[feature]}' for feature in features_with_no_strategy])

    features_with_strategy = extract_features_with_strategy(index, message, sequence, features_strategy)
    collected_features.extend(features_with_strategy)
    return collected_features


def extract_features(sequence, features, feature_strategy):
    return [
        sequence2features(index, message, sequence, features, feature_strategy)
        for
        index, message in sequence.iterrows()]


def build_sequences(data, sequence_size):
    if isinstance(data, pd.DataFrame):
        return [data.iloc[i:i + sequence_size] for i in range(0, len(data), sequence_size)]

    return [data[i:i + sequence_size] for i in range(0, len(data), sequence_size)]
