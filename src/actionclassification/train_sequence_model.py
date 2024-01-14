import itertools

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import actionclassification.conditional_random_fields as crf
import actionclassification.utilities as util


def build_data_sets(sequences, features, features_strategy, label_column):
    """Creates a data set of sequences"""
    X, y = util.create_x_y_sequences(features, features_strategy, label_column, sequences)
    return train_test_split(X, y, test_size=0.3)


def train_and_test_crf(model_name, activity_sequences, features, features_strategy, label, labels, target_names,
                       create_data_sets_method=None, silent_mode=False):
    creat_data_set = create_data_sets_method if create_data_sets_method else build_data_sets
    X_train, X_test, y_train, y_test = creat_data_set(activity_sequences, features, features_strategy, label)

    print(f'Train the CRF model for {label}')
    crf.train_crf_model(X_train, y_train, model_name=model_name, silent_mode=silent_mode)
    print(f'Finished Train the CRF model for {label}')

    y_pred = crf.classify_using_crf(model_name=model_name, test_dataset=X_test)
    util.print_classification_report(truth_y=y_test, predicted_y=y_pred, labels=labels,
                                     target_names=target_names, output_dict=silent_mode)
    return X_train, X_test, y_train, y_test, y_pred


def resolve_classification(tagger, tagging):
    try:
        return tagger.resolve_classification(tagging)
    except AttributeError:
        return itertools.chain(*tagging)


def classify_data_set(model_name, data_set, features, features_strategy, sequence_size=15,
                      classification_column='Tagging', classifier=None):
    """Classifies a data based on a model name given"""
    activity_sequences = util.build_sequences(data_set, sequence_size=sequence_size)
    tagging = classify_sequences(model_name, activity_sequences, features, features_strategy, classifier)
    tagging = resolve_classification(classifier, tagging)
    data_set[classification_column] = list(tagging)
    return data_set


def classify_sequences(model_name, m_activity_sequences, features, features_strategy, tagger=None):
    """Classifies a sequence based on a model given"""
    m_X = [util.extract_features(sequence, features, features_strategy) for sequence in
           tqdm(m_activity_sequences)]
    m_y_pred = crf.classify_using_crf(model_name=model_name, test_dataset=m_X, tagger=tagger)
    return m_y_pred
