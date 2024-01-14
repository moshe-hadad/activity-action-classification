"""This module is responsible for extracting features for the CRF algorithm
The features are extracted as a window around in index
When the window is extended beyond items in the sequence (the index is the first in the sequence or the last in the
sequence) the feature is created with  'NoMessage' to fill the gap
"""


def extract_backwards_messages_features(index, sequence, feature: str, number_of_messages_backwards):
    """Extracting features from a sequence in a backwards window

    :param index: the location in the sequence to open a window backward from
    :param sequence: the sequence to extract features from
    :param feature: the name of the column (feature) to extract value from e.g. event_with_roles
    :param number_of_messages_backwards: the backwards window size
    :return: a list of the features in the format suitable for the CRF algorithm
    """
    start = index - number_of_messages_backwards
    start = start if start > 0 else 0
    window = []
    features_backwards = sequence[feature][start:index].to_list()
    missing_elements = number_of_messages_backwards - len(features_backwards)
    collected_features = ['NoMessage'] * missing_elements
    collected_features.extend(features_backwards)

    window.extend([f'{feature}[-{number_of_messages_backwards - i}]={feature_value}' for i, feature_value in
                   enumerate(collected_features)])
    return window


def extract_forward_messages_features(index, sequence, feature: str, number_of_messages_forward):
    """Extracting features from a sequence in a forward window

    :param index: the location in the sequence to open a window forward from
    :param sequence: the sequence to extract features from
    :param feature: the name of the column (feature) to extract value from e.g. event_with_roles
    :param number_of_messages_forward: the forward window size
    :return: a list of the features in the format suitable for the CRF algorithm
    """
    end = index + number_of_messages_forward + 1
    end = end if end < len(sequence) else len(sequence)
    features_forward = []

    extracted_features = sequence[feature][index + 1:end].to_list()
    missing_elements = number_of_messages_forward - len(extracted_features)
    extracted_features.extend(missing_elements * ['NoMessage'])
    features_forward.extend(
        [f'{feature}[{i}]={feature_value}' for i, feature_value in enumerate(extracted_features, start=1)])

    return features_forward


class WindowFeatureStrategy(object):
    """A strategy class for extracting features from a sequence in a suitable format for the CRF algorithm
    """

    def __init__(self, window_backwards, window_forward):
        self.window_backwards = window_backwards
        self.window_forward = window_forward

    def __call__(self, index, message, sequence, feature):
        backwards_messages = extract_backwards_messages_features(index, sequence, feature, self.window_backwards)
        forward_messages = extract_forward_messages_features(index, sequence, feature, self.window_forward)
        collected_features = []
        collected_features.extend(backwards_messages)
        collected_features.append(f'{feature}[0]={message[feature]}')
        collected_features.extend(forward_messages)
        return collected_features
