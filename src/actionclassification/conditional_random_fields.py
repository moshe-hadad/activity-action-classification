"""This module responsible for the basic conditional random fields (CRF) implementations based on  pycrfsuite"""
import pycrfsuite


def train_crf_model(X_train, y_train, model_name='crf.model', silent_mode=False):
    """Train a CRF model

    :param X_train: the data set with the training examples
    :param y_train: the data set with the labels
    :param model_name: a name for the model to save
    :param silent_mode: if True, no logs will be printed to the screen (defaults to 'crf.model')
    :return: the model name
    """
    verbose = not silent_mode
    trainer = pycrfsuite.Trainer(verbose=verbose)
    # Submit training data to the trainer
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    # Set the parameters of the model
    trainer.set_params({
        # coefficient for L1 penalty
        'c1': 0.1,

        # coefficient for L2 penalty
        'c2': 0.01,

        # maximum number of iterations
        'max_iterations': 200,

        # whether to include transitions that
        # are possible, but not observed
        'feature.possible_transitions': True
    })
    # Provide a file name as a parameter to the train function, such that
    # the model will be saved to the file when training is finished
    trainer.train(model_name)
    return model_name


def classify_using_crf(model_name, test_dataset, tagger=None):
    """Classifies a data set based on a given model name

    :param model_name: the name of the model to load and use for classification
    :param test_dataset: the data set to classify
    :param tagger: a wrapper class for the classifier, if one is not given the method creates a new one
    :return: classified list created from the test_dataset
    """
    if not tagger:
        tagger = pycrfsuite.Tagger()
        tagger.open(model_name)
    return [tagger.tag(xseq) for xseq in test_dataset]
