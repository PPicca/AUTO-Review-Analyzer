from os import read
from string import punctuation, digits
import numpy as np
import random

# Part I


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    z = label * (np.inner(feature_vector, theta) + theta_0)
    hinge_single = max(1 - z, 0)
    return hinge_single


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    full_loss = 0
    size_matrix = len(feature_matrix)
    for i in range(size_matrix):
        full_loss += hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
        hinge_full = full_loss / size_matrix
    return hinge_full


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    if label * (np.inner(feature_vector, current_theta) + current_theta_0) <= 0:
        current_theta = current_theta + (label * feature_vector)
        current_theta_0 = current_theta_0 + label

    return current_theta, current_theta_0


def perceptron(feature_matrix, labels, T):
    n = len(feature_matrix[0])
    updated_theta = np.zeros([n, ])
    updated_theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            updated_theta, updated_theta_0 = perceptron_single_step_update(feature_matrix[i],
                                                                           labels[i],
                                                                           updated_theta,
                                                                           updated_theta_0)

    return updated_theta, updated_theta_0


def average_perceptron(feature_matrix, labels, T):
    n = len(feature_matrix[0])
    updated_theta = np.zeros([n, ])
    updated_theta_0 = 0
    thetas = []
    thetas_0 = []
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            updated_theta, updated_theta_0 = perceptron_single_step_update(feature_matrix[i],
                                                                           labels[i],
                                                                           updated_theta,
                                                                           updated_theta_0)
            thetas.append(updated_theta)
            thetas_0.append(updated_theta_0)

    avg_theta = sum(thetas) / len(thetas)
    avg_theta_0 = sum(thetas_0) / len(thetas_0)

    return avg_theta, avg_theta_0


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    if label * (np.inner(feature_vector, current_theta) + current_theta_0) <= 1:
        current_theta = (1 - (eta * L)) * current_theta + (eta * label * feature_vector)
        current_theta_0 = current_theta_0 + (label * eta)
    else:
        current_theta = (1 - (eta * L)) * current_theta

    return current_theta, current_theta_0


def pegasos(feature_matrix, labels, T, L):
    n = len(feature_matrix[0])
    pegasus_theta = np.zeros([n, ])
    pegasus_theta_0 = 0
    t = 0

    for iteration in range(T):
        for i in get_order(feature_matrix.shape[0]):
            t += 1
            eta = 1 / np.sqrt(t)
            pegasus_theta, pegasus_theta_0 = pegasos_single_step_update(feature_matrix[i],
                                                                        labels[i],
                                                                        L,
                                                                        eta,
                                                                        pegasus_theta,
                                                                        pegasus_theta_0)

    return pegasus_theta, pegasus_theta_0


def classify(feature_matrix, theta, theta_0):
    n = len(feature_matrix)
    labels_array = []

    for i in range(n):
        if np.inner(feature_matrix[i], theta) + theta_0 > 0:
            labels_array.append(+1)
        else:
            labels_array.append(-1)

    return np.array(labels_array)


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    c_train = classifier(train_feature_matrix, train_labels, **kwargs)
    theta_train = c_train[0]
    theta_0_train = c_train[1]
    t_labels = classify(train_feature_matrix, theta_train, theta_0_train)
    v_labels = classify(val_feature_matrix, theta_train, theta_0_train)

    return accuracy(t_labels, train_labels), accuracy(v_labels, val_labels)


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # with open("stopwords.txt", "r") as myfile:
    #     string = myfile.read().replace('\n', ' ')
    # stop_words = []
    # for word in string.split(' '):
    #     stop_words.append(word)

    dictionary = {}  # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix


def accuracy(preds, targets):
    return (preds == targets).mean()
