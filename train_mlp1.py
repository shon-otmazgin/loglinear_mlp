import mlp1 as mlp1
import random
import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    arr = np.zeros(len(F2I))
    for f in features:
        try:
            arr[F2I[f]] += 1
        except KeyError:
            # f not in vocab (top 600 bigram)
            continue
    return arr

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        x = feats_to_vec(features)
        y_pred = mlp1.predict(x, params)
        good += 1 if y_pred==L2I[label] else 0
        bad += 1 if y_pred!=L2I[label] else 0
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = L2I[label]                  # convert the label to number if needed.
            loss, grads = mlp1.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

            W, b, U, b_tag = params
            gW, gb, gU, gb_tag = grads

            W_new = W - (learning_rate*gW)
            b_new = b.reshape(-1,1) - (learning_rate * gb)
            U_new = U - (learning_rate * gU)
            b_tag_new = b_tag.reshape(-1,1) - (learning_rate * gb_tag)

            params = [W_new, b_new, U_new, b_tag_new]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    from utils import TRAIN as train_data
    from utils import DEV as dev_data
    from utils import TEST as test_data
    from utils import L2I, I2L, F2I

    num_iterations = 7
    learning_rate = 1e-2
    in_dim = len(F2I)
    hid_dim = 16
    out_dim = len(L2I)

    params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    preds = []
    for features in test_data:
        x = feats_to_vec(features)
        preds.append(mlp1.predict(x, trained_params))

    with open('test.pred', 'w') as f:
        for y_hat in preds:
            f.write(f'{I2L[y_hat]}\n')

