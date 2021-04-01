import mlpn
import random
import numpy as np

STUDENT = {'name': 'Royi Rassin',
           'ID': '311334734',
           'name_2': 'Shon Otzmagin',
            'ID_2': "305394975"
         }

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
        y_pred = mlpn.predict(x, params)
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
            loss, grads = mlpn.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

            new_params = []
            L = int(len(params) / 2)
            for i in range(L):
                W, b = params[i*2], params[(i*2)+1]
                gW, gb = grads[i*2], grads[(i*2) + 1]
                new_params.append(W - (learning_rate*gW))
                new_params.append(b.reshape(-1,1) - (learning_rate * gb))

            params = new_params

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

    num_iterations = 10
    learning_rate = 1e-2

    dims = [len(F2I), 40, len(L2I)]

    params = mlpn.create_classifier(dims)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    preds = []
    for features in test_data:
        x = feats_to_vec(features)
        preds.append(mlpn.predict(x, trained_params))

    with open('test.pred', 'w') as f:
        for y_hat in preds:
            f.write(f'{I2L[y_hat]}\n')

