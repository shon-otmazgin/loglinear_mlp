import random
import mlp1
import mlpn

STUDENT = {'name': 'Royi Rassin, Shon Otzmagin',
           'ID': '311334734, 305394975'
         }

def train_classifier(train_data, num_iterations, learning_rate, params):
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for y, x in train_data:
            loss, grads = mlp1.loss_and_gradients(x,y,params)
            cum_loss += loss

            W, b, U, b_tag = params
            gW, gb, gU, gb_tag = grads

            W_new = W - (learning_rate*gW)
            b_new = b.reshape(-1,1) - (learning_rate * gb)
            U_new = U - (learning_rate * gU)
            b_tag_new = b_tag.reshape(-1,1) - (learning_rate * gb_tag)

            params = [W_new, b_new, U_new, b_tag_new]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        print(I, train_loss, train_accuracy)
        if train_accuracy == 1:
            return params
    return params

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for y, x in dataset:
        y_pred = mlp1.predict(x, params)
        good += 1 if y_pred==y else 0
        bad += 1 if y_pred!=y else 0
    return good / (good + bad)

if __name__ == '__main__':
        data = [(1,[0,0]),
                (0,[0,1]),
                (0,[1,0]),
                (1,[1,1])]

        num_iterations = 100000
        learning_rate = 5e-1

        in_dim = 2
        hid_dim = 2
        out_dim = 2

        params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
        trained_params = train_classifier(data, num_iterations, learning_rate, params)

        assert mlp1.predict(x=[0,0], params=trained_params) == 1
        assert mlp1.predict(x=[0, 1], params=trained_params) == 0
        assert mlp1.predict(x=[1, 0], params=trained_params) == 0
        assert mlp1.predict(x=[1, 1], params=trained_params) == 1



