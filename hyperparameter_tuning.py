import mlp1
from train_mlp1 import train_classifier
from utils import TRAIN as train_data
from utils import DEV as dev_data
from utils import L2I, F2I
import loglinear as ll
import itertools

STUDENT = {'name': 'Royi Rassin, Shon Otzmagin',
           'ID': '311334734, 305394975'
           }

param_grid = {
    'epochs': [300],
    'hid_dim': [32, 128],
    'lr': [1e-3, 2e-3, 3e-3]
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

in_dim = len(F2I)
out_dim = len(L2I)

# Use cross validation to evaluate all parameters
for h_params in all_params:
    print(h_params)
    params = mlp1.create_classifier(in_dim, h_params['hid_dim'], out_dim)
    trained_params = train_classifier(train_data, dev_data, h_params['epochs'], h_params['lr'], params)
    print()

