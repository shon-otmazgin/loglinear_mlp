from train_loglin import train_classifier
from utils import TRAIN as train_data
from utils import DEV as dev_data
from utils import L2I, F2I
import loglinear as ll
import itertools

param_grid = {
    'epochs': [100],
    'lr': [1e-5],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

in_dim = len(F2I)
out_dim = len(L2I)
params = ll.create_classifier(in_dim, out_dim)
# Use cross validation to evaluate all parameters
for h_params in all_params:
    print(h_params)
    trained_params = train_classifier(train_data, dev_data, h_params['epochs'], h_params['lr'], params)
    print()

