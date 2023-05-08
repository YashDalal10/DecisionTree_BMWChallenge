import os
import pickle
from contextlib import redirect_stdout

import keras_tuner as kt
import numpy as np
from sklearn.datasets import make_regression
