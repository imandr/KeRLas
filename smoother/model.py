from keras.layers import LSTM, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def model():
    inp = Input((None, 1))
    l1 = LSTM(100, return_sequences=True)(inp)
    l2 = LSTM(100, return_sequences=True, activation="softplus")(l1)
    out = Dense(2, activation="linear")(l2)
    
    model = Model(inputs=[inp], outputs=[out])
    model.compile(optimizer=Adam(lr=0.001), loss="mse")
    return model