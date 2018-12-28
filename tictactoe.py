import network
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.layers import BatchNormalization, Activation, Add, Dense, Flatten


def create_input():
    # One plane for player 1
    # One plane for player 1 previous
    # One plane for player 2
    # One plane for player 2 previous
    # One plane for colour to play
    return Input(shape=(3, 3, 5))


def create_policy(model):
    model = Convolution2D(2, 1, padding="same", data_format="channels_last")(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Flatten()(model)
    model = Dense(9)(model)
    return Activation("softmax", name="policy_h")(model)


def create_model() -> Model:
    return network.create_model(create_input(), create_policy)


def evaluate(model: Model, state):
    prediction = model.predict(state)
    print(prediction)