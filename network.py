from typing import Callable
from keras.models import Model
from keras.layers import BatchNormalization, Activation, Add, Dense, Convolution2D, Flatten

def convolution_block(input: Model) -> Model:
    model = Convolution2D(256, 3, padding="same", data_format="channels_first")(input)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    return model


def residual_block(model: Model) -> Model:
    input = model
    model = Convolution2D(256, 3, padding="same", data_format="channels_first")(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Convolution2D(256, 3, padding="same", data_format="channels_first")(model)
    model = BatchNormalization()(model)
    model = Add()([model, input])
    model = Activation('relu')(model)
    return model


def value_head(model: Model) -> Model:
    model = Convolution2D(1, 1, padding="same", data_format="channels_first")(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Flatten()(model)
    model = Dense(256)(model)
    model = Activation('relu')(model)
    model = Dense(1)(model)
    model = Activation('tanh', name="value_h")(model)
    return model


def create_model(input: Model, policy_factory: Callable[[Model], Model]) -> Model:
    model = convolution_block(input)
    for i in range(5):
        print("Residual block: %s" % i)
        model = residual_block(model)

    policy_h = policy_factory(model)
    value_h = value_head(model)

    model = Model(inputs=[input], outputs=[policy_h, value_h])

    model.compile(optimizer="rmsprop",
                  loss= {"policy_h": "binary_crossentropy", "value_h": "binary_crossentropy"},
                  loss_weights={"policy_h": 0.2, "value_h": 0.2})

    return model
