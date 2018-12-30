from typing import Callable
from keras.models import Model
from keras.layers import BatchNormalization, Activation, Add, Dense, Convolution2D, Flatten
from keras.optimizers import SGD
from keras.regularizers import l2

def convolution_block(input: Model, reg_constant: float) -> Model:
    model = Convolution2D(256, 3, padding="same", data_format="channels_first", kernel_regularizer=l2(reg_constant))(input)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    return model


def residual_block(model: Model, reg_constant: float) -> Model:
    input = model
    model = Convolution2D(256, 3, padding="same", data_format="channels_first", kernel_regularizer=l2(reg_constant))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Convolution2D(256, 3, padding="same", data_format="channels_first", kernel_regularizer=l2(reg_constant))(model)
    model = BatchNormalization()(model)
    model = Add()([model, input])
    model = Activation('relu')(model)
    return model


def value_head(model: Model, reg_constant: float) -> Model:
    model = Convolution2D(1, 1, padding="same", data_format="channels_first", kernel_regularizer=l2(reg_constant))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Flatten()(model)
    model = Dense(256)(model)
    model = Activation('relu')(model)
    model = Dense(1)(model)
    model = Activation('tanh')(model)
    return model

def create_model(input: Model, policy_factory: Callable[[Model, float], Model]) -> Model:
    reg_const = 0.1
    model = convolution_block(input, reg_const)
    for i in range(5):
        print("Residual block: %s" % i)
        model = residual_block(model, reg_const)

    policy_h = policy_factory(model, reg_const)
    value_h = value_head(model, reg_const)

    model = Model(inputs=[input], outputs=[policy_h, value_h])

    sgd = SGD(lr=0.2, momentum=0, decay=0, nesterov=False)
    model.compile(optimizer=sgd,
                  loss= ['categorical_crossentropy', 'mean_squared_error'],
                  loss_weights=[0.2, 0.2])

    return model
