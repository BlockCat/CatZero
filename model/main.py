
# if __name__ == "__main__":
import tensorflow as tf
import numpy as np
import keras
import builtins as bb
import datetime
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Dense, Convolution2D, Flatten, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def input_block(bs):
    # One plane for player 1
    # One plane for player 2
    # One plane for player 1 previous
    # One plane for player 2 previous
    # One plane for colour to play
    return Input(batch_shape=bs)


def convolution_block(input: Model, reg_constant) -> Model:
    model = Convolution2D(256, 3, padding="same", data_format="channels_first",
                          kernel_regularizer=l2(reg_constant))(input)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    return model


def residual_block(model: Model, reg_constant) -> Model:
    input = model
    model = Convolution2D(256, 3, padding="same", data_format="channels_first",
                          kernel_regularizer=l2(reg_constant))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Convolution2D(256, 3, padding="same", data_format="channels_first",
                          kernel_regularizer=l2(reg_constant))(model)
    model = BatchNormalization()(model)
    model = Add()([model, input])
    model = Activation('relu')(model)
    return model


def value_head(model: Model, reg_constant) -> Model:
    model = Convolution2D(1, 1, padding="same", data_format="channels_first",
                          kernel_regularizer=l2(reg_constant))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Flatten()(model)
    model = Dense(256)(model)
    model = Activation('relu')(model)
    model = Dense(1)(model)
    model = Activation('tanh', name="value_head")(model)
    return model


def policy_head(model, reg_constant, flatten_size):
    model = Convolution2D(2, 1, padding="same",
                          data_format="channels_first", kernel_regularizer=l2(reg_constant))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Flatten()(model)
    model = Dense(flatten_size)(model)
    return Activation("softmax", name="policy_head")(model)


def tune(model: Model, inputs, probs, values, hyper_epochs, epochs):
    probs = np.array(probs)
    log_dir = "data/logs/tune/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    tuner = kt.Hyperband(model,
                         objective='val_accuracy',
                         max_epochs=hyper_epochs,
                         factor=3,
                         directory='data/tune',
                         project_name='catzero_tune'
                         )

    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(
        np.array(inputs),
        [probs.reshape((probs.shape[0], -1)), np.array(values)],
        epochs=epochs,
        validation_split=0.2,
        callbacks=[tensorboard_callback, stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""The hyperparameter search is complete. The optimal number of units in the first densely-connected layer is {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}. """)


def learn(model: Model, inputs, probs, values, batch_size, epochs):
        

    probs = np.array(probs)

    log_dir = "data/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    return model.fit(
        x=np.array(inputs),
        y=[probs.reshape((probs.shape[0], -1)), np.array(values)],
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[tensorboard_callback]
    )


def evaluate(tensor, model: Model, output_shape):
    result = model.predict(np.array([tensor]))
    action_probs = result[0].reshape(output_shape).tolist()

    reward = result[1][0, 0]

    return (reward, action_probs)


def save_model(path, model: Model):

    model.save(path)


def load_model(path):
    return keras.models.load_model(path)


def create_model(input_shape, output_shape, learning_rate, reg_const, res_blocks) -> Model:

    # Create the input layer
    nn_input = input_block(input_shape)

    # Apply the convolutional layer
    model = convolution_block(nn_input, reg_const)

    # Apply residual blocks
    for i in bb.range(res_blocks):
        bb.print("Residual block: {}".format(i))
        model = residual_block(model, reg_const)

    # Create the policy head
    policy_h = policy_head(model, reg_const, output_shape)

    # Create the value head
    value_h = value_head(model, reg_const)

    # To a model with multiple outputs
    model = Model(inputs=[nn_input], outputs=[policy_h, value_h])

    # Apply optimizer
    sgd = SGD(lr=learning_rate, momentum=0, decay=0, nesterov=False)

    # Compile model
    model.compile(optimizer=sgd,
                  loss=['categorical_crossentropy', 'mean_squared_error'],
                  loss_weights=[0.2, 0.2])

    return model
