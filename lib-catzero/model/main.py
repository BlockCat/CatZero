
if __name__ == "__main__":
    import numpy as np
    import keras
    import builtins as bb
    from keras.models import Model
    from keras.layers import BatchNormalization, Activation, Add, Dense, Convolution2D, Flatten, Input
    from keras.optimizers import SGD
    from keras.regularizers import l2

def input_block(bs):
    # One plane for player 1
    # One plane for player 2
    # One plane for player 1 previous
    # One plane for player 2 previous
    # One plane for colour to play
    return Input(batch_shape=bs)


def convolution_block(input: Model, reg_constant) -> Model:
    model = Convolution2D(256, 3, padding="same", data_format="channels_first", kernel_regularizer=l2(reg_constant))(input)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    return model


def residual_block(model: Model, reg_constant) -> Model:
    input = model
    model = Convolution2D(256, 3, padding="same", data_format="channels_first", kernel_regularizer=l2(reg_constant))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Convolution2D(256, 3, padding="same", data_format="channels_first", kernel_regularizer=l2(reg_constant))(model)
    model = BatchNormalization()(model)
    model = Add()([model, input])
    model = Activation('relu')(model)
    return model


def value_head(model: Model, reg_constant) -> Model:
    model = Convolution2D(1, 1, padding="same", data_format="channels_first", kernel_regularizer=l2(reg_constant))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Flatten()(model)
    model = Dense(256)(model)
    model = Activation('relu')(model)
    model = Dense(1)(model)
    model = Activation('tanh')(model)
    return model

def policy_head(model, reg_constant, flatten_size):
    model = Convolution2D(2, 1, padding="same",
                          data_format="channels_first", kernel_regularizer=l2(reg_constant))(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Flatten()(model)
    model = Dense(flatten_size)(model)
    return Activation("softmax", name="policy_h")(model)

def learn(model: Model, inputs, probs, values, batch_size, epochs):
    return model.fit(
        x = np.array([inputs]),
        y = [np.array(probs), np.array(values)],
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True
    )
    

def evaluate(tensor, model: Model, output_shape):     
    result = model.predict(np.array([tensor]))    
    action_probs = result[0].reshape(output_shape).tolist()

    reward = result[1][0, 0]
    
    return (reward, action_probs)

def save_model(path, model: Model):
    model.save(path)

def load_model(path):
    keras.models.load_model(path)

def create_model(input_shape, output_shape, reg_const, res_blocks) -> Model:
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
    sgd = SGD(lr=0.2, momentum=0, decay=0, nesterov=False)

    # Compile model
    model.compile(optimizer=sgd,
                  loss= ['categorical_crossentropy', 'mean_squared_error'],
                  loss_weights=[0.2, 0.2])

    return model