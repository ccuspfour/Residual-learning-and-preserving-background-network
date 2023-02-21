from keras.models import Model
from keras.layers import Input, Add,Subtract, ReLU, Conv2DTranspose, multiply
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf




    def background(inputs):
        def block(inputs):
        #residual block
            for i in range(4):
             a = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
             a = BatchNormalization()(a)
             a = ReLU(shared_axes=[1, 2])(a)
             a = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(a)
             a = BatchNormalization()(a)
             a = Add()([a, inputs])
             return a

        for i in range(8):
            a = block(a)
            
        a = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(a)
        return a

    def residual_net(inputs):
        def block(inputs):
        #residual block
            for i in range(4):
             b = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
             b = BatchNormalization()(b)
             b = ReLU(shared_axes=[1, 2])(b)
             b = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(b)
             b = BatchNormalization()(b)
             b = Add()([b, inputs])
             return b

        for i in range(8):
            b = residual_block(inputs)

        b = Add()([b, inputs])
        b = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(b)
        return b

    inputs = Input(shape=(None, None, 32))
    Residual = residual_net(inputs)

    out1 = Subtract()([inputs,Residual])
    out2 = background(inputs)
    end = Add()([out1, out2])

    return end

