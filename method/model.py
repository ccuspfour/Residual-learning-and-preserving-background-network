from keras.models import Model
from keras.layers import Input, Add,Subtract, ReLU, Conv2DTranspose, multiply
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf

class L0Loss:
    def __init__(self):
        self.gamma = K.variable(2.)

    def __call__(self):
        def calc_loss(y_true, y_pred):
            loss = K.pow(K.abs(y_true - y_pred) + 1e-8, self.gamma)
            return loss
        return calc_loss


class UpdateAnnealingParameter(Callback):
    def __init__(self, gamma, nb_epochs, verbose=0):
        super(UpdateAnnealingParameter, self).__init__()
        self.gamma = gamma
        self.nb_epochs = nb_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_gamma = 2.0 * (self.nb_epochs - epoch) / self.nb_epochs
        K.set_value(self.gamma, new_gamma)

        if self.verbose > 0:
            print('\nEpoch %05d: UpdateAnnealingParameter reducing gamma to %s.' % (epoch + 1, new_gamma))


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    y_true_t = y_true *255.0
    max_pixel = 255.0
    y_pred_t = K.clip(y_pred*255.0, 0.0, 255.0)
    # y_pred = K.clip(y_pred, 0.0, 1.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred_t - y_true_t))))


def get_model(model_name="the_end"):

    if model_name == "the_end":
        return get_the_end_model()
    else:
        raise ValueError("model_name should be 'srresnet'or 'unet'")


def get_the_end_model(input_channel_num=3, feature_dim=64, resunit_num=16):



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

def main():
    # model = get_model()
    model = get_model("unet")
    model.summary()


if __name__ == '__main__':
    main()