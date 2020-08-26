import sys

from keras import Input, Model
from keras.layers import LeakyReLU, Flatten, Dense, Conv2D, Lambda
from keras.optimizers import Adam

from layers.spectralnorm import Spectral
from models.basenet import BaseNet
import tensorflow as tf

sys.path.append('../layers')


class Discriminator(BaseNet):
    '''
    LS-GAN Discriminator
    '''

    def __init__(self, conf):
        super(Discriminator, self).__init__(conf)

    def build(self):
        inp_shape         = self.conf.input_shape
        downsample_blocks = self.conf.downsample_blocks
        output            = self.conf.output
        name              = self.conf.name
        spectral          = self.conf.spectral
        f                 = self.conf.filters

        d_input = Input(inp_shape)
        l = conv2d(f, 4, 2, False, None, d_input)
        l = LeakyReLU(0.2)(l)

        for i in range(downsample_blocks):
            s = 1 if i == downsample_blocks - 1 else 2
            spectral_params = f * (2 ** i)
            l = self._downsample_block(l, f * 2 * (2 ** i), s, spectral, spectral_params)

        l_before_output = l
        if output == '2D':
            spectral_params = f * (2 ** downsample_blocks) * 4 * 4
            l_discrimination = conv2d(1, 4, 1, spectral, spectral_params, l_before_output,
                                      spectrum_regularization = self.conf.spectrum_regularization)
        elif output == '1D':
            l_discrimination = Flatten()(l_before_output)
            l_discrimination = Dense(1, activation='linear')(l_discrimination)

        l_triplet = Flatten()(l_before_output)
        l_triplet = Dense(int(f / 4), activation='linear')(l_triplet)
        self.model = Model(d_input, [l_discrimination, l_triplet], name=name)


    def _downsample_block(self, l0, f, stride, spectral, spectral_params, name=''):
        l = conv2d(f, 4, stride, spectral, spectral_params * 4 * 4, l0, name,
                   spectrum_regularization = self.conf.spectrum_regularization)
        return LeakyReLU(0.2)(l)

    def compile(self):
        assert self.model is not None, 'Model has not been built'
        self.model.compile(optimizer=Adam(lr=self.conf.lr, beta_1=0.5, decay=self.conf.decay), loss='mse')


def conv2d(filters, kernel, stride, spectral, spectral_params, l0, name='', spectrum_regularization=10.):
    if spectral:
        l = Conv2D(filters, kernel, strides=stride, padding='same',
                   kernel_regularizer=Spectral(spectral_params, spectrum_regularization), name=name)(l0)
    else:
        l = Conv2D(filters, kernel, strides=stride, padding='same', name=name)(l0)
    return l
