from keras import Input, Model
from keras.layers import Concatenate, Conv2D, MaxPooling2D, LeakyReLU, Add, Activation, UpSampling2D, \
    BatchNormalization, Lambda, Dense, Flatten
from keras_contrib.layers import InstanceNormalization
from keras.backend import concatenate

from models.basenet import BaseNet
import logging
log = logging.getLogger('unet')
from keras.layers import concatenate # harric added to incorporate the segementation correction when segmentation_option=1
from keras.backend import expand_dims # because mi only in my
from keras import regularizers



class UNet(BaseNet):
    """
    UNet Implementation of 4 downsampling and 4 upsampling blocks.
    Each block has 2 convolutions, batch normalisation and relu.
    The number of filters for the 1st layer is 64 and at every block, this is doubled. Each upsampling block halves the
    number of filters.
    """
    def __init__(self, conf):
        """
        Constructor.
        :param conf: the configuration object
        """
        super(UNet, self).__init__(conf) # inherent from the BaseNet Class
        self.input_shape  = conf.input_shape
        self.residual     = conf.residual
        self.out_channels = conf.out_channels
        self.normalise    = conf.normalise
        self.f            = conf.filters
        self.downsample   = conf.downsample
        self.regularizer = conf.regularizer
        assert self.downsample > 0, 'Unet downsample must be over 0.'

    def build(self):
        """
        Build the model
        """
        self.input_shape = [self.input_shape[0], self.input_shape[1], 1]


        self.input = Input(shape=self.input_shape)
        l = self.unet_downsample(self.input, self.normalise)
        self.unet_bottleneck(l, self.normalise)
        l = self.unet_upsample(self.bottleneck, self.normalise)

        # harric modified to incorporate with segmentation_option=2 case
        # when the mask prediction is performed in a channel-wised manner
        # possibly useless
        out = self.normal_seg_out(l)


        # # harric added to ensure the output infarction mask is within the myocardium region
        # if self.conf.segmentation_option == '1':
        #     out = Lambda(self.infarction_mask_correction)(out)

        naming_layer1 = Lambda(lambda x: x, name='dice')
        naming_layer2 = Lambda(lambda x: x, name='cret')
        out1 = naming_layer1(out)
        out2 = naming_layer2(out)

        self.model = Model(inputs=self.input, outputs=[out1,out2])
        self.model.summary(print_fn=log.info)
        self.load_models() # loaded already trained model
                            # or pre-trained model
                            # or the training is started in the middle stage

    def unet_downsample(self, inp, normalise):
        """
        Build downsampling path
        :param inp:         input layer
        :param normalise:   normalise type. Can be one of [batch, instance, None]
        :return:            last layer of the downsampling path
        """
        self.d_l0 = conv_block(inp, self.f, normalise, residual=self.residual, regularizer=self.regularizer)
        l = MaxPooling2D(pool_size=(2, 2))(self.d_l0)

        if self.downsample > 1:
            self.d_l1 = conv_block(l, self.f * 2, normalise, residual=self.residual, regularizer=self.regularizer)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l1)

        if self.downsample > 2:
            self.d_l2 = conv_block(l, self.f * 4, normalise, residual=self.residual, regularizer=self.regularizer)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l2)

        if self.downsample > 3:
            self.d_l3 = conv_block(l, self.f * 8, normalise, residual=self.residual, regularizer=self.regularizer)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l3)
        return l

    def unet_bottleneck(self, l, normalise, name=''):
        """
        Build bottleneck layers
        :param inp:         input layer
        :param normalise:   normalise type. Can be one of [batch, instance, None]
        :param name:        name of the layer
        """
        flt = self.f * 2
        if self.downsample > 1:
            flt = flt * 2
        if self.downsample > 2:
            flt = flt * 2
        if self.downsample > 3:
            flt = flt * 2
        self.bottleneck = conv_block(l, flt, normalise, self.residual, name, regularizer=self.regularizer)

    def unet_upsample(self, l, normalise):
        """
        Build upsampling path
        :param l:           the input layer
        :param normalise:   normalise type. Can be one of [batch, instance, None]
        :return:            the last layer of the upsampling path
        """

        if self.downsample > 3:
            l = upsample_block(l, self.f * 8, normalise, activation='linear', regularizer=self.regularizer)
            l = Concatenate()([l, self.d_l3])
            l = conv_block(l, self.f * 8, normalise, self.residual, regularizer=self.regularizer)

        if self.downsample > 2:
            l = upsample_block(l, self.f * 4, normalise, activation='linear', regularizer=self.regularizer)
            l = Concatenate()([l, self.d_l2])
            l = conv_block(l, self.f * 4, normalise, self.residual, regularizer=self.regularizer)

        if self.downsample > 1:
            l = upsample_block(l, self.f * 2, normalise, activation='linear', regularizer=self.regularizer)
            l = Concatenate()([l, self.d_l1])
            l = conv_block(l, self.f * 2, normalise, self.residual, regularizer=self.regularizer)

        if self.downsample > 0:
            l = upsample_block(l, self.f, normalise, activation='linear', regularizer=self.regularizer)
            l = Concatenate()([l, self.d_l0])
            l = conv_block(l, self.f, normalise, self.residual, regularizer=self.regularizer)

        return l

    def normal_seg_out(self, l, out_activ=None, out_channels=-1):
        """
        Build ouput layer
        :param l: last layer from the upsampling path
        :return:  the final segmentation layer
        """
        if out_activ is None:
            out_activ = 'sigmoid' if out_channels == 1 else 'softmax'
        return Conv2D(out_channels, 1, activation=out_activ,
                      kernel_regularizer=regularizers.l2(self.regularizer))(l)




    # # harric added to ensure the output infarction mask is within the myocardium region
    # def infarction_mask_correction(self,output_mask):
    #     my_out = expand_dims(output_mask[..., 0], axis=-1)
    #     mi_out = expand_dims(output_mask[..., 1], axis=-1)
    #     back_out = expand_dims(output_mask[..., 2], axis=-1)
    #     mi_out = mi_out * my_out
    #     output_mask_corrected = concatenate([my_out, mi_out, back_out], axis=3)
    #     return output_mask_corrected```````````

def conv_block(l0, f, norm_name, residual=False, name='', regularizer=0):
    """
    Convolutional block
    :param l0:        the input layer
    :param f:         number of feature maps
    :param residual:  True/False to define residual connections
    :return:          the last layer of the convolutional block
    """
    l = Conv2D(f, 3, strides=1, padding='same',
               kernel_regularizer=regularizers.l2(regularizer))(l0)
    l = normalise(norm_name)(l)
    l = Activation('relu')(l)
    l = Conv2D(f, 3, strides=1, padding='same',
               kernel_regularizer=regularizers.l2(regularizer))(l)
    l = normalise(norm_name)(l)
    if residual:
        Activation('relu')(l)
        # return Add(name=name)([l0, l])
        return Concatenate()([l0, l])
    return Activation('relu', name=name)(l)


def upsample_block(l0, f, norm_name, activation='relu', regularizer=0):
    """
    Upsampling block.
    :param l0:          input layer
    :param f:           number of feature maps
    :param activation:  activation name
    :return:            the last layer of the upsampling block
    """
    l = UpSampling2D(size=2)(l0)
    l = Conv2D(f, 3, padding='same',
               kernel_regularizer=regularizers.l2(regularizer))(l)
    l = normalise(norm_name)(l)

    if activation == 'leakyrelu':
        return LeakyReLU()(l)
    else:
        return Activation(activation)(l)


def normalise(norm=None, **kwargs):
    if norm == 'instance':
        return InstanceNormalization(**kwargs)
    elif norm == 'batch':
        return BatchNormalization()
    else:
        return Lambda(lambda x : x)