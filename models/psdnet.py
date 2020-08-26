eps = 1e-12 # harric added to engage the smooth factor
import logging
import os

import keras.backend as K
from keras import Input, Model
from keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Lambda, \
    Concatenate, BatchNormalization, Reshape, Add
from keras.optimizers import Adam
from keras.backend import concatenate, ones_like

from keras.layers import concatenate # harric added to incorporate the segementation correction when segmentation_option=1


import costs
from costs import make_dice_loss_fnc, \
    weighted_softmax_cross_entropy, \
    make_tversky_loss_func, \
    make_focal_loss_func,make_weighted_mae_loss_func, make_triplet_loss
from layers.film import FiLM
from layers.rounding import Rounding
from models.basenet import BaseNet
from models.discriminator import Discriminator
from models.unet import UNet
from utils.sdnet_utils import sampling, make_trainable, get_net

log = logging.getLogger('sdnet')

class SDNet(BaseNet):
    """
    The SDNet model builder.
    """
    def __init__(self, conf):
        """
        SDNet constructor
        :param conf: configuration object
        """
        super(SDNet, self).__init__(conf)

        self.Enc_Anatomy  = None  # Anatomy Encoder
        self.Enc_Pathology = None
        self.Enc_Modality = None  # Modality Encoder
        self.Segmentor    = None  # Segmentor
        self.Decoder      = None  # Decoder

        self.G_trainer_lp_rp    = None  #
        self.G_trainer_lp_pppa = None  # Trainer when using data with labels.
        self.G_trainer_lp_ppra = None  # Trainer when using data with labels.
        self.G_trainer_up_pppa = None  # Trainer when using data with labels.
        self.G_trainer_up_ppra = None
        self.G_trainer_up_anatomy  = None  # Trainer for reconstructing a sampled Z

        self.G_trainer_lp_pppa_reconst = None  # Trainer for reconstructing a sampled Z
        self.G_trainer_lp_ppra_reconst = None  # Trainer for reconstructing a sampled Z
        self.G_trainer_up_pppa_reconst = None  # Trainer for reconstructing a sampled Z
        self.G_trainer_up_ppra_reconst = None  # Trainer for reconstructing a sampled Z

        self.D_Reconstruction_trainer_lp_rp = None  # Trainer for reconstruction discriminator
        self.D_Reconstruction_trainer_pppa = None
        self.D_Reconstruction_trainer_ppra = None

        self.z_reconstructor = None

        self.out_pathology_channels = conf.out_pathology_channels
        self.out_anatomy_channels = conf.out_anatomy_channels

    def build(self):
        """
        Build the model's components
        """
        self.build_discriminators()
        self.build_generators()

        self.build_supervised_trainer()
        self.build_latent_factor_regressor()
        self.build_triple_trainer()
        mark = True
        if not mark:
            log.info("Initial Model didnt find")
            return False
        self.load_models()
        return True

    def load_models(self):
        """
        Load weights from saved model files
        """
        if os.path.exists(self.conf.folder + '/Enc_Pathology'):
            log.info('Loading trained models from file')
            self.Enc_Anatomy.load_weights(self.conf.folder + '/Enc_Anatomy')
            self.Enc_Modality.load_weights(self.conf.folder+'/Enc_Modality')
            self.Segmentor.load_weights(self.conf.folder+'/Segmentor_Anatomy')
            self.Enc_Pathology.load_weights(self.conf.folder+'/Enc_Pathology')
            self.Decoder.load_weights(self.conf.folder+'/Reconstructor')
            self.D_Reconstruction.load_weights(self.conf.folder+'/D_Reconstructor')
            self.build_latent_factor_regressor()
        else:
            log.info('No trained model found !!!')

    def save_models(self, postfix=''):
        """
        Save model weights in files.
        """
        if not postfix=='':
            postfix = '_' + postfix
        log.debug('Saving trained models')

        self.Enc_Anatomy.save_weights(self.conf.folder + '/Enc_Anatomy' + postfix)
        self.Enc_Modality.save_weights(self.conf.folder + '/Enc_Modality' + postfix)
        self.Segmentor.save_weights(self.conf.folder + '/Segmentor_Anatomy' + postfix)
        self.Enc_Pathology.save_weights(self.conf.folder + '/Enc_Pathology' + postfix)
        self.Decoder.save_weights(self.conf.folder + '/Reconstructor' + postfix)
        self.D_Reconstruction.save_weights(self.conf.folder + '/D_Reconstructor' + postfix)

    def build_reconstruction_discriminator(self):
        """
        Build a Keras model for training a mask discriminator.
        """
        # Build a discriminator for masks.
        D = Discriminator(self.conf.d_reconstruct_params)
        D.build()
        log.info('Reconstruction Discriminator D_Reconstruction')
        D.model.summary(print_fn=log.info)
        self.D_Reconstruction = D.model

        real_Reconstruct = Input(self.conf.d_reconstruct_params.input_shape)
        fake_Reconstruct = Input(self.conf.d_reconstruct_params.input_shape)

        real_discrimination,  real_triplet = self.D_Reconstruction(real_Reconstruct)
        fake_discrimination,  fake_triplet = self.D_Reconstruction(fake_Reconstruct)

        naming1 = Lambda(lambda x: x, name='Adv_Reconstruction_ActualPathology_Real_ActualPathology')
        naming2 = Lambda(lambda x: x, name='Adv_Reconstruction_ActualPathology_Fake_ActualPathology')
        #naming3 = Lambda(lambda x: x, name='Clsf_Reconstruction_ActualPathology_Real_ActualPathology')
        #naming4 = Lambda(lambda x: x, name='Clsf_Reconstruction_ActualPathology_Fake_ActualPathology')
        real_discrimination = naming1(real_discrimination)
        fake_discrimination = naming2(fake_discrimination)
        #real_classification = naming3(real_classification)
        #fake_classification = naming4(fake_classification)

        # trainer for reconstruction discriminator for real pathology (rp)
        # it is only valid with the labeled pathology case
        self.D_Reconstruction_trainer_lp_rp = Model([real_Reconstruct, fake_Reconstruct],
                                                    [real_discrimination, fake_discrimination,
                                                     real_triplet, fake_triplet],
                                                    name='D_Reconstruction_Trainer')
        self.D_Reconstruction_trainer_lp_rp.compile(Adam(lr=self.conf.d_reconstruct_params.lr,
                                                         beta_1=0.5, decay=self.conf.d_reconstruct_params.decay),
                                                    loss=['mse','mse',costs.ypred,costs.ypred],
                                                    loss_weights=[0.5*self.conf.w_adv_X*self.conf.real_pathology_weight_rate+eps,
                                                                  0.5*self.conf.w_adv_X*self.conf.real_pathology_weight_rate+eps,
                                                                  0,0])
        self.D_Reconstruction_trainer_lp_rp.summary(print_fn=log.info)


        # trainer for reconstruction discriminator of predicted pathology predicted anatomy (pppa)
        # it has nothing to do with real pathology label
        # so it is put in the unlabeled pathology branch in order to make use of more data
        self.D_Reconstruction_trainer_up_pppa = Model([real_Reconstruct, fake_Reconstruct],
                                                      [real_discrimination, fake_discrimination,
                                                       real_triplet, fake_triplet],
                                                      name='D_Reconstruction_Trainer_Supervised')
        self.D_Reconstruction_trainer_up_pppa.compile(Adam(lr=self.conf.d_reconstruct_params.lr,
                                                           beta_1=0.5,
                                                           decay=self.conf.d_reconstruct_params.decay),
                                                      loss=['mse', 'mse',costs.ypred,costs.ypred],
                                                      loss_weights=[0.5 * self.conf.w_adv_X * self.conf.pred_pathology_weight_rate * self.conf.pred_anatomy_weight_rate + eps,
                                                                    0.5 * self.conf.w_adv_X * self.conf.pred_pathology_weight_rate * self.conf.pred_anatomy_weight_rate + eps,
                                                                    0,0])
        self.D_Reconstruction_trainer_up_pppa.summary(print_fn=log.info)

        # trainer for reconstruction discriminator of predicted pathology real anatomy (ppra)
        # it has nothing to do with real pathology label
        # so it is put in the unlabeled pathology branch in order to make use of more data
        self.D_Reconstruction_trainer_up_ppra = Model([real_Reconstruct, fake_Reconstruct],
                                                      [real_discrimination, fake_discrimination,
                                                       real_triplet, fake_triplet],
                                                      name='D_Reconstruction_Trainer_Supervised')
        self.D_Reconstruction_trainer_up_ppra.compile(Adam(lr=self.conf.d_reconstruct_params.lr,
                                                           beta_1=0.5,
                                                           decay=self.conf.d_reconstruct_params.decay),
                                                      loss=['mse', 'mse',costs.ypred,costs.ypred],
                                                      loss_weights=[
                                                          0.5 * self.conf.w_adv_X * self.conf.pred_pathology_weight_rate * self.conf.real_anatomy_weight_rate + eps,
                                                          0.5 * self.conf.w_adv_X * self.conf.pred_pathology_weight_rate * self.conf.real_anatomy_weight_rate + eps,
                                                          0,0])
        self.D_Reconstruction_trainer_up_ppra.summary(print_fn=log.info)


    def build_generators(self):
        """
        Build encoders, segmentor, decoder and training models.
        """
        assert self.D_Reconstruction is not None,  'Discriminator has not been built yet'
        make_trainable(self.D_Reconstruction_trainer_lp_rp, False)
        make_trainable(self.D_Reconstruction_trainer_up_pppa, False)
        make_trainable(self.D_Reconstruction_trainer_up_ppra, False)

        self.build_anatomy_encoder() # unet encoder
        self.build_pathology_encoder_unet()

        self.build_modality_encoder() # vae architecture
        self.build_anatomy_segmentor()
        self.build_decoder_film() # FiLM


    def build_discriminators(self):
        self.build_reconstruction_discriminator()


    def build_anatomy_encoder(self):
        """
        Build an encoder to extract anatomical information from the image.
        """
        # Manually build UNet to add Rounding as a last layer
        spatial_encoder = UNet(self.conf.anatomy_encoder_params)
        spatial_encoder.input = Input(shape=self.conf.input_shape)
        l1 = spatial_encoder.unet_downsample(spatial_encoder.input, spatial_encoder.normalise)
        spatial_encoder.unet_bottleneck(l1, spatial_encoder.normalise)
        l2 = spatial_encoder.unet_upsample(spatial_encoder.bottleneck, spatial_encoder.normalise)
        anatomy = spatial_encoder.normal_seg_out(l2, out_activ='softmax',
                                                 out_channels=self.conf.anatomy_encoder_params.out_channels)
        if self.conf.rounding == 'encoder':
            anatomy = Rounding()(anatomy)
        s_factors = Lambda(lambda x: x, name='s')(anatomy)


        self.Enc_Anatomy = Model(inputs=spatial_encoder.input, outputs=s_factors, name='Enc_Anatomy')
        log.info('Enc_Anatomy')
        self.Enc_Anatomy.summary(print_fn=log.info)

    def build_modality_encoder(self):
        """
        Build an encoder to extract intensity information from the image.
        """
        anatomy = Input(self.Enc_Anatomy.output_shape[1:])
        pathology = Input(shape=tuple(self.conf.input_shape[:-1]) + (self.conf.num_pathology_masks,))
        image = Input(self.conf.input_shape)

        l = Concatenate(axis=-1)([anatomy, pathology, image])
        l = Conv2D(16, 3, strides=2)(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)
        l = Conv2D(16, 3, strides=2)(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)
        l = Conv2D(16, 3, strides=2)(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)
        l = Conv2D(16, 3, strides=2)(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)
        l = Flatten()(l)
        l = Dense(32)(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)

        z_mean = Dense(self.conf.num_z, name='z_mean')(l)
        z_log_var = Dense(self.conf.num_z, name='z_log_var')(l)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, name='z')([z_mean, z_log_var])
        divergence = Lambda(costs.kl, name='divergence')([z_mean, z_log_var])

        self.Enc_Modality = Model(inputs=[anatomy, pathology, image], outputs=[z, divergence], name='Enc_Modality')
        log.info('Enc_Modality')
        self.Enc_Modality.summary(print_fn=log.info)


    def build_pathology_encoder_unet(self):
        """
        Build an encoder to extract anatomical information from the image.
        """
        # Manually build UNet to add Rounding as a last layer
        spatial_encoder = UNet(self.conf.pathology_encoder_params)
        spatial_encoder.input = Input(shape=self.conf.input_shape[:2]+[self.conf.out_anatomy_channels+self.conf.input_shape[-1]])
        l1 = spatial_encoder.unet_downsample(spatial_encoder.input, spatial_encoder.normalise)
        spatial_encoder.unet_bottleneck(l1, spatial_encoder.normalise)
        l2 = spatial_encoder.unet_upsample(spatial_encoder.bottleneck, spatial_encoder.normalise)
        pathology = spatial_encoder.normal_seg_out(l2, out_activ='softmax', out_channels=self.out_pathology_channels)


        self.Enc_Pathology = Model(inputs=spatial_encoder.input, outputs=pathology, name='Enc_Pathology')
        log.info('Enc_Pathology')
        self.Enc_Pathology.summary(print_fn=log.info)


    def build_pathology_encoder_plain(self):
        inp = Input(shape=self.conf.input_shape[:2]+[self.conf.out_anatomy_channels+self.conf.input_shape[-1]])
        l = Conv2D(64, 3, strides=1, padding='same')(inp)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)
        l = Conv2D(64, 3, strides=1, padding='same')(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)

        pathology = Conv2D(self.out_pathology_channels, 1, padding='same', activation='softmax')(l)

        self.Enc_Pathology = Model(inputs=inp, outputs=pathology, name='Enc_Pathology')
        log.info('Anatomy_Segmentor')
        self.Enc_Pathology.summary(print_fn=log.info)




    def build_anatomy_segmentor(self):
        """
        Build a segmentation network that converts anatomical maps to segmentation masks.
        """
        inp = Input(self.Enc_Anatomy.output_shape[1:])
        l = Conv2D(64, 3, strides=1, padding='same')(inp)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)
        l = Conv2D(64, 3, strides=1, padding='same')(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)

        # conv_channels = self.loader.num_masks + 1  # +1 for softmax
        conv_channels = self.out_anatomy_channels
        output = Conv2D(conv_channels, 1, padding='same', activation='softmax')(l)

        # output = Lambda(lambda x: x[..., 0:conv_channels - 1])(output)

        self.Segmentor = Model(inputs=inp, outputs=output, name='Anatomy_Segmentor')
        log.info('Anatomy_Segmentor')
        self.Segmentor.summary(print_fn=log.info)

    def build_decoder_film(self):
        def _film_pred(z, num_chn):
            """
            Given a z-sample, predict gamma and beta to apply FiLM.
            :param z:           a modality sample
            :param num_chn:     number of channels of the spatial feature maps
            :return:            the FiLM parameters
            """
            film_pred = Dense(num_chn)(z)
            film_pred = LeakyReLU()(film_pred)
            film_pred = Dense(num_chn)(film_pred)
            gamma = Lambda(lambda x: x[:, :int(num_chn / 2)])(film_pred)
            beta = Lambda(lambda x: x[:, int(num_chn / 2):])(film_pred)
            return gamma, beta

        def _film_layer(spatial_input, resd_input):
            """
            A FiLM layer. Modulates the spatial input by the residual input.
            :param spatial_input:   the spatial features of the anatomy
            :param resd_input:      the modality features
            :return:                a modulated anatomy
            """
            l1 = Conv2D(self.conf.num_mask_channels, 3, padding='same')(spatial_input)
            l1 = LeakyReLU()(l1)

            l2 = Conv2D(self.conf.num_mask_channels, 3, strides=1, padding='same')(l1)
            gamma_l2, beta_l2 = _film_pred(resd_input, 2 * self.conf.num_mask_channels)
            l2 = FiLM()([l2, gamma_l2, beta_l2])
            l2 = LeakyReLU()(l2)

            l = Add()([l1, l2])
            return l

        """
        Build a decoder that generates an image by combining an anatomical and a modality
        representation.
        """
        spatial_shape = tuple(self.conf.input_shape[:-1]) + (self.conf.num_mask_channels,)
        spatial_input = Input(shape=spatial_shape)
        pathology_factor_input = Input(shape=tuple(self.conf.input_shape[:-1]) + (1,))

        resd_input = Input((self.conf.num_z,))  # (batch_size, 16)
        l1 = _film_layer(concatenate([spatial_input, pathology_factor_input],axis=-1),
                         resd_input)
        l2 = _film_layer(l1, resd_input)
        l3 = _film_layer(l2, resd_input)
        l4 = _film_layer(l3, resd_input)

        l = Conv2D(self.conf.input_shape[-1], 3, activation='tanh', padding='same')(l4)
        log.info('Reconstructor')
        self.Decoder = Model(inputs=[spatial_input, pathology_factor_input, resd_input], outputs=l, name='Reconstructor')
        self.Decoder.summary(print_fn=log.info)

    def build_decoder_film_mod(self):
        def _film_pred(z, num_chn):
            """
            Given a z-sample, predict gamma and beta to apply FiLM.
            :param z:           a modality sample
            :param num_chn:     number of channels of the spatial feature maps
            :return:            the FiLM parameters
            """
            film_pred = Dense(num_chn)(z)
            film_pred = LeakyReLU()(film_pred)
            film_pred = Dense(num_chn)(film_pred)
            gamma = Lambda(lambda x: x[:, :int(num_chn / 2)])(film_pred)
            beta = Lambda(lambda x: x[:, int(num_chn / 2):])(film_pred)
            return gamma, beta

        def _film_layer(spatial_input, resd_input, resd_feature_map):
            """
            A FiLM layer. Modulates the spatial input by the residual input.
            :param spatial_input:   the spatial features of the anatomy
            :param resd_input:      the modality features
            :return:                a modulated anatomy
            """
            l1 = Conv2D(self.conf.num_mask_channels, 3, padding='same')(spatial_input)
            l1 = LeakyReLU()(l1)

            l2 = Conv2D(self.conf.num_mask_channels, 3, strides=1, padding='same')(l1)
            gamma_l2, beta_l2 = _film_pred(resd_input, 2 * self.conf.num_mask_channels)
            l2 = FiLM()([l2, gamma_l2, beta_l2])
            l2 = LeakyReLU()(l2)

            l = Add()([l1, l2])

            l = Add()([l, spatial_input])
            l = Add()([l, resd_feature_map])

            return l

        """
        Build a decoder that generates an image by combining an anatomical and a modality
        representation.
        """
        spatial_shape = tuple(self.conf.input_shape[:-1]) + (self.conf.num_mask_channels,)
        spatial_input = Input(shape=spatial_shape)
        pathology_factor_input = Input(shape=tuple(self.conf.input_shape[:-1]) + (self.conf.num_pathology_masks,))
        resd_input = Input((self.conf.num_z,))  # (batch_size, 16)

        l0 = Conv2D(self.conf.num_mask_channels, 3, padding='same')(concatenate([spatial_input, pathology_factor_input]))
        l0 = LeakyReLU()(l0)

        h, w = int(spatial_input.shape[1]), \
               int(spatial_input.shape[2])
        c = self.conf.num_mask_channels
        l = Dense(2 * c)(resd_input)
        l = LeakyReLU()(l)
        l = Dense(h * w * c)(l)
        l = LeakyReLU()(l)
        resd_feature_map = Reshape(target_shape=(h, w, c))(l)

        l1 = _film_layer(l0, resd_input, resd_feature_map)
        l2 = _film_layer(l1, resd_input, resd_feature_map)
        l3 = _film_layer(l2, resd_input, resd_feature_map)
        l4 = _film_layer(l3, resd_input, resd_feature_map)

        l = Conv2D(self.conf.input_shape[-1], 3, activation='tanh', padding='same')(l4)
        log.info('Reconstructor')
        self.Decoder = Model(inputs=[spatial_input, pathology_factor_input, resd_input], outputs=l,
                             name='Reconstructor')
        self.Decoder.summary(print_fn=log.info)

    def build_supervised_trainer(self):
        def _build_graph(pathology):

            fake_Z, divergence = self.Enc_Modality([fake_S, pathology, real_X])

            mapping_layer1_dice = Lambda(lambda x: x, name='Dice_Anato')
            mapping_layer2_crossentropy = Lambda(lambda x: x, name='CrossEntropy_Anato')

            mapping_layer4_dice = Lambda(lambda x: x, name='Dice_Patho')
            mapping_layer5_crossentropy = Lambda(lambda x: x, name='CrossEntropy_Patho')
            fake_M_Dice = mapping_layer1_dice(fake_anatomy_mask)
            fake_M_CrossEntropy = mapping_layer2_crossentropy(fake_anatomy_mask)

            fake_P_Dice = mapping_layer4_dice(pathology)
            fake_P_CrossEntropy = mapping_layer5_crossentropy(pathology)

            rec_x = self.Decoder([fake_S, pathology, fake_Z])
            adv_reconstruction_pathology,  _ = self.D_Reconstruction(rec_x)

            return fake_M_Dice, fake_M_CrossEntropy, \
                   fake_P_Dice, fake_P_CrossEntropy, \
                   divergence, \
                   rec_x, adv_reconstruction_pathology
        """
        Model for training SDNet given labelled images. In addition to the unsupervised trainer, a direct segmentation
        cost is also minimised.
        """

        ce_anatomy = weighted_softmax_cross_entropy(self.conf.num_anatomy_masks)  # cross entropy loss
        dice_anatomy = make_dice_loss_fnc(self.conf.num_anatomy_masks)
        ce_pathology = make_focal_loss_func(self.conf.num_pathology_masks, gamma=2)  # cross entropy loss
        dice_pathology = make_tversky_loss_func(self.conf.num_pathology_masks, beta=0.7)

        real_X = Input(self.conf.input_shape)
        anatomy_real = Input(tuple(self.conf.input_shape[:-1]) + (self.conf.num_anatomy_masks,))
        def _anatomy_real_correction(input):
            for ii in range(self.conf.num_anatomy_masks):
                if ii == 0:
                    added = input[:,:,:,ii:ii+1]
                else:
                    added = added+input[:,:,:,ii:ii+1]
            background = ones_like(input[:,:,:,0:1]) - added
            output = concatenate([input,background], axis=-1)
            return output
        anatomy_real_with_background = Lambda(_anatomy_real_correction)(anatomy_real)

        pathology_real = Input(tuple(self.conf.input_shape[:-1]) + (self.conf.num_pathology_masks,))
        pseudo_healthy = Input(tuple(self.conf.input_shape[:-1]) + (self.conf.num_pathology_masks,))

        fake_S = self.Enc_Anatomy(real_X)
        fake_anatomy_mask = self.Segmentor(fake_S)
        predicted_pathology_from_predicted_anatomy = self.Enc_Pathology(Concatenate(axis=-1)([real_X, fake_anatomy_mask]))
        predicted_pathology_from_true_anatomy = self.Enc_Pathology(Concatenate(axis=-1)([real_X, anatomy_real_with_background]))


        pathology_correction = Lambda(lambda x: x[..., 0:self.conf.num_pathology_masks])
        predicted_pathology_from_predicted_anatomy = pathology_correction(predicted_pathology_from_predicted_anatomy)
        predicted_pathology_from_true_anatomy = pathology_correction(predicted_pathology_from_true_anatomy)


        # graph from real pathology (rp)
        # only exists for labeled pathology case
        _, _, \
        _, _, \
        divergence_rp, \
        rec_x_rp, \
        adv_reconstruction_pathology_rp \
            = _build_graph(pathology_real)
        weighted_mae_rp_reconst = make_weighted_mae_loss_func([rec_x_rp, real_X, pathology_real,
                                                               self.conf.w_sup_PathoMask / (self.conf.w_sup_AnatoMask + eps), '_RP'])
        self.G_trainer_lp_rp \
            = Model(inputs=[real_X, pathology_real],
                    outputs=[divergence_rp,
                             # rec_x_rp,
                             weighted_mae_rp_reconst,
                             adv_reconstruction_pathology_rp,
                             ])
        self.G_trainer_lp_rp.summary(print_fn=log.info)
        self.G_trainer_lp_rp.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                     loss=[costs.ypred,
                                           costs.ypred,
                                           'mse'],
                                     loss_weights=[
                                         self.conf.w_kl * self.conf.real_pathology_weight_rate + eps,
                                         self.conf.w_rec_X * self.conf.real_pathology_weight_rate * self.conf.pe_weight + eps,
                                         self.conf.w_adv_X * self.conf.real_pathology_weight_rate + eps])  # harric modified

        # graph from predicted pathology from predicted anatomy (pppa)
        fake_A_Dice, fake_A_CrossEntropy, \
        fake_P_Dice_pppa, fake_P_CrossEntropy_pppa, \
        divergence_pppa, \
        rec_x_pppa, \
        adv_reconstruction_pathology_pppa \
            = _build_graph(predicted_pathology_from_predicted_anatomy)
        weighted_mae_pppa_reconst = make_weighted_mae_loss_func([rec_x_pppa, real_X, pathology_real,
                                                                 self.conf.w_sup_PathoMask / (self.conf.w_sup_AnatoMask + eps), '_PPPA'])
        # for labeled pathology case
        self.G_trainer_lp_pppa \
            = Model(inputs=[real_X, pathology_real],
                    outputs=[fake_P_Dice_pppa, fake_P_CrossEntropy_pppa,
                             ])
        self.G_trainer_lp_pppa.summary(print_fn=log.info)
        self.G_trainer_lp_pppa.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                       loss=[dice_pathology, ce_pathology],
                                       loss_weights=[
                                           self.conf.w_sup_PathoMask * self.conf.pred_pathology_weight_rate * self.conf.pred_anatomy_weight_rate + eps,
                                           self.conf.w_sup_PathoMask * self.conf.pred_pathology_weight_rate * self.conf.pred_anatomy_weight_rate * self.conf.ce_focal_patho_weight + eps])

        make_trainable(self.Decoder, False)
        self.G_trainer_lp_pppa_reconst \
            = Model(inputs=[real_X, pathology_real],
                    outputs=[weighted_mae_pppa_reconst])
        self.G_trainer_lp_pppa_reconst.summary(print_fn=log.info)
        self.G_trainer_lp_pppa_reconst.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                       loss=[costs.ypred],
                                       loss_weights=[self.conf.w_rec_X
                                                     * self.conf.pred_pathology_weight_rate
                                                     * self.conf.pred_anatomy_weight_rate
                                                     * self.conf.pe_weight + eps])
        make_trainable(self.Decoder, True)

        # unlabelled pathology
        self.G_trainer_up_pppa \
            = Model(inputs=[real_X],
                    outputs=[divergence_pppa,
                             adv_reconstruction_pathology_pppa,
                             ])
        self.G_trainer_up_pppa.summary(print_fn=log.info)
        self.G_trainer_up_pppa.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                       loss=[costs.ypred,
                                             'mse',
                                             ],
                                       loss_weights=[
                                           self.conf.w_kl * self.conf.pred_pathology_weight_rate * self.conf.pred_anatomy_weight_rate + eps,
                                           self.conf.w_adv_X * self.conf.pred_pathology_weight_rate * self.conf.pred_anatomy_weight_rate + eps,
                                       ])

        make_trainable(self.Decoder, False)
        self.G_trainer_up_pppa_reconst \
            = Model(inputs=[real_X],
                    outputs=[rec_x_pppa])
        self.G_trainer_up_pppa_reconst.summary(print_fn=log.info)
        self.G_trainer_up_pppa_reconst.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                       loss=['mse'],
                                       loss_weights=[self.conf.w_rec_X
                                                     * self.conf.pred_pathology_weight_rate
                                                     * self.conf.pred_anatomy_weight_rate + eps])
        make_trainable(self.Decoder, True)


        # for anatomy only
        self.G_trainer_up_anatomy \
            = Model(inputs=[real_X],
                    outputs=[fake_A_Dice, fake_A_CrossEntropy,
                             # adv_Anatomy_Segmentation
                             ])
        self.G_trainer_up_anatomy.summary(print_fn=log.info)

        self.G_trainer_up_anatomy.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                          loss=[dice_anatomy, ce_anatomy,
                                                ],
                                          loss_weights=[self.conf.w_sup_AnatoMask + eps,
                                                        self.conf.w_sup_AnatoMask * self.conf.ce_focal_anato_weight + eps,
                                                        ])


        # graph from predicted pathology from real anatomy (ppra)
        _, _, \
        fake_P_Dice_ppta, fake_P_CrossEntropy_ppta, \
        divergence_ppta, \
        rec_x_ppta, \
        adv_reconstruction_pathology_ppta \
            = _build_graph(predicted_pathology_from_true_anatomy)
        weighted_mae_ppra_reconst = make_weighted_mae_loss_func([rec_x_ppta, real_X, pathology_real,
                                                                 self.conf.w_sup_PathoMask / (self.conf.w_sup_AnatoMask + eps), '_PPRA'])
        # for labeled pathology case
        self.G_trainer_lp_ppra \
            = Model(inputs=[real_X, anatomy_real, pathology_real],
                    outputs=[fake_P_Dice_ppta, fake_P_CrossEntropy_ppta,
                             ])
        self.G_trainer_lp_ppra.summary(print_fn=log.info)
        self.G_trainer_lp_ppra.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                       loss=[dice_pathology, ce_pathology],
                                       loss_weights=[
                                           self.conf.w_sup_PathoMask * self.conf.pred_pathology_weight_rate * self.conf.real_anatomy_weight_rate + eps,
                                           self.conf.w_sup_PathoMask * self.conf.pred_pathology_weight_rate * self.conf.real_anatomy_weight_rate * self.conf.ce_focal_patho_weight + eps])

        make_trainable(self.Decoder, False)
        self.G_trainer_lp_ppra_reconst \
            = Model(inputs=[real_X, anatomy_real, pathology_real],
                    outputs=[weighted_mae_ppra_reconst])
        self.G_trainer_lp_ppra_reconst.summary(print_fn=log.info)
        self.G_trainer_lp_ppra_reconst.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                       loss=[costs.ypred],
                                       loss_weights=[self.conf.w_rec_X
                                                     * self.conf.pred_pathology_weight_rate
                                                     * self.conf.real_anatomy_weight_rate
                                                     * self.conf.pe_weight + eps])
        make_trainable(self.Decoder, True)



        # for unlabeled pathology case
        self.G_trainer_up_ppra \
            = Model(inputs=[real_X, anatomy_real],
                    outputs=[divergence_ppta,
                             adv_reconstruction_pathology_ppta,
                             ])
        self.G_trainer_up_ppra.summary(print_fn=log.info)
        self.G_trainer_up_ppra.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                       loss=[costs.ypred,
                                             'mse',
                                             ],
                                       loss_weights=[
                                           self.conf.w_kl * self.conf.pred_pathology_weight_rate * self.conf.real_anatomy_weight_rate + eps,
                                           self.conf.w_adv_X * self.conf.pred_pathology_weight_rate * self.conf.real_anatomy_weight_rate + eps,
                                       ])

        make_trainable(self.Decoder, False)
        self.G_trainer_up_ppra_reconst \
            = Model(inputs=[real_X, anatomy_real],
                    outputs=[rec_x_ppta])
        self.G_trainer_up_ppra_reconst.summary(print_fn=log.info)
        self.G_trainer_up_ppra_reconst.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                       loss=['mae'],
                                       loss_weights=[self.conf.w_rec_X * self.conf.pred_pathology_weight_rate * self.conf.real_anatomy_weight_rate + eps])
        make_trainable(self.Decoder, True)


    def build_triple_trainer(self):
        make_trainable(self.D_Reconstruction_trainer_lp_rp, True)
        make_trainable(self.D_Reconstruction_trainer_up_pppa, True)
        make_trainable(self.D_Reconstruction_trainer_up_ppra, True)

        def _build_graph(pathology):

            fake_Z, _ = self.Enc_Modality([fake_S, pathology, real_X])
            rec_x = self.Decoder([fake_S, pathology, fake_Z])
            tmp1, triplet_output = self.D_Reconstruction(rec_x)
            return  triplet_output, tmp1
        """
        Model for training SDNet given labelled images. In addition to the unsupervised trainer, a direct segmentation
        cost is also minimised.
        """

        real_X = Input(self.conf.input_shape)
        anatomy_real = Input(tuple(self.conf.input_shape[:-1]) + (self.conf.num_anatomy_masks,))
        def _anatomy_real_correction(input):
            for ii in range(self.conf.num_anatomy_masks):
                if ii == 0:
                    added = input[:,:,:,ii:ii+1]
                else:
                    added = added+input[:,:,:,ii:ii+1]
            background = ones_like(input[:,:,:,0:1]) - added
            output = concatenate([input,background], axis=-1)
            return output
        anatomy_real_with_background = Lambda(_anatomy_real_correction)(anatomy_real)

        pathology_real = Input(tuple(self.conf.input_shape[:-1]) + (self.conf.num_pathology_masks,))
        pseudo_healthy = Input(tuple(self.conf.input_shape[:-1]) + (self.conf.num_pathology_masks,))

        fake_S = self.Enc_Anatomy(real_X)
        fake_anatomy_mask = self.Segmentor(fake_S)
        predicted_pathology_from_predicted_anatomy = self.Enc_Pathology(Concatenate(axis=-1)([real_X, fake_anatomy_mask]))
        predicted_pathology_from_true_anatomy = self.Enc_Pathology(Concatenate(axis=-1)([real_X, anatomy_real_with_background]))


        pathology_correction = Lambda(lambda x: x[..., 0:self.conf.num_pathology_masks])
        predicted_pathology_from_predicted_anatomy = pathology_correction(predicted_pathology_from_predicted_anatomy)
        predicted_pathology_from_true_anatomy = pathology_correction(predicted_pathology_from_true_anatomy)


        tmp1, real_x_triplet = self.D_Reconstruction(real_X)
        triplet_rp, _ = _build_graph(pathology_real)
        triplet_ph,tmp3 = _build_graph(pseudo_healthy)
        triplet_pppa,_ = _build_graph(predicted_pathology_from_predicted_anatomy)
        triplet_ppra,_ = _build_graph(predicted_pathology_from_true_anatomy)

        triplet_op_rp = make_triplet_loss([real_x_triplet, triplet_rp, triplet_ph,
                                           self.conf.triplet_margin], name='RP')
        triplet_op_pppa = make_triplet_loss([real_x_triplet, triplet_pppa, triplet_ph,
                                             self.conf.triplet_margin], name='PPPA')
        triplet_op_ppra = make_triplet_loss([real_x_triplet, triplet_ppra, triplet_ph,
                                             self.conf.triplet_margin], name='PPRA')

        self.G_trainer_lp_triplet = Model([real_X, pathology_real, pseudo_healthy],
                                          outputs=[triplet_op_rp, tmp1])
        self.G_trainer_lp_triplet.summary(print_fn=log.info)
        self.G_trainer_lp_triplet.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                          loss=[costs.ypred, costs.ypred],
                                          loss_weights=[
                                              self.conf.triplet_weight * self.conf.real_pathology_weight_rate + eps,
                                              0])

        self.G_trainer_up_triplet = Model([real_X, pseudo_healthy, anatomy_real],
                                          outputs=[triplet_op_pppa,
                                                   triplet_op_ppra,
                                                   tmp3])
        self.G_trainer_up_triplet.summary(print_fn=log.info)
        self.G_trainer_up_triplet.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                          loss=[costs.ypred, costs.ypred,costs.ypred],
                                          loss_weights=[
                                              self.conf.triplet_weight * self.conf.pred_pathology_weight_rate * self.conf.pred_anatomy_weight_rate + eps,
                                              self.conf.triplet_weight * self.conf.pred_pathology_weight_rate * self.conf.real_anatomy_weight_rate + eps,
                                              0])



    def build_latent_factor_regressor(self):

        sample_S = Input(self.Enc_Anatomy.output_shape[1:])
        sample_Z = Input((self.conf.num_z,))
        pathology = Input(tuple(self.conf.input_shape[:-1]) + (1,))
        reconstruct_sample_X = self.Decoder([sample_S, pathology, sample_Z])

        z_model = Model(self.Enc_Modality.inputs, self.Enc_Modality.get_layer('z').output)

        rec_Z = z_model([sample_S, pathology, reconstruct_sample_X])
        self.z_reconstructor = Model(inputs=[sample_S, pathology, sample_Z], outputs=rec_Z, name='z_reconstructor')
        self.z_reconstructor.summary(print_fn=log.info)
        self.z_reconstructor.loss_weight = K.variable(eps)
        self.z_reconstructor.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay), loss=['mae'],
                                     loss_weights=[self.z_reconstructor.loss_weight])

        log.info('latent factor Regressor')


    def get_anatomy_segmentor(self):
        inp = Input(self.conf.input_shape)
        return Model(inputs=inp, outputs=self.Segmentor(self.Enc_Anatomy(inp)))

    def get_pathology_encoder(self):
        inp = Input(self.conf.input_shape)
        return Model(inputs=inp,outputs=self.Enc_Pathology(Concatenate(axis=-1)([inp, self.Segmentor(self.Enc_Anatomy(inp))])))



