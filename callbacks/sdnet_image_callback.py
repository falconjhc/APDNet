import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from keras import Input, Model
#from scipy.misc import imsave
from imageio import imwrite as imsave # harric modified
from utils.image_utils import image_show, generate_mask_on_img # harric modified

import utils.data_utils
import utils.image_utils
from callbacks.image_callback import BaseSaveImage
from layers.rounding import Rounding
from utils import sdnet_utils
from utils.distributions import NormalDistribution
from utils.sdnet_utils import get_net

log = logging.getLogger('SDNetImageCallback')


class SDNetImageCallback(BaseSaveImage):
    def __init__(self, conf, sdnet, data_gen_lb,  mask_gen, img_channels, anato_mask_channels, patho_mask_channels):
        '''
        Callback for printint various types of images during SDNet training.

        :param folder:         location of callback images
        :param generators:     a list of "generator-tye" NN: usually [Decomposer, Reconstructor, Segmentor]
        :param discriminators: a list of discriminator NN: usually  [D_Images, D_Masks, D_Z]
        :param data_gen_lb:    a python iterator of images+masks
        :param data_gen_ul:    a python iterator of images
        :param mask_gen:       a python iterator of additional masks with full anatomy used in discriminator: can be None
        '''
        self.conf = conf
        super(SDNetImageCallback, self).__init__(conf.folder, sdnet)

        self._make_dirs(self.folder)
        self.data_gen_lb = data_gen_lb
        self.mask_gen = mask_gen
        self.img_channels = img_channels
        self.anato_mask_channels = anato_mask_channels
        self.patho_mask_channels = patho_mask_channels
        self.init_models()

    def _make_dirs(self, folder):
        self.lr_folder = folder + '/images_lr'
        if not os.path.exists(self.lr_folder):
            os.makedirs(self.lr_folder)

        self.anato_segm_folder = folder + '/images_anato_segm'
        if not os.path.exists(self.anato_segm_folder):
            os.makedirs(self.anato_segm_folder)

        self.patho_segm_folder = folder + '/images_patho_segm'
        if not os.path.exists(self.patho_segm_folder):
            os.makedirs(self.patho_segm_folder)

        self.rec_folder = folder + '/images_rec'
        if not os.path.exists(self.rec_folder):
            os.makedirs(self.rec_folder)

        self.reconstruct_discr_folder = folder + '/images_reconstruct_discr'
        if not os.path.exists(self.reconstruct_discr_folder):
            os.makedirs(self.reconstruct_discr_folder)
        self.reconstruct_classifier_folder = folder + '/images_reconstruct_classifier'
        if not os.path.exists(self.reconstruct_classifier_folder):
            os.makedirs(self.reconstruct_classifier_folder)

        self.interp_folder = folder + '/images_interp'
        if not os.path.exists(self.interp_folder):
            os.makedirs(self.interp_folder)

    def init_models(self):
        self.enc_anatomy = self.model.Enc_Anatomy
        self.reconstructor = self.model.Decoder
        self.segmentor = self.model.Segmentor
        self.discr_reconstruct_mask = self.model.D_Reconstruction
        self.enc_modality = self.model.Enc_Modality
        self.enc_pathology = self.model.Enc_Pathology

        mean = get_net(self.enc_modality, 'z_mean')
        var = get_net(self.enc_modality, 'z_log_var')
        self.z_mean = Model(self.enc_modality.inputs, mean.output)
        self.z_var = Model(self.enc_modality.inputs, var.output)

        inp = Input(self.conf.input_shape)
        self.round_model = Model(inp, Rounding()(self.enc_anatomy(inp)))

    def on_epoch_end(self, epoch=None, logs=None):
        '''
        Plot training images from the real_pool. For SDNet the real_pools will contain images paired with masks,
        and also unlabelled images.
        :param epoch:       current training epoch
        :param real_pools:  pool of images. Each element might be an image or a real mask
        :param logs:
        '''
        lb_image_mask_pack = next(self.data_gen_lb)

        # we usually plot 4 image-rows.
        # If we have less, it means we've reached the end of the data, so iterate from the beginning
        if len(lb_image_mask_pack[0]) < 4:
            lb_image_mask_pack = next(self.data_gen_lb)

        ims = lb_image_mask_pack[:,:,:,0:self.img_channels]
        anato_mks = lb_image_mask_pack[:,:,:,self.img_channels:self.img_channels+self.anato_mask_channels]
        patho_mks = lb_image_mask_pack[:, :, :, self.img_channels + self.anato_mask_channels:]
        lb_images = [ims, anato_mks, patho_mks]

        anato_masks = None if self.mask_gen is None else next(self.mask_gen[0])
        patho_masks = None if self.mask_gen is None else next(self.mask_gen[1])
        if patho_masks is not None:
            if len(anato_masks) < 4:
                anato_masks = next(self.mask_gen[0])
            _, b = utils.data_utils.crop_same([anato_masks], [anato_masks], size=(lb_images[0].shape[1], lb_images[0].shape[2]))
            anato_masks = b[0]
        if patho_masks is not None:
            if len(patho_masks) < 4:
                patho_masks = next(self.mask_gen[1])
            _, b = utils.data_utils.crop_same([patho_masks], [patho_masks], size=(lb_images[0].shape[1], lb_images[0].shape[2]))
            patho_masks = b[0]

        # self.plot_anatomy_mask_discriminator_outputs(lb_images, [anato_masks, patho_masks], epoch)
        # self.plot_pathology_mask_discriminator_outputs(lb_images,  [anato_masks, patho_masks], epoch)
        self.plot_anatomy_segmentations(lb_images,  epoch)
        self.plot_pathology_segmentations(lb_images,  epoch)
        self.plot_reconstructions(lb_images,  epoch)
        self.plot_reconstruction_classifier_outputs(lb_images,  [anato_masks, patho_masks], epoch)
        self.plot_reconstruction_discriminator_outputs(lb_images,  [anato_masks, patho_masks], epoch)
        self.plot_latent_representation(lb_images,  epoch)
        self.plot_image_switch_lr(lb_images,  epoch)
        self.plot_image_interpolation(lb_images,  epoch)




    def plot_latent_representation(self, lb_images,  epoch):
        """
        Plot a 4-row image, where the first column shows the input image and the following columns
        each of the 8 channels of the spatial latent representation.
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param epoch    :   the epoch number
        """

        # combine labelled and unlabelled images and randomly sample 4 examples
        images = lb_images[0]  # [el[0] for el in lb_images]
        anato_masks = lb_images[1]
        patho_masks = lb_images[2]
        current_select = epoch % images.shape[3]

        x = images

        # plot S
        s = self.enc_anatomy.predict(x)
        predicted_anatomy = self.segmentor.predict(s)
        predicted_pathology_from_predicted_anatomy = self.enc_pathology.predict(np.concatenate([x,predicted_anatomy],
                                                                                               axis=-1))

        rows = [np.concatenate([x[i, :, :, current_select]]
                               + [s[i, :, :, s_chn] for s_chn in range(s.shape[-1])]
                               + [predicted_pathology_from_predicted_anatomy[i,:,:,chn]
                                  for chn in range(predicted_pathology_from_predicted_anatomy.shape[-1])],
                               axis=1)
                for i in range(x.shape[0])]
        im_plot = np.concatenate(rows, axis=0)
        imsave(self.lr_folder + '/s_lr_epoch_%d.png' % epoch, im_plot)
        # harric modified

        plt.figure()
        plt.imshow(im_plot, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.close()

        if self.conf.rounding == 'decoder':
            s = self.round_model.predict(x)
            rows = [np.concatenate([x[i, :, :, 0]] + [s[i, :, :, s_chn] for s_chn in range(s.shape[-1])], axis=1)
                   for i in range(x.shape[0])]
            im_plot = np.concatenate(rows, axis=0)
            imsave(self.lr_folder + '/srnd_lr_epoch_%d.png' % epoch, im_plot)
            # harric modifiedd

            plt.figure()
            plt.imshow(im_plot, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.close()

        # plot Z
        enc_modality_inputs = [self.enc_anatomy.predict(images),
                               predicted_pathology_from_predicted_anatomy[:,:,:,:-1], images]
        z, _ = self.enc_modality.predict(enc_modality_inputs)
        gaussian = NormalDistribution()
        real_z = gaussian.sample(z.shape)

        fig, axes = plt.subplots(nrows=z.shape[1], ncols=2, sharex=True, sharey=True, figsize=(10, 8))
        axes[0, 0].set_title('Predicted Z')
        axes[0, 1].set_title('Real Z')
        for i in range(len(axes)):
            axes[i, 0].hist(z[:, i], normed=True, bins=11, range=(-3, 3))
            axes[i, 1].hist(real_z[:, i], normed=True, bins=11, range=(-3, 3))
        axes[0, 0].plot(0, 0)

        plt.savefig(self.lr_folder + '/z_lr_epoch_%d.png' % epoch)
        plt.close()

        means = self.z_mean.predict(enc_modality_inputs)
        variances  = self.z_var.predict(enc_modality_inputs)
        means = np.var(means, axis=0)
        variances = np.mean(np.exp(variances), axis=0)
        with open(self.lr_folder + '/z_means.csv', 'a+') as f:
            f.writelines(', '.join([str(means[i]) for i in range(means.shape[0])]) + '\n')
        with open(self.lr_folder + '/z_vars.csv', 'a+') as f:
            f.writelines(', '.join([str(variances[i]) for i in range(variances.shape[0])]) + '\n')

    def plot_anatomy_segmentations(self, lb_images,  epoch):
        '''
        Plot an image for every sample, where every row contains a channel of the spatial LR and a channel of the
        predicted mask.
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param epoch:       the epoch number
        '''

        imags = lb_images[0]  # [el[0] for el in lb_images]
        anato_masks = lb_images[1]  # [el[1] for el in lb_images]
        patho_masks = lb_images[2]  # [el[1] for el in lb_images]
        current_select = epoch % imags.shape[3]
        x = imags
        m_anato = anato_masks
        m_patho = patho_masks

        assert x.shape[:-1] == m_anato.shape[:-1] == m_patho.shape[:-1], \
            'Incompatible shapes: %s vs %s vsv %s' % (str(x.shape), str(m_anato.shape), str(m_patho.shape))

        s = self.enc_anatomy.predict(x)
        y = self.segmentor.predict(s)

        rows = []
        for i in range(x.shape[0]):
            y_list = [y[i, :, :, chn] for chn in range(y.shape[-1])]
            m_anato_list = [m_anato[i, :, :, chn] for chn in range(m_anato.shape[-1])]
            if m_anato.shape[-1] < y.shape[-1]:
                m_anato_list += [np.zeros(shape=(m_anato.shape[1], m_anato.shape[2]))] * (y.shape[-1] - m_anato.shape[-1])
            assert len(y_list) == len(m_anato_list), 'Incompatible sizes: %d vs %d' % (len(y_list), len(m_anato_list))
            rows += [np.concatenate([x[i, :, :, current_select]] + y_list + m_anato_list, axis=1)]

        im_plot = np.concatenate(rows, axis=0)
        imsave(self.anato_segm_folder + '/segmentations_epoch_%d.png' % (epoch), im_plot)

    def plot_pathology_segmentations(self, lb_images,  epoch):
        '''
        Plot an image for every sample, where every row contains a channel of the spatial LR and a channel of the
        predicted mask.
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param epoch:       the epoch number
        '''

        imags = lb_images[0]  # [el[0] for el in lb_images]
        anato_masks = lb_images[1]  # [el[1] for el in lb_images]
        patho_masks = lb_images[2]  # [el[1] for el in lb_images]
        current_select = epoch % imags.shape[3]
        x = imags
        m_anato = anato_masks
        m_patho = patho_masks

        assert x.shape[:-1] == m_anato.shape[:-1] == m_patho.shape[:-1], \
            'Incompatible shapes: %s vs %s vsv %s' % (str(x.shape), str(m_anato.shape), str(m_patho.shape))

        s = self.enc_anatomy.predict(x)
        predicted_anatomy = self.segmentor.predict(s)
        y = self.enc_pathology.predict(np.concatenate([x, predicted_anatomy], axis=-1))

        rows = []
        for i in range(x.shape[0]):
            y_list = [y[i, :, :, chn] for chn in range(y.shape[-1])]
            m_patho_list = [m_patho[i, :, :, chn] for chn in range(m_patho.shape[-1])]
            if m_patho.shape[-1] < y.shape[-1]:
                m_patho_list += [np.zeros(shape=(m_patho.shape[1], m_patho.shape[2]))] * (y.shape[-1] - m_patho.shape[-1])
            assert len(y_list) == len(m_patho_list), 'Incompatible sizes: %d vs %d' % (len(y_list), len(m_patho_list))
            rows += [np.concatenate([x[i, :, :, current_select]] + y_list + m_patho_list, axis=1)]

        im_plot = np.concatenate(rows, axis=0)
        imsave(self.patho_segm_folder + '/segmentations_epoch_%d.png' % (epoch), im_plot)


    def plot_reconstructions(self, lb_images,  epoch):
        def _create_row(pathology_masks, z):
            y = self.reconstructor.predict([s, pathology_masks, z])
            y_s0 = self.reconstructor.predict([s, pathology_masks, np.zeros(z.shape)])
            all_bkg = np.concatenate([np.zeros(s.shape[:-1] + (s.shape[-1] - 1,)), np.ones(s.shape[:-1] + (1,))],
                                     axis=-1)
            y_0z = self.reconstructor.predict([all_bkg, pathology_masks, z])
            y_00 = self.reconstructor.predict([all_bkg, pathology_masks, np.zeros(z.shape)])
            z_random = gaussian.sample(z.shape)
            y_random = self.reconstructor.predict([s,pathology_masks, z_random])

            rows = [np.concatenate([_generate_mask_on_img(x[i, :, :, current_select], pathology_masks[i, :, :, :]),
                                    _expand(x[i, :, :, current_select]),
                                    _expand(y[i, :, :, current_select]),
                                    _expand(y_random[i, :, :, current_select]),
                                    _expand(y_s0[i, :, :, current_select])] +
                                   [_expand(self.reconstructor.predict(
                                       [self._get_s0chn(k, s), pathology_masks, z])[i, :, :,
                                            current_select])
                                    for k in range(s.shape[-1] - 1)] +
                                   [_expand(y_0z[i, :, :, current_select]),
                                    _expand(y_00[i, :, :, current_select])], axis=1) for i in range(x.shape[0])]

            return rows

        def _expand(img):
            return np.tile(np.expand_dims(img, axis=-1), [1,1,3])
        def _generate_mask_on_img(img,mask):
            img = np.expand_dims(img,axis=-1)
            img_cpy = np.copy(img)
            mask_pixel = np.copy(img)
            mask_pixel[np.where(mask==1)]=1
            return np.concatenate([mask_pixel,img_cpy,img_cpy], axis=-1)
        """
        Plot two images showing the combination of the spatial and modality LR to generate an image. The first
        image uses the predicted S and Z and the second samples Z from a Gaussian.
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param epoch:       the epoch number
        """

        # combine labelled and unlabelled images and randomly sample 4 examples
        images = lb_images[0]  # [el[0] for el in lb_images]
        anato_masks = lb_images[1]
        patho_masks = lb_images[2]
        pseudo_masks = np.zeros(shape=patho_masks.shape,dtype=patho_masks.dtype)
        current_select = epoch % images.shape[3]
        # if len(ul_images) > 0:
        #     images = np.concatenate([images, ul_images], axis=0)
        # x = utils.data_utils.sample(images, nb_samples=4)
        x = images

        # S + Z -> Image
        gaussian = NormalDistribution()

        s = self.enc_anatomy.predict(x)
        predicted_anatomy = self.segmentor.predict(s)
        predicted_pathology = self.enc_pathology.predict(np.concatenate([x,predicted_anatomy], axis=-1))
        z_with_real_pathology, _ = self.enc_modality.predict([s, patho_masks, x])
        z_with_pseodu_health, _ = self.enc_modality.predict([s, pseudo_masks, x])
        z_with_predicted_pathology,_ = self.enc_modality.predict([s, predicted_pathology[:,:,:,
                                                                     0:patho_masks.shape[3]], x])

        rows_with_real_pathology = _create_row(pathology_masks=patho_masks, z=z_with_real_pathology)
        rows_with_pseodu_health = _create_row(pathology_masks=pseudo_masks, z=z_with_pseodu_health)
        rows_with_predicted_pathology = _create_row(pathology_masks=predicted_pathology[:,:,:,
                                                                    0:patho_masks.shape[3]],
                                                    z=z_with_predicted_pathology)

        header = utils.image_utils.makeTextHeaderImage(x.shape[2],
                                                       ['pathology', 'X', 'rec(s,z)', 'rec(s,~z)', 'rec(s,0)'] +
                                                       ['rec(s0_%d, z)' % k for k in range(s.shape[-1] - 1)] + [
                                                        'rec(0, z)', 'rec(0,0)'])
        header = _expand(header)
        im_plot_with_actual_pathology = np.concatenate([header] + rows_with_real_pathology, axis=0)
        im_plot_with_pseudo_healthy = np.concatenate([header] + rows_with_pseodu_health, axis=0)
        im_plot_with_predicted_pathology = np.concatenate([header] + rows_with_predicted_pathology, axis=0)
        im_plot_actual_pathology = np.clip(im_plot_with_actual_pathology, -1, 1)
        im_plot_pseudo_healthy = np.clip(im_plot_with_pseudo_healthy, -1, 1)
        im_plot_predicted_pathology = np.clip(im_plot_with_predicted_pathology, -1, 1)
        im = np.concatenate([im_plot_actual_pathology, im_plot_predicted_pathology,im_plot_pseudo_healthy], axis=0)
        imsave(self.rec_folder + '/rec_epoch_%d.png' % epoch, im)

        plt.figure()
        plt.imshow(im, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.close()

    def _get_s0chn(self, k, s):
        s_res = s.copy()
        chnk = s_res[..., k]
        # move channel k 1s to the background
        s_res[..., -1][chnk == 1] = 1
        s_res[..., k] = 0
        return s_res


    def plot_reconstruction_classifier_outputs(self, lb_images,  other_masks, epoch):
        '''
        Plot a histogram of predicted values by the discriminator
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param other_masks: a 4-dim array of masks with full anatomy: can be None
        :param epoch:       the epoch number
        '''
        imags = lb_images[0]  # [el[0] for el in lb_images]
        anato_masks = lb_images[1]  # [el[1] for el in lb_images]
        patho_masks = lb_images[2]  # [el[1] for el in lb_images]
        current_selected = epoch % imags.shape[3]

        x = imags
        m_anato = anato_masks
        m_patho = patho_masks
        m_pseudo_health = np.zeros(shape=m_patho.shape,dtype=m_patho.dtype)

        print(m_anato.shape)
        print(m_patho.shape)
        print(self.discr_reconstruct_mask.input_shape[-1], self.model.Decoder.output_shape[-1])

        s = self.enc_anatomy.predict(x)
        predicted_anatomy = self.segmentor.predict(s)
        predicted_pathology = self.enc_pathology.predict(np.concatenate([x, predicted_anatomy], axis=-1))

        pred_z_actual_pathology, _ = self.enc_modality.predict([s, m_patho, x])
        pred_z_pseudo_health, _ = self.enc_modality.predict([s, m_pseudo_health, x])
        pred_z_predicted_pathology,_ = self.enc_modality.predict([s, predicted_pathology[:,:,:,
                                                                     0:patho_masks.shape[3]], x])
        pred_i_actual_pathology = self.reconstructor.predict([s, m_patho, pred_z_actual_pathology])
        pred_i_pseudo_health = self.reconstructor.predict([s, m_pseudo_health, pred_z_pseudo_health])
        pred_i_predicted_pathology = self.reconstructor.predict([s, predicted_pathology[:,:,:,
                                                                    0:patho_masks.shape[3]],
                                                                 pred_z_predicted_pathology])


        plt.figure()
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0], 2, 2 * i + 1)
            m_allchn = np.concatenate([x[i, :, :, chn] for chn in range(x.shape[-1])], axis=1)
            plt.imshow(m_allchn, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f / %.3f' % (self.discr_reconstruct_mask.predict(x[i:i + 1])[1][0][0],
                                             self.discr_reconstruct_mask.predict(x[i:i + 1])[1][0][1]))

            plt.subplot(x.shape[0], 2, 2 * i + 2)
            pred_m_allchn = pred_i_actual_pathology[i:i + 1]
            pred_m_allchn_img = np.concatenate([pred_m_allchn[0, :, :, chn] for chn in range(pred_m_allchn.shape[-1])],
                                               axis=1)
            plt.imshow(pred_m_allchn_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f / %.3f' % (self.discr_reconstruct_mask.predict(pred_m_allchn)[1][0][0],
                                             self.discr_reconstruct_mask.predict(pred_m_allchn)[1][0][1]))
        plt.tight_layout()
        plt.savefig(self.reconstruct_classifier_folder +
                    '/classifier_reconstruction_epoch_%d_actual_pathology.png' % epoch)
        plt.close()

        plt.figure()
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0], 2, 2 * i + 1)
            m_allchn = np.concatenate([x[i, :, :, chn] for chn in range(x.shape[-1])], axis=1)
            plt.imshow(m_allchn, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f / %.3f' % (self.discr_reconstruct_mask.predict(x[i:i + 1])[1][0][0],
                                             self.discr_reconstruct_mask.predict(x[i:i + 1])[1][0][1]))

            plt.subplot(x.shape[0], 2, 2 * i + 2)
            pred_m_allchn = pred_i_pseudo_health[i:i + 1]
            pred_m_allchn_img = np.concatenate([pred_m_allchn[0, :, :, chn] for chn in range(pred_m_allchn.shape[-1])],
                                               axis=1)
            plt.imshow(pred_m_allchn_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f / %.3f' % (self.discr_reconstruct_mask.predict(pred_m_allchn)[1][0][0],
                                             self.discr_reconstruct_mask.predict(pred_m_allchn)[1][0][1]))
        plt.tight_layout()
        plt.savefig(self.reconstruct_classifier_folder + '/classifier_reconstruction_epoch_%d_pseudo_health.png' % epoch)
        plt.close()

        plt.figure()
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0], 2, 2 * i + 1)
            m_allchn = np.concatenate([x[i, :, :, chn] for chn in range(x.shape[-1])], axis=1)
            plt.imshow(m_allchn, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f / %.3f' % (self.discr_reconstruct_mask.predict(x[i:i + 1])[1][0][0],
                                             self.discr_reconstruct_mask.predict(x[i:i + 1])[1][0][1]))

            plt.subplot(x.shape[0], 2, 2 * i + 2)
            pred_m_allchn = pred_i_predicted_pathology[i:i + 1]
            pred_m_allchn_img = np.concatenate([pred_m_allchn[0, :, :, chn] for chn in range(pred_m_allchn.shape[-1])],
                                               axis=1)
            plt.imshow(pred_m_allchn_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f / %.3f' % (self.discr_reconstruct_mask.predict(pred_m_allchn)[1][0][0],
                                             self.discr_reconstruct_mask.predict(pred_m_allchn)[1][0][1]))
        plt.tight_layout()
        plt.savefig(
            self.reconstruct_classifier_folder +
            '/classifier_reconstruction_epoch_%d_predicted_pathology.png' % epoch)
        plt.close()

    def plot_reconstruction_discriminator_outputs(self, lb_images,  other_masks, epoch):
        '''
        Plot a histogram of predicted values by the discriminator
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param other_masks: a 4-dim array of masks with full anatomy: can be None
        :param epoch:       the epoch number
        '''
        imags = lb_images[0]  # [el[0] for el in lb_images]
        anato_masks = lb_images[1]  # [el[1] for el in lb_images]
        patho_masks = lb_images[2]  # [el[1] for el in lb_images]
        current_selected = epoch % imags.shape[3]

        x = imags
        m_anato = anato_masks
        m_patho = patho_masks
        m_pseudo_health = np.zeros(shape=m_patho.shape,dtype=m_patho.dtype)

        print(m_anato.shape)
        print(m_patho.shape)
        print(self.discr_reconstruct_mask.input_shape[-1], self.model.Decoder.output_shape[-1])

        s = self.enc_anatomy.predict(x)
        predicted_anatomy = self.segmentor.predict(s)
        predicted_pathology = self.enc_pathology.predict(np.concatenate([x, predicted_anatomy], axis=-1))


        pred_z_actual_pathology, _ = self.enc_modality.predict([s, m_patho, x])
        pred_z_pseudo_health, _ = self.enc_modality.predict([s, m_pseudo_health, x])
        pred_z_predicted_pathology, _ = self.enc_modality.predict([s,
                                                                   predicted_pathology[:, :, :,
                                                                   0:patho_masks.shape[3]], x])

        pred_i_actual_pathology = self.reconstructor.predict([s, m_patho, pred_z_actual_pathology])
        pred_i_pseudo_health = self.reconstructor.predict([s, m_pseudo_health, pred_z_pseudo_health])
        pred_i_predicted_pathology = self.reconstructor.predict([s,
                                                                 predicted_pathology[:, :, :,
                                                                 0:patho_masks.shape[3]],
                                                                 pred_z_predicted_pathology])

        dm_input_fake_actual_pathology = pred_i_actual_pathology
        dm_input_fake_pseudo_health = pred_i_pseudo_health
        dm_input_fake_predicted_pathology = pred_i_predicted_pathology
        dm_true = self.discr_reconstruct_mask.predict(x)[0].reshape(x.shape[0], -1).mean(axis=1)
        dm_pred_actual_pathology = self.discr_reconstruct_mask.predict(dm_input_fake_actual_pathology)[0].\
            reshape(pred_i_actual_pathology.shape[0], -1).mean(axis=1)
        dm_pred_pseudo_health = self.discr_reconstruct_mask.predict(dm_input_fake_pseudo_health)[0].\
            reshape(pred_i_pseudo_health.shape[0], -1).mean(axis=1)
        dm_pred_predicted_pathology = self.discr_reconstruct_mask.predict(dm_input_fake_predicted_pathology)[0].\
            reshape(dm_input_fake_predicted_pathology.shape[0], -1).mean(axis=1)

        plt.figure()
        plt.subplot(1, 1, 1)
        plt.title('Reconstruction Discriminator with Actual Pathology')
        plt.hist([dm_true, dm_pred_actual_pathology], stacked=True, normed=True)
        plt.savefig(self.reconstruct_discr_folder +
                    '/discriminator_reconstruction_hist_epoch_%d_actual_pathology.png' % epoch)
        plt.close()

        plt.figure()
        plt.subplot(1, 1, 1)
        plt.title('Reconstruction Discriminator with Pseudo Health')
        plt.hist([dm_true, dm_pred_pseudo_health], stacked=True, normed=True)
        plt.savefig(self.reconstruct_discr_folder +
                    '/discriminator_reconstruction_hist_epoch_%d_pseudo_health.png' % epoch)
        plt.close()

        plt.figure()
        plt.subplot(1, 1, 1)
        plt.title('Reconstruction Discriminator with Pseudo Health')
        plt.hist([dm_true, dm_pred_predicted_pathology], stacked=True, normed=True)
        plt.savefig(
            self.reconstruct_discr_folder +
            '/discriminator_reconstruction_hist_epoch_%d_predicted_pathology.png' % epoch)
        plt.close()

        plt.figure()
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0], 2, 2 * i + 1)
            m_allchn = np.concatenate([x[i, :, :, chn] for chn in range(x.shape[-1])], axis=1)
            plt.imshow(m_allchn, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_reconstruct_mask.predict(x[i:i + 1])[0].reshape(1, -1).mean(axis=1))

            plt.subplot(x.shape[0], 2, 2 * i + 2)
            pred_m_allchn = pred_i_actual_pathology[i:i + 1]
            pred_m_allchn_img = np.concatenate([pred_m_allchn[0, :, :, chn]
                                                for chn in range(pred_m_allchn.shape[-1])],
                                               axis=1)
            plt.imshow(pred_m_allchn_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_reconstruct_mask.predict(pred_m_allchn)[0].reshape(1, -1).mean(axis=1))
        plt.tight_layout()
        plt.savefig(self.reconstruct_discr_folder
                    + '/discriminator_reconstruction_epoch_%d_actual_pathology.png' % epoch)
        plt.close()

        plt.figure()
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0], 2, 2 * i + 1)
            m_allchn = np.concatenate([x[i, :, :, chn] for chn in range(x.shape[-1])], axis=1)
            plt.imshow(m_allchn, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_reconstruct_mask.predict(x[i:i + 1])[0].reshape(1, -1).mean(axis=1))

            plt.subplot(x.shape[0], 2, 2 * i + 2)
            pred_m_allchn = pred_i_pseudo_health[i:i + 1]
            pred_m_allchn_img = np.concatenate([pred_m_allchn[0, :, :, chn]
                                                for chn in range(pred_m_allchn.shape[-1])],
                                               axis=1)
            plt.imshow(pred_m_allchn_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_reconstruct_mask.predict(pred_m_allchn)[0].reshape(1, -1).mean(axis=1))
        plt.tight_layout()
        plt.savefig(self.reconstruct_discr_folder
                    + '/discriminator_reconstruction_epoch_%d_pseudo_health.png' % epoch)
        plt.close()

        plt.figure()
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0], 2, 2 * i + 1)
            m_allchn = np.concatenate([x[i, :, :, chn] for chn in range(x.shape[-1])], axis=1)
            plt.imshow(m_allchn, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_reconstruct_mask.predict(x[i:i + 1])[0].reshape(1, -1).mean(axis=1))

            plt.subplot(x.shape[0], 2, 2 * i + 2)
            pred_m_allchn = pred_i_predicted_pathology[i:i + 1]
            pred_m_allchn_img = np.concatenate([pred_m_allchn[0, :, :, chn]
                                                for chn in range(pred_m_allchn.shape[-1])],
                                               axis=1)
            plt.imshow(pred_m_allchn_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_reconstruct_mask.predict(pred_m_allchn)[0].reshape(1, -1).mean(axis=1))
        plt.tight_layout()
        plt.savefig(self.reconstruct_discr_folder
                    + '/discriminator_reconstruction_epoch_%d_predicted_pathology.png' % epoch)
        plt.close()

    def plot_anatomy_mask_discriminator_outputs(self, lb_images,  other_masks, epoch):
        '''
        Plot a histogram of predicted values by the discriminator
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param other_masks: a 4-dim array of masks with full anatomy: can be None
        :param epoch:       the epoch number
        '''
        imags = lb_images[0]  # [el[0] for el in lb_images]
        anato_masks = lb_images[1]  # [el[1] for el in lb_images]
        patho_masks = lb_images[2]  # [el[1] for el in lb_images]
        current_selected = epoch % imags.shape[3]

        x = imags
        m_anato = anato_masks

        print(m_anato.shape)
        print(self.discr_anato_mask.input_shape[-1], self.model.Decoder.output_shape[-1])

        s = self.enc_anatomy.predict(x)
        pred_m = self.segmentor.predict(s)

        dm_input_fake = pred_m[:,:,:,:-1]
        dm_true = self.discr_anato_mask.predict(m_anato).reshape(m_anato.shape[0], -1).mean(axis=1)
        dm_pred = self.discr_anato_mask.predict(dm_input_fake).reshape(pred_m.shape[0], -1).mean(axis=1)

        plt.figure()
        plt.subplot(1, 1, 1)
        plt.title('Anatomy Mask Discriminator')
        plt.hist([dm_true, dm_pred], stacked=True, normed=True)
        plt.savefig(self.anatomy_mask_discr_folder + '/discriminator_hist_epoch_%d.png' % epoch)
        plt.close()

        plt.figure()
        for i in range(m_anato.shape[0]):
            plt.subplot(4, 2, 2 * i + 1)
            m_allchn = np.concatenate([m_anato[i, :, :, chn] for chn in range(m_anato.shape[-1])], axis=1)
            plt.imshow(m_allchn, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_anato_mask.predict(m_anato[i:i + 1]).reshape(1, -1).mean(axis=1))

            plt.subplot(4, 2, 2 * i + 2)
            pred_m_allchn = pred_m[i:i + 1,:,:,:-1]
            pred_m_allchn_img = np.concatenate([pred_m_allchn[0, :, :, chn] for chn in range(pred_m_allchn.shape[-1])],
                                               axis=1)
            plt.imshow(pred_m_allchn_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_anato_mask.predict(pred_m_allchn).reshape(1, -1).mean(axis=1))
        plt.tight_layout()
        plt.savefig(self.anatomy_mask_discr_folder + '/discriminator_mask_epoch_%d.png' % epoch)
        plt.close()

    def plot_pathology_mask_discriminator_outputs(self, lb_images,  other_masks, epoch):
        '''
        Plot a histogram of predicted values by the discriminator
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param other_masks: a 4-dim array of masks with full anatomy: can be None
        :param epoch:       the epoch number
        '''
        imags = lb_images[0]  # [el[0] for el in lb_images]
        anato_masks = lb_images[1]  # [el[1] for el in lb_images]
        patho_masks = lb_images[2]  # [el[1] for el in lb_images]
        current_selected = epoch % imags.shape[3]

        anato_background = np.ones(shape=anato_masks.shape[:-1]+(1,),dtype=anato_masks.dtype)
        for ii in range(anato_masks.shape[-1]):
            anato_background = anato_background - np.expand_dims(anato_masks[:,:,:,ii], axis=-1)
        anato_masks = np.concatenate([anato_masks, anato_background], axis=-1)

        x = imags
        m_anato = anato_masks
        m_patho = patho_masks

        print(m_anato.shape)
        print(self.discr_patho_mask.input_shape[-1], self.model.Decoder.output_shape[-1])

        s = self.enc_anatomy.predict(x)
        predicted_anatomy = self.segmentor.predict(s)
        predicted_pathology_from_predicted_anatomy = self.enc_pathology.predict(np.concatenate([x,predicted_anatomy],axis=-1))
        predicted_pathology_from_real_anatomy = self.enc_pathology.predict(np.concatenate([x,m_anato],axis=-1))

        dm_true = self.discr_patho_mask.predict(m_patho).reshape(m_patho.shape[0], -1).mean(axis=1)

        dm_input_fake = predicted_pathology_from_predicted_anatomy[:, :, :, :-1]
        dm_pred = self.discr_patho_mask.predict(dm_input_fake).reshape(predicted_pathology_from_predicted_anatomy.shape[0], -1).mean(axis=1)

        plt.figure()
        plt.subplot(1, 1, 1)
        plt.title('Pathology Mask Discriminator: From Predicted Anatomy')
        plt.hist([dm_true, dm_pred], stacked=True, normed=True)
        plt.savefig(self.pathology_mask_discr_folder
                    + '/discriminator_hist_FromPredictedAnatomy_epoch_%d.png' % epoch)
        plt.close()

        plt.figure()
        for i in range(m_patho.shape[0]):
            plt.subplot(4, 2, 2 * i + 1)
            m_allchn = np.concatenate([m_patho[i, :, :, chn] for chn in range(m_patho.shape[-1])], axis=1)
            plt.imshow(m_allchn, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_patho_mask.predict(m_patho[i:i + 1]).reshape(1, -1).mean(axis=1))

            plt.subplot(4, 2, 2 * i + 2)
            pred_m_allchn = predicted_pathology_from_predicted_anatomy[i:i + 1,:,:,:-1]
            pred_m_allchn_img = np.concatenate([pred_m_allchn[0, :, :, chn] for chn in range(pred_m_allchn.shape[-1])],
                                               axis=1)
            plt.imshow(pred_m_allchn_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_patho_mask.predict(pred_m_allchn).reshape(1, -1).mean(axis=1))
        plt.tight_layout()
        plt.savefig(self.anatomy_mask_discr_folder
                    + '/discriminator_mask_FromPredictedAnatomy_epoch_%d.png' % epoch)
        plt.close()

        dm_input_fake = predicted_pathology_from_real_anatomy[:, :, :, :-1]
        dm_pred = self.discr_patho_mask.predict(dm_input_fake).reshape(predicted_pathology_from_real_anatomy.shape[0], -1).mean(axis=1)
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.title('Pathology Mask Discriminator: From Real Anatomy')
        plt.hist([dm_true, dm_pred], stacked=True, normed=True)
        plt.savefig(self.pathology_mask_discr_folder
                    + '/discriminator_hist_FromRealAnatomy_epoch_%d.png' % epoch)
        plt.close()

        plt.figure()
        for i in range(m_patho.shape[0]):
            plt.subplot(4, 2, 2 * i + 1)
            m_allchn = np.concatenate([m_patho[i, :, :, chn] for chn in range(m_patho.shape[-1])], axis=1)
            plt.imshow(m_allchn, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_patho_mask.predict(m_patho[i:i + 1]).reshape(1, -1).mean(axis=1))

            plt.subplot(4, 2, 2 * i + 2)
            pred_m_allchn = predicted_pathology_from_real_anatomy[i:i + 1, :, :, :-1]
            pred_m_allchn_img = np.concatenate([pred_m_allchn[0, :, :, chn] for chn in range(pred_m_allchn.shape[-1])],
                                               axis=1)
            plt.imshow(pred_m_allchn_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_patho_mask.predict(pred_m_allchn).reshape(1, -1).mean(axis=1))
        plt.tight_layout()
        plt.savefig(self.anatomy_mask_discr_folder
                    + '/discriminator_mask_FromRealAnatomy_epoch_%d.png' % epoch)
        plt.close()

    def plot_image_switch_lr(self, lb_images,  epoch):
        '''
        Switch anatomy between two images and plot the synthetic result
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param epoch:       the epoch number
        '''
        imags = lb_images[0]  # [el[0] for el in lb_images]
        anato_masks = lb_images[1]
        patho_masks = lb_images[2]
        pseudo_masks = np.zeros(shape=patho_masks.shape,dtype=patho_masks.dtype)
        current_selected = epoch % imags.shape[3]
        x = imags
        s = self.enc_anatomy.predict(x)
        predicted_anatomy = self.segmentor.predict(s)
        predicted_pathology_from_predicted_anatomy = self.enc_pathology.\
            predict(np.concatenate([x,predicted_anatomy],
                                   axis=-1))

        if anato_masks.shape[0]!=4 or patho_masks.shape[0]!=4:
            return

        rows = []
        for i in range(0, imags.shape[0], 2):
            x1 = x[i: i + 1]
            x2 = x[i + 1: i + 2]

            s1 = self.enc_anatomy.predict(x1)
            z1, _ = self.enc_modality.predict([s1, predicted_pathology_from_predicted_anatomy[:,:,:,0:-1], x1])
            s2 = self.enc_anatomy.predict(x2)
            z2, _ = self.enc_modality.predict([s2, predicted_pathology_from_predicted_anatomy[:,:,:,0:-1], x2])

            x11 = self.reconstructor.predict([s1, predicted_pathology_from_predicted_anatomy[:,:,:,0:-1], z1])
            x12 = self.reconstructor.predict([s1, predicted_pathology_from_predicted_anatomy[:,:,:,0:-1],z2])
            x21 = self.reconstructor.predict([s2, predicted_pathology_from_predicted_anatomy[:,:,:,0:-1],z1])
            x22 = self.reconstructor.predict([s2, predicted_pathology_from_predicted_anatomy[:,:,:,0:-1],z2])

            row = np.concatenate([x1[0, :, :, current_selected],
                                  x11[0, :, :, current_selected],
                                  x12[0, :, :, current_selected],
                                  x21[0, :, :, current_selected],
                                  x22[0, :, :, current_selected],
                                  x2[0, :, :, current_selected]], axis=1)
            rows.append(row)

        header = utils.image_utils.makeTextHeaderImage(x.shape[2],
                                                       ['X1', 'Rec(s1,z1)', 'Rec(s1,z2)', 'Rec(s2,z1)', 'Rec(s2,z2)',
                                                     'X2'])
        image = np.concatenate([header] + rows, axis=0)
        imsave(self.interp_folder + '/switch_lr_epoch_%d_actual_pathology.png' % (epoch), image)

        rows = []
        for i in range(0, imags.shape[0], 2):
            x1 = x[i: i + 1]
            x2 = x[i + 1: i + 2]

            s1 = self.enc_anatomy.predict(x1)
            z1, _ = self.enc_modality.predict([s1, pseudo_masks, x1])
            s2 = self.enc_anatomy.predict(x2)
            z2, _ = self.enc_modality.predict([s2, pseudo_masks, x2])

            x11 = self.reconstructor.predict([s1, pseudo_masks, z1])
            x12 = self.reconstructor.predict([s1, pseudo_masks, z2])
            x21 = self.reconstructor.predict([s2, pseudo_masks, z1])
            x22 = self.reconstructor.predict([s2, pseudo_masks, z2])

            row = np.concatenate([x1[0, :, :, current_selected],
                                  x11[0, :, :, current_selected],
                                  x12[0, :, :, current_selected],
                                  x21[0, :, :, current_selected],
                                  x22[0, :, :, current_selected],
                                  x2[0, :, :, current_selected]], axis=1)
            rows.append(row)

        header = utils.image_utils.makeTextHeaderImage(x.shape[2],
                                                       ['X1', 'Rec(s1,z1)', 'Rec(s1,z2)', 'Rec(s2,z1)', 'Rec(s2,z2)',
                                                        'X2'])
        image = np.concatenate([header] + rows, axis=0)
        imsave(self.interp_folder + '/switch_lr_epoch_%d_pseudo_health.png' % (epoch), image)

    def plot_image_interpolation(self, lb_images,  epoch):
        '''
        Interpolate between two images and plot the transition in reconstructing the image.
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param epoch:       the epoch number
        '''
        imags = lb_images[0]  # [el[0] for el in lb_images]
        anato_masks = lb_images[1]
        patho_masks = lb_images[2]
        pseudo_health_masks = np.zeros(shape=patho_masks.shape,dtype=patho_masks.dtype)
        current_selected = epoch % imags.shape[3]
        # if len(ul_images) > 0:
        #     imags = np.concatenate([imags, ul_images], axis=0)

        # x = utils.data_utils.sample(imags, 4, seed=self.conf.seed)
        x = imags
        s = self.enc_anatomy.predict(x)
        predicted_anatomy = self.segmentor.predict(s)
        predicted_pathology_from_predicted_anatomy = self.enc_pathology.predict(np.concatenate([x,predicted_anatomy], axis=-1))
        predicted_pathology_from_predicted_anatomy = predicted_pathology_from_predicted_anatomy[:,:,:,0:-1]
        if anato_masks.shape[0] != 4 or patho_masks.shape[0] != 4:
            return

        for i in range(0, x.shape[0], 2):
            x1 = x[i: i + 1]
            s1 = self.enc_anatomy.predict(x1)
            x2 = x[i + 1: i + 2]
            s2 = self.enc_anatomy.predict(x2)

            z1_actual_pathology = sdnet_utils.vae_sample([self.z_mean.
                                                         predict([s1,
                                                                  predicted_pathology_from_predicted_anatomy, x1]),
                                                          self.z_var.
                                                         predict([s1,
                                                                  predicted_pathology_from_predicted_anatomy, x1])])
            z2_actual_pathology = sdnet_utils.vae_sample([self.z_mean.
                                                         predict([s2,
                                                                  predicted_pathology_from_predicted_anatomy, x2]),
                                                          self.z_var.
                                                         predict([s2,
                                                                  predicted_pathology_from_predicted_anatomy, x2])])

            z1_pseudo_health = sdnet_utils.vae_sample([self.z_mean.predict([s1, pseudo_health_masks, x1]),
                                                       self.z_var.predict([s1, pseudo_health_masks, x1])])
            z2_pseudo_health = sdnet_utils.vae_sample([self.z_mean.predict([s2, pseudo_health_masks, x2]),
                                                       self.z_var.predict([s2, pseudo_health_masks, x2])])

            imsave(self.interp_folder + '/interpolation1_epoch_%d_actual_pathology.png' % epoch,
                   self._interpolate(s1, z1_actual_pathology, z2_actual_pathology, current_selected, patho_masks))
            imsave(self.interp_folder + '/interpolation2_epoch_%d_actual_pathology.png' % epoch,
                   self._interpolate(s2, z2_actual_pathology, z1_actual_pathology, current_selected, patho_masks))
            imsave(self.interp_folder + '/interpolation1_epoch_%d_pseudo_health.png' % epoch,
                   self._interpolate(s1, z1_pseudo_health, z2_pseudo_health, current_selected, pseudo_health_masks))
            imsave(self.interp_folder + '/interpolation2_epoch_%d_pseudo_health.png' % epoch,
                   self._interpolate(s2, z2_pseudo_health, z1_pseudo_health, current_selected, pseudo_health_masks))

    def _interpolate(self, s, z1, z2, current_selected, m):
        row1, row2 = [], []
        for w1, w2 in zip(np.arange(0, 1, 0.1), np.arange(1, 0, -0.1)):
            sum = w1 * z1 + w2 * z2
            rec = self.reconstructor.predict([s, m, sum])[0, :, :, current_selected]
            if w1 < 0.5:
                row1.append(rec)
            else:
                row2.append(rec)
        return np.concatenate([np.concatenate(row1, axis=1), np.concatenate(row2, axis=1)], axis=0)