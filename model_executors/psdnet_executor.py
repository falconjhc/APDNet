eps = 1e-12 # harric added to engage the smooth factor
import itertools
import logging
import os
import math

import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Progbar

import costs
import utils.data_utils
from keras.callbacks import LearningRateScheduler
from callbacks.loss_callback import SaveLoss
from callbacks.sdnet_image_callback import SDNetImageCallback
from callbacks.sdnet_executor_loss_weights_call_back import SingleWeights_Callback
from model_executors.base_executor import Executor
from utils.distributions import NormalDistribution
from utils.image_utils import image_show, generate_mask_on_img # harric added

log = logging.getLogger('sdnet_executor')
check_batch_iters = 150

import csv
import shutil


class SDNetExecutor(Executor):
    """
    Executor for training SDNet.
    """
    def __init__(self, conf, model):
        super(SDNetExecutor, self).__init__(conf, model)

        self.lr_schedule_coef = -math.log(0.1)/self.conf.epochs

        l_mix = self.conf.l_mix
        self.conf.l_mix = float(l_mix.split('-')[0])
        self.conf.pctg_per_volume = float(l_mix.split('-')[1])

        self.model = model

        self.gen_labelled = None
        self.discriminator_anato_masks_labeled_patho = None
        self.discriminator_patho_masks_labeled_patho = None
        self.discriminator_mask_images_labeled_patho = None

        self.gen_unlabelled = None
        self.discriminator_anato_masks_unlabeled_patho = None
        self.discriminator_patho_masks_unlabeled_patho = None
        self.discriminator_mask_images_unlabeled_patho = None
        self.img_clb = None

        self.data_labelled = None
        self.data_unlabelled = None


    def write_csv(self,path, epoch,iter,performance):
        with open(path, 'a') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow([epoch, iter, performance])

    # def test(self, mark=''):
    #     """
    #     Evaluate a model on the test data.
    #     """
    #     if self.conf.modality == 'all':
    #         for modality in self.loader.modalities:
    #             log.info('Evaluating model on test data %s' % modality)
    #             folder = os.path.join(self.conf.folder, 'test_results_%s_%s' % (self.conf.test_dataset, modality))
    #             if not os.path.exists(folder):
    #                 os.makedirs(folder)
    #
    #             self.test_modality(folder, modality)
    #     else:
    #         super(SDNetExecutor, self).test(mark=mark)

    def init_train_data(self):
        """
        Initialise data iterators.
        :param split_type: training/validation/test
        """
        self.gen_labelled         = self._init_labelled_data_generator()
        self.gen_unlabelled       = self._init_unlabelled_data_generator()

        self.discriminator_anato_masks_labeled_patho, \
        self.discriminator_patho_masks_labeled_patho, \
        self.discriminator_mask_images_labeled_patho \
            = self._init_disciminator_mask_generator(sample=True)

        self.discriminator_images, \
        self.discriminator_anatomasks = self._init_discriminator_image_generator()

        self.discriminator_anato_masks_unlabeled_patho, \
        self.discriminator_patho_masks_unlabeled_patho, \
        self.discriminator_mask_images_unlabeled_patho \
            = self._init_disciminator_mask_generator(sample=False)

        self.conf.batches = int(np.ceil(self.conf.data_len / self.conf.batch_size))

    def _init_labelled_data_generator(self):
        """
        Initialise a data generator (image, mask, scanner) for labelled data
        """
        if self.conf.l_mix == 0:
            return

        log.info('Initialising labelled datagen. Loading %s data' % self.conf.dataset_name)
        self.data_labelled = \
            self.loader.load_labelled_data(self.conf.split, 'training',
                                           modality=self.conf.modality,
                                           downsample=self.conf.image_downsample)
        # harric added modality and segmentation_option auguments
        self.data_labelled.sample_per_volume(-1,self.conf.pctg_per_volume, seed=self.conf.seed)
        self.data_labelled.sample_by_volume(int(self.conf.l_mix * self.data_labelled.num_volumes), seed=self.conf.seed)

        self.data_labelled.crop(self.conf.input_shape[:2]) # crop data to input shape: useful in transfer learning
        # self.conf.data_len = self.data.size()

        datagen_dict1 = self.get_datagen_params()
        datagen_dict2 = self.get_datagen_params()
        datagen_dict3 = self.get_datagen_params()
        img_gen = ImageDataGenerator(**datagen_dict1).flow(self.data_labelled.images, batch_size=self.conf.batch_size,
                                                           seed=self.conf.seed)
        anato_msk_gen = ImageDataGenerator(**datagen_dict2).flow(self.data_labelled.anato_masks, batch_size=self.conf.batch_size,
                                                                 seed=self.conf.seed)
        patho_msk_gen = ImageDataGenerator(**datagen_dict3).flow(self.data_labelled.patho_masks, batch_size=self.conf.batch_size,
                                                                 seed=self.conf.seed)
        scn_gen = utils.data_utils.generator(self.conf.batch_size, self.conf.seed, 'no_overflow', self.data_labelled.scanner)
        return itertools.zip_longest(img_gen, anato_msk_gen, patho_msk_gen, scn_gen)

    def _init_unlabelled_data_generator(self):
        """
        Initialise a data generator (image) for unlabelled data
        """
        if self.conf.l_mix == 0:
            return

        log.info('Initialising labelled datagen. Loading %s data' % self.conf.dataset_name)
        self.data_unlabelled = \
            self.loader.load_labelled_data(self.conf.split, 'training',
                                           modality=self.conf.modality,
                                           downsample=self.conf.image_downsample)

        self.data_unlabelled.sample_per_volume(-1, self.conf.pctg_per_volume, seed=self.conf.seed)

        self.data_unlabelled.crop(self.conf.input_shape[:2])  # crop data to input shape: useful in transfer learning
        self.conf.data_len = self.data_unlabelled.size()

        datagen_dict1 = self.get_datagen_params()
        datagen_dict2 = self.get_datagen_params()
        datagen_dict3 = self.get_datagen_params()
        img_gen = ImageDataGenerator(**datagen_dict1).flow(self.data_unlabelled.images, batch_size=self.conf.batch_size,
                                                           seed=self.conf.seed)
        anato_msk_gen = ImageDataGenerator(**datagen_dict2).flow(self.data_unlabelled.anato_masks, batch_size=self.conf.batch_size,
                                                                 seed=self.conf.seed)
        patho_msk_gen = ImageDataGenerator(**datagen_dict3).flow(self.data_unlabelled.patho_masks, batch_size=self.conf.batch_size,
                                                                 seed=self.conf.seed)
        scn_gen = utils.data_utils.generator(self.conf.batch_size, self.conf.seed, 'no_overflow', self.data_unlabelled.scanner)
        return itertools.zip_longest(img_gen, anato_msk_gen, patho_msk_gen, scn_gen)

    # def _load_unlabelled_data(self, data_type):
    #     '''
    #     Create a Data object with unlabelled data. This will be used to train the unlabelled path of the
    #     generators and produce fake masks for training the discriminator
    #     :param data_type:   can be one ['ul', 'all']. The second includes images that have masks.
    #     :return:            a data object
    #     '''
    #     log.info('Loading unlabelled images of type %s' % data_type)
    #     log.info('Estimating number of unlabelled images from %s data' % self.conf.dataset_name)
    #
    #     num_all_volumes = len(self.loader.splits()[self.conf.split]['training'])
    #     # ul_mix = 1 if self.conf.ul_mix > 1 else self.conf.ul_mix
    #
    #     log.info('Initialising unlabelled datagen. Loading %s data' % self.conf.dataset_name)
    #     if data_type == 'ul':
    #         ul_data = self.loader.load_unlabelled_data(self.conf.split, 'training', modality=self.conf.modality, segmentation_option=self.conf.segmentation_option)
    #         ul_data.crop(self.conf.input_shape[:2])
    #         self.conf.num_ul_volumes = int(num_all_volumes * ul_mix)
    #         log.info('Sampling %d unlabelled images out of total %d.' % (self.conf.num_ul_volumes, num_all_volumes))
    #         ul_data.sample_by_volume(self.conf.num_ul_volumes, seed=self.conf.seed)
    #     elif data_type == 'all':
    #         ul_data = self.loader.load_all_data(self.conf.split, 'training', modality=self.conf.modality, segmentation_option=self.conf.segmentation_option)
    #         ul_data.crop(self.conf.input_shape[:2])
    #     else:
    #         raise Exception('Invalid data_type: %s' % str(data_type))
    #
    #     # Use 1200 unlabelled images maximum, to be comparable with the total number of labelled images of ACDC (~1200)
    #     # for rohan, it is 450
    #     if self.conf.dataset_name=='acdc':
    #         max_ul_images_limit = 1200
    #     elif self.conf.dataset_name == 'rohan':
    #         max_ul_images_limit = 450
    #     elif self.conf.dataset_name == 'miccai':
    #         max_ul_images_limit = 150
    #     elif self.conf.dataset_name == 'cmr':
    #         max_ul_images_limit = 200
    #     elif self.conf.dataset_name =='liverct':
    #         max_ul_images_limit = 3500
    #     elif self.conf.dataset_name == 'isles':
    #         max_ul_images_limit = 2100
    #     max_ul_images = max_ul_images_limit # if self.conf.ul_mix <= 1 else max_ul_images_limit * self.conf.ul_mix
    #     # max_ul_images = max_ul_images_limit if self.conf.ul_mix > 1 else max_ul_images_limit * self.conf.ul_mix
    #     ##########################?????????????????????#################################
    #     ##########################?????????????????????#################################
    #     ##########################?????????????????????#################################
    #
    #     if ul_data.size() > max_ul_images:
    #         samples_per_volume = int(np.ceil(max_ul_images / ul_data.num_volumes))
    #         ul_data.sample_per_volume(samples_per_volume, seed=self.conf.seed)
    #     log.info('Unlabeled Data Size: %d' % ul_data.size())
    #     return ul_data

    def _init_disciminator_mask_generator(self, batch_size=None, sample=False):
        """
        Init a generator for masks to use in the discriminator.
        """
        log.info('Initialising discriminator maskgen.')
        anato_masks, patho_masks, images, index = self._load_discriminator_masks()

        volumes = sorted(set(index))
        if sample and self.conf.l_mix * self.data_unlabelled.num_volumes < self.data_unlabelled.num_volumes:
            np.random.seed(self.conf.seed)
            volumes = np.random.choice(volumes,
                                       size=int(self.conf.l_mix * self.data_unlabelled.num_volumes),
                                       replace=False)
            anato_masks = np.concatenate([anato_masks[index==v] for v in volumes], axis=0)
            patho_masks = np.concatenate([patho_masks[index==v] for v in volumes], axis=0)
            images = np.concatenate([images[index == v] for v in volumes], axis=0)
            index = np.concatenate([index[index == v] for v in volumes], axis=0)


        datagen_dict = self.get_datagen_params()
        other_datagen_anato = ImageDataGenerator(**datagen_dict)
        other_datagen_patho = ImageDataGenerator(**datagen_dict)
        other_datagen_image = ImageDataGenerator(**datagen_dict)
        bs = self.conf.batch_size if batch_size is None else batch_size
        return other_datagen_anato.flow(anato_masks, batch_size=bs, seed=self.conf.seed), \
               other_datagen_patho.flow(patho_masks, batch_size=bs, seed=self.conf.seed), \
               other_datagen_image.flow(images,batch_size=bs,seed=self.conf.seed)

    def _load_discriminator_masks(self):
        """
        :return: dataset masks
        """
        if self.conf.seed > -1:
            np.random.seed(self.conf.seed)
        temp = \
            self.loader.load_labelled_data(self.conf.split, 'training',
                                           modality=self.conf.modality,
                                           downsample=self.conf.image_downsample)
        # harric added modality and segmentation_option arguments

        temp.sample_per_volume(-1, self.conf.pctg_per_volume, seed=self.conf.seed)

        temp.crop(self.conf.input_shape[:2])

        anato_masks = temp.anato_masks
        patho_masks = temp.patho_masks
        images = temp.images
        index = temp.index
        # volumes = temp.volumes()



        im_shape = self.conf.input_shape[:2]
        assert anato_masks.shape[1] == im_shape[0] and anato_masks.shape[2] == im_shape[1], anato_masks.shape
        assert patho_masks.shape[1] == im_shape[0] and patho_masks.shape[2] == im_shape[1], patho_masks.shape
        return anato_masks, patho_masks, images, index

    def _init_discriminator_image_generator(self):
        """
        Init a generator for images to train a discriminator (for fake masks)
        """
        log.info('Initialising discriminator imagegen.')
        # data = self._load_unlabelled_data('all')
        data = \
            self.loader.load_labelled_data(self.conf.split, 'training',
                                           modality=self.conf.modality,
                                           downsample=self.conf.image_downsample)
        data.sample_per_volume(-1,self.conf.pctg_per_volume, seed=self.conf.seed)
        images = data.images
        anato_masks = data.anato_masks

        datagen_dict = self.get_datagen_params()
        datagen_anatomask = ImageDataGenerator(**datagen_dict)
        datagen_image = ImageDataGenerator(**datagen_dict)
        return datagen_image.flow(images, batch_size=self.conf.batch_size, seed=self.conf.seed), \
               datagen_anatomask.flow(anato_masks, batch_size=self.conf.batch_size, seed=self.conf.seed)

    def init_image_callback(self):
        log.info('Initialising a data generator to use for printing.')
        datagen_dict1 = self.get_datagen_params()

        data_mask_pack = np.concatenate([self.data_unlabelled.images,
                                         self.data_unlabelled.anato_masks,
                                         self.data_unlabelled.patho_masks], axis=-1)
        gen = ImageDataGenerator(**datagen_dict1).flow(data_mask_pack, batch_size=4, seed=self.conf.seed)
        other_anato_masks_gen, other_patho_masks_gen, other_images_gen = self._init_disciminator_mask_generator(batch_size=4)
        self.img_clb = SDNetImageCallback(self.conf, self.model, gen, [other_anato_masks_gen, other_patho_masks_gen],
                                          self.data_unlabelled.images.shape[-1],
                                          self.data_unlabelled.anato_masks.shape[-1],
                                          self.data_unlabelled.patho_masks.shape[-1])

    def get_loss_names(self):
        """
        :return: loss names to report.
        """
        return ['Loss_Total_Generator_LP',
                'Loss_Total_Generator_UP',
                'Loss_Total_Discriminator_LP',
                'Loss_Total_Discriminator_UP',

                'SegDice_Anato', 'SegCrossEntropy_Anato',
                'SegDice_Patho_PPPA', 'SegCrossEntropy_Patho_PPPA',
                'SegDice_Patho_PPRA', 'SegCrossEntropy_Patho_PPRA',

                'Triplet_PPPA','Triplet_PPRA','Triplet_RP',

                'Adv_Reconstruction_Generator_RP',
                'Adv_Reconstruction_Generator_PPPA',
                'Adv_Reconstruction_Generator_PPRA',



                'Adv_Reconstruction_Discriminator_RP',
                'Adv_Reconstruction_Discriminator_PPPA',
                'Adv_Reconstruction_Discriminator_PPRA',

                # 'Clsf_Reconstruction_Discriminator_RP',
                # 'Clsf_Reconstruction_Discriminator_PPPA',
                # 'Clsf_Reconstruction_Discriminator_PPRA',

                'KL_ActualPathology_RP',
                'KL_ActualPathology_PPPA',
                'KL_ActualPathology_PPRA',

                'Reconstruct_X_RP',
                'Reconstruct_X_PPPA_LP','Reconstruct_X_PPPA_UP',
                'Reconstruct_X_PPRA_LP','Reconstruct_X_PPRA_UP',

                'Reconstruct_Z_RP',
                'Reconstruct_Z_PPPA','Reconstruct_Z_PPRA',



                'Validate_Dice', 'Test_Performance_Dice']


    def train(self):
        def _learning_rate_schedule(epoch):
            return self.conf.lr * math.exp(self.lr_schedule_coef * (-epoch - 1))

        if os.path.exists(os.path.join(self.conf.folder, 'test-performance.csv')):
            os.remove(os.path.join(self.conf.folder, 'test-performance.csv'))
        if os.path.exists(os.path.join(self.conf.folder, 'validation-performance.csv')):
            os.remove(os.path.join(self.conf.folder, 'validation-performance.csv'))

        log.info('Training Model')
        dice_record = 0
        self.eval_train_interval = int(max(1, self.conf.epochs/50))

        self.init_train_data()
        lr_callback = LearningRateScheduler(_learning_rate_schedule)

        self.init_image_callback()
        sl = SaveLoss(self.conf.folder)
        cl = CSVLogger(self.conf.folder + '/training.csv')
        cl.on_train_begin()

        es = EarlyStopping('Validate_Dice', self.conf.min_delta, self.conf.patience)
        es.model = self.model.Segmentor
        es.on_train_begin()

        loss_names = self.get_loss_names()
        loss_names.sort()
        total_loss = {n: [] for n in loss_names}

        progress_bar = Progbar(target=self.conf.batches)
        # self.img_clb.on_epoch_end(self.epoch)

        best_performance = 0.
        test_performance = 0.
        total_iters = 0
        for self.epoch in range(self.conf.epochs):
            total_iters+=1
            log.info('Epoch %d/%d' % (self.epoch+1, self.conf.epochs))

            epoch_loss = {n: [] for n in loss_names}
            epoch_loss_list = []

            for self.batch in range(self.conf.batches):
                total_iters += 1
                self.train_batch(epoch_loss, lr_callback)
                progress_bar.update(self.batch + 1)

            val_dice = self.validate(epoch_loss)
            if val_dice > dice_record:
                dice_record = val_dice

            cl.model = self.model.D_Reconstruction
            cl.model.stop_training = False

            self.model.save_models()

            # Plot some example images
            if self.epoch % self.eval_train_interval == 0 or self.epoch == self.conf.epochs - 1:
                self.img_clb.on_epoch_end(self.epoch)
                folder = os.path.join(os.path.join(self.conf.folder, 'test_during_train'),
                                      'test_results_%s_epoch%d'
                                      % (self.conf.test_dataset, self.epoch))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                test_performance = self.test_modality(folder, self.conf.modality, 'test', False)
                if test_performance > best_performance:
                    best_performance = test_performance
                    self.model.save_models('BestModel')
                    log.info("BestModel@Epoch%d" % self.epoch)

                folder = os.path.join(os.path.join(self.conf.folder, 'test_during_train'),
                                      'validation_results_%s_epoch%d'
                                      % (self.conf.test_dataset, self.epoch))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                validation_performance = self.test_modality(folder, self.conf.modality, 'validation', False)
                if self.conf.batches>check_batch_iters:
                    self.write_csv(os.path.join(self.conf.folder, 'test-performance.csv'),
                                   self.epoch, self.batch, test_performance)
                    self.write_csv(os.path.join(self.conf.folder, 'validation-performance.csv'),
                                   self.epoch, self.batch, validation_performance)
            epoch_loss['Test_Performance_Dice'].append(test_performance)

            for n in loss_names:
                epoch_loss_list.append((n, np.mean(epoch_loss[n])))
                total_loss[n].append(np.mean(epoch_loss[n]))

            if self.epoch<5:
                log.info(str('Epoch %d/%d:\n' + ''.join([l + ' Loss = %.3f\n' for l in loss_names])) %
                         ((self.epoch, self.conf.epochs) + tuple(total_loss[l][-1] for l in loss_names)))
            else:
                info_str = str('Epoch %d/%d:\n' % (self.epoch, self.conf.epochs))
                loss_info = ''
                for l in loss_names:
                    loss_info = loss_info + l + ' Loss = %.3f->%.3f->%.3f->%.3f->%.3f\n' % \
                                (total_loss[l][-5],
                                 total_loss[l][-4],
                                 total_loss[l][-3],
                                 total_loss[l][-2],
                                 total_loss[l][-1])
                log.info(info_str + loss_info)
            log.info("BestTest:%f" % best_performance)
            log.info('Epoch %d/%d' % (self.epoch + 1, self.conf.epochs))
            logs = {l: total_loss[l][-1] for l in loss_names}
            cl.on_epoch_end(self.epoch, logs)
            sl.on_epoch_end(self.epoch, logs)

            if self.stop_criterion(es, logs) and self.epoch > self.conf.epochs / 2:
                log.info('Finished training from early stopping criterion')
                self.img_clb.on_epoch_end(self.epoch)
                break


    def validate(self, epoch_loss):
        # Report validation error
        valid_data = \
            self.loader.load_labelled_data(self.conf.split, 'validation',
                                           modality=self.conf.modality,
                                           downsample=self.conf.image_downsample)

        # harric added modality and segmentation_option arguments
        valid_data.crop(self.conf.input_shape[:2])
        anatomy_segmentor = self.model.Segmentor
        anatomy_encoder = self.model.Enc_Anatomy
        pathology_segmentor = self.model.Enc_Pathology

        s = anatomy_encoder.predict(valid_data.images)
        anatomy_pred_mask = anatomy_segmentor.predict(s)
        pathology_pred_mask = pathology_segmentor.predict(np.concatenate([valid_data.images,anatomy_pred_mask], axis=-1))

        assert pathology_pred_mask.shape[:-1] == valid_data.images.shape[:-1], \
            str(valid_data.images.shape) + ' ' + str(pathology_pred_mask.shape)
        epoch_loss['Validate_Dice'].append(1 - costs.dice(valid_data.patho_masks,
                                                          pathology_pred_mask)[0])
        return costs.dice(valid_data.patho_masks,pathology_pred_mask)[0]

    def discriminator_train_with_labeled_pathology(self, epoch_loss, lr_callback):
        # train with labelled
        m_anato = next(self.discriminator_anato_masks_labeled_patho)
        m_patho = next(self.discriminator_patho_masks_labeled_patho)
        x_masks = next(self.discriminator_mask_images_labeled_patho)

        batch_size = x_masks.shape[0]  # maybe this differs from conf.batch_size at the last batch.
        m_anato_backgroud = np.ones(shape=m_anato.shape[:-1] + (1,))
        for ii in range(m_anato.shape[-1]):
            m_anato_backgroud = m_anato_backgroud - m_anato[:, :, :, ii:ii + 1]

        fake_s = self.model.Enc_Anatomy.predict(x_masks)
        fake_m_real_pathology = self.model.Enc_Modality.predict([fake_s, m_patho, x_masks])[0]

        fake_rec_real_pathology_real_pathology = self.model.Decoder.predict([fake_s, m_patho, fake_m_real_pathology])

        # Train Discriminator
        image_shape = (batch_size,) + self.model.D_Reconstruction.get_output_shape_at(0)[0][1:]

        h_reconstruction_discriminator_rp = \
            self.model.D_Reconstruction_trainer_lp_rp. \
                fit([x_masks, fake_rec_real_pathology_real_pathology],
                    [np.ones(image_shape),
                     np.zeros(image_shape),
                     np.zeros(batch_size),np.zeros(batch_size)],
                    epochs=1, verbose=0, callbacks= [lr_callback])

        epoch_loss['Adv_Reconstruction_Discriminator_RP']. \
            append([(h_reconstruction_discriminator_rp.history[
                         'Adv_Reconstruction_ActualPathology_Real_ActualPathology_loss'][-1] +
                     h_reconstruction_discriminator_rp.history[
                         'Adv_Reconstruction_ActualPathology_Fake_ActualPathology_loss'][-1])])


        return h_reconstruction_discriminator_rp.history['loss'][-1]


    def discriminator_train_with_unlabeled_pathology(self, epoch_loss, lr_callback):
        m_anato = next(self.discriminator_anato_masks_unlabeled_patho)
        x_masks = next(self.discriminator_mask_images_unlabeled_patho)

        m_anato_backgroud = np.ones(shape=m_anato.shape[:-1] + (1,))
        for ii in range(m_anato.shape[-1]):
            m_anato_backgroud = m_anato_backgroud - m_anato[:, :, :, ii:ii + 1]
        m_anato_with_background = np.concatenate([m_anato, m_anato_backgroud], axis=-1)

        batch_size = x_masks.shape[0]
        fake_s = self.model.Enc_Anatomy.predict(x_masks)
        fake_m = self.model.Segmentor.predict(fake_s)
        predicted_pathology_ppra = self.model.Enc_Pathology.predict(np.concatenate([x_masks, m_anato_with_background], axis=-1))
        fake_z_ppra = self.model.Enc_Modality.predict([fake_s, predicted_pathology_ppra[:, :, :, 0:-1], x_masks])[0]
        fake_rec_ppra = self.model.Decoder.predict([fake_s, predicted_pathology_ppra[:, :, :, 0:-1], fake_z_ppra])

        predicted_pathology_pppa = self.model.Enc_Pathology.predict(np.concatenate([x_masks, fake_m], axis=-1))
        fake_z_pppa = self.model.Enc_Modality.predict([fake_s, predicted_pathology_pppa[:, :, :, 0:-1], x_masks])[0]
        fake_rec_pppa = self.model.Decoder.predict([fake_s, predicted_pathology_pppa[:, :, :, 0:-1], fake_z_pppa])

        image_shape = (batch_size,) + self.model.D_Reconstruction.get_output_shape_at(0)[0][1:]
        actual_classification_label_batch = np.array(np.array(np.sum(m_anato, axis=(1, 2, 3)), dtype=bool),
                                                     dtype=np.uint8)
        one_hot_actual_target = np.eye(2)[actual_classification_label_batch]

        h_reconstruction_discriminator_pppa_supervised = \
            self.model.D_Reconstruction_trainer_up_pppa. \
                fit([x_masks, fake_rec_pppa],
                    [np.ones(image_shape),
                     np.zeros(image_shape),

                     np.zeros(batch_size),np.zeros(batch_size)],
                    epochs=1, verbose=0, callbacks = [lr_callback])

        h_reconstruction_discriminator_ppra_supervised = \
            self.model.D_Reconstruction_trainer_up_ppra. \
                fit([x_masks, fake_rec_ppra],
                    [np.ones(image_shape),
                     np.zeros(image_shape),

                     np.zeros(batch_size),np.zeros(batch_size)],
                    epochs=1, verbose=0, callbacks = [lr_callback])

        epoch_loss['Adv_Reconstruction_Discriminator_PPPA']. \
            append([(h_reconstruction_discriminator_pppa_supervised.history[
                         'Adv_Reconstruction_ActualPathology_Real_ActualPathology_loss'][-1] +
                     h_reconstruction_discriminator_pppa_supervised.history[
                         'Adv_Reconstruction_ActualPathology_Fake_ActualPathology_loss'][-1])])
        epoch_loss['Adv_Reconstruction_Discriminator_PPRA']. \
            append([(h_reconstruction_discriminator_ppra_supervised.history[
                         'Adv_Reconstruction_ActualPathology_Real_ActualPathology_loss'][-1] +
                     h_reconstruction_discriminator_ppra_supervised.history[
                         'Adv_Reconstruction_ActualPathology_Fake_ActualPathology_loss'][-1])])




        return h_reconstruction_discriminator_pppa_supervised.history['loss'][-1] \
               + h_reconstruction_discriminator_ppra_supervised.history['loss'][-1]



    def train_batch(self, epoch_loss, lr_callback):

        discriminator_lp_loss = discriminator_up_loss = 0.
        if self.conf.l_mix>0:
            generator_lp_loss = self.train_batch_generators_labeled_pathology(epoch_loss, lr_callback)
            discriminator_lp_loss = self.discriminator_train_with_labeled_pathology(epoch_loss, lr_callback)
        generator_up_loss = self.\
            train_batch_generators_unlabeld_pathology(epoch_loss, lr_callback)
        discriminator_up_loss = self.discriminator_train_with_unlabeled_pathology(epoch_loss, lr_callback)

        epoch_loss['Loss_Total_Generator_LP'].append(generator_lp_loss)
        epoch_loss['Loss_Total_Discriminator_LP'].append(discriminator_lp_loss)
        epoch_loss['Loss_Total_Generator_UP'].append(generator_up_loss)
        epoch_loss['Loss_Total_Discriminator_UP'].append(discriminator_up_loss)



    def train_batch_generators_unlabeld_pathology(self, epoch_loss, lr_callback):
        if self.gen_unlabelled is not None:
            x, anato_m, _, scanner = next(self.gen_unlabelled)
            pseudo_health_m = np.zeros(shape=tuple(anato_m.shape[:-1]
                                                   + (self.conf.num_pathology_masks,)), dtype=anato_m.dtype)
            batch_size = x.shape[0]

            # actual_classification_label_batch = np.array(np.array(np.sum(_, axis=(1, 2, 3)), dtype=bool),
            #                                              dtype=np.uint8)
            # one_hot_actual_target = np.eye(2)[actual_classification_label_batch]
            # one_hot_pseudo_health = np.concatenate([np.ones(shape=(x.shape[0], 1), dtype=one_hot_actual_target.dtype),
            #                                         np.zeros(shape=(x.shape[0], 1), dtype=one_hot_actual_target.dtype)],
            #                                        axis=1)

            h_pppa = self.model.G_trainer_up_pppa. \
                fit([x],
                    [np.zeros(batch_size),
                     np.ones((batch_size,) + self.model.D_Reconstruction.output_shape[0][1:])],
                    epochs=1, verbose=0, callbacks= [lr_callback])
            h_pppa_reconst = self.model.G_trainer_up_pppa_reconst.\
                fit([x], x,
                    epochs=1, verbose=0, callbacks=[lr_callback])


            h_ppra = self.model.G_trainer_up_ppra. \
                fit([x, anato_m],
                    [np.zeros(batch_size),
                     np.ones((batch_size,) + self.model.D_Reconstruction.output_shape[0][1:]),
                     ],
                    epochs=1, verbose=0, callbacks= [lr_callback])
            h_ppra_reconst = self.model.G_trainer_up_ppra_reconst. \
                fit([x, anato_m],
                    [x],
                    epochs=1, verbose=0, callbacks=[lr_callback])

            h_anatomy = self.model.G_trainer_up_anatomy. \
                fit([x],
                    [anato_m, anato_m],
                    epochs=1, verbose=0, callbacks= [lr_callback])

            h_triplet = self.model.G_trainer_up_triplet. \
                fit([x, pseudo_health_m, anato_m],
                    [np.zeros(batch_size),
                     np.zeros(batch_size),
                     np.ones((batch_size,) + self.model.D_Reconstruction.output_shape[0][1:])],
                    epochs=1, verbose=0, callbacks= [lr_callback])

            epoch_loss['Triplet_PPPA'].append(h_triplet.history['Triplet_PPPA_loss'][-1])
            epoch_loss['Triplet_PPRA'].append(h_triplet.history['Triplet_PPRA_loss'][-1])

            epoch_loss['SegDice_Anato'].append(h_anatomy.history['Dice_Anato_loss'][-1])
            epoch_loss['SegCrossEntropy_Anato'].append(h_anatomy.history['CrossEntropy_Anato_loss'][-1])

            epoch_loss['Adv_Reconstruction_Generator_PPPA'].append([h_pppa.history['D_Reconstruction_loss'][-1]])
            epoch_loss['KL_ActualPathology_PPPA'].append([h_pppa.history['Enc_Modality_loss'][-1]])
            epoch_loss['Reconstruct_X_PPPA_UP'].append(h_pppa_reconst.history['loss'][-1])
            # #
            epoch_loss['Adv_Reconstruction_Generator_PPRA'].append([h_ppra.history['D_Reconstruction_loss'][-1]])
            epoch_loss['KL_ActualPathology_PPRA'].append([h_ppra.history['Enc_Modality_loss'][-1]])
            epoch_loss['Reconstruct_X_PPRA_UP'].append(h_ppra_reconst.history['loss'][-1])

            s = self.model.Enc_Anatomy.predict(x)
            predicted_anatomy = self.model.Segmentor.predict(s)
            predicted_pathology_from_predicted_anatomy \
                = self.model.Enc_Pathology.predict(np.concatenate([x, predicted_anatomy], axis=-1))
            m_anato_backgroud = np.ones(shape=anato_m.shape[:-1] + (1,))
            for ii in range(anato_m.shape[-1]):
                m_anato_backgroud = m_anato_backgroud - anato_m[:, :, :, ii:ii + 1]
            m_anato_with_background = np.concatenate([anato_m, m_anato_backgroud], axis=-1)
            predicted_pathology_from_real_anatomy = self.model.Enc_Pathology.predict(
                np.concatenate([x, m_anato_with_background], axis=-1))
            sample_z2 = NormalDistribution().sample((batch_size, self.conf.num_z))
            sample_z3 = NormalDistribution().sample((batch_size, self.conf.num_z))

            h_pppa_z = self.model.z_reconstructor. \
                fit([s, predicted_pathology_from_predicted_anatomy[:, :, :, :-1],
                     sample_z2], sample_z2, epochs=1, verbose=0,
                    callbacks=[SingleWeights_Callback(self.conf.w_rec_Z * self.conf.pred_pathology_weight_rate * self.conf.pred_anatomy_weight_rate + eps,
                                                      model=self.model.z_reconstructor),lr_callback])


            h_ppra_z = self.model.z_reconstructor. \
                fit([s, predicted_pathology_from_real_anatomy[:, :, :, :-1],
                     sample_z3], sample_z3, epochs=1, verbose=0,
                    callbacks=[SingleWeights_Callback(self.conf.w_rec_Z * self.conf.pred_pathology_weight_rate * self.conf.real_anatomy_weight_rate + eps,
                                                      model=self.model.z_reconstructor),lr_callback])


            epoch_loss['Reconstruct_Z_PPPA']. \
                append([h_pppa_z.history['loss'][-1] / (self.conf.w_rec_Z
                                                        * self.conf.pred_pathology_weight_rate
                                                        * self.conf.pred_anatomy_weight_rate + eps)])
            epoch_loss['Reconstruct_Z_PPRA']. \
                append([h_ppra_z.history['loss'][-1] / (self.conf.w_rec_Z
                                                        * self.conf.pred_pathology_weight_rate
                                                        * self.conf.real_anatomy_weight_rate + eps)])

            return h_anatomy.history['loss'][-1] \
                   + h_pppa_reconst.history['loss'][-1] + h_ppra_reconst.history['loss'][-1] \
                   + h_pppa.history['loss'][-1] + h_ppra.history['loss'][-1] \
                   + h_triplet.history['loss'][-1] \
                   + h_pppa_z.history['loss'][-1] + h_ppra_z.history['loss'][-1]


    def train_batch_generators_labeled_pathology(self, epoch_loss, lr_callback):
        """
        Train generator for labelled pathology networks.
        :param epoch_loss:  Dictionary of losses for the epoch
        """
        if self.gen_labelled is not None:
            x, anato_m, patho_m, scanner = next(self.gen_labelled)
            pseudo_health_m = np.zeros(shape=patho_m.shape, dtype=anato_m.dtype)

            batch_size = x.shape[0]  # maybe this differs from conf.batch_size at the last batch.

            # generate pathology labels
            actual_classification_label_batch = np.array(np.array(np.sum(patho_m,axis=(1,2,3)), dtype=bool), dtype=np.uint8)
            one_hot_actual_target = np.eye(2)[actual_classification_label_batch]


            # train for real pathology
            h_rp = self.model.G_trainer_lp_rp. \
                fit([x,patho_m],
                    [np.zeros(batch_size),
                     np.zeros(x.shape[0],dtype=x.dtype),
                     np.ones((batch_size,) + self.model.D_Reconstruction.output_shape[0][1:])],
                    epochs=1, verbose=0, callbacks= [lr_callback])

            s = self.model.Enc_Anatomy.predict(x)
            sample_z1 = NormalDistribution().sample((batch_size, self.conf.num_z))
            h_rp_z = self.model.z_reconstructor. \
                fit([s, patho_m, sample_z1], sample_z1, epochs=1, verbose=0,
                    callbacks=[SingleWeights_Callback(self.conf.w_rec_Z * self.conf.real_pathology_weight_rate + eps,
                                                      model=self.model.z_reconstructor)])

            # train for predicted pathology and predicted anatomy
            h_pppa = self.model.G_trainer_lp_pppa. \
                fit([x, patho_m],
                    [patho_m, patho_m],
                    epochs=1, verbose=0, callbacks= [lr_callback])
            h_pppa_reconst = self.model.G_trainer_lp_pppa_reconst. \
                fit([x, patho_m],
                    [np.zeros(x.shape[0], dtype=x.dtype)],
                    epochs=1, verbose=0, callbacks=[lr_callback])

            # train for predicted pathology and real anatony
            h_ppra = self.model.G_trainer_lp_ppra. \
                fit([x, anato_m, patho_m],
                    [patho_m, patho_m],
                    epochs=1, verbose=0, callbacks= [lr_callback])
            h_ppra_reconst = self.model.G_trainer_lp_ppra_reconst. \
                fit([x, anato_m, patho_m],
                    [np.zeros(x.shape[0], dtype=x.dtype)],
                    epochs=1, verbose=0, callbacks=[lr_callback])

            # train for ratio-based triplet loss
            h_triplet = self.model.G_trainer_lp_triplet. \
                fit([x, patho_m, pseudo_health_m],
                    [np.zeros(batch_size),
                     np.ones((batch_size,) + self.model.D_Reconstruction.output_shape[0][1:])],
                    epochs=1, verbose=0, callbacks= [lr_callback])

            epoch_loss['Triplet_RP'].append(h_triplet.history['Triplet_RP_loss'][-1])

            epoch_loss['SegDice_Patho_PPPA'].append(h_pppa.history['Dice_Patho_loss'][-1])
            epoch_loss['SegCrossEntropy_Patho_PPPA'].append(h_pppa.history['CrossEntropy_Patho_loss'][-1])
            epoch_loss['SegDice_Patho_PPRA'].append(h_ppra.history['Dice_Patho_loss'][-1])
            epoch_loss['SegCrossEntropy_Patho_PPRA'].append(h_ppra.history['CrossEntropy_Patho_loss'][-1])

            epoch_loss['Adv_Reconstruction_Generator_RP'].append([h_rp.history['D_Reconstruction_loss'][-1]])
            epoch_loss['KL_ActualPathology_RP'].append([h_rp.history['Enc_Modality_loss'][-1]])

            epoch_loss['Reconstruct_X_RP'].append(h_rp.history['Reconstructor_RP_loss'][-1])
            epoch_loss['Reconstruct_X_PPPA_LP'].append(h_pppa_reconst.history['loss'][-1])
            epoch_loss['Reconstruct_X_PPRA_LP'].append(h_ppra_reconst.history['loss'][-1])





            epoch_loss['Reconstruct_Z_RP'].append([h_rp_z.history['loss'][-1] /
                                                   (self.conf.w_rec_Z * self.conf.real_pathology_weight_rate + eps)])


            return h_pppa.history['loss'][-1] + h_ppra.history['loss'][-1] \
                   + h_pppa_reconst.history['loss'][-1] + h_ppra_reconst.history['loss'][-1] \
                   + h_rp.history['loss'][-1] \
                   + h_triplet.history['loss'][-1] + h_rp_z.history['loss'][-1]