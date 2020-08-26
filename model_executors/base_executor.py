import logging
import numpy as np
import os
from abc import abstractmethod
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from callbacks.image_callback import SaveEpochImages
from loaders import loader_factory
from utils.image_utils import save_segmentation
from utils.image_utils import image_show
from costs import dice, calculate_false_negative # harric added regression2segmentation to incorporate with segmentation_option=4 case
log = logging.getLogger('executor')
from imageio import imwrite as imsave # harric modified


class Executor(object):
    """
    Base class for executor objects.
    """
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
        self.loader = loader_factory.init_loader(self.conf.dataset_name)
        self.epoch = 0
        self.models_folder = self.conf.folder + '/models'
        self.train_data = None
        self.valid_data = None
        self.train_folder = None

    @abstractmethod
    def init_train_data(self):
        self.train_data = \
            self.loader.load_labelled_data(self.conf.split, 'training',
                                           downsample=self.conf.image_downsample,
                                           modality=self.conf.modality,
                                           segmentation_option=self.conf.segmentation_option
                                           if self.conf.segmentation_option is not None
                                           else -1)
        self.valid_data = \
            self.loader.load_labelled_data(self.conf.split, 'validation',
                                           downsample=self.conf.image_downsample,
                                           modality=self.conf.modality,
                                           segmentation_option=self.conf.segmentation_option if self.conf.segmentation_option is not None
                                           else -1)

        self.train_data.select_masks(self.conf.num_masks)
        self.valid_data.select_masks(self.conf.num_masks)

        self.train_data.sample(int(self.conf.l_mix * self.train_data.num_volumes), seed=self.conf.seed)
        self.conf.data_len = self.train_data.size()

    @abstractmethod
    def get_loss_names(self):
        pass

    @abstractmethod
    def train(self):
        log.info('Training Model')
        self.init_train_data()

        self.train_folder = os.path.join(self.conf.folder, 'training_results')
        if not os.path.exists(self.train_folder):
            os.mkdir(self.train_folder)

        callbacks = self.init_callbacks()

        train_images = self.get_inputs(self.train_data)
        train_labels = self.get_labels(self.train_data)

        valid_images = self.get_inputs(self.valid_data)
        valid_labels = self.get_labels(self.valid_data)

        if self.conf.outputs > 1:
            train_labels = [self.train_data.masks[..., i:i+1] for i in range(self.conf.outputs)]
            valid_labels = [self.valid_data.masks[..., i:i+1] for i in range(self.conf.outputs)]
        if self.conf.augment:
            datagen_dict = self.get_datagen_params()
            datagen = ImageDataGenerator(**datagen_dict)
            gen = data_generator_multiple_outputs(datagen, self.conf.batch_size, train_images, [train_labels,train_labels])

            self.model.model.fit_generator(gen, steps_per_epoch=len(train_images) / self.conf.batch_size,
                                           epochs=self.conf.epochs, callbacks=callbacks,
                                           validation_data=(valid_images, [valid_labels,valid_labels]))
        else:
            self.model.model.fit(train_images, train_labels,
                                 validation_data=(valid_images, valid_labels),
                                 epochs=self.conf.epochs, callbacks=callbacks, batch_size=self.conf.batch_size)

    def init_callbacks(self):
        datagen_dict = self.get_datagen_params()

        data_pack = np.concatenate([self.train_data.images,self.train_data.masks], axis=-1)
        image_channels = self.train_data.images.shape[-1]
        mask_chananels = self.train_data.masks.shape[-1]
        gen = ImageDataGenerator(**datagen_dict).flow(x=data_pack,
                                                      batch_size=self.conf.batch_size,
                                                      seed=self.conf.seed)

        es = EarlyStopping(min_delta=0.01, patience=50)
        si = SaveEpochImages(self.conf, self.model, gen,image_channels,mask_chananels)
        cl = CSVLogger(self.train_folder + '/training.csv')
        mc = ModelCheckpoint(self.conf.folder + '/model', monitor='val_loss', verbose=0, save_best_only=False,
                             save_weights_only=True, mode='min', period=1)
        mc_best = ModelCheckpoint(self.conf.folder + '/model_best', monitor='val_loss', verbose=0, save_best_only=True,
                                  save_weights_only=True, mode='min', period=1)
        return [es, si, cl, mc, mc_best]

    def get_labels(self, data):
        """
        :param data: the Data object used in training
        :return:     the network's target, usually the masks
        """
        return data.masks

    def get_inputs(self, data):
        """
        :param data: the Data object used in training
        :return:     the network's input, usually the images
        """
        return data.images

    @abstractmethod
    def test(self):
        """
        Evaluate a model on the test data.
        """
        self.model.load_models()

        # evaluate on test set
        log.info('Evaluating model on test data')
        folder = os.path.join(self.conf.folder, 'test_results_%s' % self.conf.test_dataset)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.test_modality(folder, self.conf.modality, 'test')

        # evaluate on train set
        log.info('Evaluating model on training data')
        folder = os.path.join(self.conf.folder, 'training_results_%s'% self.conf.test_dataset)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.test_modality(folder, self.conf.modality, 'training')

        # evaluate on the validation set
        log.info('Evaluating model on validation data')
        folder = os.path.join(self.conf.folder, 'validation_results_%s' % self.conf.test_dataset)
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.test_modality(folder, self.conf.modality, 'validation')




    def test_modality(self, folder, modality, group, save_figs=True):
        test_loader = loader_factory.init_loader(self.conf.test_dataset)
        test_data = test_loader.load_labelled_data(self.conf.split, group,
                                                   modality=modality,
                                                   downsample=self.conf.image_downsample)

        anatomy_segmentor = self.model.get_anatomy_segmentor()
        pathology_segmentator = self.model.get_pathology_encoder()

        synth = []
        im_dice_anato, im_false_negative_anato = {}, {}
        im_dice_patho, im_false_negative_patho = {}, {}

        sep_dice_list_anato, sep_false_negative_list_anato = [], []
        sep_dice_list_patho, sep_false_negative_list_patho = [], []
        anato_mask_num = len(test_data.anato_mask_names)
        patho_mask_num = len(test_data.patho_mask_names)
        for ii in range(anato_mask_num):
            sep_dice_list_anato.append([])
            sep_false_negative_list_anato.append([])
        for ii in range(patho_mask_num):
            sep_dice_list_patho.append([])
            sep_false_negative_list_patho.append([])

        f = open(os.path.join(folder, 'results.csv'), 'w')
        for vol_i in test_data.volumes():
            vol_image = test_data.get_images(vol_i)
            vol_anato_mask = test_data.get_anato_masks(vol_i)
            vol_patho_mask = test_data.get_patho_masks(vol_i)
            vol_slice = test_data.get_slice(vol_i)
            assert vol_image.shape[0] > 0 and vol_image.shape[:-1] == vol_anato_mask.shape[:-1] and vol_image.shape[:-1] == vol_patho_mask.shape[:-1]
            anato_pred = anatomy_segmentor.predict(vol_image)
            patho_pred = pathology_segmentator.predict(vol_image)
            pred = [anato_pred, patho_pred]
            synth.append(pred)

            model_type = 'sdnet'

            im_dice_anato[vol_i], sep_dice_anato \
                = dice(vol_anato_mask, pred[0])
            im_false_negative_anato[vol_i], sep_false_negative_anato \
                = calculate_false_negative(vol_anato_mask, pred[0])

            im_dice_patho[vol_i], sep_dice_patho \
                = dice(vol_patho_mask, pred[1])
            im_false_negative_patho[vol_i], sep_false_negative_patho \
                = calculate_false_negative(vol_patho_mask, pred[1])

            # harric added to specify dice scores across different masks
            assert anato_mask_num == len(sep_dice_anato), 'Incorrect mask num !'
            assert patho_mask_num == len(sep_dice_patho), 'Incorrect mask num !'
            for ii in range(anato_mask_num):
                sep_dice_list_anato[ii].append(sep_dice_anato[ii])
                sep_false_negative_list_anato[ii].append(sep_false_negative_anato[ii])
            for ii in range(patho_mask_num):
                sep_dice_list_patho[ii].append(sep_dice_patho[ii])
                sep_false_negative_list_patho[ii].append(sep_false_negative_patho[ii])

            # harric added to specify dice scores across different masks
            s = 'Volume:%s, AnatomyDice:%.3f, AnatomyFN:%.3f, ' \
                + 'PathologyDice:%.3f, PathologyFN:%.3f, ' \
                + ', '.join(['%s, %.3f, %.3f, '] * len(test_data.anato_mask_names)) \
                + ', '.join(['%s, %.3f, %.3f, '] * len(test_data.patho_mask_names)) \
                + '\n'
            d = (str(vol_i), im_dice_anato[vol_i], im_false_negative_anato[vol_i])
            d += (im_dice_patho[vol_i], im_false_negative_patho[vol_i])
            for info_travesal in range(anato_mask_num):
                d += (test_data.anato_mask_names[info_travesal],
                      sep_dice_anato[info_travesal],
                      sep_false_negative_anato[info_travesal])
            for info_travesal in range(patho_mask_num):
                d += (test_data.patho_mask_names[info_travesal],
                      sep_dice_patho[info_travesal],
                      sep_false_negative_patho[info_travesal])
            f.writelines(s % d)

            if save_figs:
                for i in range(vol_image.shape[0]):
                    d, m, mm = vol_image[i], vol_anato_mask[i], vol_patho_mask[i]
                    # d, m, mm = vol_image[10], vol_anato_mask[10], vol_patho_mask[10]
                    s = vol_slice[i]
                    im1 = save_segmentation(pred[0][i, :, :, :], d, m)
                    im2 = save_segmentation(pred[1][i, :, :, :], d, mm)

                    if im1.shape[1] > im2.shape[1]:
                        im2 = np.concatenate([im2, np.zeros(shape=(im2.shape[0], im1.shape[1] - im2.shape[1]),
                                                            dtype=im2.dtype)], axis=1)
                    elif im1.shape[1] < im2.shape[1]:
                        im1 = np.concatenate([im1, np.zeros(shape=(im1.shape[0], im2.shape[1] - im1.shape[1]),
                                                            dtype=im1.dtype)], axis=1)

                    im = np.concatenate([im1, im2], axis=0)
                    imsave(os.path.join(folder, "vol%s_slice%s" % (str(vol_i), s) + '.png'), im)

        # harric added to specify dice scores across different masks
        print_info = group + ', AnatomyDice:%.3f, AnatoFN:%.3f, PathoDice:%.3f, PathoFN:%.3f,' % \
                     (np.mean(list(im_dice_anato.values())),
                      np.mean(list(im_false_negative_anato.values())),
                      np.mean(list(im_dice_patho.values())),
                      np.mean(list(im_false_negative_patho.values())))
        for ii in range(anato_mask_num):
            print_info += '%s, %.3f, %.3f,' % \
                          (test_data.anato_mask_names[ii],
                           np.mean(sep_dice_list_anato[ii]),
                           np.mean(sep_false_negative_list_anato[ii]))
        for ii in range(patho_mask_num):
            print_info += '%s, %.3f, %.3f' % \
                          (test_data.patho_mask_names[ii],
                           np.mean(sep_dice_list_patho[ii]),
                           np.mean(sep_false_negative_list_patho[ii]))
        print(print_info)
        f.write(print_info)
        f.close()
        return np.mean(list(im_dice_patho.values()))

    def stop_criterion(self, es, logs):
        es.on_epoch_end(self.epoch, logs)
        if es.stopped_epoch > 0:
            return True

    def get_datagen_params(self):
        """
        Construct a dictionary of augmentations.
        :param augment_spatial:
        :param augment_intensity:
        :return: a dictionary of augmentation parameters to use with a keras image processor
        """
        result = dict(horizontal_flip=False, vertical_flip=False, rotation_range=0.)

        if self.conf.augment:
            result['rotation_range'] = 90.
            result['horizontal_flip'] = True
            result['vertical_flip'] = True
            result['width_shift_range'] = 0.15
            result['height_shift_range'] = 0.15
        return result

    def align_batches(self, array_list):
        """
        Align the arrays of the input list, based on batch size.
        :param array_list: list of 4-d arrays to align
        """
        mn = np.min([x.shape[0] for x in array_list])
        new_list = [x[0:mn] for x in array_list]
        return new_list

    def get_fake(self, pred, fake_pool, sample_size=-1):
        sample_size = self.conf.batch_size if sample_size == -1 else sample_size

        if pred.shape[0] > 0:
            fake_pool.extend(pred)

        fake_pool = fake_pool[-self.conf.pool_size:]
        sel = np.random.choice(len(fake_pool), size=(sample_size,), replace=False)
        fake_A = np.array([fake_pool[ind] for ind in sel])
        return fake_pool, fake_A


def data_generator_multiple_outputs(datagen, batch_size, inp, outs):
    x = inp
    in_channels = inp.shape[3]
    out_channels = outs[0].shape[3]
    for ii in range(len(outs)):
        if ii ==0:
            y = outs[ii]
        else:
            y = np.concatenate([y,outs[ii]], axis=-1)
    x_and_y = np.concatenate([x,y],axis=-1)

    gen_iterator = datagen.flow(x=x_and_y, batch_size=batch_size, seed=1)
    while True:
        batch_x_and_y = next(gen_iterator)
        batch_x = batch_x_and_y[:,:,:,0:in_channels]
        batch_y_raw = batch_x_and_y[:,:,:,in_channels:]
        batch_y = []
        for ii in range(len(outs)):
            batch_y.append(batch_y_raw[:, :, :, ii * out_channels:(ii + 1) * out_channels])
        yield batch_x, batch_y
