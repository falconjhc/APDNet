from keras.callbacks import Callback
import keras.backend as K


class SingleWeights_Callback(Callback):
    def __init__(self, weight, model):
        super(SingleWeights_Callback, self).__init__()
        self.weight = weight
        # K.set_value(model.loss_weight, self.weight)
        self.model = model
       #  model.loss_weight = self.weight
    def on_epoch_begin(self, epoch=None, logs=None):
       #  self.weight = self.weight+0.
        K.set_value(self.model.loss_weight, self.weight)
    def on_epoch_end(self, epoch=None, logs=None):
        K.set_value(self.model.loss_weight, self.weight)