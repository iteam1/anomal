import os
from model import *
from data import *
import tensorflow as tf
import tensorflow as tf

src = '/content/anomal/datasets/segment/train/1'
dst = 'model/unet/model.hdf5'

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2,src,'image','label',data_gen_args,save_to_dir = None)

model = unet()

model_checkpoint = ModelCheckpoint(dst, monitor='loss',verbose=1, save_best_only=True)

model.fit_generator(myGene,steps_per_epoch=20,epochs=50,callbacks=[model_checkpoint])
