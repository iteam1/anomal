import os
from model import *
from data import *
from keras.models import load_model

test_dir = "datasets/unet/test" 
model_dir = "model/unet/model.hdf5"

testGene = testGenerator(test_dir)

trained_model = load_model(model_dir)

results = trained_model.predict_generator(testGene,2,verbose=1)

# saveResult("data/membrane/test",results)