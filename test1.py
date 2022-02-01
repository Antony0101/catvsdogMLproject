import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

testdir="test1"
modelfile="model1.h5"
Image_Width=128
Image_Height=128
Image_Size=(Image_Width,Image_Height)
batch_size=15

def append_ext(fn):
    return fn+".jpg"

model = load_model(modelfile)
testdf=pd.read_csv("sampleSubmission.csv",dtype=str)
testdf["id"]=testdf["id"].apply(append_ext)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(testdf,
                                                 testdir,
                                                 x_col='id',
                                                 y_col=None,
                                                 target_size=Image_Size,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                  class_mode=None)

nb_samples = testdf.shape[0]
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
testdf['category'] = np.argmax(predict, axis=-1)
print(testdf)
