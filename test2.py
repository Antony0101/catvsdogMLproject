import numpy as np
from keras.models import load_model
from PIL import Image

imagefile="23.jpg"
modelfile="model1.h5"
Image_Width=128
Image_Height=128
Image_Size=(Image_Width,Image_Height)

results={
    0:'cat',
    1:'dog'
}

model = load_model(modelfile)

im=Image.open(imagefile)
im=im.resize(Image_Size)
im=np.expand_dims(im,axis=0)
im=np.array(im)
im=im/255
pred=model.predict([im])
predict=np.argmax(pred, axis=1)
print(predict,results[predict[0]],pred)
