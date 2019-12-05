import numpy as np
from PIL import Image
import PIL.ImageOps
import tensorflow as tf
import glob

modelLoaded = tf.keras.models.load_model('mnistconv.h5')
np.set_printoptions(linewidth=500,precision=2)


filenames = glob.glob('./handwritten/*.png')
images = np.empty((len(filenames),28,28,1))
for i,filename in enumerate(filenames):
    pilImage = Image.open(filename).convert('L')
    pilImage = PIL.ImageOps.invert(pilImage)

    image = np.array(pilImage)
    #print(image)

    image = image / 255.0
    images[i,:,:] = image[...,tf.newaxis]



# need to wrap it in an array 
result = modelLoaded.predict(images)
print(result)
ans = tf.argmax(result,1)

print(ans)


#width, height = image.size

#print(image.size)