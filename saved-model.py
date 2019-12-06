import numpy as np
from PIL import Image
import PIL.ImageOps
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
modelLoaded = tf.keras.models.load_model('mnistconv.h5')
np.set_printoptions(linewidth=500,precision=2)

classes = ['zero','one','two','three','four','five','six','seven','eight','nine']

filenames = glob.glob('./handwritten/*.png')

plt.figure(figsize=(4,4))

images = np.empty((len(filenames),28,28,1))
for i,filename in enumerate(filenames):
    pilImage = Image.open(filename).convert('L')
    pilImage = PIL.ImageOps.invert(pilImage)

    image = np.array(pilImage)
    #print(image)
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    images[i,:,:] = image[...,tf.newaxis]




# need to wrap it in an array 
result = modelLoaded.predict(images)
print(result)
ans = tf.argmax(result,1)

print(ans)
for i,label in enumerate(list(map(lambda x:classes[x],ans))):
    plt.subplot(4,4,i+1)
    plt.xlabel(label)

plt.show()

#width, height = image.size

#print(image.size)