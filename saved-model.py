import numpy as np
from PIL import Image
import PIL.ImageOps
import tensorflow as tf
import glob
import matplotlib.pyplot as plt


modelLoaded = tf.keras.models.load_model('mnistconv.h5')

#This line is useful for printing/visualizing numpy arrays of images nicely.
np.set_printoptions(linewidth=500,precision=2)

classes = ['zero','one','two','three','four','five','six','seven','eight','nine']


filenames = glob.glob('./handwritten/*.png')

# honestly not sure if I need figsize parameter here, since subplot is called later.
plt.figure(figsize=(4,4))

images = np.empty((len(filenames),28,28,1))
for i,filename in enumerate(filenames):
    # The 'L" converts the image into a single-channel grayscale image, I think
    pilImage = Image.open(filename).convert('L')
    # I need to invert the image since in the PNG realm, 255 is white in 0 is black.
    pilImage = PIL.ImageOps.invert(pilImage)

    # converting to a numpy array, because that is what tensorflow likes.
    image = np.array(pilImage)
    
    #normalizing the image to the scale of 0 to 1.
    image = image / 255.0

    # print(image)
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