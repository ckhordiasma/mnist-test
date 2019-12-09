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

images = np.empty((len(filenames),28,28))
for i,filename in enumerate(filenames):
    pilImage = Image.open(filename).convert('L')
    pilImage = PIL.ImageOps.invert(pilImage)

    image = np.array(pilImage) / 255.0
    #print(image)

    images[i,:] = image




# need to wrap it in an array 
result = modelLoaded.predict(images[...,tf.newaxis])
print(result)
ans = tf.argmax(result,1)

print(ans)



def plot_image(image, prob):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel("{:2.5f}%".format(np.max(prob)*100))

def plot_probs(prob):
    plt.grid(False)
    plt.xticks(range(len(prob)))
    plt.yticks([])
    thisplot = plt.bar(range(len(prob)), prob, color="#777777")
    plt.ylim([0,1])
    prediction = np.argmax(prob)
    thisplot[prediction].set_color('red')

for i,label in enumerate(list(map(lambda x:classes[x],ans))):
    plt.subplot(5,4,2*i+1)
    plot_image(images[i,...],result[i])
    plt.subplot(5,4,2*i+2)
    plot_probs(result[i])
    

plt.show()

#width, height = image.size

#print(image.size)