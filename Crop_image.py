# Crops the image according to the given co-ordinates by use of TensorFlow

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "test.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.constant(raw_image_data)
slice_op = tf.slice(image,[200,0,0],[100,-1,-1])

with tf.Session() as session:
	result = session.run(slice_op)
	print(result.shape)

plt.imshow(result)
plt.show()