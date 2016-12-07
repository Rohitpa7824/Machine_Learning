# Rotates an image 180 degree to the right by use of TensorFlow

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "test.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8",[None,None,3])
reverse = tf.reverse(image,[True,False,False])

with tf.Session() as session:
	result = session.run(reverse,feed_dict={image:raw_image_data})
	print(result.shape)

plt.imshow(result)
plt.show()