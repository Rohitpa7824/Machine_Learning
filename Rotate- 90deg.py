# Rotates an image 90 degree to the right by use of TensorFlow

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "test.jpg"
raw_image_data = mpimg.imread(filename)

x = tf.Variable(raw_image_data,name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
	transpose = tf.transpose(x,perm=[1,0,2])
	session.run(model)
	result = session.run(transpose)
	# print(result)

plt.imshow(result)
plt.show()