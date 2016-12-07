import tensorflow as tf

x = tf.constant(35,name='x')
y=x+5

print(y)

with tf.Session() as session:
	print(session.run(y))