import tensorflow as tf
sess = tf.Session()

checkpoint_path = './models/naturally_trained/checkpoint-70000'
new_saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
_ = new_saver.restore(sess, checkpoint_path)

all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = None)
for v in all_vars:
  v_ = sess.run(v)
  print('Name: ' + v.name)
  print('Shape: ' + str(v_.shape))
