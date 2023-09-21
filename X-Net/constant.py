import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('input_image_height', 256,
                            """Input image height.""")
tf.app.flags.DEFINE_integer('input_image_width', 256,
                            """Input image width.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")