import tensorflow as tf
from tensorflow.keras import layers
import string

img_height, img_width = 50, 200
characters = sorted(list(set(string.ascii_letters + string.digits)))
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def encode_single_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = 1.0 - img
    def remove_lines(x, ksize=2):
        eroded = -tf.nn.max_pool(-x[None, ...], ksize=[1,ksize,ksize,1], strides=[1,1,1,1], padding="SAME")
        opened = tf.nn.max_pool(eroded, ksize=[1,ksize,ksize,1], strides=[1,1,1,1], padding="SAME")
        return opened[0]
    img = remove_lines(img)
    img = 1 - img
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1,0,2])
    return img
