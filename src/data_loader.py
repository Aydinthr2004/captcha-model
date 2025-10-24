import os
import pandas as pd
import tensorflow as tf

def load_data_paths(csv_path, folder_name):
    if not os.path.exists(csv_path):
        return [], []
    df = pd.read_csv(csv_path)
    image_paths = []
    labels = []
    actual_folder = folder_name
    if os.path.exists(os.path.join(folder_name, folder_name)):
        actual_folder = os.path.join(folder_name, folder_name)
    for _, row in df.iterrows():
        img_path = os.path.join(actual_folder, row['filename'])
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(str(row['text'])[:5])
    return image_paths, labels

def split_data(images, labels, train_size=0.9, shuffle=True):
    size = tf.shape(images)[0]
    indices = tf.range(size)
    if shuffle:
        indices = tf.random.shuffle(indices)
    train_samples = tf.cast(tf.floor(tf.cast(size, tf.float32) * train_size), tf.int32)
    train_idx, valid_idx = indices[:train_samples], indices[train_samples:]
    x_train, y_train = tf.gather(images, train_idx), tf.gather(labels, train_idx)
    x_valid, y_valid = tf.gather(images, valid_idx), tf.gather(labels, valid_idx)
    return x_train, x_valid, y_train, y_valid
