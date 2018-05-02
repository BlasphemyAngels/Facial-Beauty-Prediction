import os
import sys

import tensorflow as tf

def read_record(filepath):
    """
    读取tfrecord的内容

    @param filepath: tfrecord的文件路径
    @return: img 图像 和 rating 打分
    """

    reader = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer([filepath])

    _, serialized_record = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_record,
        features={
            "img": tf.FixedLenFeature([], tf.string),
            "rating": tf.FixedLenFeature([], tf.float32)
        }
    )

    img = tf.decode_raw(features["img"], tf.uint8)
    img = tf.reshape(img, [350, 350, 3], name="img_reshape")

    rating = tf.cast(features["rating"], tf.float32)

    return img, rating

def get_batch(filepath, batch_size):
    """
    从给定文件中读取一个batch的数据

    @param filepath: 文件路径
    @param batch_size: 批数据的大小
    """
    img, rating = read_record(filepath)
    # min_after_dequeue代表当队列中的数据数目小于min_after_dequeue的值时，不再进行shuffle
    min_after_dequeue = 10000
    # capacity队列的容量，它的值必须大于min_after_dequeue的值
    capacity = min_after_dequeue + 3 * batch_size

    img_batch, rating_batch = tf.train.shuffle_batch([img, rating], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return img_batch, rating_batch
