import os
import sys

import re

import tensorflow as tf


def read_record(filepattern):
    """
    读取tfrecord的内容

    @param filepath: tfrecord的文件路径
    @return: img 图像 和 rating 打分
    """

    reader = tf.TFRecordReader()

    filenames = tf.train.match_filenames_once(filepattern)

    # 测试tf.train.match_filenames_once是否功能正常的代码
    #  init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    #  print(tfrecord_path)
    #  with tf.Session() as sess:
        #  sess.run(init)
        #  print(sess.run(filenames))
    #  sys.exit(0)

    filename_queue = tf.train.string_input_producer(filenames)

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

def get_batch(filepattern, batch_size):
    """
    从给定文件中读取一个batch的数据

    @param filepath: 文件路径
    @param batch_size: 批数据的大小
    @return: 一个批次的数据
    """
    img, rating = read_record(filepattern)
    # min_after_dequeue代表当队列中的数据数目小于min_after_dequeue的值时，不再进行shuffle
    min_after_dequeue = 10000
    # capacity队列的容量，它的值必须大于min_after_dequeue的值
    capacity = min_after_dequeue + 3 * batch_size

    img_batch, rating_batch = tf.train.shuffle_batch([img, rating], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return img_batch, rating_batch

def get_filenames_by_regexp(filepattern):
    """
    根据给定的模式查找到所有的文件名

    @param filepattern: 文件名模式
    @return: 一个列表，里面包含了满足此模式的所有文件名
    """

    filepattern = filepattern.replace("*", ".*")

    dir_path = filepattern.split("/")[:-1]
    dir_path = "/".join(dir_path)
    
    file_list = os.listdir(dir_path)

    file_list = map(lambda filename: os.path.join(dir_path, filename), file_list)

    def filter_pattern(filename):
        return re.match(filepattern, filename)
    
    file_list_filter = [filename for filename in file_list if filter_pattern(filename)]
    return file_list_filter



def get_batch_dataset_iterator(filepattern, batch_size):
    """
    通过tf.data.DataSet的方式读取一个batch

    @param filepath: 文件路径
    @param batch_size: 一个批次的大小
    @return: 一个批次的数据
    """


    filenames = get_filenames_by_regexp(filepattern)

    #  filenames = shuffle(filenames)

    dataset = tf.data.TFRecordDataset(filenames)

    def parse_func(example_proto):
        """
        对TFRecord中的每一个example进行解析
        """
        parse_example = tf.parse_single_example(
            example_proto,
            features = {
                "img": tf.FixedLenFeature([], tf.string),
                "rating": tf.FixedLenFeature([], tf.float32)
            }
        )
        parse_example["img"] = tf.decode_raw(parse_example["img"], tf.uint8)
        parse_example["rating"] = tf.cast(parse_example["rating"], tf.float32)

        parse_example["img"] = tf.reshape(parse_example["img"], [350, 350, 3])

        return parse_example
    
    parse_dataset = dataset.map(parse_func)
    shuffle_dataset = parse_dataset.shuffle(buffer_size=10000)

    batch_dataset = shuffle_dataset.batch(batch_size=batch_size)
    
    iterator = batch_dataset.make_one_shot_iterator()

    return iterator.get_next()


# 使用方式
'''
# epoch_dataset = batch_dataset.repeat(10) 也可以使用epoch_dataset
iterator = batch_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

i = 1
while True:
    # 不断的获得下一个样本
    try:
        scalar = sess.run(next_element['scalar'])
    except tf.errors.OutOfRangeError:
        print("End of dataset")
        break
    else:
        print('example %s | scalar: value: %s' %(i,scalar))
    i+=1
'''
