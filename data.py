import os
import sys

import argparse

import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cv2 import imread
from tqdm import tqdm

def loadDataList(filename):
    dataList = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            lineSplits = line.split()
            dataList.append(lineSplits)
    return dataList


def store_to_tfrecord(filename, datadir, tfrecord_name, img_data_path):
    """
    存储数据到TFRecord中

    @param filename: 数据的文件名文件
    @param datadir: 数据的实际存储文件
    @param tfrecord_name: 要存储到的文件的文件名
    """

    dataListPath = os.path.join(datadir, filename)
    datalist = tqdm(loadDataList(dataListPath))

    tfrecord_store_path = os.path.join(datadir, tfrecord_name)
    img_data_path = os.path.join(datadir, img_data_path)
    writer = tf.python_io.TFRecordWriter(tfrecord_store_path)
    for data in datalist:
        datalist.set_description("Processing %s" % data)
        img_path = os.path.join(img_data_path, data[0])
        img_raw = imread(img_path).tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    "rating": tf.train.Feature(float_list=tf.train.FloatList(value=[np.float64(data[1])]))
                }
            )
        )
        writer.write(record=example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument('--data_dir', type=str, default="data/", help="数据的实际存储路径")
    parser.add_argument("--img_data_path", type=str, default='Images/', help="图像的存储目录")
    parser.add_argument('--data_list_path', type=str, required=True, help="数据名存储文件")
    parser.add_argument('--store_path', type=str, default='data.tfrecord', help="存储到的TFRecord文件名")

    FLAGS, _ = parser.parse_known_args()
    store_to_tfrecord(FLAGS.data_list_path, FLAGS.data_dir, FLAGS.store_path, FLAGS.img_data_path)
