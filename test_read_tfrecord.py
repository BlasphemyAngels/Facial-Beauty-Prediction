import unittest

import tensorflow as tf

import numpy as np
import utils
import cv2

class TestReadTFRecord(unittest.TestCase):
    """
    测试读取TFRecord文件的函数功能是否正常
    """

    def test_read_one(self):
        """
        测试读取一条数据
        """
        img, rating = utils.read_record("data/train/*.tfrecord")
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print(sess.run(rating))
            img = sess.run(img)
            img = np.reshape(img, [350, 350, 3])
            cv2.imshow("a", img)
            cv2.waitKey()

    def test_read_batch(self):
        """
        测试读取一个批次的数据
        """

        img_batch, rating_batch = utils.get_batch("data/train/*.tfrecord", 32)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            img_b, rating_b = sess.run([img_batch, rating_batch])

            print(rating_b)
            print(len(img_b))

            img_x = img_b[2]

            cv2.imshow("haha", img_x)
            cv2.waitKey()

    def test_batch_dataset(self):
        """
        测试使用dataset得到batch功能是否正常
        """

        batch_dataset_next = utils.get_batch_dataset_iterator("data/train/*.tfrecord", batch_size=32)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            img_batch = sess.run(batch_dataset_next["img"])
            img = img_batch[2]
            cv2.imshow("hehe", img)
            cv2.waitKey()

        

if __name__ == '__main__':

    # unittest运行的几种方式
    # unittest.main()

    suite = unittest.TestSuite()
    #  suite.addTest(TestReadTFRecord("test_read_one"))
    #  suite.addTest(TestReadTFRecord("test_read_batch"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
