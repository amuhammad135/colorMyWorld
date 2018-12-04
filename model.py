import tensorflow as tf
import nn
import numpy as np
import conf


def low_level_feature_network(x):
    conv1 = nn.Convolution_Layer(shape=[3, 3, 1, 64], std=.1, v=.1)
    out = conv1.feed_forward(x=x, stride=[1, 2, 2, 1])

    conv2 = nn.Convolution_Layer(shape=[3, 3, 64, 128], std=.1, v=.1)
    out = conv2.feed_forward(x=out, stride=[1, 1, 1, 1])

    conv3 = nn.Convolution_Layer(shape=[3, 3, 128, 128], std=.1, v=.1)
    out = conv3.feed_forward(x=out, stride=[1, 2, 2, 1])

    conv4 = nn.Convolution_Layer(shape=[3, 3, 128, 256], std=.1, v=.1)
    out = conv4.feed_forward(x=out, stride=[1, 1, 1, 1])

    conv5 = nn.Convolution_Layer(shape=[3, 3, 256, 256], std=.1, v=.1)
    out = conv5.feed_forward(x=out, stride=[1, 2, 2, 1])

    conv6 = nn.Convolution_Layer(shape=[3, 3, 256, 512], std=.1, v=.1)
    out = conv6.feed_forward(x=out, stride=[1, 1, 1, 1])

    return out


def mid_level_feature_network(x):
    conv1 = nn.Convolution_Layer(shape=[3, 3, 512, 512], std=.1, v=.1)
    out = conv1.feed_forward(x=x, stride=[1, 1, 1, 1])

    conv2 = nn.Convolution_Layer(shape=[3, 3, 512, 256], std=.1, v=.1)
    out = conv2.feed_forward(x=out, stride=[1, 1, 1, 1])

    return out

def global_level_feature_network(x):
    # Convolution Layer
    conv1 = nn.Convolution_Layer(shape=[3, 3, 512, 512], std=.1, v=.1)
    out = conv1.feed_forward(x=x, stride=[1, 2, 2, 1])

    conv2 = nn.Convolution_Layer(shape=[3, 3, 512, 512], std=.1, v=.1)
    out = conv2.feed_forward(x=out, stride=[1, 1, 1, 1])

    conv3 = nn.Convolution_Layer(shape=[3, 3, 512, 512], std=.1, v=.1)
    out = conv3.feed_forward(x=out, stride=[1, 2, 2, 1])

    conv4 = nn.Convolution_Layer(shape=[3, 3, 512, 512], std=.1, v=.1)
    out = conv4.feed_forward(x=out, stride=[1, 1, 1, 1])

    # Fully Connected Layer


    return out

class Model():
    def __init__(self):
        self.input = tf.placeholder(shape=[conf.BATCH_SIZE, conf.IMAGE_SIZE, conf.IMAGE_SIZE, 1], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[conf.BATCH_SIZE, conf.IMAGE_SIZE, conf.IMAGE_SIZE, 2], dtype=tf.float32)
        self.loss = None
        self.output = None



    def construct(self):
        out_low = low_level_feature_network(self.input)
        out_mid = mid_level_feature_network(out_low)
        out_global = global_level_feature_network(out_low)
