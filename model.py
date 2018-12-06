import tensorflow as tf
import nn
import numpy as np
import conf


def low_level_feature_network(x):
    conv1 = nn.ConvolutionLayer(shape=[3, 3, 1, 64], std=.1, v=.1)
    out = conv1.feed_forward(x=x, stride=[1, 2, 2, 1])

    conv2 = nn.ConvolutionLayer(shape=[3, 3, 64, 128], std=.1, v=.1)
    out = conv2.feed_forward(x=out, stride=[1, 1, 1, 1])

    conv3 = nn.ConvolutionLayer(shape=[3, 3, 128, 128], std=.1, v=.1)
    out = conv3.feed_forward(x=out, stride=[1, 2, 2, 1])

    conv4 = nn.ConvolutionLayer(shape=[3, 3, 128, 256], std=.1, v=.1)
    out = conv4.feed_forward(x=out, stride=[1, 1, 1, 1])

    conv5 = nn.ConvolutionLayer(shape=[3, 3, 256, 256], std=.1, v=.1)
    out = conv5.feed_forward(x=out, stride=[1, 2, 2, 1])

    conv6 = nn.ConvolutionLayer(shape=[3, 3, 256, 512], std=.1, v=.1)
    out = conv6.feed_forward(x=out, stride=[1, 1, 1, 1])

    return out


def mid_level_feature_network(x):
    conv1 = nn.ConvolutionLayer(shape=[3, 3, 512, 512], std=.1, v=.1)
    out = conv1.feed_forward(x=x, stride=[1, 1, 1, 1])

    conv2 = nn.ConvolutionLayer(shape=[3, 3, 512, 256], std=.1, v=.1)
    out = conv2.feed_forward(x=out, stride=[1, 1, 1, 1])

    return out


def global_level_feature_network(x):
    # Convolution Layer
    conv1 = nn.ConvolutionLayer(shape=[3, 3, 512, 512], std=.1, v=.1)
    out = conv1.feed_forward(x=x, stride=[1, 2, 2, 1])

    conv2 = nn.ConvolutionLayer(shape=[3, 3, 512, 512], std=.1, v=.1)
    out = conv2.feed_forward(x=out, stride=[1, 1, 1, 1])

    conv3 = nn.ConvolutionLayer(shape=[3, 3, 512, 512], std=.1, v=.1)
    out = conv3.feed_forward(x=out, stride=[1, 2, 2, 1])

    conv4 = nn.ConvolutionLayer(shape=[3, 3, 512, 512], std=.1, v=.1)
    out = conv4.feed_forward(x=out, stride=[1, 1, 1, 1])

    # Fully Connected Layer
    flatten = tf.reshape(out, [conf.BATCH_SIZE, -1])

    fc1 = nn.FCLayer(shape=[flatten.get_shape()[1].value, 1024], std=.04, v=.1)
    out = fc1.feed_forward(x=flatten)

    fc2 = nn.FCLayer(shape=[1024, 512], std=.04, v=.1)
    out = fc2.feed_forward(x=out)

    fc3 = nn.FCLayer(shape=[512, 256], std=.04, v=.1)
    out = fc3.feed_forward(x=out)

    return out

def colorization_network(out_mid, out_global):
    # fusion -> conv
    fusion_layer = nn.FusionLayer(shape=[1, 1 , 512, 256], std=.1, v=.1)
    out = fusion_layer.feed_forward(out_mid,out_global,stride=[1, 1, 1, 1])

    conv1 = nn.ConvolutionLayer(shape=[3, 3, 256, 128], std=.1, v=.1)
    out = conv1.feed_forward(x=out, stride=[1, 1, 1, 1])

    # upsample -> conv -> conv
    out = tf.image.resize_images(out, [56,56], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv2 = nn.ConvolutionLayer(shape=[3, 3, 128, 64], std=.1, v=.1)
    out = conv2.feed_forward(x=out, stride=[1, 1, 1, 1])

    conv3 = nn.ConvolutionLayer(shape=[3, 3, 64, 64], std=.1, v=.1)
    out = conv3.feed_forward(x=out, stride=[1, 1, 1, 1])


    # upsample -> conv -> output
    out = tf.image.resize_images(out, [112,112], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv4 = nn.ConvolutionLayer(shape=[3, 3, 64, 32], std=.1, v=.1)
    out = conv4.feed_forward(x=out, stride=[1, 1, 1, 1])

    out_layer = nn.OutLayer(shape=[3, 3, 32, 2], std=.1, v=.1)
    out = out_layer.feed_forward(x=out, stride=[1, 1, 1, 1])

    output = tf.image.resize_images(out, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return output



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
        self.output = colorization_network(out_mid,out_global)
        self.loss = tf.reduce_mean(tf.squared_difference(self.labels, self.output))
