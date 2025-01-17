import os
import tensorflow as tf
import nn
import conf
import numpy as np
import cv2


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
    fusion_layer = nn.FusionLayer(shape=[1, 1, 512, 256], std=.1, v=.1)
    out = fusion_layer.feed_forward(out_mid, out_global, stride=[1, 1, 1, 1])

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


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY, filelist):
    for i in range(conf.BATCH_SIZE):
        result = np.concatenate((batchX[i], predictedY[i]), axis=2)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join(conf.OUT_DIR, filelist[i][:-4] + "reconstructed.jpg")
        print(save_path)
        cv2.imwrite(save_path, result)


class Model():
    def __init__(self):
        self.inputs = tf.placeholder(shape=[conf.BATCH_SIZE, conf.IMAGE_SIZE, conf.IMAGE_SIZE, 1], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[conf.BATCH_SIZE, conf.IMAGE_SIZE, conf.IMAGE_SIZE, 2], dtype=tf.float32)
        self.loss = None
        self.output = None

    def construct(self):
        out_low = low_level_feature_network(self.inputs)
        out_mid = mid_level_feature_network(out_low)
        out_global = global_level_feature_network(out_low)
        self.output = colorization_network(out_mid,out_global)
        self.loss = tf.reduce_mean(tf.squared_difference(self.labels, self.output))

    def train(self, data):
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        saver = tf.train.Saver()
        with tf.Session() as session:
            # Initialize variables
            session.run(tf.global_variables_initializer())

            if conf.USE_PRETRAINED:
                saver.restore(session, os.path.join(conf.MODEL_DIR, conf.PRETRAINED))
                print('Pretrained weights loaded')

            for epoch in range(conf.NUM_EPOCHS):
                avg_cost = 0
                for batch in range(int(data.size/conf.BATCH_SIZE)):
                    batch_x, batch_y, _ = data.get_batch()
                    feed_dict = {self.inputs: batch_x, self.labels: batch_y}
                    _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
                    print("batch:", batch, " loss: ", loss_val)
                    avg_cost += loss_val / int(data.size/conf.BATCH_SIZE)
                print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost))

            save_path = saver.save(session, os.path.join(conf.MODEL_DIR, "model" + str(conf.BATCH_SIZE) + "_" +
                                                         str(conf.NUM_EPOCHS) + ".ckpt"))
            print("save_path: ", save_path)
            print("Model saved in path: %s" % save_path)

    def test(self, data, output_path):
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, os.path.join(conf.MODEL_DIR, "model" + str(conf.BATCH_SIZE) + "_" +
                                                str(conf.NUM_EPOCHS) + ".ckpt"))
            avg_cost = 0
            for _ in range(data.size):
                batch_x, batch_y, images = data.get_batch()
                feed_dict = {self.inputs: batch_x, self.labels: batch_y}
                pred_y, loss = session.run([self.output, self.loss], feed_dict=feed_dict)
                reconstruct(deprocess(batch_x), deprocess(pred_y), images)
                avg_cost += loss/data.size
            print("cost =", "{:.3f}".format(avg_cost))







