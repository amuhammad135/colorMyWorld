import tensorflow as tf
import conf

class Layer:
    def __init__(self, shape, std, v):
        self.w = tf.Variable(tf.truncated_normal(shape=shape, stddev=std))
        self.b = tf.Variable(tf.constant(value=v, shape=[shape[-1]]))


class ConvolutionLayer(Layer):
    def __init__(self, shape, std, v):
        super(ConvolutionLayer, self).__init__(shape, std, v)

    def feed_forward(self, x, stride):
        wx = tf.nn.conv2d(x, self.w, stride, padding="SAME")
        return tf.nn.tanh(tf.nn.bias_add(wx, self.b))


class FCLayer(Layer):
    def __init__(self, shape, std, v):
        super(FCLayer, self).__init__(shape, std, v)

    # relu(W*x + b)
    def feed_forward(self, x):
        wx = tf.matmul(x, self.w)
        return tf.nn.relu(tf.nn.bias_add(wx, self.b))


class FusionLayer(ConvolutionLayer):
    def __init__(self, shape, std, v):
        super(FusionLayer, self).__init__(shape, std, v)

    def feed_forward(self, mid_features, global_features, stride):
        mid_features_shape = mid_features.get_shape().as_list()
        mid_features_reshaped = tf.reshape(mid_features,
                                           [conf.BATCH_SIZE, mid_features_shape[1] * mid_features_shape[2], 256])
        fusion_level = []
        for j in range(mid_features_reshaped.get_shape()[0]):
            for i in range(mid_features_reshaped.get_shape()[1]):
                see_mid = mid_features_reshaped[j, i, :]
                see_mid_shape = see_mid.get_shape().as_list()
                see_mid = tf.reshape(see_mid, [1, see_mid_shape[0]])
                global_features_shape = global_features[j, :].get_shape().as_list()
                see_global = tf.reshape(global_features[j, :], [1, global_features_shape[0]])
                fusion = tf.concat([see_mid, see_global], 1)
                fusion_level.append(fusion)
        fusion_level = tf.stack(fusion_level, 1)
        fusion_level = tf.reshape(fusion_level, [conf.BATCH_SIZE, 28, 28, 512])
        return super(FusionLayer, self).feed_forward(fusion_level, stride)


class OutLayer(Layer):
    def __init__(self, shape, std, v):
        super(OutLayer, self).__init__(shape,std,v)

    def feed_forward(self, x, stride):
        wx = tf.nn.conv2d(x, self.w, stride, padding='SAME')
        return tf.nn.sigmoid(tf.nn.bias_add(wx, self.b))
