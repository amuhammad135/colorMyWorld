import tensorflow as tf

class Layer():
    def __init__(self, shape, std, v):
        self.w = tf.Variable(tf.truncated_normal(shape=shape, stddev=std))
        self.b = tf.Variable(tf.constant(value=v, shape=[shape[-1]]))

class Convolution_Layer(Layer):
    def __init__(self, shape, std, v):
        super(Convolution_Layer, self).__init__(shape,std, v)

    def feed_forward(self, x, stride):
        wx = tf.nn.conv2d(x, self.w, stride, padding="SAME")
        return tf.nn.tanh(tf.nn.bias_add(wx, self.b))

class FC_Layer(Layer):
    def __init__(self, shape, std, v):
        super(FC_Layer, self).__init__(shape, std, v)

    # relu(W*x + b)
    def feed_forward(self, x):
        wx = tf.matmul(x, self.w)
        return tf.nn.relu(tf.nn.bias_add(wx, self.b)

class Fusion_Layer(Layer):
    def __init__(self, shape, std, v):
        super(Fusion_Layer, self).__init__(shape,std,v)

    def feed_forward(self, mid_level, global_level, stride):
        return ""

class Out_Layer(Layer):
    def __init__(self, shape, std, v):
        super(Out_Layer, self).__init__(shape,std,v)

    def feed_forward(self, x, stride):
        wx = tf.nn.conv2d(x, self.w, stride, padding='SAME')
        return tf.nn.sigmoid(tf.nn.bias_add(wx, self.b))