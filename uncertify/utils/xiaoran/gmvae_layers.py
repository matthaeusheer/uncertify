import tensorflow as tf

def weight_variable(shape, std = 0.001):#, mode = 'nature'):
#    if mode == 'nature':
    initial = tf.truncated_normal(shape, stddev=std,dtype=tf.float32)
    return tf.Variable(initial, dtype = tf.float32)

def bias_variable(shape, bias = 0.1):
    initial = tf.constant(bias, shape=shape,dtype=tf.float32)
    return tf.Variable(initial, dtype = tf.float32)

def variable_summaries(var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      #tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # tf.summary.scalar('stddev', stddev)
      # tf.summary.scalar('max', tf.reduce_max(var))
      # tf.summary.scalar('min', tf.reduce_min(var))
      # tf.summary.histogram('histogram', var)

def conv(x, W , all_stride = 1, padding = 'SAME'):
    return tf.nn.conv2d(x, W, strides = [1, all_stride, all_stride, 1], padding = padding)

def upconv(x, W , outputshape , all_stride = 1):
    batchsize = tf.shape(x)[0]
    outputshape1 = tf.stack([batchsize, outputshape[1], outputshape[2], outputshape[3]])
    return tf.nn.conv2d_transpose(x, W, strides = [1, all_stride, all_stride, 1], output_shape = outputshape1, padding = "VALID")


def conv_layer(input, kernelx, kernely, output_channel, layer_name, act=tf.nn.relu, stride=1, bias=0.01,
               padding='SAME'):
    input_channel = input.get_shape().as_list()[-1]
    std_na = 0.01
    #    std_he=np.sqrt(2 / np.prod(input.get_shape().as_list()[1:]))
    #    std_xa=np.sqrt(2 / (np.prod(input.get_shape().as_list()[1:])+(np.prod(input.get_shape().as_list()[1:-1])*output_channel)))
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('kernel'):
            kernel = weight_variable([kernelx, kernely, input_channel, output_channel], std=std_na)
            #variable_summaries(kernel)
        with tf.name_scope('convbiases'):
            convbiases = bias_variable([output_channel], bias)
            #variable_summaries(convbiases)
        with tf.name_scope('conv2d_x_W'):
            preactivate = conv(input, kernel, all_stride=stride, padding=padding) + convbiases
            #            tf.summary.histogram('pre_activations_afterconv', preactivate)
            activations = act(preactivate, name='activation_afterconv')
            #tf.summary.histogram('conv_output', activations)
    return activations


def upconv_layer(input, kernelx, kernely, outputshape, layer_name, act=tf.nn.relu, stride = 1, bias = 0.01):
    input_channel = input.get_shape().as_list()[-1]
    std_na = 0.01
    with tf.name_scope(layer_name):
     # This Variable will hold the state of the weights for the layer
        with tf.name_scope('upkernel'):
            kernel = weight_variable([kernelx, kernely, outputshape[-1], input_channel], std = std_na)
            #variable_summaries(kernel)
        with tf.name_scope('upconvbiases'):
            convbiases = bias_variable([outputshape[-1]], bias)
            #variable_summaries(convbiases)
        with tf.name_scope('upconv2d_x_W'):
            preactivate = upconv(input, kernel, outputshape, all_stride = stride) + convbiases
#            tf.summary.histogram('pre_activations_afterconv', preactivate)
            activations = act(preactivate, name='activation_afterconv')
            #tf.summary.histogram('upconv_output', activations)
    return activations