import numpy as np
import tensorflow as tf
# import tensorlayer as tl
# from tensorlayer.layers import *
from pdb import set_trace as bp


# from tensorflow.image import ResizeMethod


def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def resblock_down(inputs, filters_in, filters_out, scope_name, reuse, phase_train, act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        # tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        # input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d(inputs, filters_in, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="rbd_conv1", reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                              gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rbd_bn1', reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d(conv1, filters_out, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="rbd_conv2", reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn2', reuse=reuse)
        if act:
            conv2 = tf.nn.relu(conv2)
        conv3 = tf.layers.conv2d(inputs, filters_out, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="conv3", reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn3', reuse=reuse)
        if act:
            conv3 = tf.nn.relu(conv3)
        conv_out = tf.add(conv2, conv3)
    return conv_out


def resblock_up(inputs, filters_in, filters_out, scope_name, reuse, phase_train,
                act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        # tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        # input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d_transpose(inputs, filters_in, (3, 3), (2, 2), padding='same',
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_deconv1", reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1, center=True,
                                              scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbu_bn1', reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d_transpose(conv1, filters_out, (3, 3), (1, 1), padding='same',
                                           kernel_initializer=w_init, bias_initializer=b_init, trainable=True,
                                           name="rbu_deconv2",
                                           reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                              gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train,
                                              name='rbu_bn2', reuse=reuse)
        if act:
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
        conv3 = tf.layers.conv2d_transpose(inputs, filters_out, (3, 3), (2, 2), padding='same',
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_conv3",
                                           reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbu_bn3',
                                              reuse=reuse)
        if act:
            conv3 = tf.nn.leaky_relu(conv3, 0.2)
        conv_out = tf.add(conv2, conv3)
    return conv_out


def resblock_valid_enc(inputs, filters_in, filters_out,
                       scope_name, reuse, phase_train, act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        # tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        # input_layer = InputLayer(inputs, name='e_inputs')
        conv1 = tf.layers.conv2d(inputs, filters_in, (3, 3), padding='valid', kernel_initializer=w_init,
                                 bias_initializer=b_init, name="rb_conv1")
        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                              gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rb_bn1')
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d(conv1, filters_out, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, name="rb_conv2")
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                              gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rb_bn2')
        # conv2 = tf.nn.leaky_relu(conv2, 0.2)

        conv3 = tf.layers.conv2d(inputs, filters_out, (3, 3), (1, 1), padding='valid', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="conv3", reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn3', reuse=reuse)

        h = tf.shape(conv2)[1]
        w = tf.shape(conv2)[2]
        # inputs = tf.image.resize_images(inputs, tf.cast([h,w], tf.int32),
        #                                method=tf.image.ResizeMethod.BILINEAR)
        conv_out = tf.add(conv2, conv3)
        if act:
            conv_out = lrelu(conv_out, 0.2)
    return conv_out


def resblock_valid_dec(inputs, filters_in, filters_out,
                       scope_name, reuse, phase_train,
                       act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        # tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        # input_layer = InputLayer(inputs, name='e_inputs')
        conv1 = tf.layers.conv2d_transpose(inputs, filters_in, (3, 3), padding='valid', kernel_initializer=w_init,
                                           bias_initializer=b_init, name="rb_conv1")
        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                              gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rb_bn1')
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d_transpose(conv1, filters_out, (3, 3), padding='same', kernel_initializer=w_init,
                                           bias_initializer=b_init, name="rb_conv2")
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                              gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rb_bn2')
        # conv2 = tf.nn.leaky_relu(conv2, 0.2)
        conv3 = tf.layers.conv2d_transpose(inputs, filters_out, (3, 3), padding='valid', kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="conv3", reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn3', reuse=reuse)
        conv_out = tf.add(conv2, conv3)
        if act:
            conv_out = lrelu(conv_out, 0.2)
    return conv_out


def resblock_down_bilinear(inputs, filters_in, filters_out, scope_name, reuse, phase_train, act=True):
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    with tf.variable_scope(scope_name, reuse=reuse):
        # tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        # input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d(inputs, filters_in, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="rbd_conv1", reuse=reuse)
        conv1 = tf.image.resize_images(conv1, tf.cast([h / 2, w / 2], tf.int32), method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=True
                                       )
        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                              gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rbd_bn1', reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d(conv1, filters_out, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="rbd_conv2", reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn2', reuse=reuse)
        # conv2_leaky = tf.nn.leaky_relu(conv2, 0.2)
        # conv3 = tf.layers.conv2d(inputs, filters_out, (3, 3), (1, 1), padding='same', kernel_initializer=w_init,
        #                          bias_initializer=b_init, trainable=True, name="conv3",reuse=reuse)
        # conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
        #                                       trainable=True, training=phase_train,
        #                                       gamma_initializer=gamma_init, name='rbd_bn3', reuse=reuse)
        # input_identity = tf.image.resize_images(conv3, tf.cast([h / 2, w / 2], tf.int32), method=tf.image.ResizeMethod.BILINEAR)
        # # conv3 = tf.nn.leaky_relu(conv3, 0.2)
        # conv_out = tf.add(conv2,input_identity)
        # conv_out = tf.nn.leaky_relu(conv_out, 0.2)

        if act:
            conv2 = tf.nn.relu(conv2)
        conv3 = tf.layers.conv2d(inputs, filters_out, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="conv3", reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn3', reuse=reuse)
        if act:
            conv3 = tf.nn.relu(conv3)
        conv_out = tf.add(conv2, conv3)

    return conv_out


def resblock_up_bilinear(inputs, filters_in, filters_out, scope_name, reuse, phase_train, act=True):
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    with tf.variable_scope(scope_name, reuse=reuse):
        # tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        # input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d(inputs, filters_in, (3, 3), (1, 1), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="rbu_deconv1", reuse=reuse)
        conv1 = tf.image.resize_images(conv1, tf.cast([h * 2, w * 2], tf.int32), method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=True
                                       )
        conv1 = tf.layers.batch_normalization(conv1, center=True,
                                              scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbu_bn1', reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d(conv1, filters_out, (3, 3), (1, 1), padding='same',
                                 kernel_initializer=w_init, bias_initializer=b_init, trainable=True, name="rbu_deconv2",
                                 reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                              gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train,
                                              name='rbu_bn2', reuse=reuse)
        # conv2_leaky = tf.nn.leaky_relu(conv2, 0.2)
        # conv3 = tf.layers.conv2d_transpose(inputs, filters_out, (3, 3), (1, 1), padding='same', kernel_initializer=w_init,
        #                                    bias_initializer=b_init, trainable=True, name="rbu_conv3",
        #                                     reuse=reuse)
        # conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
        #                                       trainable=True, training=phase_train,
        #                                       gamma_initializer=gamma_init, name='rbu_bn3',
        #                                       reuse=reuse)
        # input_identity = tf.image.resize_images(conv3, tf.cast([h * 2, w * 2], tf.int32), method=tf.image.ResizeMethod.BILINEAR)
        # #conv3 = tf.nn.leaky_relu(conv3, 0.2)
        # conv_out = tf.add(conv2, input_identity)
        # conv_out = tf.nn.leaky_relu(conv_out, 0.2)

        if act:
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
        conv3 = tf.layers.conv2d(inputs, filters_out, (3, 3), (1, 1), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="rbu_conv3",
                                 reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbu_bn3',
                                              reuse=reuse)
        conv3 = tf.image.resize_images(conv3, tf.cast([h * 2, w * 2], tf.int32), method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=True
                                       )
        if act:
            conv3 = tf.nn.leaky_relu(conv3, 0.2)
        conv_out = tf.add(conv2, conv3)

    return conv_out


class VariationalAutoencoder():
    def __init__(self, model_name=None, image_size=32):
        self.model_name = model_name
        self.image_size = image_size

    def encoder(self, x, reuse=False, is_train=True, n_layers=4, gf_dim=16, output_dim=2):
        """
        Encode part of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """
        image_size = self.image_size
        gf_dim = gf_dim  # Dimension of gen filters in first conv layer. [64]
        with tf.variable_scope(self.model_name + "_encoder", reuse=reuse):
            # x,y,z,_ = tf.shape(input_images)
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)
            gamma_init = tf.random_normal_initializer(0.5, 0.01)
            # inputs = InputLayer(x, name='e_inputs')
            conv1 = tf.layers.conv2d(x, gf_dim, (3, 3), padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init,
                                     trainable=True,
                                     name="e_conv1",
                                     reuse=reuse)
            conv1 = tf.layers.batch_normalization(conv1, center=True,
                                                  scale=True, trainable=True,
                                                  gamma_initializer=gamma_init,
                                                  training=is_train,
                                                  name='e_bn1',
                                                  reuse=reuse)
            conv1 = tf.nn.leaky_relu(conv1, 0.2)
            # self._conv1 = conv1
            # image_size * image_size
            res = resblock_down(conv1, gf_dim, gf_dim, "res", reuse=reuse, phase_train=is_train)
            # self._activation_value_res1=res1
            for n in range(n_layers):
                res = resblock_down(res, gf_dim * 2 ** n, gf_dim * 2 ** (n + 1), "res" + str(n), reuse=reuse,
                                    phase_train=is_train)
            # s2*s2
            # res2 = resblock_down(res1, gf_dim, gf_dim * 2, "res2", reuse, is_train)
            # self._activation_value_res2=res2
            # s4*s4
            # res3 = resblock_down(res2, gf_dim * 2, gf_dim * 4, "res3", reuse, is_train)

            # self._activation_value_res3=res3
            # s8*s8
            # res4 = resblock_down(res3, gf_dim * 4, gf_dim * 8, "res4", reuse, is_train)
            # self._activation_value_res4=res4
            # s16*s16
            # res5 = resblock_down(res4, gf_dim * 8, gf_dim * 16, "res5", reuse, is_train)
            # self._activation_value_res5=res5
            # s32*s32
            res2 = resblock_down(res, gf_dim * 2 ** (n + 1), gf_dim * 2 ** (n + 2), "res" + str(n + 1),
                                 reuse, is_train, act=False)
            res2_stddev = resblock_down(res, gf_dim * 2 ** (n + 1), gf_dim * 2 ** (n + 2), "res_stddev" + str(n + 1),
                                        reuse, is_train, act=False)
            # s64*s64
            # res7 = resblock_valid_enc(res6, gf_dim * 16, gf_dim * 32, "res7", reuse, is_train)

            # res7 = resblock_down(res6, gf_dim * 32, gf_dim*16,
            #                               "res7", reuse, is_train)

            # enc_mean = tf.layers.conv2d(res6, gf_dim*32, (3, 3), (1, 1),
            #                             padding='same', kernel_initializer=w_init,
            #                      bias_initializer=b_init, trainable=True,
            #                             name="enc_mean", reuse=reuse)
            #
            # enc_stddev = tf.layers.conv2d(res6, gf_dim*32, (3, 3), (1, 1),
            #                             padding='same', kernel_initializer=w_init,
            #                      bias_initializer=b_init, trainable=True,
            #                             name="enc_stddev", reuse=reuse)
            # 40 x 40

            # res6_res = ResBlockDown(res5, gf_dim * 16, gf_dim * 4, "res6_res", reuse, is_train)
            conv2 = tf.layers.conv2d(conv1, gf_dim, (3, 3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="e_conv2",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                                  trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='e_bn2',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, gf_dim * 2, (3, 3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="e_conv3",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                                  trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='e_bn3',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, gf_dim, (3, 3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="e_conv4",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                                  trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='e_bn4',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, output_dim, (3, 3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="e_conv5",
                                     reuse=reuse)
            # conv7_mean_flat = tf.contrib.layers.flatten(res_mean)
            # conv7_std_flat = tf.contrib.layers.flatten(res_stddev)
            # conv7_mean_res = tf.contrib.layers.flatten(res6_res)
        return res2, res2_stddev, conv2

    def decoder(self, x, name, reuse=False, is_train=True, n_layers=4, gf_dim=16, output_dim=2):
        """
        Decoder part of the autoencoder.
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """
        image_size = self.image_size
        # s2, s4, s8, s16, s32, s64 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16), \
        #                        int(image_size/32), int(image_size/64)
        gf_dim = gf_dim  # Dimension of gen filters in first conv layer. [64]
        with tf.variable_scope(self.model_name + "_decoder_" + name, reuse=reuse):
            # tl.layers.set_name_reuse(reuse)
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)
            gamma_init = tf.random_normal_initializer(0.5, 0.01)
            # 1*1
            # z_develop = tf.reshape(x, [-1, dim, dim, gf_dim*10])
            # s64

            resp1 = resblock_up(x, gf_dim * 32, gf_dim * 16, "gresp1", reuse, is_train)

            for n in range(n_layers):
                resp1 = resblock_up(resp1, gf_dim * 2 ** (n_layers - n),
                                    gf_dim * 2 ** (n_layers - n - 1), "gres" + str(n),
                                    reuse=reuse, phase_train=is_train)

            # s32*s32
            # res0 = resblock_up(resp1, gf_dim*16, gf_dim * 8, "gres0", reuse, is_train)

            # res1 = resblock_up(res0, gf_dim * 8, gf_dim * 4, "gres1", reuse, is_train)

            # res2 = resblock_up(res1, gf_dim * 4, gf_dim * 2, "gres2", reuse, is_train)

            # res3 = resblock_up(res2, gf_dim * 2, gf_dim, "gres3", reuse, is_train)

            res4 = resblock_up(resp1, gf_dim, gf_dim, "gres" + str(n + 1), reuse, is_train)

            conv1 = tf.layers.conv2d(res4, gf_dim, (3, 3), (1, 1),
                                     padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True,
                                     name="g_conv1", reuse=reuse)
            conv1 = tf.layers.batch_normalization(conv1, center=True,
                                                  scale=True, trainable=True,
                                                  gamma_initializer=gamma_init,
                                                  training=is_train,
                                                  name='g_bn1',
                                                  reuse=reuse)
            conv1 = tf.nn.leaky_relu(conv1, 0.2)

            conv2 = tf.layers.conv2d(conv1, output_dim, (3, 3), padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="g_conv2",
                                     reuse=reuse)
        return conv2  # , res4
