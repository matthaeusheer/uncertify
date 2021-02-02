import tensorflow as tf
import numpy as np
import math
from pdb import set_trace as bp
from utils import deeploss

def gaussian_negative_log_likelihood(x, mu, std):
    sq_x = (x - mu)**2
    #sq_std = std**2
    sq_std = tf.exp(std)
    #sq_std = (tf.log(std**2+1e-10))**2
    log_x = - sq_x / (sq_std+1e-10)
    C = -0.5 * tf.log(2.*math.pi*sq_std+1e-10)
    return -(C + log_x)

def l2loss(x,y, axis=(1,2,3)):
    dim = tf.shape(x)
    summed = tf.reduce_sum((y - x)**2,
                           axis=axis)
    sqrt_summed = tf.sqrt(summed)
    l2_loss = sqrt_summed
    return l2_loss

def l2loss_np(x,y):
    summed = (y - x)**2
    sqrt_summed = tf.sqrt(summed + 1e-10)
    l2 = sqrt_summed
    return l2

def l1loss(x,y):
    summed = tf.reduce_sum(tf.abs(y - x), axis=[1, 2, 3])
    #sqrt_summed = tf.sqrt(summed + 1e-10)
    l1_loss = summed
    return l1_loss

def l1loss_np(x,y):
    l1 = np.abs(y - x)
    #sqrt_summed = tf.sqrt(summed + 1e-10)
    return l1


def kl_loss_1d(z_mean,z_log_sigma, axis=(1,2,3)):
    latent_loss = -0.5 * tf.reduce_sum(
        1 + 2*z_log_sigma - tf.square(z_mean) -
        tf.exp(2*z_log_sigma), axis=axis)

    return latent_loss

def kl_loss_1d_1d(z_mean,z_stddev):
    latent_loss = tf.reduce_sum(
        tf.square(z_mean) + tf.square(z_stddev) - tf.log(z_stddev + 1e-10) - 1, [1])
    return latent_loss

def batch_transpose(x):
    # only for dim(x)==4
    x = tf.transpose(x, perm = [0,1,3,2])
    return x

def kl_cov_gaussian(mu, A):
    n = tf.shape(mu)[0]
    c = tf.shape(mu)[1]
    h = tf.shape(mu)[-1]
    sigma = tf.matmul(batch_transpose(A), A)+tf.eye(h, h, batch_shape=[n, c]) * 1e-8

    mu = batch_transpose(mu)

    mu0 = tf.zeros_like(mu)
    sigma0 = tf.eye(h, h, batch_shape=[n, c])
    #eps = tf.eye(h, h, batch_shape=[n, c])*1e-10

    #mu0 = [[0, 0, 0, 0]]
    #sigma0 = tf.eye(4)

    sigma_inv = tf.linalg.inv(sigma0)
    _dot = tf.matmul(sigma_inv, sigma)
    _dot = tf.trace(_dot)

    _matmul = tf.matmul(batch_transpose(mu0-mu), tf.linalg.inv(sigma0))
    _matmul = tf.matmul(_matmul, (mu0-mu))
    _matmul = tf.reshape(_matmul, [n,c])

    _k = tf.linalg.trace(sigma0)

    _log = tf.log(tf.linalg.det(sigma0)+1e-8)-tf.log(tf.linalg.det(sigma)+1e-8)

    #_log = tf.linalg.logdet(sigma0)-tf.linalg.logdet(sigma)

    kl = 0.5*(_dot + _matmul - _k + _log)
    kl = tf.reduce_sum(kl, [1])
    return kl

#
# def kl_cov_gaussian_nd(mu, sigma):
#     # mu = tf.transpose(mu, [0,-1,1,2])
#     # sigma = tf.transpose(sigma, [0, -1, 1, 2])
#     #
#     # n = tf.shape(mu)[0]
#     # c = tf.shape(mu)[1]
#     # h = tf.shape(mu)[2]
#     #
#     # mu = tf.reshape(mu,(n*c, h, h))
#     # sigma = tf.reshape(sigma,(n*c, h*2, h*2))
#     sigma_T = tf.transpose(sigma, perm=[0, 2, 1])
#
#     _sigma = tf.matmul(sigma_T, sigma)
#     #_sigma = tf.map_fn(lambda x,y: tf.matmul(x, y), sigma_T, sigma)
#     #bp()
#
#     summed_kl = tf.map_fn(lambda x: kl_cov_gaussian(x[0], x[1]), (mu, _sigma))
#     return summed_kl


def perceputal_loss(x, params, vgg19):
    """
    see vunet repository
    :param x:
    :param params:
    :return:
    """
    return 5.0 * vgg19.make_loss_op(x, params)


# def negative_nllh(x, mu, sigma):
#     sigma_max = tf.maximum(1e-10, sigma)
#     sum_sigma = tf.reduce_sum(sigma_max, axis=[1,2,3])
#     sum_frac = tf.reduce_sum(tf.square(x-mu)/(tf.exp(sigma_max)), axis=[1,2,3])
#     llh = sum_frac + sum_sigma
#     return llh

def negative_nllh(x, mu, sigma):
    #sigma = tf.minimum(sigma, np.log(np.sqrt(2.)))
    #sigma_max = tf.maximum(1e-10, sigma)
    sum_sigma = -tf.reduce_sum(sigma, axis=[1,2,3])
    sum_frac = tf.reduce_sum(tf.square(x-mu)*(tf.exp(sigma)**2/2.), axis=[1,2,3])
    llh = sum_frac + sum_sigma
    return llh

def negative_llh_var(x, mu, sigma):
    #sigma = tf.minimum(sigma, np.log(np.sqrt(2.)))
    #sigma_max = tf.maximum(1e-10, sigma)
    sum_sigma = -tf.reduce_sum(sigma, axis=[1,2,3])
    sum_frac = tf.reduce_sum((x-mu)**2*tf.exp(sigma), axis=[1,2,3])
    llh = sum_frac + sum_sigma
    return llh

def llh(x, mu, sigma):
    #sigma_min = np.minimum(sigma, np.log(1e4))
    sum_sigma = -sigma
    sum_frac = ((x - mu)**2) * np.exp(sigma)
    llh = sum_frac + sum_sigma
    return llh

def aggregate_var_loss(mu, true_image, pred_var):
    decoder_err_mu = tf.zeros_like(mu)
    for i in range(25):
        # err = (mu - true_image)**2
        err = tf.abs(mu-true_image)
        decoder_err_mu += err
    decoder_err_mu = decoder_err_mu / 25.
    loss = l1loss(pred_var, decoder_err_mu)
    return loss, decoder_err_mu





