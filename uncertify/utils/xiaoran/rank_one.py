import tensorflow as tf
import numpy as np
from pdb import set_trace as bp

def dot(a,b):
    return tf.einsum('ij,jk->ik', a, b)


class RankOne():
    def __init__(self, sess, n_latent, batch_size, img_size):
        self.sess = sess
        self.b = batch_size
        self.n = n_latent
        self.img_size = img_size

        self.u = tf.Variable(tf.zeros([self.n, self.img_size**2, 1]), trainable=True)
        self.w = tf.Variable(tf.zeros([self.n]), trainable=True)
        self.u = tf.cast(self.u, tf.float32)
        self.w = tf.cast(self.w, tf.float32)

        self.A = tf.eye(self.img_size**2)
        #self.A = tf.linalg.inv(A)

        self.M_i = self.rank_one_update(self.u, self.w)

    def rank_one(self, A, u, v):
        u = tf.reshape(u, [128*128,1])
        v = tf.reshape(v, [128*128,1])
        x = dot(dot(dot(A, u), tf.transpose(v)), A)
        y = 1 + dot(dot(tf.transpose(v), A), u)
        return A - x/y

    def step_one(self):
        self.M_i = self.A

    def rank_one_update(self, u, w):
        if tf.shape(u)[0] != 1:
            self.M_i = self.rank_one(self.rank_one_update(self.u[:-1], self.w[:-1]),
                                     self.w[-1] * self.u[-1], self.u[-1])
            return self.M_i
        else:
            A = self.A
            return self.rank_one(A, w * u, u)

    # def rank_one_update_np(self,u, w, b, n, A_inv):
    #     if u.shape[0] != 1:
    #         self.M_i = self.rank_one_np(rank_one_update_np(u[:-1], w[:-1], b, n, A_inv),
    #                           w[-1] * u[-1], u[-1])
    #         return self.M_i
    #     else:
    #         return self.rank_one_np(A_inv, w * u, u)

    # def get_grad(self):
    #     self.grad_j = tf.reshape(dot(self.M_i, tf.reshape(tf.transpose(self.mu),[128*128,1])),
    #                           [128,128,1])
            #self.assign_op = self.grad[j].assign(grad_j)
            #self.sess.run(self.assign_op)
        #return grad_j


def rank_one_np(A, u ,v):
    u = u.reshape(1,128*128)
    v = v.reshape(1,128*128)
    u = u.T
    v = v.T
    x = np.dot(np.dot(np.dot(A, u), np.transpose(v)), A)
    y = 1 + np.dot(np.dot(np.transpose(v), A), u)
    return A - x/y

# def rank_one_update_np(u, mu, w, b, n, A_inv):
#     grad = np.zeros((b, 128, 128, 1))
#
#     for j in range(b):
#         M_i = rank_one_np(A_inv, w[0, j] * np.transpose(u[0, j]),
#                        np.transpose(u[0, j]))
#         for i in range(n-1):
#             M_i = rank_one_np(M_i, w[i+1,j]*np.transpose(u[i+1,j]),
#                             np.transpose(u[i+1,j]))
#         grad_j= np.reshape(np.dot(M_i, np.reshape(np.transpose(mu[j]),[128*128,1])),
#                               [1,128,128,1])
#         grad[j] = grad_j
#
#     return grad


def rank_one_update_np(u, w, b, n):
    if u.shape[0]!=1:
        M_i = rank_one_np(rank_one_update_np(u[:-1], w[:-1], b, n),
                            w[-1]*u[-1], u[-1])
        return M_i
    else:
        A=np.eye(128*128)
        return rank_one_np(A, w*u, u)
