import PIL.Image
from multiprocessing.pool import ThreadPool
from uncertify.utils.xiaoran.IndexFlow_camcan import *
import numpy as np
import pickle
import os
import cv2
import math
import matplotlib.pyplot as plt

n_boxes = 8


class BufferedWrapper(object):
    """Fetch next batch asynchronuously to avoid bottleneck during GPU
    training."""

    def __init__(self, gen):
        self.gen = gen
        self.n = gen.n
        self.pool = ThreadPool(1)
        self._async_next()

    def _async_next(self):
        self.buffer_ = self.pool.apply_async(next, (self.gen,))

    def __next__(self):
        result = self.buffer_.get()
        self._async_next()
        return result


#
def tile(X, rows, cols):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype=X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx, ...]
                tiling[
                i * X.shape[1]:(i + 1) * X.shape[1],
                j * X.shape[2]:(j + 1) * X.shape[2],
                :] = img
    return tiling


def plot_batch(X, out_path, vmax=0, vmin=0):
    """Save batch of images tiled."""
    X = np.stack(X)
    n_channels = X.shape[3]
    if n_channels > 3:
        X = X[:, :, :, np.random.choice(n_channels, size=3)]
    # X = postprocess(X)
    rc = math.sqrt(X.shape[0])
    rows = cols = math.ceil(rc)
    canvas = tile(X, rows, cols)
    canvas = np.squeeze(canvas)
    # plt.imsave(out_path, canvas)
    if vmax == vmin:
        plt.imsave(out_path, canvas)
    else:
        plt.imsave(out_path, canvas, vmin=vmin, vmax=vmax)


def get_camcan_batches(
        shape,
        index_path,
        train,
        box_factor,
        len_consecutive=64,
        fill_batches=True,
        shuffle=True,
        consecutive=False):
    """Buffered IndexFlow."""
    if not consecutive:
        flow = IndexFlowCamCAN(shape, index_path, train, box_factor, fill_batches, shuffle)
    # print(flow.n)
    else:
        # flow = IndexFlowCamCAN_consecutive(shape, index_path, train, box_factor, len_consecutive, fill_batches, shuffle)
        flow = IndexFlowCamCAN_3D(shape, index_path, train, box_factor, fill_batches, shuffle)
    return BufferedWrapper(flow)


def get_brats_batches(
        shape,
        index_path,
        train,
        box_factor,
        fill_batches=True,
        shuffle=True):
    """Buffered IndexFlow."""
    flow = IndexFlowBrats(shape, index_path, train, box_factor, fill_batches, shuffle)
    return BufferedWrapper(flow)


def get_brats_batch_subject(
        shape,
        index_path,
        train,
        box_factor,
        fill_batches=True,
        shuffle=True):
    """Buffered IndexFlow."""
    flow = IndexFlowBrats(shape, index_path, train, box_factor, fill_batches, shuffle)
    return BufferedWrapper(flow)


def get_multichannel_batches(
        shape,
        index_path,
        train,
        box_factor,
        len_consecutive=64,
        fill_batches=True,
        shuffle=True):
    """Buffered IndexFlow."""
    flow = IndexFlow_consecutive(shape=shape,
                                 index_path=index_path,
                                 train=train,
                                 number=len_consecutive,
                                 box_factor=box_factor,
                                 fill_batches=fill_batches,
                                 shuffle=shuffle)
    return BufferedWrapper(flow)


def get_ucsd_batches(
        shape,
        index_path,
        fill_batches=True,
        shuffle=True):
    """Buffered IndexFlow."""
    flow = IndexFlowUCSD(shape=shape, index_path=index_path, fill_batches=fill_batches,
                         shuffle=shuffle)
    return BufferedWrapper(flow)
