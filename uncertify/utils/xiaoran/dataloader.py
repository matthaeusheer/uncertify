import nibabel as nib
from scipy.ndimage import zoom
import numpy as np
from preprocess.preprocess import *
from multiprocessing.pool import ThreadPool
from pdb import set_trace as bp

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


class dataloader_brats(object):
    """Batches from index file."""
    def __init__(self, test_sub, shape_train):

        self.shape = shape
        self.batch_size = shape_train[0]
        img_size = shape_train[1:]

        img = nib.load(test_sub).get_data()

        image_original_size = img.shape[1]
        self.original = image_original_size

        seg = nib.load(test_sub.replace("normalized_cropped_mask", "seg_cropped")).get_data()

        seg[seg != 0] = 1

        mask = nib.load(test_sub.replace("normalized_cropped_mask", "mask_cropped_mask")).get_data()

        idx = [i for i in range(len(mask)) if len(set(mask[i, :, :].flatten())) > 1]

        img = img[idx]
        seg = seg[idx]
        mask = mask[idx]

        len0 = len(img)

        if len0%batch_size:
            fill_len = (int(len0/batch_size)+1)*batch_size-len0

            fill_img = np.zeros((fill_len, image_original_size, image_original_size, c_dims))+img[:fill_len]
            fill_mask = np.zeros((fill_len, image_original_size, image_original_size, c_dims)) + mask[:fill_len]

            img_filled = np.append(img, fill_img, axis=0)
            mask_filled = np.append(mask, fill_mask, axis=0)
        else:
            img_filled = img
            mask_filled = mask

        self.img_resized = resize(img_filled[:,:,:,0], img_size / image_original_size, "bilinear")
        self.mask_resized = resize(mask_filled[:,:,:,0], img_size / image_original_size, "nearest")

        self.indices = np.array([i for i in range(len(self.img_resized))])
        self.n = self.indices.shape[0]
        self.batch_start = 0


    def __next__(self):
        batch = dict()
        # get indices for batch
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        if self.fill_batches and batch_indices.shape[0] != self.batch_size:
            n_missing = self.batch_size - batch_indices.shape[0]
            batch_indices = np.concatenate([batch_indices, self.indices[:n_missing]], axis = 0)
            assert batch_indices.shape[0] == self.batch_size
        batch_indices = np.array(batch_indices)
        batch["indices"] = batch_indices

        # prepare next batch
        if batch_end >= self.n:
            pass
        else:
            self.batch_start = batch_end

        # load images
        batch_imgs = list()
        for i in batch_indices:
            scan = self.img_resized[i]
            scan = resize(scan, self.img_shape, "bilinear")
            scan = scan[:, :, np.newaxis]
            batch_imgs.append(scan)
        imgs = np.stack(batch_imgs)

        # load masks
        batch_masks = list()
        for i in batch_indices:
            mask = self.mask_resized[i]
            mask = resize(mask, self.img_shape, 'nearest')
            mask = mask[:, :, np.newaxis]
            batch_masks.append(mask)
        masks = np.stack(batch_masks)

        return (imgs, masks)


def get_brats_multi_test(
        test_sub, shape_train):
    """Buffered IndexFlow."""
    flow = dataloader_brats_multi(test_sub, shape_train)
    return BufferedWrapper(flow)

class dataloader_brats_multi(object):
    """Batches from index file."""
    def __init__(self, test_sub, shape_train):
        self.fill_batches = False
        self.batch_size = shape_train[0]
        self.img_size = shape_train[1]
        c_dim = shape_train[-1]

        img = nib.load(test_sub).get_data()
        self.img = img

        image_original_size = img.shape[1]
        self.original = image_original_size

        img_concat = list()

        for c in range(c_dim):
            x_img = np.concatenate((img[c:], np.zeros((c, self.original, self.original))-3.5),
                                   axis=0)
            x_img = resize(x_img, self.img_size / image_original_size, "bilinear")
            x_img = np.transpose(x_img, (1,2,0))
            img_concat.append(x_img)
        self.img_concat = np.asarray(img_concat)
        self.img_concat = np.transpose(self.img_concat, (3,1,2,0))

        seg = nib.load(test_sub.replace("normalized_cropped_mask", "seg_cropped")).get_data()
        seg[seg != 0] = 1
        self.seg = seg

        mask = nib.load(test_sub.replace("normalized_cropped_mask", "mask_cropped_mask")).get_data()
        self.mask = mask

        mask_concat = list()

        for c in range(c_dim):
            x_img = np.concatenate((mask[c:], np.zeros((c, self.original, self.original))),
                                   axis=0)
            x_img = resize(x_img, self.img_size / image_original_size, "nearest")
            x_img = np.transpose(x_img, (1,2,0))
            mask_concat.append(x_img)
        self.mask_concat = np.asarray(mask_concat)
        self.mask_concat = np.transpose(self.mask_concat, (3, 1, 2, 0))

        idx = [i for i in range(len(mask)) if len(set(mask[i].flatten())) > 1]

        self.img_concat = self.img_concat[idx]
        seg = seg[idx]
        self.mask_concat = self.mask_concat[idx]

        len0 = len(self.img_concat)

        if len0%self.batch_size:
            fill_len = (int(len0/self.batch_size)+1)*self.batch_size-len0

            fill_img = np.zeros((fill_len, self.img_size, self.img_size, c_dim))+self.img_concat[:fill_len]
            fill_mask = np.zeros((fill_len, self.img_size, self.img_size, c_dim)) + self.mask_concat[:fill_len]

            self.img_filled = np.append(self.img_concat, fill_img, axis=0)
            self.mask_filled = np.append(self.mask_concat, fill_mask, axis=0)
        else:
            self.img_filled = self.img_concat
            self.mask_filled = self.mask_concat

        #self.img_resized = resize(img_filled[:,:,:,1], img_size / image_original_size, "bilinear")
        #self.mask_resized = resize(mask_filled[:,:,:,1], img_size / image_original_size, "bilinear")
        #img_2_resized = resize(img_filled[:,:,:,1], img_size / image_original_size, "bilinear")
        #img_1_resized = resize(img_filled[:,:,:,0], img_size / image_original_size, "bilinear")

        #img_resized = t2_img_resized
        #self.img_resized = np.concatenate((img_1_resized[:, :, :, np.newaxis], img_2_resized[:, :, :, np.newaxis]),
        #                     axis=-1)


        #mask1_resized = resize(mask[:,:,:,0], img_size / image_original_size, "nearest")
        #mask2_resized = resize(mask[:,:,:,1], img_size / image_original_size, "nearest")
        #self.mask_resized = np.concatenate((mask1_resized[:, :, :, np.newaxis], mask2_resized[:, :, :, np.newaxis]),
        #                             axis=-1)

        #seg_resized = resize(seg, self.img_size/image_original_size,"nearest")

        self.indices = np.array([i for i in range(len(self.img_filled))])
        self.n = self.indices.shape[0]
        self.batch_start = 0

    def get_shape(self):
        return self.img.shape

    def __iter__(self):
        return self

    def __next__(self):
        batch = dict()
        # get indices for batch
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        if self.fill_batches and batch_indices.shape[0] != self.batch_size:
            n_missing = self.batch_size - batch_indices.shape[0]
            batch_indices = np.concatenate([batch_indices, self.indices[:n_missing]], axis=0)
            assert batch_indices.shape[0] == self.batch_size
        batch_indices = np.array(batch_indices)
        batch["indices"] = batch_indices

        # prepare next batch
        if batch_end >= self.n:
            self.batch_start = 0
        else:
            self.batch_start = batch_end

        # load images
        batch_imgs = list()
        for i in batch_indices:
            scan = self.img_filled[i]
            #scan = resize(scan, (self.img_size,self.img_size), "bilinear")
            #scan = scan[:, :, np.newaxis]
            batch_imgs.append(scan)
        imgs = np.stack(batch_imgs)

        # load masks
        batch_masks = list()
        for i in batch_indices:
            mask = self.mask_filled[i]
            #mask = resize(mask, (self.img_size,self.img_size), 'nearest')
            #mask = mask[:, :, np.newaxis]
            batch_masks.append(mask)
        masks = np.stack(batch_masks)
        return (imgs, masks)


class dataloader_brats_3d(object):
    def __init__(self, test_sub, shape_train):

        self.shape = shape
        self.batch_size = shape_train[0]
        img_size = shape_train[1:]

        img = nib.load(test_sub).get_data()
        seg = nib.load(test_sub.replace("normalized_cropped_mask", "seg_cropped")).get_data()

        seg[seg != 0] = 1
        # #

        mask = nib.load(test_sub.replace("normalized_cropped_mask", "mask_cropped_mask")).get_data()

        idx = [i for i in range(len(mask)) if len(set(mask[i, :, :].flatten())) > 1]

        img = img[idx]
        seg = seg[idx]
        mask = mask[idx]

        dims = np.array([128, 128, 128])

        img = zoom(img, dims / np.array(img.shape), order=1, mode='nearest')
        seg = zoom(seg, dims / np.array(seg.shape), order=0, mode='nearest')
        mask = zoom(mask, dims / np.array(mask.shape), order=0, mode='nearest')
        len0 = len(img)

        img[mask == 0] = -3.5

        self.img = img[np.newaxis, :, :, :, np.newaxis]
        self.seg = seg[np.newaxis, :, :, :, np.newaxis]
        self.mask = mask[np.newaxis, :, :, :, np.newaxis]

        self.indices = np.array([i for i in range(len(self.img_resized))])
        self.n = self.indices.shape[0]
        self.batch_start = 0

    def get_shape(self):
        return self.img.shape

    def __iter__(self):
        return self


    def __next__(self):
        batch = dict()
        # get indices for batch
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        if self.fill_batches and batch_indices.shape[0] != self.batch_size:
            n_missing = self.batch_size - batch_indices.shape[0]
            batch_indices = np.concatenate([batch_indices, self.indices[:n_missing]], axis=0)
            assert batch_indices.shape[0] == self.batch_size
        batch_indices = np.array(batch_indices)
        batch["indices"] = batch_indices

        # prepare next batch
        if batch_end >= self.n:
            pass
        else:
            self.batch_start = batch_end

        # load images
        for i in batch_indices:
            scan = self.img[i]
        #imgs = scan

        # load masks
        for i in batch_indices:
            mask = self.mask[i]

        #masks = mask
        return (self.imgs, self.masks)


class dataloader_atlas_multi(object):
    """Batches from index file."""
    def __init__(self, test_sub, shape_train):
        self.fill_batches = False
        self.batch_size = shape_train[0]
        self.img_size = shape_train[1]
        c_dim = shape_train[-1]

        test_sub = test_sub.replace('/scratch_net/bmicdl01/Data/',
                                    '/itet-stor/chenx/chenx_bmicnas01/')

        img_file = test_sub

        img = nib.load(img_file).get_data()
        self.img = img

        image_original_size = img.shape[1]
        self.original = image_original_size

        img_concat = list()

        for c in range(c_dim):
            x_img = np.concatenate((img[c:], np.zeros((c, self.original, self.original))-3.5),
                                   axis=0)
            x_img = resize(x_img, self.img_size / image_original_size, "bilinear")
            x_img = np.transpose(x_img, (1,2,0))
            img_concat.append(x_img)
        self.img_concat = np.asarray(img_concat)
        self.img_concat = np.transpose(self.img_concat, (3,1,2,0))

        seg = nib.load(test_sub.replace("normalized_cropped_mask", "seg_cropped")).get_data()
        seg[seg != 0] = 1
        self.seg = seg

        mask = nib.load(test_sub.replace("normalized_cropped_mask", "mask_cropped_mask")).get_data()
        self.mask = mask

        mask_concat = list()

        for c in range(c_dim):
            x_img = np.concatenate((mask[c:], np.zeros((c, self.original, self.original))),
                                   axis=0)
            x_img = resize(x_img, self.img_size / image_original_size, "nearest")
            x_img = np.transpose(x_img, (1,2,0))
            mask_concat.append(x_img)
        self.mask_concat = np.asarray(mask_concat)
        self.mask_concat = np.transpose(self.mask_concat, (3, 1, 2, 0))

        idx = [i for i in range(len(mask)) if len(set(mask[i].flatten())) > 1]

        self.img_concat = self.img_concat[idx]
        seg = seg[idx]
        self.mask_concat = self.mask_concat[idx]

        len0 = len(self.img_concat)

        if len0%self.batch_size:
            fill_len = (int(len0/self.batch_size)+1)*self.batch_size-len0

            fill_img = np.zeros((fill_len, self.img_size, self.img_size, c_dim))+self.img_concat[:fill_len]
            fill_mask = np.zeros((fill_len, self.img_size, self.img_size, c_dim)) + self.mask_concat[:fill_len]

            self.img_filled = np.append(self.img_concat, fill_img, axis=0)
            self.mask_filled = np.append(self.mask_concat, fill_mask, axis=0)
        else:
            self.img_filled = self.img_concat
            self.mask_filled = self.mask_concat

        #self.img_resized = resize(img_filled[:,:,:,1], img_size / image_original_size, "bilinear")
        #self.mask_resized = resize(mask_filled[:,:,:,1], img_size / image_original_size, "bilinear")
        #img_2_resized = resize(img_filled[:,:,:,1], img_size / image_original_size, "bilinear")
        #img_1_resized = resize(img_filled[:,:,:,0], img_size / image_original_size, "bilinear")

        #img_resized = t2_img_resized
        #self.img_resized = np.concatenate((img_1_resized[:, :, :, np.newaxis], img_2_resized[:, :, :, np.newaxis]),
        #                     axis=-1)


        #mask1_resized = resize(mask[:,:,:,0], img_size / image_original_size, "nearest")
        #mask2_resized = resize(mask[:,:,:,1], img_size / image_original_size, "nearest")
        #self.mask_resized = np.concatenate((mask1_resized[:, :, :, np.newaxis], mask2_resized[:, :, :, np.newaxis]),
        #                             axis=-1)

        #seg_resized = resize(seg, self.img_size/image_original_size,"nearest")

        self.indices = np.array([i for i in range(len(self.img_filled))])
        self.n = self.indices.shape[0]
        self.batch_start = 0

    def get_shape(self):
        return self.img.shape

    def __iter__(self):
        return self

    def __next__(self):
        batch = dict()
        # get indices for batch
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        if self.fill_batches and batch_indices.shape[0] != self.batch_size:
            n_missing = self.batch_size - batch_indices.shape[0]
            batch_indices = np.concatenate([batch_indices, self.indices[:n_missing]], axis=0)
            assert batch_indices.shape[0] == self.batch_size
        batch_indices = np.array(batch_indices)
        batch["indices"] = batch_indices

        # prepare next batch
        if batch_end >= self.n:
            self.batch_start = 0
        else:
            self.batch_start = batch_end

        # load images
        batch_imgs = list()
        for i in batch_indices:
            scan = self.img_filled[i]
            #scan = resize(scan, (self.img_size,self.img_size), "bilinear")
            #scan = scan[:, :, np.newaxis]
            batch_imgs.append(scan)
        imgs = np.stack(batch_imgs)

        # load masks
        batch_masks = list()
        for i in batch_indices:
            mask = self.mask_filled[i]
            #mask = resize(mask, (self.img_size,self.img_size), 'nearest')
            #mask = mask[:, :, np.newaxis]
            batch_masks.append(mask)
        masks = np.stack(batch_masks)
        return (imgs, masks)


class dataloader_atlas_3d(object):
    def __init__(self, test_sub, shape_train):

        self.shape = shape
        self.batch_size = shape_train[0]
        img_size = shape_train[1:]

        img = nib.load(test_sub).get_data()
        seg = nib.load(test_sub.replace("normalized_cropped_mask", "seg_cropped")).get_data()

        seg[seg != 0] = 1
        # #

        mask = nib.load(test_sub.replace("normalized_cropped_mask", "mask_cropped_mask")).get_data()

        idx = [i for i in range(len(mask)) if len(set(mask[i, :, :].flatten())) > 1]

        img = img[idx]
        seg = seg[idx]
        mask = mask[idx]

        dims = np.array([128, 128, 128])

        img = zoom(img, dims / np.array(img.shape), order=1, mode='nearest')
        seg = zoom(seg, dims / np.array(seg.shape), order=0, mode='nearest')
        mask = zoom(mask, dims / np.array(mask.shape), order=0, mode='nearest')
        len0 = len(img)

        img[mask == 0] = -3.5

        img = img[np.newaxis, :, :, :, np.newaxis]
        seg = seg[np.newaxis, :, :, :, np.newaxis]
        mask = mask[np.newaxis, :, :, :, np.newaxis]

        self.indices = np.array([i for i in range(len(self.img_resized))])
        self.n = self.indices.shape[0]
        self.batch_start = 0


    def __next__(self):
        batch = dict()
        # get indices for batch
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        if self.fill_batches and batch_indices.shape[0] != self.batch_size:
            n_missing = self.batch_size - batch_indices.shape[0]
            batch_indices = np.concatenate([batch_indices, self.indices[:n_missing]], axis=0)
            assert batch_indices.shape[0] == self.batch_size
        batch_indices = np.array(batch_indices)
        batch["indices"] = batch_indices

        # prepare next batch
        if batch_end >= self.n:
            pass
        else:
            self.batch_start = batch_end

        # load images
        for i in batch_indices:
            scan = self.img_resized[i]
        imgs = scan

        # load masks
        for i in batch_indices:
            mask = self.mask_resized[i]

        masks = mask
        return (imgs, masks)