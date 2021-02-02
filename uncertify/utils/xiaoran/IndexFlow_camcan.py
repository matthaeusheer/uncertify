import os
import h5py
import numpy as np
import glob
import random
from scipy.misc import imresize
from scipy.ndimage import zoom
import nibabel as nib
from pdb import set_trace as bp


def resize(img, shape, mode):
    i_ = imresize(img, shape, interp=mode)
    i_re = (i_/255.0)*(img.max()-img.min())+img.min()
    return np.asarray(i_re)

def resize3D(img_3d,scale, method):
    img=img_3d
    res = []
    for im in img:
        i_ = imresize(im,scale, interp=method)
        i_re = (i_/255.0)*(im.max()-im.min())+im.min()
        res.append(i_re)
    return np.asarray(res)


class IndexFlowCamCAN(object):
    """Batches from index file."""
    def __init__(
            self,
            shape,
            index_path,
            train,
            box_factor,
            fill_batches = True,
            shuffle = True,
            return_keys=["imgs", "mask", "norm_imgs", "norm_mask"]):
        self.shape = shape
        self.batch_size = self.shape[0]
        self.img_shape = self.shape[1:]
        self.box_factor = box_factor
        self.nimg_shape = self.img_shape[0]//(2**self.box_factor)

        self.basepath = os.path.dirname(index_path)
        self.train = train
        self.fill_batches = fill_batches
        self.shuffle_ = shuffle

        self.data_file = h5py.File(index_path, 'r')
        self.return_keys = return_keys

        self.indices = np.array(
                [i for i in range(self.data_file['Scan_T2w'].shape[0])])

        self.n = self.indices.shape[0]
        self.shuffle()
    # def _filter(self, i):
    #     good = True
    #     good = good and (self.index["train"][i] == self.train)
    #     joints = self.index["joints"][i]
    #     required_joints = ["lshoulder","rshoulder","lhip","rhip"]
    #     joint_indices = [self.jo.index(b) for b in required_joints]
    #     joints = np.float32(joints[joint_indices])
    #     good = good and valid_joints(joints)
    #     return good


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
            self.shuffle()
        else:
            self.batch_start = batch_end

        # prepare batch data
        # load images
        batch["imgs"] = list()
        for i in batch_indices:
            scan = self.data_file['Scan_T2w'][i].reshape(200,200)
            #scan = imresize(scan, self.img_shape, "bilinear")
            scan = resize(scan, self.img_shape, "bilinear")
            scan = scan[:, :, np.newaxis]
            # bp()
            #relpath = self.index["imgs"][i]
            #path = os.path.join(self.basepath, relpath)
            #batch["imgs"].append(load_img(path, target_size = self.img_shape))
            batch["imgs"].append(scan)
        batch["imgs"] = np.stack(batch["imgs"])
        #batch["imgs"] = preprocess(batch["imgs"])

        # load masks
        batch["mask"] = list()
        for i in batch_indices:
            mask = self.data_file['Mask_1'][i].reshape(200,200)
            #mask = imresize(mask, self.img_shape, 'nearest')
            mask = resize(mask, self.img_shape, 'nearest')
            mask = mask[:, :, np.newaxis]
            batch["mask"].append(mask)
        batch["mask"] = np.stack(batch["mask"])

        # imgs, joints = normalize(batch["imgs"], batch["joints_coordinates"], batch["joints"], self.jo, self.box_factor)
        batch["norm_imgs"] = list()
        for i in batch_indices:
            scan = self.data_file['Scan_T1w'][i].reshape(200,200)
            #scan = imresize(scan+3.5, 2*[self.nimg_shape]+[1], 'bilinear')
            scan = resize(scan, 2*[self.nimg_shape]+[1],'bilinear')
            scan = scan[:, :, np.newaxis]
            batch["norm_imgs"].append(scan)
        batch["norm_imgs"] = np.stack(batch["norm_imgs"])

        batch["norm_mask"] = list()
        for i in batch_indices:
            mask = self.data_file['Mask_2'][i].reshape(200, 200)
            #mask = imresize(mask+3.5, 2*[self.nimg_shape]+[1], 'nearest')
            mask = resize(mask, 2*[self.nimg_shape]+[1], 'nearest')
            mask = mask[:,:,np.newaxis]
            batch["norm_mask"].append(mask)
        batch["norm_mask"] = np.stack(batch["norm_mask"])
        #
        # batch["norm_imgs"] = batch['imgs']
        # batch["norm_mask"] = batch['mask']
        batch_list = [batch[k] for k in self.return_keys]
        return batch_list
        self.data_file.close()


    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)

class IndexFlowCamCAN_consecutive(object):
    """Batches from index file."""
    def __init__(
            self,
            shape,
            index_path,
            train,
            box_factor,
            len_consecutive=64,
            fill_batches = True,
            shuffle = True,
            return_keys=["imgs", "mask", "norm_imgs", "norm_mask"]):
        self.shape = shape
        self.batch_size = self.shape[0]
        self.img_shape = self.shape[1:]
        self.box_factor = box_factor
        self.nimg_shape = self.img_shape[0]//(2**self.box_factor)
        # with open(index_path, "rb") as f:
        #     self.index = pickle.load(f)
        self.basepath = os.path.dirname(index_path)

        self.train = train
        self.fill_batches = fill_batches
        self.shuffle_ = shuffle

        self.data_file = [i for i in glob.glob(os.path.join(self.basepath, 'CamCAN/T2w/*')) if "normalized_cropped" in i]#h5py.File(index_path, 'r')
        self.return_keys = return_keys
        #self.return_keys = [key for key in self.data_file.keys()]

        # rescale joint coordinates to image shape
        # h,w = self.img_shape[:2]
        # wh = np.array([[[w,h]]])
        # # self.index["joints"] = self.index["joints"] * wh
        #bp()
        self.indices = np.array(
                [i for i in range(len(self.data_file))])

        self.n = self.indices.shape[0]
        self.shuffle()
        self.len_consecutive=len_consecutive


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
            self.shuffle()
        else:
            self.batch_start = batch_end

        # prepare batch data
        # load images
        batch["imgs"] = list()
        batch["mask"] = list()
        batch["norm_imgs"] = list()
        batch["norm_mask"] = list()
        #for _ in range(self.batch_size):
        for i in batch_indices:
            scan = nib.load(self.data_file[i]).get_data()
            #img_file_t1 = nib.load(self.data_file[i].replace('T2w', 'T1w')).get_data()

            #scan = self.data_file['Scan_T2w'][i].reshape(200,200)
            len_img = len(scan)-self.len_consecutive
            # p = np.array(range(len_img))/len_img
            # p = (p-p.min())/(p.max()-p.min())
            # p = p/sum(p)
            x_list = np.arange(len_img)
            #self.x = np.random.choice(len_img, 1, p=p[::-1])[0]
            self.x = np.random.choice(x_list, 1)[0]

            #scan = imresize(scan, self.img_shape, "bilinear")
            scan = resize3D(scan, self.img_shape, "bilinear")
            scan = scan[self.x:self.x+self.len_consecutive,:, :, np.newaxis]
            batch["imgs"].append(scan)
        # load masks

        for i in batch_indices:
            mask = nib.load(self.data_file[i].replace('normalized_cropped','mask_cropped')).get_data()
            mask = resize3D(mask, self.img_shape, 'nearest')
            mask = mask[self.x:self.x+self.len_consecutive, :, :, np.newaxis]
            batch["mask"].append(mask)

        # for i in batch_indices:
        #     scan = nib.load(self.data_file[i].replace('T2w', 'T1w')).get_data()
        #     scan = resize3D(scan, self.img_shape,'bilinear')
        #     scan = scan[self.x:self.x+self.len_consecutive, :, :, np.newaxis]
        #     batch["norm_imgs"].append(scan)
        #
        # for i in batch_indices:
        #     mask = nib.load(self.data_file[i].replace('normalized_cropped','mask_cropped')).get_data()
        #     mask = resize3D(mask, self.img_shape, 'nearest')
        #     mask = mask[self.x:self.x+self.len_consecutive, :, :, np.newaxis]
        #     batch["norm_mask"].append(mask)

        batch["imgs"] = np.stack(batch["imgs"])
        batch["mask"] = np.stack(batch["mask"])
        batch["norm_imgs"] = np.stack(batch["imgs"])
        batch["norm_mask"] = np.stack(batch["mask"])

        batch_list = [batch[k] for k in self.return_keys]
        return batch_list
        #self.data_file.close()


    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)


class IndexFlow_consecutive(object):
    """Batches from index file."""
    def __init__(
            self,
            shape,
            index_path,
            train,
            box_factor,
            number,
            fill_batches = True,
            shuffle = True,
            return_keys=["imgs", "mask"]):
        self.shape = shape
        self.batch_size = self.shape[0]
        self.img_shape = self.shape[1:-1]
        self.box_factor = box_factor
        self.nimg_shape = self.img_shape[0]//(2**self.box_factor)

        #self.basepath = os.path.dirname(index_path)
        self.train = train
        self.fill_batches = fill_batches
        self.shuffle_ = shuffle
        
        self.data_file = h5py.File(index_path)
        self.return_keys = return_keys
        self.num_slices = number

        self.indices = np.array([i for i in range(len(self.data_file['Scan']))])

        self.n = self.indices.shape[0]
        self.shuffle()


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
            if self.shuffle_:
                self.shuffle()
            else:
                self.batch_start = 0
        else:
            self.batch_start = batch_end

        # prepare batch data
        # load images
        batch["imgs"] = list()
        for i in batch_indices:
            scan = self.data_file['Scan'][i].reshape(-1, 200, 200)
            scan = resize3D(scan, self.img_shape, "bilinear")
            scan = np.transpose(scan, [1, 2, 0])

            batch["imgs"].append(scan)
        batch["imgs"] = np.stack(batch["imgs"])

        # load masks
        batch["mask"] = list()
        for i in batch_indices:
            mask = self.data_file['Mask'][i].reshape(-1, 200,200)
            mask = resize3D(mask, self.img_shape, 'nearest')
            mask = np.transpose(mask, [1,2,0])
            #mask = mask[:, :, np.newaxis]
            batch["mask"].append(mask)
        batch["mask"] = np.stack(batch["mask"])

        batch_list = [batch[k] for k in self.return_keys]
        return batch_list
        self.data_file.close()

    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)

class IndexFlowCamCAN_3D(object):
    """Batches from index file."""
    def __init__(
            self,
            shape,
            index_path,
            train,
            box_factor,
            fill_batches = True,
            shuffle = True,
            return_keys=["imgs", "mask"]):
        self.shape = shape
        self.batch_size = self.shape[0]
        self.img_shape = self.shape[1:]
        self.box_factor = box_factor
        self.nimg_shape = self.img_shape[0]//(2**self.box_factor)
        # with open(index_path, "rb") as f:
        #     self.index = pickle.load(f)
        self.basepath = os.path.dirname(index_path)

        self.train = train
        self.fill_batches = fill_batches
        self.shuffle_ = shuffle

        self.data_file = [i for i in glob.glob(os.path.join(self.basepath, 'CamCAN/T2w/*')) if "normalized_cropped" in i]#h5py.File(index_path, 'r')
        self.return_keys = return_keys
        self.indices = np.array(
                [i for i in range(len(self.data_file))])

        self.n = self.indices.shape[0]
        self.shuffle()

    def __next__(self):
        batch = dict()
        # get indices for batch
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        if self.fill_batches and batch_indices.shape[0] != self.batch_size:
            n_missing = self.batch_size - batch_indices.shape[0]
            batch_indices = np.concatenate([batch_indices, self.indices[:n_missing]], axis = 0)
            #assert batch_indices.shape[0] == self.batch_size
        batch_indices = np.array(batch_indices)
        batch["indices"] = batch_indices

        # prepare next batch
        if batch_end >= self.n:
            self.shuffle()
        else:
            self.batch_start = batch_end

        # prepare batch data
        # load images
        batch["imgs"] = list()
        for i in batch_indices:
            scan = nib.load(self.data_file[i]).get_data()
            #img_file_t1 = nib.load(self.data_file[i].replace('T2w', 'T1w')).get_data()

            #scan = self.data_file['Scan_T2w'][i].reshape(200,200)
            dims = scan.shape
            zoom_factor = np.asarray([128,128,128])/np.asarray(dims)
            scan = zoom(scan, zoom_factor, mode='nearest')

            #scan = scan[self.x:self.x+self.len_consecutive,:, :, np.newaxis]
            batch["imgs"].append(scan)
        batch["imgs"] = np.stack(batch["imgs"])

        # load masks
        batch["mask"] = list()
        for i in batch_indices:
            mask = nib.load(self.data_file[i].replace('normalized_cropped','mask_cropped')).get_data()
            #mask = resize3D(mask, self.img_shape, 'nearest')

            dims = mask.shape
            zoom_factor = np.asarray([128, 128, 128]) / np.asarray(dims)
            mask = zoom(mask, zoom_factor, order =0, mode='nearest')
            #mask = mask[self.x:self.x+self.len_consecutive, :, :, np.newaxis]
            batch["mask"].append(mask)
        batch["mask"] = np.stack(batch["mask"])

        # batch["norm_imgs"] = list()
        # for i in batch_indices:
        #     scan = nib.load(self.data_file[i].replace('T2w', 'T1w')).get_data()
        #
        #     dims = scan.shape
        #     zoom_factor = np.asarray([128, 128, 128]) / np.asarray(dims)
        #     scan = zoom(scan, zoom_factor, mode='nearest')
        #
        #     batch["norm_imgs"].append(scan)
        # batch["norm_imgs"] = np.stack(batch["norm_imgs"])
        #
        # batch["norm_mask"] = list()
        #
        # for i in batch_indices:
        #     mask = nib.load(self.data_file[i].replace('normalized_cropped','mask_cropped')).get_data()
        #
        #     dims = mask.shape
        #     zoom_factor = np.asarray([128, 128, 128]) / np.asarray(dims)
        #     mask = zoom(mask, zoom_factor, order=0, mode='nearest')
        #
        #     batch["norm_mask"].append(mask)
        # batch["norm_mask"] = np.stack(batch["norm_mask"])
        #
        batch_list = [batch[k] for k in self.return_keys]
        return batch_list


    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)


class IndexFlowBrats(object):
    """Batches from index file."""
    def __init__(
            self,
            shape,
            index_path,
            train,
            box_factor,
            fill_batches = True,
            shuffle = True,
            return_keys=["imgs", "seg", "norm_imgs", "norm_seg"]):
        self.shape = shape
        self.batch_size = self.shape[0]
        self.img_shape = self.shape[1:]
        self.box_factor = box_factor
        self.nimg_shape = self.img_shape[0]//(2**self.box_factor)
        # with open(index_path, "rb") as f:
        #     self.index = pickle.load(f)
        self.basepath = os.path.dirname(index_path)
        self.train = train
        self.fill_batches = fill_batches
        self.shuffle_ = shuffle
        # self.return_keys = return_keys
        #data_file = h5py(self.basepath)
        #bp()
        self.data_file = h5py.File(index_path, 'r')
        self.return_keys = return_keys
        #self.return_keys = [key for key in self.data_file.keys()]

        # rescale joint coordinates to image shape
        # h,w = self.img_shape[:2]
        # wh = np.array([[[w,h]]])
        # # self.index["joints"] = self.index["joints"] * wh
        #bp()
        self.indices = np.array(
                [i for i in range(self.data_file['Scan'].shape[0])])

        self.n = self.indices.shape[0]
        self.shuffle()
    # def _filter(self, i):
    #     good = True
    #     good = good and (self.index["train"][i] == self.train)
    #     joints = self.index["joints"][i]
    #     required_joints = ["lshoulder","rshoulder","lhip","rhip"]
    #     joint_indices = [self.jo.index(b) for b in required_joints]
    #     joints = np.float32(joints[joint_indices])
    #     good = good and valid_joints(joints)
    #     return good


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
            self.shuffle()
        else:
            self.batch_start = batch_end

        # prepare batch data
        # load images
        batch["imgs"] = list()
        for i in batch_indices:
            scan = self.data_file['Scan_T2w'][i].reshape(200,200)

            #scan = imresize(scan, self.img_shape, "bilinear")
            scan = resize(scan, self.img_shape, "bilinear")
            scan = scan[:, :, np.newaxis]
            # bp()
            #relpath = self.index["imgs"][i]
            #path = os.path.join(self.basepath, relpath)
            #batch["imgs"].append(load_img(path, target_size = self.img_shape))
            batch["imgs"].append(scan)
        batch["imgs"] = np.stack(batch["imgs"])
        #batch["imgs"] = preprocess(batch["imgs"])

        # load masks
        batch["seg"] = list()
        for i in batch_indices:
            mask = self.data_file['Seg'][i].reshape(200,200)
            #mask = imresize(mask, self.img_shape, 'nearest')
            mask = resize(mask, self.img_shape, 'nearest')
            mask = mask[:, :, np.newaxis]
            batch["seg"].append(mask)
        batch["seg"] = np.stack(batch["seg"])
        #batch["imgs"] = preprocess(batch["imgs"])

        # load joint coordinates
        #batch["joints_coordinates"] = [self.index["joints"][i] for i in batch_indices]

        # generate stickmen images from coordinates
        #batch["joints"] = list()
        # for joints in batch["joints_coordinates"]:
        #     img = make_joint_img(self.img_shape, self.jo, joints)
        #     batch["joints"].append(img)
        # batch["joints"] = np.stack(batch["joints"])
        # batch["joints"] = preprocess(batch["joints"])

        # imgs, joints = normalize(batch["imgs"], batch["joints_coordinates"], batch["joints"], self.jo, self.box_factor)
        batch["norm_imgs"] = list()
        for i in batch_indices:
            scan = self.data_file['Scan_T1w'][i].reshape(200,200)
            #scan = imresize(scan+3.5, 2*[self.nimg_shape]+[1], 'bilinear')
            scan = resize(scan, 2*[self.nimg_shape]+[1],'bilinear')
            scan = scan[:, :, np.newaxis]
            batch["norm_imgs"].append(scan)
        batch["norm_imgs"] = np.stack(batch["norm_imgs"])

        batch["norm_seg"] = list()
        for i in batch_indices:
            mask = self.data_file['Seg'][i].reshape(200, 200)
            #mask = imresize(mask+3.5, 2*[self.nimg_shape]+[1], 'nearest')
            mask = resize(mask, 2*[self.nimg_shape]+[1], 'nearest')
            mask = mask[:,:,np.newaxis]
            batch["norm_seg"].append(mask)
        batch["norm_seg"] = np.stack(batch["norm_seg"])
        #
        # batch["norm_imgs"] = batch['imgs']
        # batch["norm_mask"] = batch['mask']
        batch_list = [batch[k] for k in self.return_keys]
        return batch_list
        self.data_file.close()


    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)


class IndexFlowUCSD(object):
    """Batches from index file."""
    def __init__(
            self,
            shape,
            index_path,
            fill_batches = True,
            shuffle = True,
            return_keys=["imgs", "seg"]):

        self.shape = shape
        self.batch_size = self.shape[0]
        self.img_shape = self.shape[1:]

        self.h = self.img_shape[0]
        self.w = self.img_shape[1]

        self.fill_batches = fill_batches
        self.shuffle_ = shuffle

        self.data_file = h5py.File(index_path, 'r')
        self.return_keys = return_keys

        self.indices = np.array(
                [i for i in range(self.data_file['Scan'].shape[0])])

        self.n = self.indices.shape[0]
        self.shuffle()

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
            self.shuffle()
        else:
            self.batch_start = batch_end

        # prepare batch data
        batch["imgs"] = list()
        for i in batch_indices:
            scan = self.data_file['Scan'][i].reshape(self.h, self.w)
            batch["imgs"].append(scan)
        batch["imgs"] = np.stack(batch["imgs"])

        # load masks
        batch["seg"] = list()
        for i in batch_indices:
            mask = self.data_file['Mask'][i].reshape(self.h,self.w)
            batch["seg"].append(mask)
        batch["seg"] = np.stack(batch["seg"])

        batch_list = [batch[k] for k in self.return_keys]
        return batch_list

    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)


