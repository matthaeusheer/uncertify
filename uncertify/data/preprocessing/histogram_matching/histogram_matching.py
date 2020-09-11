import os
import sys
import numpy as np
# import SimpleITK as sitk
import nibabel as nib
import glob
from uncertify.data.preprocessing.histogram_matching.HistogramEqualization import *
from pdb import set_trace as bp


# from medpy.io import save

# class sitk_image(object):
#     def __init__(self, sitkimage):
#         self.img = sitkimage
#     def get_data(self):
#         img_array = sitk.GetArrayFromImage(self.img)
#         return img_array

def save_to_nif(img, img_data):
    img_dir = os.path.split(img)[:-1]
    img_name = img.split("/")[-1] + "_matched.nii.gz"
    x_nif = nib.Nifti1Image(img_data, np.eye(4))
    nib.save(x_nif, os.path.join(img_dir[0], img_name))


def MatchHistogramsTwoImages(I, J, L, nbins=50, skip=50, begval=0., finval=0.99, train_mask=None, test_mask=None):
    if type(I) == str:
        # I = sitk.GetArrayFromImage(sitk.ReadImage(I))
        I = nib.load(I)
        I = I.get_data()
        # I = np.transpose(I, [1, 0, 2])
    if type(J) == str:
        # J = sitk.GetArrayFromImage(sitk.ReadImage(J))
        J = nib.load(J).get_data()
    if type(train_mask) == str:
        train_mask = nib.load(train_mask).get_data()
    if type(test_mask) == str:
        test_mask = nib.load(test_mask).get_data()
    HI = ComputeImageHistogram(I, nbins=nbins, skip=skip, mask=train_mask)
    HJ = ComputeImageHistogram(J, nbins=nbins, skip=skip, mask=test_mask)
    if np.isscalar(L):
        L_ = np.linspace(begval, 1, L + 1)
        L_[-1] = finval
        L = L_
    PI = ComputePercentiles(HI, L)
    PJ = ComputePercentiles(HJ, L)
    K = MultiLinearMap(J, PJ, PI)
    return K


if __name__ == "__main__":
    # camcan_subjects = glob.glob(
    #    '/scratch-second/HCP_3T_Structural_Preprocessed/100206/T2w_acpc_dc_restore.nii.gz')
    # ref_img = '/scratch-second/HCP_3T_Structural_Preprocessed/100206/T2w_acpc_dc_restore.nii.gz'
    ref_img = '/scratch-second/CamCAN_unbiased_copies/original_CamCAN/T2w/sub-CC110033_T2w_unbiased.nii.gz'
    # ref_mask = camcan_subjects[0].split("_renormalized")[0]+"_mask_cropped_mask.nii.gz"
    ref_img_data = nib.load(ref_img).get_data()
    ref_img_data = np.transpose(ref_img_data, [2, 0, 1])

    # ref_img_data = np.transpose(ref_img_data, [2, 0, 1])

    ref_img_data = np.array([np.rot90(i, 1) for i in ref_img_data])

    ref_img_data = ref_img_data[:, 23:283, :]

    ref_mask_data = np.ones_like(ref_img_data)
    # ref_mask_data = nib.load(ref_mask).get_data()

    # brats_subjects = glob.glob(
    #    '/scratch_net/bmicdl01/Data/BraTS_unbiased_aligned/T2w-rigid-to-mni/*/*_renormalized_*')
    MS_subjects = glob.glob(
        '/scratch_net/bmicdl01/Data/MS_Lesions_2008/CHB_train_Case01/CHB_train_Case01_T2_stripped.nii.gz'
    )

    for i in MS_subjects:
        print(i.split("/")[-1])
        # mask = i.split("_renormalized")[0]+"_mask_cropped_mask.nii.gz"
        _i = nib.load(i).get_data()
        # _mask = nib.load(mask).get_data()
        _mask = np.ones_like(_i)
        _i_matched = MatchHistogramsTwoImages(ref_img_data, _i, L=200, nbins=246, begval=0.0, finval=1.,
                                              train_mask=ref_mask_data, test_mask=_mask)

        x_nif = nib.Nifti1Image(_i_matched, np.eye(4))
        i_save = i.replace('.nii.gz', '_eq_to_camcan.nii.gz')
        nib.save(x_nif, os.path.join(i_save))
