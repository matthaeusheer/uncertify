import numpy as np
from pdb import set_trace as bp

def total_variation(images, mask, name=None):
  images = np.array(images)
  #with ops.name_scope(name, 'total_variation'):
  ndims = len(images.shape)

  if ndims == 3:
    # The input is a single image with shape [height, width, channels].

    # Calculate the difference of neighboring pixel-values.
    # The images are shifted one pixel along the height and width by slicing.
    pixel_dif1 = images[1:, :, :] - images[:-1, :, :]
    pixel_dif2 = images[:, 1:, :] - images[:, :-1, :]

    # Sum for all axis. (None is an alias for all axis.)
    sum_axis = None
  elif ndims == 4:
    # The input is a batch of images with shape:
    # [batch, height, width, channels].

    # Calculate the difference of neighboring pixel-values.
    # The images are shifted one pixel along the height and width by slicing.
    pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
    pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]

    # Only sum for the last 3 axis.
    # This results in a 1-D tensor with the total variation for each image.
    sum_axis = (1, 2, 3)
  else:
    raise ValueError('\'images\' must be either 3 or 4-dimensional.')

  # Calculate the total variation by taking the absolute value of the
  # pixel-differences and summing over the appropriate axis.
  dif1 = np.abs(pixel_dif1)
  dif2 = np.abs(pixel_dif2)
  mask1 = mask[:-1,:,:]
  mask2 = mask[:,:-1,:]
  #bp()
  tot_var = (
      np.sum(dif1[mask1==1], axis=sum_axis) +
      np.sum(dif2[mask2==1], axis=sum_axis))

  return tot_var
