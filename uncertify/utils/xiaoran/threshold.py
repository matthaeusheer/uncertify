import h5py
import random
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SGE_GPU']
import tensorflow as tf
import tensorlayer as tl
#from networks.vae_res_bilinear_conv_lof import VariationalAutoencoder
sys.path.append("/scratch_net/bmicdl01/chenx/PycharmProjects/refine_vae")
from preprocess.preprocess import *
from utils import losses

# atlas=0
#
#
# if atlas==1:
#     crop_size = 210.
# else:
#     crop_size = 200.
# # to use different test dataset, change between atlas_size and crop_size
#
# EPOCH=400
# LEARNING_RATE =1e-4
# BATCH_SIZE=64
# IMAGE_SIZE=128
# Z_DIM=1024


def determine_threshold(phi, fprate):
    phi = np.asarray(phi)
    """
    determines the lowest threshold on Phi that provides at max FP rate on the Phi values.
    all the samples need to be controls for this function
    """
    nums = len(phi)
    #numf = phi.shape[1]

    def func(threshold):
        phi_ = phi > threshold
        fprate_ = np.sum(phi_) / np.float(nums)
        return np.sqrt((fprate - fprate_) ** 2)
    return gss(func, phi.min(), phi.mean(), phi.max(), tau=1e-8)


def gss(f, a, b, c, tau=1e-3):
    """
    Python recursive version of Golden Section Search algorithm

    tau is the tolerance for the minimal value of function f
    b is any number between the interval a and c
    """

    goldenRatio = (1 + 5 ** 0.5) / 2
    if c - b > b - a:
        x = b + (2 - goldenRatio) * (c - b)
    else:
        x = b - (2 - goldenRatio) * (b - a)
    if abs(c - a) < tau * (abs(b) + abs(x)): return (c + a) / 2
    if f(x) < f(b):
        if c - b > b - a:
            return gss(f, b, x, c, tau)
        return gss(f, a, x, b, tau)
    else:
        if c - b > b - a:
            return gss(f, a, b, x, tau)
        return gss(f, x, b, c, tau)

def mad_score(x, med):
    #minval = (x-mean).min()
    score = np.median(losses.l2loss_np(x, med), axis=-1, keepdims=True)
    return score

def modified_z_score(x, x_hat):
    median_x = np.median(x_hat, axis=-1, keepdims=True)
    MAD = np.median(np.abs(x_hat - median_x), axis=-1, keepdims=True)
    M = np.abs(x-median_x)/(MAD+1e-9)
    return M


def compute_threshold(fprate, model, img_size, batch_size, n_latent_samples, n_random_sub = 100,
                      renormalized = False):
    fprate = fprate
    if renormalized:
        n = "_renormalized"
        data = h5py.File('/scratch_net/bmicdl01/Data/camcan_test'+str(n)+'.hdf5')
    else:
        data = h5py.File('/scratch_net/bmicdl01/Data/camcan_train_set.hdf5')
    indices = random.sample(range(len(data['Scan']))[::batch_size], n_random_sub)

    image_size = img_size
    image_original_size = 200
    batch_size = batch_size
    dif = []
    dif_vae = []
    dif_vae_rel = []
    dif_prob = []
    dif_naive = []
    num = 0
    n_latent_samples = n_latent_samples
    for ind in indices:
        print(num, ind)
        res = data['Scan'][ind:ind + batch_size]
        res = res.reshape(-1, image_original_size, image_original_size)
        mask = data['Mask'][ind:ind + batch_size]
        mask = mask.reshape(-1, image_original_size, image_original_size)

        dim_res = res.shape
        image_original_size = res.shape[1]
        res_minval = res.min()

        if dim_res[0] % batch_size:
            dim_res_expand = batch_size - (dim_res[0] % batch_size)
            res_expand = np.zeros((dim_res_expand, dim_res[1], dim_res[2])) + res_minval
            res_exp = np.append(res, res_expand, axis=0)
            mask_exp = np.append(mask, np.zeros((dim_res_expand, dim_res[1], dim_res[2])), axis=0)
        else:
            res_exp = res
            mask_exp = mask

        res_exp = resize(res_exp, image_size / image_original_size)

        cnt = 0
        predicted_residuals = []
        predicted_residuals_vae = []
        predicted_residuals_vae_relative = []
        prob_map = []

        for batch in tl.iterate.minibatches(inputs=res_exp, targets=mask_exp,
                                            batch_size=batch_size, shuffle=False):
            b_images, _ = batch
            b_images = b_images[:, :, :, np.newaxis]
            decoded = []
            for i in range(n_latent_samples):
                model.validate(b_images)
                decoded_vae = model.out_mu_test
                #decoded_std = model.out_std_test
                decoded_vae_res = np.abs(b_images-decoded_vae) #np.abs(b_images - decoded_vae)
                decoded.append(decoded_vae_res)

            # predicted model error
            decoded_res = model.residual_output_test
            decoded.append(decoded_res)

            decoded = np.asarray(decoded).reshape(n_latent_samples+1, batch_size, image_size, image_size)
            decoded = np.transpose(decoded, (1, 2, 3, 0))
            # z_mu, z_dev = sess.run([z_mean, z_stddev], {image_matrix: b_images})
            # decoded_vae_mu = sess.run(decoded_mu, {guessed_z:z_mu})
            # decoded_vae_mu = np.mean(decoded, axis=-1, keepdims=True)
            # decoded_vae_std = np.std(decoded, axis=-1, keepdims=True)

            batch_median = np.median(decoded, axis=-1, keepdims=True)

            # print(decoded_vae_mu.shape, decoded_vae_std.shape)
            #residuals_vae = decoded_vae_res
            residuals_raw = np.abs(b_images - decoded_vae)
            #posterior_xz = losses.llh(b_images, decoded_vae, decoded_std)

            # decoded_res = sess.run(resid, {residuals_matrix: residuals_vae})
            residuals = np.abs(residuals_raw - decoded_res)
            #residuals_relative = b_images - decoded_vae #residuals_vae - decoded_res

            # corrected
            predicted_residuals.extend(residuals)
            # raw
            predicted_residuals_vae.extend(residuals_raw)
            # signed
            #predicted_residuals_vae_relative.extend(residuals_relative)

            # evaluate if predicted loss fits error distribution during test time
            residuals_mad_map = np.median(np.abs(decoded-batch_median), axis=-1, keepdims=True)

            prob_map.extend(residuals_mad_map)
            cnt += 1
        # prob_map_n = 1.-prob_map

        predicted_residuals_vae = np.asarray(predicted_residuals_vae).reshape(res_exp.shape[0], 128, 128)
        predicted_residuals = np.asarray(predicted_residuals).reshape(res_exp.shape[0], 128, 128)

        predicted_residuals_vae = resize(predicted_residuals_vae[:dim_res[0]], image_original_size / 128.)
        predicted_residuals = resize(predicted_residuals[:dim_res[0]], image_original_size / 128.)

        prob_map = np.asarray(prob_map).reshape(res_exp.shape[0], 128, 128)
        prob_map = resize(prob_map[:dim_res[0]], image_original_size / 128.)

        dif.extend(predicted_residuals[mask == 1])
        dif_vae.extend(predicted_residuals_vae[mask == 1])
        #dif_vae_rel.extend(predicted_residuals_vae_relative[mask == 1])
        dif_prob.extend(prob_map[mask == 1])
        #dif_naive.extend(res[mask == 1])
        num += 1

    thr_error = determine_threshold(dif_vae, fprate)
    thr_error_corr = determine_threshold(dif, fprate)
    thr_MAD = determine_threshold(dif_prob, fprate)

    return thr_error, thr_error_corr, thr_MAD


def compute_brats_threshold(fprate, model, img_size, batch_size, n_latent_samples, n_random_sub = 100,
                      renormalized = False):
    fprate = fprate
    if renormalized:
        n = "_renormalized"
        data = h5py.File('/scratch_net/bmicdl01/Data/camcan_test'+str(n)+'.hdf5')
    else:
        data = h5py.File('/scratch_net/bmicdl01/Data/brats_healthy_train.hdf5')
    indices = random.sample(range(len(data['Scan']))[::batch_size], n_random_sub)

    image_size = img_size
    image_original_size = 200
    batch_size = batch_size
    dif = []
    dif_vae = []
    dif_vae_rel = []
    dif_prob = []
    dif_naive = []
    num = 0
    n_latent_samples = n_latent_samples
    for ind in indices:
        print(num, ind)
        res = data['Scan'][ind:ind + batch_size]
        res = res.reshape(-1, image_original_size, image_original_size)
        mask = data['Mask'][ind:ind + batch_size]
        mask = mask.reshape(-1, image_original_size, image_original_size)

        dim_res = res.shape
        image_original_size = res.shape[1]
        res_minval = res.min()

        if dim_res[0] % batch_size:
            dim_res_expand = batch_size - (dim_res[0] % batch_size)
            res_expand = np.zeros((dim_res_expand, dim_res[1], dim_res[2])) + res_minval
            res_exp = np.append(res, res_expand, axis=0)
            mask_exp = np.append(mask, np.zeros((dim_res_expand, dim_res[1], dim_res[2])), axis=0)
        else:
            res_exp = res
            mask_exp = mask

        res_exp = resize(res_exp, image_size / image_original_size)

        cnt = 0
        predicted_residuals = []
        predicted_residuals_vae = []
        predicted_residuals_vae_relative = []
        prob_map = []

        for batch in tl.iterate.minibatches(inputs=res_exp, targets=mask_exp,
                                            batch_size=batch_size, shuffle=False):
            b_images, _ = batch
            b_images = b_images[:, :, :, np.newaxis]
            decoded = []
            for i in range(n_latent_samples):
                model.validate(b_images)
                decoded_vae = model.out_mu_test
                #decoded_std = model.out_std_test
                decoded_vae_res = np.abs(b_images-decoded_vae) #np.abs(b_images - decoded_vae)
                decoded.append(decoded_vae_res)

            # predicted model error
            decoded_res = model.residual_output_test
            decoded.append(decoded_res)

            decoded = np.asarray(decoded).reshape(n_latent_samples+1, batch_size, image_size, image_size)
            decoded = np.transpose(decoded, (1, 2, 3, 0))

            batch_median = np.median(decoded, axis=-1, keepdims=True)

            # print(decoded_vae_mu.shape, decoded_vae_std.shape)
            #residuals_vae = decoded_vae_res
            residuals_raw = np.abs(b_images - decoded_vae)
            #posterior_xz = losses.llh(b_images, decoded_vae, decoded_std)

            # decoded_res = sess.run(resid, {residuals_matrix: residuals_vae})
            residuals = np.abs(residuals_raw - decoded_res)
            #residuals_relative = b_images - decoded_vae #residuals_vae - decoded_res

            # corrected
            predicted_residuals.extend(residuals)
            # raw
            predicted_residuals_vae.extend(residuals_raw)
            # signed
            #predicted_residuals_vae_relative.extend(residuals_relative)

            # evaluate if predicted loss fits error distribution during test time
            residuals_mad_map = np.median(np.abs(decoded-batch_median), axis=-1, keepdims=True)

            prob_map.extend(residuals_mad_map)
            cnt += 1
        # prob_map_n = 1.-prob_map

        predicted_residuals_vae = np.asarray(predicted_residuals_vae).reshape(res_exp.shape[0], 128, 128)
        predicted_residuals = np.asarray(predicted_residuals).reshape(res_exp.shape[0], 128, 128)

        predicted_residuals_vae = resize(predicted_residuals_vae[:dim_res[0]], image_original_size / 128.)
        predicted_residuals = resize(predicted_residuals[:dim_res[0]], image_original_size / 128.)

        prob_map = np.asarray(prob_map).reshape(res_exp.shape[0], 128, 128)
        prob_map = resize(prob_map[:dim_res[0]], image_original_size / 128.)

        dif.extend(predicted_residuals[mask == 1])
        dif_vae.extend(predicted_residuals_vae[mask == 1])
        #dif_vae_rel.extend(predicted_residuals_vae_relative[mask == 1])
        dif_prob.extend(prob_map[mask == 1])
        #dif_naive.extend(res[mask == 1])
        num += 1

    thr_error = determine_threshold(dif_vae, fprate)
    thr_error_corr = determine_threshold(dif, fprate)
    thr_MAD = determine_threshold(dif_prob, fprate)

    return thr_error, thr_error_corr, thr_MAD

# if __name__=="__main__":
#     fprate = 0.1
#     data = h5py.File('/scratch_net/bmicdl01/Data/camcan_test.hdf5')
#     indices = random.sample(range(len(data['Scan']))[::64], 100)
#
#     model_folder = "logs/saved/"
#     #model_name = 'vae_res_bilinear_ims200_outsize2_zdim_1024_concat'
#     #epoch = 33000
#     model_name = 'vae_convres_bilinear_z1024_'
#     epoch = 99000
#     model_dir = '/scratch_net/bmicdl01/chenx/PycharmProjects/refine_vae/'
#     model = VariationalAutoencoder(model_name=model_name, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
#                                    z_dim=Z_DIM)
#     sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#     image_matrix = tf.placeholder('float32', [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')
#     z_mean, z_stddev, z_res = model.encoder(image_matrix, is_train=True, reuse=False)
#     z_mean_v, z_stddev_v, z_res_v = model.encoder(image_matrix, is_train=False, reuse=True)
#     samples = tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
#     guessed_z = z_mean + (z_stddev * samples)
#     guessed_z_v = z_mean_v + (z_stddev_v + samples)
#     mu, _ = model.decoder(guessed_z, name='img', is_train=True, reuse=False)
#     mu_, _ = model.decoder(guessed_z_v, name='img', is_train=False, reuse=True)
#
#     saver = tf.train.Saver()
#     saver.restore(sess, model_dir + model_folder + model_name + ".ckpt-" + str(epoch))
#
#     image_size = 200.
#     dif = []
#     dif_vae = []
#     dif_vae_rel=[]
#     dif_prob = []
#     dif_naive = []
#     num=0
#     for ind in indices:
#         print(num, ind)
#         res = data['Scan'][ind:ind+64]
#         res = res.reshape(-1, 200, 200)
#         mask = data['Mask'][ind:ind+64]
#         mask = mask.reshape(-1, 200, 200)
#
#         dim_res = res.shape
#
#         if dim_res[0] % 64:
#             dim_res_expand = 64 - (dim_res[0] % 64)
#             res_expand = np.zeros((dim_res_expand, dim_res[1], dim_res[2])) - 3.5
#             res_exp = np.append(res, res_expand, axis=0)
#             mask_exp = np.append(mask, np.zeros((dim_res_expand, dim_res[1], dim_res[2])), axis=0)
#         else:
#             res_exp = res
#             mask_exp = mask
#
#         res_exp = resize(res_exp, 128. / image_size)
#
#         cnt = 0
#         predicted_residuals = []
#         predicted_residuals_vae = []
#         predicted_residuals_vae_relative = []
#         prob_map = []
#
#         for batch in tl.iterate.minibatches(inputs=res_exp, targets=mask_exp,
#                                             batch_size=64, shuffle=False):
#             b_images, _ = batch
#             b_images = b_images[:, :, :, np.newaxis]
#             decoded = []
#             for i in range(100):
#                 decoded_vae = sess.run(mu, {image_matrix: b_images})
#                 decoded_vae_res = np.abs(b_images - decoded_vae)
#                 decoded.append(decoded_vae_res)
#             decoded = np.asarray(decoded).reshape(100, 64, 128, 128)
#             decoded = np.transpose(decoded, (1, 2, 3, 0))
#             # z_mu, z_dev = sess.run([z_mean, z_stddev], {image_matrix: b_images})
#             # decoded_vae_mu = sess.run(decoded_mu, {guessed_z:z_mu})
#             #decoded_vae_mu = np.mean(decoded, axis=-1, keepdims=True)
#             #decoded_vae_std = np.std(decoded, axis=-1, keepdims=True)
#
#             decoded_vae_median = np.median(decoded, axis=-1, keepdims=True)
#
#             #print(decoded_vae_mu.shape, decoded_vae_std.shape)
#             residuals_vae = np.abs(b_images - decoded_vae)
#             residuals_vae = residuals_vae.astype("float32")
#             decoded_res = sess.run(z_res, {image_matrix: b_images})
#             # decoded_res = sess.run(resid, {residuals_matrix: residuals_vae})
#             residuals = np.abs(residuals_vae - decoded_res)
#             residuals_relative = residuals_vae - decoded_res
#             predicted_residuals.extend(residuals)
#             predicted_residuals_vae.extend(residuals_vae)
#             predicted_residuals_vae_relative.extend(residuals_relative)
#             # probability_map = np.log(normpdf(decoded_res, decoded_vae_mu, decoded_vae_std) + 1e-10)
#             # probability_map = z_score(decoded_res, decoded_vae_mu, decoded_vae_std)
#             # probability_map[b_images == -3.5] = probability_map.min()
#
#             #decoded_mad_map = np.median(np.abs(decoded - decoded_vae_median), axis=-1, keepdims=True)
#             residuals_mad_map = mad_score(decoded_res, decoded_vae_median)
#             #dif_mad_map = np.abs(decoded_mad_map - residuals_mad_map)
#
#             #mad_map = np.array([nldenoise(i) for i in mad_map])
#             #mad_map = mad_map[:, :, :, np.newaxis]
#             # probability_map[b_images == -3.5] = probability_map.min()
#             residuals_mad_map[b_images == -3.5] = 0
#
#             prob_map.extend(residuals_mad_map)
#             cnt += 1
#         # prob_map_n = 1.-prob_map
#
#         predicted_residuals_vae = np.asarray(predicted_residuals_vae).reshape(res_exp.shape[0], 128, 128)
#         predicted_residuals = np.asarray(predicted_residuals).reshape(res_exp.shape[0], 128, 128)
#
#         predicted_residuals_vae = resize(predicted_residuals_vae[:dim_res[0]], image_size / 128.)
#         predicted_residuals = resize(predicted_residuals[:dim_res[0]], image_size / 128.)
#
#         predicted_residuals_vae_relative = np.asarray(predicted_residuals_vae_relative).reshape(res_exp.shape[0], 128,
#                                                                                                 128)
#         predicted_residuals_vae_relative = resize(predicted_residuals_vae_relative[:dim_res[0]], image_size / 128.)
#
#         prob_map = np.asarray(prob_map).reshape(res_exp.shape[0], 128, 128)
#         prob_map = resize(prob_map[:dim_res[0]], image_size / 128.)
#
#         dif.extend(predicted_residuals[mask==1])
#         dif_vae.extend(predicted_residuals_vae[mask==1])
#         dif_vae_rel.extend(predicted_residuals_vae_relative[mask==1])
#         dif_prob.extend(prob_map[mask==1])
#         dif_naive.extend(res[mask==1])
#         num+=1
#
#     segment_threshold = determine_threshold(dif, fprate)
#     print("res prediction threshold:{}".format(segment_threshold))
#     segment_threshold = determine_threshold(dif_vae, fprate)
#     print("vae threshold:{}".format(segment_threshold))
#     segment_threshold = determine_threshold(dif_vae_rel, fprate)
#     print("relative threshold:{}".format(segment_threshold))
#     segment_threshold = determine_threshold(dif_prob, fprate)
#     print("prob map threshold:{}".format(segment_threshold))
#     segment_threshold = determine_threshold(dif_naive, fprate)
#     print("naive threshold:{}".format(segment_threshold))
