import numpy as np
import nibabel as nib
import scipy.ndimage as ND
import uncertify.data.preprocessing.histogram_matching.KernelDensityEstimation as KDE


def ComputeHistograms(I, nbins=50, skip=50, kernel_size=50.):
    my = I.min()
    My = I.max()
    dy = My - my
    y = np.linspace(my - dy / 20., My, nbins)
    h = KDE.SilvermanWidthEstimate(np.float(I.size) / skip, 1) * kernel_size
    P = KDE.KernelDensityEstimate(KDE.GaussianKernel, I[range(0, I.size, skip), np.newaxis], y[:, np.newaxis], h)
    return y, P


def MeanIntensityThreshold(V):
    W = V[V > V.mean()]
    return W


def ComputeImageHistogram(I, nbins=50, skip=50, mean_threshold=True, mask=None, kernel_size=50.):
    if mask is not None:  # meaning there is a binary mask that will be applied to I before computing histogram
        J = I * mask
    else:
        J = I

    if mean_threshold:
        W = MeanIntensityThreshold(J.astype(np.double))
    else:
        W = J.reshape(np.prod(J.shape)).astype(np.double)

    y, P = ComputeHistograms(W, nbins=nbins, skip=skip, kernel_size=kernel_size)
    return (y, P)


def ComputePercentiles(H, L):
    # H is a histogram
    # L is a list of percentile values
    C = np.cumsum(H[1])
    C = C / C[-1]
    p = np.zeros(len(L))
    for n in range(len(L)):
        r = np.where(C >= L[n])[0][0]
        if r == 0 or r == len(H[0]):
            p[n] = H[0][r]
        else:
            # then we should do linear interpolation to avoid jagged results
            p[n] = H[0][r - 1] + (L[n] - C[r - 1]) / (C[r] - C[r - 1]) * (H[0][r] - H[0][r - 1])
    return p


def MultiLinearMap(J, PJ, Target):
    K = J.copy().astype(np.double)
    # for normal values that are within range
    for n in range(len(PJ) - 1):
        rows = (J >= PJ[n]) * (J <= PJ[n + 1])
        K[rows] = (J[rows] - PJ[n]).astype(np.double) / (PJ[n + 1] - PJ[n]) * (Target[n + 1] - Target[n]) + Target[n]
    # for values that are outside the ranges of the percentiles
    # if you use the entire image this normally should not happen.
    # if you use only the masks to compute PJ then this is likely to happen
    # in this case we simply extend the line towards the values that are outside.
    # lower values
    rows = J < PJ[0]
    K[rows] = (J[rows] - PJ[0]).astype(np.double) / (PJ[1] - PJ[0]) * (Target[1] - Target[0]) + Target[0]
    # higher values
    rows = J > PJ[-1]
    K[rows] = (J[rows] - PJ[-2]).astype(np.double) / (PJ[-1] - PJ[-2]) * (Target[-1] - Target[-2]) + Target[-2]
    return K


def MatchHistogramsTwoImages(I, J, L, nbins=50, skip=50, begval=0., finval=0.998, train_mask=None, test_mask=None):
    if type(I) == str:
        I = nib.load(I).get_data()
    if type(J) == str:
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


def MatchHistogramsWithMultipleTargets(I, J, L, nbins=50, skip=50, begval=0., finval=0.998, train_mask=None,
                                       test_mask=None):
    print("Computing population histogram")
    P = PopulationPercentiles(I, L, nbins=nbins, skip=skip, begval=begval, finval=finval, mask=train_mask)
    print("Matching test image")
    K = MatchHistogramsWithPopulation(P, J, nbins=nbins, skip=skip, begval=begval, finval=finval, mask=test_mask)
    return K


def MatchHistogramsWithMultipleTargetsMultipleMasks(I, J, L, train_masks, test_masks, nbins=50, skip=50, begval=0.,
                                                    finval=0.998, h=25.):
    print("Computing population histogram")
    Pmm = PopulationPercentilesMultipleMasks(I, L, train_masks, nbins=nbins, skip=skip, begval=begval, finval=finval)
    print("Matching test image to each mask")
    Kl = MatchHistogramsWithPopulationMultipleMasks(Pmm, J, test_masks, nbins=nbins, skip=skip, begval=begval,
                                                    finval=finval)
    print("Interpolating the final image from individual ones")
    K = InterpolateHistogramCorrection(Kl, test_masks, h=h)[0]
    return (K)


def MatchHistogramsWithPopulation(P, J, nbins=50, skip=50, begval=0., finval=0.998, mask=None):
    # if there is a mask then you compute the histogram on the mask but apply the
    # correction to everywhere in the image.
    if type(J) == str:
        J = nib.load(J).get_data()
    if type(mask) == str:
        mask = nib.load(mask).get_data()
    HJ = ComputeImageHistogram(J, nbins=nbins, skip=skip, mask=mask)
    L = P[1]
    PJ = ComputePercentiles(HJ, L)
    K = MultiLinearMap(J, PJ, P[0])
    return K


def MatchHistogramsWithPopulationMultipleMasks(P, J, masks, nbins=50, skip=50, begval=0., finval=0.998):
    numMasks = len(masks)
    K = []
    for l in range(numMasks):
        Kl = MatchHistogramsWithPopulation(P[l], J, nbins=nbins, skip=skip, begval=begval, finval=finval, mask=masks[l])
        K = K + [Kl]
    return K


def PopulationPercentilesMultipleMasks(J, L, masks, nbins=50, skip=50, begval=0., finval=0.998):
    numMasks = len(masks)
    P = []
    for l in range(numMasks):
        P = P + [PopulationPercentiles(J, L, nbins=nbins, skip=skip, begval=begval, mask=masks[l])]
    return P


def PopulationPercentiles(I, L, nbins=50, skip=50, begval=0., finval=0.998, mask=None):
    numIm = len(I)
    HI = []
    PI = []
    M = []
    if np.isscalar(L):
        L_ = np.linspace(begval, 1, L + 1)
        L_[-1] = finval
        L = L_
    # computing the percentiles for each image in the set
    for n in range(numIm):
        if type(I[n]) == str:
            I[n] = nib.load(I[n]).get_data()
        if mask is None:
            M = M + [None]
        elif type(mask[n]) == str:
            M = M + [nib.load(mask[n]).get_data()]
        else:
            M[n] = mask[n]

        HI = HI + [ComputeImageHistogram(I[n], nbins=nbins, skip=skip, mask=M[n])]

        PI = PI + [ComputePercentiles(HI[n], L)]

    # computing the average end points.
    PI = np.asarray(PI)
    m_begval = np.mean(PI[:, 0])
    m_finval = np.mean(PI[:, -1])
    # mapping the volumes to the average frame and computing the new values

    for n in range(numIm):
        K = MultiLinearMap(I[n], [PI[n, 0], PI[n, -1]], [m_begval, m_finval])
        HI[n] = ComputeImageHistogram(K, nbins=nbins, skip=skip, mask=M[n])
        PI[n, :] = ComputePercentiles(HI[n], L)
    PI = np.mean(PI, axis=0)
    return (PI, L)


def InterpolateHistogramCorrection(Kl, masks, h=25.):
    # reading masks:
    numMasks = len(masks)
    M = []
    for l in range(numMasks):
        if type(masks[l]) == str:
            M = M + [nib.load(masks[l]).get_data()]
        else:
            M[l] = masks[l]
    # computing the distance transforms for the masks:
    DM = DistanceTransformsForMasks(M)
    # computing the sum of distances
    SW = np.zeros(DM[0].shape)
    for l in range(numMasks):
        SW += np.exp(-DM[l] / h)
    # computing the interpolated and corrected image
    K = np.zeros(SW.shape)
    for l in range(numMasks):
        K += Kl[l] * np.exp(-DM[l] / h)
    SW[SW == 0] = 1e-15
    K = K / SW
    return K, SW


def DistanceTransformsForMasks(masks):
    numMasks = len(masks)
    Dmask = []
    for l in range(numMasks):
        Dmask = Dmask + [ND.distance_transform_edt(1 - masks[l])]
    return Dmask


def Map2UINT8(K):
    J = K.copy()
    J[K < 0] = 0.
    J[K > 255.] = 255.
    return J.astype(np.uint8)
