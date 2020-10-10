import numpy as np
import nibabel as nib
import sys
sys.path.append('/Users/medimwiz/Development/pymed/Segmentation/')
sys.path.append('/Users/medimwiz/Development/pymed/Tools/')
import HistogramEqualization as HE
import Tools as TO
import EM as EM
from scipy import special, weave

def PVPosteriors(v_j, t, MU, SIGMA, sig_noise, M):
    # assuming unimodal
    mu = np.zeros(M)
    C  = np.eye(M)
    sofar = 0
    for i in range(len(t)):
        num = (t[i]*M).astype(np.int)
        mu[sofar:sofar+num] = MU[0,i]
        for j in range(sofar,sofar+num):
            C[j,j] = SIGMA[0,0,i] + 0.1
        sofar += num
    mu = mu[:,np.newaxis]
    OneV = np.ones(M)[:,np.newaxis]
    OneM = np.dot(OneV, OneV.T)
    numerator = (sig_noise + np.dot(OneV.T, np.dot(C,OneV)))
    Cv = C - np.dot(C,np.dot(OneM,C)) / numerator
    muv = mu + v_j[np.newaxis,:]*np.dot(C,OneV)/numerator
    muv = muv - np.dot(np.dot(C, OneM),mu) / numerator

    return muv, Cv

def IPosteriorGivent(v_i, v_j, t, MU, SIGMA, sig_noise, M, trans):
    # assuming unimodal
    muv, Cv = PVPosteriors(v_j, t, MU, SIGMA, sig_noise, M)
    a = np.zeros([M,1])
    b = np.zeros([M,1])
    sofar = 0
    for i in range(len(t)):
        num = (t[i]*M).astype(np.int)
        a[sofar:sofar+num,0] = trans[0,i]
        b[sofar:sofar+num,0] = trans[1,i]
        sofar += num
    mui = np.dot(muv.T, a) + b.sum()
    Ci  = np.dot(a.T, np.dot(Cv,a))
    D = v_i[np.newaxis,:] - mui
    
    return mui, Ci, 1./np.sqrt(2*np.pi*Ci[0,0])*np.exp(-0.5 * D**2 / Ci[0,0])

def GMMPVPosteriors(v_j, t, MU, SIGMA, PI, M, outLambda=None):
    # assuming unimodal
    ncomp = t.shape[0] # number of components
    nprop = t.shape[1] # number of proportions
    nsamp = v_j.shape[0] # number of samples
    if outLambda is None:
        T = np.zeros([nsamp,nprop])
    else:
        T = np.zeros([nsamp, nprop+1]) # outlier class is at the end
    for k in range(nprop):
        SIGMAt = M*(SIGMA * t[:,k][np.newaxis,np.newaxis,:]).sum(axis=2)
        MUt = M*(MU * t[:,k][np.newaxis,:]).sum(axis=1)
        T[:,k] = PI[k] * EM.Gaussian(v_j, MUt, SIGMAt)
    if outLambda is not None:
        # Outlier component
        T[:,-1] = PI[-1] * outLambda
    T = T / T.sum(axis=1)[:,np.newaxis]
    return T

def IPosteriorGivenJ(v_i, v_j, t, MU, SIGMA, PI, sig_noise, M, trans, outLambda=None):
    # assuming unimodal.
    p_TgivenJ = GMMPVPosteriors(v_j[:,np.newaxis], t, MU, SIGMA + sig_noise / np.float(M), PI, M, outLambda=outLambda)
    nprop = t.shape[1]
    if outLambda is None:
        p_TandIgivenJ = np.zeros([v_j.size,v_i.size,nprop])
    else:
        p_TandIgivenJ = np.zeros([v_j.size,v_i.size,nprop+1])
    for n in range(nprop):
        p_IgivenTandJ = IPosteriorGivent(v_i, v_j, t[:,n], MU, SIGMA, sig_noise, M, trans)[2]
        p_TandIgivenJ[:,:,n] = p_IgivenTandJ * p_TgivenJ[:,n][:,np.newaxis]
    if outLambda is not None:
        #D = v_i[np.newaxis,:] - v_j[:,np.newaxis]
        kernel_eval = np.exp(-(v_j[:,np.newaxis] - M*MU[0])**2/((SIGMA[0,:,:]+1)*M))
        vj_outlier = ((v_j[:,np.newaxis] * trans[0,:] + trans[1,:]) * \
        ( kernel_eval / kernel_eval.sum(axis=1)[:,np.newaxis])).sum(axis=1)
        D = v_i[np.newaxis,:] - vj_outlier[:,np.newaxis]
        p_IgivenTandJ = 1./np.sqrt(2*np.pi*100.)*np.exp(-0.5 * D**2 / 100.)
        p_TandIgivenJ[:,:,-1] = p_IgivenTandJ * p_TgivenJ[:,-1][:,np.newaxis]
    p_IgivenJ = p_TandIgivenJ.sum(axis=2)
    IgivenJ_map = v_i[p_IgivenJ.argmax(axis=1)] # MAP estimate
    IgivenJ_mean = (p_IgivenJ * v_i[np.newaxis,:]).sum(axis=1) / (p_IgivenJ.sum(axis=1)) # Mean estimate
    IgivenJ_std = np.sqrt((p_IgivenJ * (v_i[np.newaxis,:]-IgivenJ_mean[:,np.newaxis])**2).sum(axis=1)/(p_IgivenJ.sum(axis=1)))

    return IgivenJ_map, IgivenJ_mean, IgivenJ_std

def TransformIntensities(IgivenJ, VI):
    # applies the transformation IgivenJ to VI
    VI_FIN = VI.copy()
    code = """
    for (int x = 0; x < nx; x++){
        for (int y = 0; y < ny; y++){
            for (int z = 0; z < nz; z++){
                if (VI(x,y,z) > 0){
                    VI_FIN(x,y,z) = IgivenJ((int)(VI_FIN(x,y,z)));
                }
            }
        }
    }
    """
    nx,ny,nz = VI_FIN.shape
    weave.inline(code, ['VI_FIN','VI','nx','ny','nz','IgivenJ'],\
    type_converters=weave.converters.blitz,compiler='gcc')
    return VI_FIN

def TransformIntensitiesFloatingPoint(IgivenJ, VI, vj):
    # applies the transformation IgivenJ to VI
    VI_FIN = VI.copy()
    code = """
    int min_ind; float min_val;
    for (int x = 0; x < nx; x++){
        for (int y = 0; y < ny; y++){
            for (int z = 0; z < nz; z++){
                if (VI(x,y,z) > 0){
                    min_val = 100000;
                    for (int i = 0; i < nvj; i++){
                        if (((vj(i) - VI(x,y,z))*(vj(i) - VI(x,y,z))) < min_val){
                            min_val = (vj(i) - VI(x,y,z))*(vj(i) - VI(x,y,z));
                            min_ind = i;
                        }
                    }
                    VI_FIN(x,y,z) = IgivenJ(min_ind);
                }
            }
        }
    }
    """
    nx,ny,nz = VI_FIN.shape
    nvj = vj.size
    weave.inline(code, ['VI_FIN','VI','nx','ny','nz','IgivenJ','vj','nvj'],\
    type_converters=weave.converters.blitz,compiler='gcc')
    return VI_FIN
