# %%
import numpy as np
import math, pandas
import scipy
from scipy import sparse
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
import gudhi as gd
from gudhi.weighted_rips_complex import WeightedRipsComplex
from gudhi.dtm_rips_complex import DTMRipsComplex
from ripser import ripser
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers
from gudhi.tensorflow import RipsLayer
from sys import exit
import seaborn as sns
from utilyze import *
import pickle




def Collect_jacobian_v2(dataset, jacobian_method, xrange, yrange, number_pixel=20, gauss_sigma=0.01, weight_func = ConstantWeight(), max_edge_length=2., homology_dimensions=[1],normalize=False, jacobian_normalize=True):
    l = len(dataset)
    number_pts = dataset[0].shape[0]
    jacobians = np.zeros((l, number_pixel**2, (number_pts)*2))
    for index, pc in enumerate(dataset): 

        J = jacobian_method(pc, xrange, yrange, number_pixel, gauss_sigma, weight_func, max_edge_length, homology_dimensions,normalize)
        _, Sigma, _ = scipy.linalg.svd(J)
        if np.max(Sigma)>0 and jacobian_normalize:
            J /= np.max(Sigma)
        jacobians[index,:,:] = J
    return jacobians



def compute_pbnorm_perturb_Rips(dataset, perturbed_dataset, xrange, yrange, number_pixel, gauss_sigma, weight_func, max_edge_length):
    '''
    Computes the average pull-back norm for a given perturbation
    '''
    JAC = Collect_jacobian_v2(dataset = dataset,
                            jacobian_method = get_jacobian_rips,
                            xrange = xrange,
                            yrange = yrange, 
                            number_pixel = number_pixel,
                            gauss_sigma = gauss_sigma,
                            weight_func = weight_func,
                            max_edge_length = max_edge_length, 
                            homology_dimensions = [1],
                            normalize = False) 
    Pbn =[]
    for i, pc in enumerate(dataset):
        jacobian = JAC[i]
        perturbed_pc = perturbed_dataset[i]
        vec_perturb = (perturbed_pc-pc).reshape(-1,1)
        vec_perturb /= np.linalg.norm(vec_perturb, 2)
        pullback_norm = np.linalg.norm(jacobian @ vec_perturb, 2)
        Pbn.append(pullback_norm)
    return np.mean(np.array(Pbn))




# Example of usage:

# import numpy as np
# import math, pandas
# import scipy
# from scipy import sparse
# from sklearn.neighbors import KDTree
# from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.model_selection import train_test_split
# import gudhi as gd
# from gudhi.weighted_rips_complex import WeightedRipsComplex
# from gudhi.dtm_rips_complex import DTMRipsComplex
# from ripser import ripser
# import matplotlib.pyplot as plt
# import time
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# import tensorflow.keras.layers as layers
# from gudhi.tensorflow import RipsLayer
# from sys import exit
# import seaborn as sns
# from utilyze import *
# import pickle

# # Generate original data
# number_pts = 150
# alist = np.linspace(.50, .9, 5)
# wlist = np.linspace(3, 10, 8)
# Pointclouds = [ RFP(a, w, 0, number_pts,normalize=True) for a in alist for w in wlist]

# # Generate perturbed data
# Pointclouds_perturb = []
# for i, pc in enumerate(Pointclouds):
#     Pointclouds_perturb.append(noising(pc,10))

# # Parameters for Rips filtration
# xrange = [0.,1.]
# yrange = [0.,1.]
# number_pixel = 20
# gauss_sigma = 0.001
# maximal_edge_length_rips = 1.
# linear_weighting = Weight(method='linear',b=yrange[1])

# pbn_perturb_func.compute_pbnorm_perturb_Rips(Pointclouds, Pointclouds_perturb, xrange, yrange, number_pixel, gauss_sigma, linear_weighting, maximal_edge_length_rips) 
