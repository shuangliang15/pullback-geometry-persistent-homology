import numpy as np
import math, pandas, pickle
import scipy
from scipy import sparse
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import euclidean_distances
import gudhi as gd
from gudhi.weighted_rips_complex import WeightedRipsComplex
from gudhi.dtm_rips_complex import DTMRipsComplex
from ripser import ripser
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from gudhi.tensorflow import RipsLayer
import plotly.graph_objects as go
import ot.plot
import matplotlib.pylab as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import open3d as o3d
from scipy.special import factorial


def random_seed():
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value= 0

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value)
    # for later versions: 
    # tf.compat.v1.set_random_seed(seed_value)



def transform(pd):
    '''transfer (b,d) to (b,p)'''
    birth = tf.reshape(pd[:, 0], (-1,1))
    pers = tf.reshape(pd[:, 1]-pd[:,0], (-1,1))
    pd_transformed = tf.concat([birth, pers], axis=1)
    return pd_transformed

def normalizer(pd):
    '''normalize pd to unit square (use after transform)'''
    pd_vec = tf.reshape(pd, (-1,))
    pd_normalized = (pd_vec-tf.math.reduce_min(pd_vec)) / (tf.math.reduce_max(pd_vec)-tf.math.reduce_min(pd_vec))
    return tf.reshape(pd_normalized,(-1,2))


# weight_functions
# Note it takes transformed pd as inputs
class LinearWeight:
    def __init__(self, b):
        self.b = b
    
    def compute_weight(self, x):
        x = tf.stop_gradient(x)
        weights = x[:,1] / self.b
        return tf.identity(weights)

class ConstantWeight:
    def __init__(self):
        pass
    
    def compute_weight(self, x):
        x = tf.stop_gradient(x)
        return tf.ones_like(x[:,1])

class BetaWeighting:
    def __init__(self, m, s):
        self.m = m # mean
        self.s = s # concentration
    
    def compute_weight(self, x):
        scale = 1 # support = [0, 1/scale]

        x = tf.stop_gradient(x).numpy()
        pers = x[:,1]
        n = self.m * (1 - self.m) / self.s**2
        a = self.m * n
        b = (1-self.m) * n
        beta = factorial(a-1) * factorial(b-1)/factorial(a+b-1)
        weights = []
        for idx in range(len(pers)):
            if pers[idx] * scale >= 1:
                weights.append(0)
            else:    
                weights.append((scale * pers[idx])**(a-1) * (1- scale * pers[idx])**(b-1) / beta)
        
        weights = tf.constant(weights, dtype=float)
        return weights
    
class Weight:
    def __init__(self, method='linear', b=None, m=None, s=None):
        self.method = method
        if method == 'beta':
            self.weight_func = BetaWeighting(m, s)
        elif method == 'linear':
            self.weight_func = LinearWeight(b)
        elif method == 'constant':
            self.weight_func = ConstantWeight()
        else:
            raise ValueError("Invalid method specified: {}".format(method))
    
    def compute_weight(self, x):
        return self.weight_func.compute_weight(x)
    

def pd2pi(pd, xrange, yrange, number_pixel=20, gauss_sigma=0.01, weight_func = ConstantWeight(), normalize=False):
    pd_transformed = transform(pd)
    x0,x1 = xrange
    y0,y1 = yrange
    if normalize:
        pd_transformed = normalizer(pd_transformed)
        x0,x1 = [0.,1.]
        y0,y1 = [0.,1.]

    weights = weight_func.compute_weight(pd_transformed)
    pi = tf.zeros((number_pixel,number_pixel), dtype=tf.float32)

    xx ,yy = tf.cast(tf.linspace(x0, x1, number_pixel), tf.float32), tf.cast(tf.linspace(y0, y1, number_pixel), tf.float32)
    grid_x, grid_y = tf.meshgrid(xx, yy, indexing='ij')
    for i in range(pd.shape[0]):
        pt = pd_transformed[i]
        pi += weights[i] * tf.exp( ((grid_x-pt[0])**2 + (grid_y-pt[1])**2)/(-2*gauss_sigma) )
    pi = tf.image.rot90(tf.reshape(pi,(number_pixel,number_pixel,1)), k=1)
    pi = tf.reshape(pi, [-1])
    return pi

def plotpc(pc,color='blue'):
    fig = go.Figure(data=[go.Scatter3d(
        x=pc[:,0],
        y=pc[:,1],
        z=pc[:,2],
        mode='markers',
        marker=dict(
            size=1,
            color=color
        )
    )])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def get_jacobian_rips(pc, xrange, yrange, number_pixel=20, gauss_sigma=0.01, weight_func = ConstantWeight(), max_edge_length=2., homology_dimensions=[0],normalize=False,output_pi=False):
    x = tf.Variable(pc.reshape(-1,),dtype=tf.float32,trainable=True)
    rl = RipsLayer(maximum_edge_length=max_edge_length, homology_dimensions=homology_dimensions)
    with tf.GradientTape() as tape:
        pc = tf.reshape(x,(-1,3))
        pd = rl.call(pc)[0][0]
        y = pd2pi(pd,xrange, yrange,number_pixel,gauss_sigma,weight_func,normalize)
    J = tape.jacobian(y, x)
    if output_pi:
        output = (J, y)
    else:
        output = J
    return output


def Collect_jacobian_and_rank_and_pi(dataset, jacobian_method, xrange, yrange, number_pixel=20, gauss_sigma=0.01, weight_func = ConstantWeight(), max_edge_length=2., homology_dimensions=[1],normalize=False, normalize_jac=True, lapack_driver='gesdd'):
    '''
    Computes the jacobians for a whole dataset.
    Each of the jacobian is computed with jacobian_method.
    Input:
        dataset:    Pointcloud list
    Output: 
        jacobians:  len(dataset) x number_pixel**2 x number_pts*3 numpy array
    '''
    number_pt = dataset[0].shape[0]
    l = len(dataset)
    jacobians = np.zeros((l, number_pixel**2, (number_pt)*3))
    pis = np.zeros((l, number_pixel**2))
    ranks = []
    for index, pc in enumerate(dataset): 
        J, pi = jacobian_method(pc, xrange, yrange, number_pixel, gauss_sigma, weight_func, max_edge_length, homology_dimensions,normalize, output_pi=True)
        _, Sigma, _ = scipy.linalg.svd(J, lapack_driver=lapack_driver)
        # Normalize
        if normalize_jac ==True and np.max(Sigma)>0:
            J /= np.max(Sigma)
        jacobians[index,:,:] = J
        pis[index,:] = pi
        ranks.append(np.where(Sigma>1e-3)[0].shape[0])
    return jacobians, pis, ranks


def Collect_pi(dataset, xrange, yrange, number_pixel=20, gauss_sigma=0.01, weight_func = ConstantWeight(), max_edge_length=2., homology_dimensions=[1], normalize=False):
    l = len(dataset)
    pis = np.zeros((l, number_pixel**2))
    for index, pc in enumerate(dataset): 
        x = tf.Variable(pc.reshape(-1,),dtype=tf.float32,trainable=False)
        rl = RipsLayer(maximum_edge_length=max_edge_length, homology_dimensions=homology_dimensions)
        pc = tf.reshape(x,(-1,3))
        pd = rl.call(pc)[0][0]
        pi = pd2pi(pd,xrange, yrange,number_pixel,gauss_sigma,weight_func,normalize)
        pis[index,:] = pi
    return pis


def find_align(source_pc, target_pc, output_icp = False, max_correspondence_distance=.1):
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_pc)
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_pc)
    # Perform ICP
    icp_result = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    # Get the aligned source cloud
    aligned_source_cloud = source_cloud.transform(icp_result.transformation)
    pc1_regis = np.asarray(aligned_source_cloud.points)
    if output_icp:
        output = (pc1_regis, icp_result)
    else:
        output = pc1_regis
    return output
