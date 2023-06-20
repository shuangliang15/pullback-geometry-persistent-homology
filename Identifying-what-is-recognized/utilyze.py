import numpy as np
import math, pandas
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
from sys import exit

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

def RFP(a,w,t,number_pts,normalize=False):
    theta = np.linspace(0,2*np.pi-1e-5,number_pts) #Avoid two same pts
    r = 1+a*np.sin(w*(theta-t))
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    X = np.array([x, y]).T
    if normalize:
        X = (X-np.min(X))/(np.max(X)-np.min(X))
    return X

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


class LinearWeight:
    def __init__(self, b):
        self.b = b
    
    def compute_weight(self, x):
        x = tf.stop_gradient(x)
        weights = x[:,1] / self.b
        return weights

class ConstantWeight:
    def __init__(self):
        pass
    
    def compute_weight(self, x):
        x = tf.stop_gradient(x)
        return tf.ones_like(x[:,1])

class GaussianWeight:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def compute_weight(self, x):
        x = tf.stop_gradient(x)
        pers = x[:,1]
        weights = tf.exp(-(pers - self.mu) ** 2 /(2 * self.sigma))
        return weights
    
class Weight:
    def __init__(self, method='linear', mu=None, sigma=None, b=None):
        self.method = method
        if method == 'gaussian':
            self.weight_func = GaussianWeight(mu,sigma)
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


def pd2pi_inf(pd_inf, xrange, yrange, number_pixel=20, gauss_sigma=0.01, weight_func = ConstantWeight(), normalize=False):
    x0,x1 = xrange
    y0,y1 = yrange

    # weight no grad
    a = tf.reshape(pd_inf,(-1,1))
    pd = tf.concat((a, tf.ones_like(a)),1) # pc is normalized in [0,1]
    pd_transformed = transform(pd)
    weights = weight_func.compute_weight(pd_transformed)


    pi = tf.zeros((number_pixel,number_pixel), dtype=tf.float32)
    xx ,yy = tf.cast(tf.linspace(x0, x1, number_pixel), tf.float32), tf.cast(tf.linspace(y0, y1, number_pixel), tf.float32)
    grid_x, grid_y = tf.meshgrid(xx, yy, indexing='ij')
    for i in range(pd_inf.shape[0]):
        birth = pd_inf[i]
        pers = 1.-birth
        pi += weights[i] * tf.exp( ((grid_x-birth)**2 + (grid_y-pers)**2)/(-2*gauss_sigma) )
    pi = tf.image.rot90(tf.reshape(pi,(number_pixel,number_pixel,1)), k=1)
    pi = tf.reshape(pi, [-1])
    return pi


# plot functions
def plotpc(pc, simplex=[], legend=False, colormap = False, color=[]):
    if colormap:
        plt.scatter(pc[:,0], pc[:,1], s=8, c=color)
    else:
        plt.scatter(pc[:,0], pc[:,1], s=8)
    for pt in simplex:
        plt.scatter(pc[pt][0], pc[pt][1], label=str(pt), s=8)
    limit = np.max(abs(pc))
    plt.xlim(-limit-0.1,limit+0.1)
    plt.ylim(-limit-0.1,limit+0.1)
    if legend:
        plt.legend()

def plot_lines(pc, indices):
    fig, ax = plt.subplots()
    for idx in indices:
        x_vals = [pc[idx[0]][0], pc[idx[1]][0]]
        y_vals = [pc[idx[0]][1], pc[idx[1]][1]]
        ax.plot(x_vals, y_vals, color='lightblue')
    ax.scatter(pc[:,0],pc[:,1],s=5,c='grey')
    plt.show()

def plot_filtration_under(pc, st, fil_ts):
    st2 = st.copy()
    st2.prune_above_filtration(fil_ts)
    edges = []
    for simplex, filtration_value in st2.get_filtration():
        if len(simplex) == 2:  # Only consider 1-dimensional simplices
            edges.append(simplex)
    plot_lines(pc, edges)


# lowerstar
def lowerstar_fil_val(st, fil_vtx):
    F = {}
    for simplex,_ in st.get_filtration():
        if len(simplex)==1:
            F[str(simplex)] = fil_vtx[simplex]
        else:
            fil_val = tf.math.reduce_max([fil_vtx[vertex] for vertex in simplex])
            F[str(simplex)] = fil_val
    return F

def fil_val_edge_triangle(st, fil_vtx, DX, p=np.inf):
    F = {}
    for simplex,_ in st.get_filtration():
        if len(simplex)==1:
            F[str(simplex)] = fil_vtx[simplex] * 2
        elif len(simplex)==2:
            if p == np.inf:
                max_vtx_fil_val = tf.math.reduce_max([fil_vtx[vertex] for vertex in simplex])
                F[str(simplex)] = tf.math.reduce_max([max_vtx_fil_val, DX[simplex[0],simplex[1]]])
            elif p == 1:
                sum_vtx_fil_val = tf.math.reduce_sum([fil_vtx[vertex] for vertex in simplex])
                F[str(simplex)] = sum_vtx_fil_val + DX[simplex[0],simplex[1]]
                
        elif len(simplex)==3:
            fil_val = tf.math.reduce_max([F[str(edge)] for edge,_ in st.get_boundaries(simplex)])
            F[str(simplex)] = fil_val
    return F


def select_essential(pairs):
    pairs_ess = []
    pairs_inf = []
    for pair in pairs:
        if len(pair[1])!=0:
            pairs_ess.append(pair)
        else:
            pairs_inf.append(pair)
    return pairs_ess, pairs_inf

def dtm_fil_vtx(x, m=0.02):
    # idx
    points = tf.reshape(x,(-1,2)).numpy()
    k = math.floor(m * points.shape[0]) + 1
    kdt = KDTree(points, leaf_size = 30, metric = "euclidean")
    idxs = kdt.query(points, k, return_distance = False)

    # distmat
    X = tf.reshape(x,(-1,2))
    D = tf.norm(tf.expand_dims(X, 1)-tf.expand_dims(X, 0), axis=2)
    
    #compute fil val
    F = []
    for i,idx in enumerate(idxs): # eg idx = [0 149 148 1]
        F.append( tf.math.reduce_mean([D[i,j] for j in idx if j!= i], axis=0, keepdims=True))
    return tf.concat(F,0)



# Features
phi = np.array(np.pi/2)
s = np.sin(phi)
c = np.cos(phi)
rot = np.array([[c, -s],
                [s, c]])

def rotate(pc, t):
    phi = np.array(t)
    s = np.sin(phi)
    c = np.cos(phi)
    rott = np.array([[c, -s],
                    [s, c]])
    return pc @ rott.T

def dilation(pc, t):
    A = np.array([[1+t, 0],
                  [0, 1+t]])
    return pc @ A.T

def translation(pc, t):
    return pc + t

def shear(pc, t):
    A = np.array([[1, t],
                  [0, 1]])
    return pc @ A.T

def stretch_x(pc, t):
    A = np.array([[1+t, 0],
                  [0, 1]])
    return pc @ A.T

def noising(pc, s):
    np.random.seed(seed=1)
    mean = [0, 0]
    cov = [[s/10000, 0], [0, s/10000]]
    PC = np.copy(pc)
    for i in range(pc.shape[0]):
        pt = pc[i]
        PC[i] = pt + np.random.multivariate_normal(mean, cov)
    return PC

def tangent(pc,s):
    vec = np.zeros_like(pc)
    for i in range(pc.shape[0]):
        if i==pc.shape[0]-1:
            vector = pc[0,:] - pc[i,:]
        else:
            vector = pc[i+1,:] - pc[i,:]
        if np.linalg.norm(vector,2) == 0:
            vec[i,:] = np.array([0.,0.])
        else:
            vec[i,:] = vector/(np.linalg.norm(vector,2))
    vec /= np.linalg.norm(vec)
    PC = np.copy(pc)
    PC = pc+s*vec
    return PC

def normal(pc,s):
    vec = np.zeros_like(pc)
    for i in range(pc.shape[0]):
        if i==pc.shape[0]-1:
            vector = pc[i,:]-pc[0,:]
        else:
            vector = pc[i+1,:] - pc[i,:]
        vector = vector @ rot.T
        if np.linalg.norm(vector,2) == 0:
            vec[i,:] = np.array([0.,0.])
        else:
            vec[i,:] = vector/(np.linalg.norm(vector,2))
    vec /= np.linalg.norm(vec)
    PC = np.copy(pc)
    PC = pc+s*vec
    return PC

def wiggly(pc, s, w=31):
    PC = np.zeros_like(pc)
    l = pc.shape[0]
    wave = s*np.sin(w*np.linspace(0,2*np.pi,l))
    normal_vec = normal(pc,1)-pc
    for i in range(l):
        vec = wave[i]*normal_vec[i,:]
        PC[i]=pc[i]+vec
    return PC

from scipy.spatial import ConvexHull
def convex(pc,t):
    hull = ConvexHull(pc)
    hullpts = np.sort(hull.vertices)
    index = -1
    vec = np.zeros_like(pc)
    #print(hullpts)
    for i in range(pc.shape[0]):
        if i not in hullpts:
            pt = pc[i]
            if index == hullpts.shape[0]-1:
                hullpt0 = pc[hullpts[index]]
                hullpt1 = pc[hullpts[0]]
                #print(i,hullpts[index],hullpts[0])
            else:
                hullpt0 = pc[hullpts[index]]
                hullpt1 = pc[hullpts[index+1]]
                #print(i,hullpts[index],hullpts[index+1])
            normal = (hullpt0-hullpt1) @ rot.T
            normal /= np.linalg.norm(normal)
            vec[i] = ((hullpt0-pt).T @ normal )* normal
        elif i in hullpts:
            index += 1
        #print(i, hullpts[index])
    PC = np.copy(pc)
    PC = pc + t*vec
    return PC



def get_jacobian_rips(pc, xrange, yrange, number_pixel=20, gauss_sigma=0.01, weight_func = ConstantWeight(), max_edge_length=2., homology_dimensions=[0],normalize=False, output_PI = False):
    x = tf.Variable(pc.reshape(-1,),dtype=tf.float32,trainable=True)
    rl = RipsLayer(maximum_edge_length=max_edge_length, homology_dimensions=homology_dimensions)
    with tf.GradientTape() as tape:
        pc = tf.reshape(x,(-1,2))
        pd = rl.call(pc)[0][0]
        y = pd2pi(pd,xrange, yrange,number_pixel,gauss_sigma,weight_func,normalize)
    J = tape.jacobian(y, x)
    if output_PI:
        output = (J, y, transform(pd))
    else:
        output = J
    return output

def get_jacobian_height(pc, xrange, yrange, number_pixel=20, gauss_sigma=0.01, weight_func = ConstantWeight(), max_edge_length=.3, homology_dimensions=[1], normalize=False, output_PI = False):
    # complex
    sc = gd.RipsComplex(pc, max_edge_length = max_edge_length)
    st = sc.create_simplex_tree(max_dimension = 2)
    st.reset_filtration(0.) #set all filval to 0.

    x = tf.Variable(pc.reshape(-1,), dtype=tf.float32, trainable=True)
    with tf.GradientTape() as tape:
        # vertex fil val
        fil_vtx = tf.reshape(x,(-1,2))[:,0] - 0.0
        # all simplex fil val
        F = lowerstar_fil_val(st, fil_vtx)
        # paring
        for i in range(st.num_vertices()):
            st.assign_filtration([i], fil_vtx[i].numpy())
        st.make_filtration_non_decreasing()
        st.compute_persistence()#min_persistence=0.
        pairs_raw = st.persistence_pairs() # e.g. [([2], [2, 1]), ([0], [])]
        _, pairs_inf = select_essential(pairs_raw) # e.g. [([2], [2, 1])]
        # plug in
        pd_inf = [F[str(sorted(pair[0]))] for pair in pairs_inf if len(pair[0])-1 == homology_dimensions[0]]
        pd_inf = tf.reshape(pd_inf,-1) # this is a vector
        # pi
        y = pd2pi_inf(pd_inf,xrange, yrange,number_pixel,gauss_sigma,weight_func,normalize)
    J = tape.jacobian(y, x)

    a = tf.reshape(pd_inf,(-1,1))
    pd = tf.concat((a, tf.ones_like(a)),1) # pc is normalized in [0,1]
    pd = transform(pd)

    if output_PI:
        output = (J, y, pd)
    else:
        output = J
    return output

def get_jacobian_dtm(pc, xrange, yrange, number_pixel=20, gauss_sigma=0.01, weight_func = ConstantWeight(), max_edge_length=.3, homology_dimensions=[1], normalize=False, output_PI = False):
    # complex
    sc = gd.RipsComplex(pc, max_edge_length = max_edge_length)
    st = sc.create_simplex_tree(max_dimension = 2)
    st.reset_filtration(0.) #set all filval to 0.

    x = tf.Variable(pc.reshape(-1,), dtype=tf.float32, trainable=True)
    with tf.GradientTape() as tape:
        # Distance matrix
        X = tf.reshape(x,(-1,2))
        DX = tf.norm(tf.expand_dims(X, 1)-tf.expand_dims(X, 0), axis=2)
        # vertex fil val
        fil_vtx = dtm_fil_vtx(x, m=0.02)
        # all simplex fil val
        F = fil_val_edge_triangle(st, fil_vtx, DX, p=1)
        # paring
        for simplex,_ in st.get_filtration():
            st.assign_filtration(simplex, F[str(simplex)].numpy())
        st.compute_persistence()#min_persistence=0.
        pairs_raw = st.persistence_pairs() # e.g. [([2], [2, 1]), ([0], [])]
        pairs_ess, _ = select_essential(pairs_raw) # e.g. [([2], [2, 1])]

        # plug in
        pd_raw_ess = [[F[str(sorted(simplex))] for simplex in pair] for pair in pairs_ess if len(pair[0])-1 == homology_dimensions[0]]
        pd_ess = tf.reshape(tf.concat(pd_raw_ess,0),(-1,2))
        # pi
        y = pd2pi(pd_ess,xrange,yrange,number_pixel,gauss_sigma,weight_func,normalize)
    J = tape.jacobian(y, x)
    pd = transform(pd_ess)
    if output_PI:
        output = (J, y, pd)
    else:
        output = J
    return output

def Collect_jacobian(dataset, jacobian_method, xrange, yrange, number_pixel=20, gauss_sigma=0.01, weight_func = ConstantWeight(), max_edge_length=2., homology_dimensions=[1],normalize=False, jacobian_normalize=True):
    '''
    Computes the jacobians for a whole dataset.
    Each of the jacobian is computed with jacobian_method.
    Input:
        dataset:    Pointcloud dictionary          
    Output: 
        jacobians:  len(dataset) x number_pixel**2 x number_pts*2 numpy array
    '''
    l = len(dataset)
    number_pts = dataset[list(dataset.keys())[0]].shape[0]
    jacobians = np.zeros((l, number_pixel**2, (number_pts)*2))
    for index, (_, pc) in enumerate(dataset.items()): 
        #print(index)
        J = jacobian_method(pc, xrange, yrange, number_pixel, gauss_sigma, weight_func, max_edge_length, homology_dimensions,normalize)
        _, Sigma, _ = scipy.linalg.svd(J)
        if np.max(Sigma)>0 and jacobian_normalize:
            J /= np.max(Sigma)
        jacobians[index,:,:] = J
    return jacobians


def Collect_jacobian_and_rank_and_pi(dataset, jacobian_method, xrange, yrange, number_pixel=20, gauss_sigma=0.01, weight_func = ConstantWeight(), max_edge_length=2., homology_dimensions=[1],normalize=False):

    number_pt = dataset[0].shape[0]
    l = len(dataset)
    jacobians = np.zeros((l, number_pixel**2, (number_pt)*2))
    pis = np.zeros((l, number_pixel**2))
    ranks = []
    for index, pc in enumerate(dataset): 
        J, pi, _ = jacobian_method(pc, xrange, yrange, number_pixel, gauss_sigma, weight_func, max_edge_length, homology_dimensions,normalize, output_PI=True)
        _, Sigma, _ = scipy.linalg.svd(J)
        # Normalize
        if np.max(Sigma)>0:
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
        pc = tf.reshape(x,(-1,2))
        pd = rl.call(pc)[0][0]
        pi = pd2pi(pd,xrange, yrange,number_pixel,gauss_sigma,weight_func,normalize)
        pis[index,:] = pi
    return pis
