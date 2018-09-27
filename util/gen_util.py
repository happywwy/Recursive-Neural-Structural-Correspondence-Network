
import numpy as np

# - given a vector containing all parameters, return a list of unrolled parameters
# - specifically, these parameters are:
#   - rel_dict, dictionary of {dependency relation r: composition matrix W_r}
#   - Wv, the matrix for lifting a word embedding to the hidden space
#   - Wh, the matrix for lifting a hidden embedding to the relation space
#   - Wc, classification matrix
#   - b, b_c bias term
#   - We, the word embedding matrix

    
def unroll_scl_auto_params(arr, d, c, rc, len_voc, rel_list):

    mat_size = d * d
    #classification
    matClass_size = c * d
    relClass_size = rc * d
    rel_dict = {}
    ind = 0

    for r in rel_list:
        rel_dict[r] = arr[ind: ind + mat_size].reshape( (d, d) )
        ind += mat_size

    Wv = arr[ind : ind + mat_size].reshape( (d, d) )
    ind += mat_size
    
    Wh = arr[ind : ind + mat_size].reshape( (d, d) )
    ind += mat_size
    
    #Wc
    Wc = arr[ind : ind + matClass_size].reshape( (c, d) )
    ind += matClass_size

    b = arr[ind : ind + d].reshape( (d, 1) )
    ind += d
    
    #b_c
    b_c = arr[ind : ind + c].reshape( (c, 1) )
    ind += c

    We = arr[ind : ind + len_voc * d].reshape( (d, len_voc))

    return [rel_dict, Wv, Wh, Wc, b, b_c, We]


    
def unroll_params_auto_joint(arr, d, c, len_voc, rel_list):

    mat_size = d * d
    rel_dict = {}
    ind = 0

    for r in rel_list:
        rel_dict[r] = arr[ind: ind + mat_size].reshape( (d, d) )
        ind += mat_size

    Wv = arr[ind : ind + mat_size].reshape( (d, d) )
    ind += mat_size
    
    Wh = arr[ind : ind + mat_size].reshape( (d, d) )
    ind += mat_size
    
    b = arr[ind : ind + d].reshape( (d, 1) )
    ind += d

    
    We = arr[ind : ind + len_voc * d].reshape( (d, len_voc))

    return [rel_dict, Wv, Wh, b, We]
    
    

def roll_scl_auto_params(params, rel_list):
    (rel_dict, Wv, Wh, Wc, b, b_c, We) = params
    rels = np.concatenate( [rel_dict[key].ravel() for key in rel_list] )
    return np.concatenate( (rels, Wv.ravel(), Wh.ravel(), Wc.ravel(), b.ravel(), b_c.ravel(), We.ravel() ) )

    
def roll_params_auto_joint(params, rel_list):
    (rel_dict, Wv, Wh, b, We) = params
    rels = np.concatenate( [rel_dict[key].ravel() for key in rel_list] )
    return np.concatenate( (rels, Wv.ravel(), Wh.ravel(), b.ravel(), We.ravel() ) )
    

# randomly initialize all parameters
def gen_dtrnn_scl_auto_params(d, c, rc, rels):
    """
    Returns (dict{rels:[mat]}, Wv, Wh, Wc, b, b_c)
    """
    r = np.sqrt(6) / np.sqrt(2 * d + 1)
    r_Wc = 1.0 / np.sqrt(d)
    rel_dict = {}
    np.random.seed(3)
    for rel in rels:
	   rel_dict[rel] = np.random.rand(d, d) * 2 * r - r

    return (
	    rel_dict,
	    np.random.rand(d, d) * 2 * r - r,
        np.random.rand(d, d) * 2 * r - r,
          #Wc
          np.random.rand(c, d) * 2 * r_Wc - r_Wc,
          #b
	     np.zeros((d, 1)),
          #b_c
          np.random.rand(c, 1)
          )
          

          
# returns list of zero gradients which backprop modifies
def init_dtrnn_scl_auto_grads(rel_list, d, c, len_voc):

    rel_grads = {}
    for rel in rel_list:
	  rel_grads[rel] = np.zeros( (d, d) )

    return [
	    rel_grads,
	    np.zeros((d, d)),
        np.zeros((d, d)),
          np.zeros((c, d)),
	    np.zeros((d, 1)),
          np.zeros((c, 1)),
	    np.zeros((d, len_voc))
	    ]     


    
     
def init_joint_auto_grads(rel_list, d, c, len_voc):

    rel_grads = {}
    for rel in rel_list:
        rel_grads[rel] = np.zeros( (d, d) )

    return [
	    rel_grads,
	    np.zeros((d, d)),
        np.zeros((d, d)),
	    np.zeros((d, 1)),
	    np.zeros((d, len_voc))
	    ]
     

     


# random embedding matrix for gradient checks
def gen_rand_we(len_voc, d):
    r = np.sqrt(6) / np.sqrt(51)
    we = np.random.rand(d, len_voc) * 2 * r - r
    return we
