from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# NGM model options
__C.NGM = edict()
__C.NGM.FEATURE_CHANNEL = 512
__C.NGM.SK_ITER_NUM = 10
__C.NGM.SK_EPSILON = 1e-10
__C.NGM.SK_TAU = 0.005
__C.NGM.MGM_SK_TAU = 0.005
__C.NGM.GNN_FEAT = [16, 16, 16]
__C.NGM.GNN_LAYER = 3
__C.NGM.GAUSSIAN_SIGMA = 1.
__C.NGM.SIGMA3 = 1.
__C.NGM.WEIGHT2 = 1.
__C.NGM.WEIGHT3 = 1.
__C.NGM.EDGE_FEATURE = 'cat' # 'cat' or 'geo'
__C.NGM.ORDER3_FEATURE = 'none' # 'cat' or 'geo' or 'none'
__C.NGM.FIRST_ORDER = True
__C.NGM.EDGE_EMB = False
__C.NGM.SK_EMB = 1
__C.NGM.GUMBEL_SK = 0 # 0 for no gumbel, other wise for number of gumbel samples
__C.NGM.UNIV_SIZE = -1
__C.NGM.POSITIVE_EDGES = True

__C.GMT = edict()

__C.GMT.MODE = 'node filter + edge bilinear'
__C.GMT.FILTER_LOW = 11
__C.GMT.FILTER_HIGH = 12
__C.GMT.BILINEAR_LOW = 11
__C.GMT.BILINEAR_HIGH = 12

__C.ATTN = edict()

__C.ATTN.SWITCH = False
__C.ATTN.DEPTH = 3
__C.ATTN.IN_DIM = [1, 16, 16]
__C.ATTN.HEAD = [1, 1, 2]
__C.ATTN.OUT_DIM = [16, 16, 16]
__C.ATTN.SK_EMB = 0
__C.ATTN.RECURRENCE = 1