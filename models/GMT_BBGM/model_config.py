from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# BBGM model options
__C.BBGM = edict()
__C.BBGM.SOLVER_NAME = 'LPMP'
__C.BBGM.LAMBDA_VAL = 80.0
__C.BBGM.SOLVER_PARAMS = edict()
__C.BBGM.SOLVER_PARAMS.timeout = 1000
__C.BBGM.SOLVER_PARAMS.primalComputationInterval = 10
__C.BBGM.SOLVER_PARAMS.maxIter = 100
__C.BBGM.FEATURE_CHANNEL = 1024

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