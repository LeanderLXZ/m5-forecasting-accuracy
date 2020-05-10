from easydict import EasyDict

__C = EasyDict()

# ===========================================

__C.data_path = '../data/'
__C.source_path = __C.data_path + 'source_data'
__C.preprocessed_path = __C.data_path + 'preprocessed_data/'
__C.result_path = '../results'
__C.log_path = '../logs'

# ===========================================

# get config by: from config import cfg
cfg = __C
