from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.emb = edict()
__C.emb.JIEBA_FLAG = True
__C.emb.DEL_STOPWORD = False
