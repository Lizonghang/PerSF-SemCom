import os
import torch
import numpy as np


class Constant():
    base_dir = '/root/TextMatch/modelfile/'
    STOPWORDS_FILE = os.path.join(base_dir, 'text_model_file/stop_words/stop_words.txt')
    NGRAM_TFIDF_DIC_PATH = os.path.join(base_dir, 'text_model_file/ngram_tfidf_modelfile/ths_dict.dict')
    NGRAM_TFIDF_MODEL_PATH = os.path.join(base_dir, 'text_model_file/ngram_tfidf_modelfile/ths_tfidf.model')
    NGRAM_TFIDF_INDEX_PATH = os.path.join(base_dir, 'text_model_file/ngram_tfidf_modelfile/ths_tfidf.index')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
