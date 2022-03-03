from ...config.constant import Constant as const
from .ngram_tf_idf_sklearn import NgramTfIdf


class ModelFactory(object):
    def __init__(self, ngram_tf_idf_model=NgramTfIdf):
        self.model = ngram_tf_idf_model(
            dic_path=const.NGRAM_TFIDF_DIC_PATH,
            tfidf_model_path=const.NGRAM_TFIDF_MODEL_PATH,
            tfidf_index_path=const.NGRAM_TFIDF_INDEX_PATH)

    def init(self, words_dict=None, update=False):
        if words_dict != None:
            self.id_lists, self.words_list = self._dic2list(words_dict)
        else:
            self.id_lists, self.words_list = None, None
        self.model.init(self.words_list, update)

    def _dic2list(self, words_dict):
        return list(words_dict.keys()), list(words_dict.values())

    def predict(self, words):
        return [round(score, 3)
                for score in self.model.predict(words)]
