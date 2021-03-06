from ...config.constant import Constant as const


class StopWords(object):
    def __init__(self, stopwords_file=const.STOPWORDS_FILE):
        self.stopwords = set([word.strip() for word in open(stopwords_file, 'r')])

    def del_stopwords(self, words):
        return [word for word in words if word not in self.stopwords]
