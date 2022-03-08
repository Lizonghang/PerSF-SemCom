import numpy as np
from .textmatch.models.text_embedding.model_factory_sklearn import ModelFactory


class TextMatcher:
    def __init__(self, original_output, query_text, args):
        self.num_persons = args.num_persons
        self.args = args
        self.mf = ModelFactory()
        self.triplet_dict = self._collect_all_words(original_output)
        self.triplet_dict_per_person = None
        self.query_text = query_text
        self.mf.init(words_dict=self.triplet_dict, update=True)
        self.output = {}
        self.repeat_counter = 0

    def _collect_all_words(self, semsal_output):
        triplet_list_all = []

        for img_name in semsal_output.keys():
            merged_output_ = semsal_output[img_name]

            for pid in range(self.num_persons):
                query_ids = merged_output_[pid]["queries"]

                triplet_list = [merged_output_["reltr_output"][qid.item()]['semantic']
                                for qid in query_ids]

                for tri in triplet_list:
                    if tri not in triplet_list_all:
                        triplet_list_all.append(tri)

        triplet_dict_all = {}
        for idx in range(len(triplet_list_all)):
            triplet_dict_all[idx] = triplet_list_all[idx]
        return triplet_dict_all

    def _reformat_received_output(self, received_output):
        triplet_dict_per_person = {}
        for pid in range(self.num_persons):
            triplet_dict_per_person[pid] = {}

        for img_name in received_output.keys():
            merged_output_ = received_output[img_name]

            for pid in range(self.num_persons):
                triplet_received = merged_output_[pid]["triplet_received"]
                triplet_dropped = merged_output_[pid]["triplet_dropped"]

                triplet_dict_per_person[pid][img_name] = {}
                triplet_dict_per_person[pid][img_name]["triplet_received"] = triplet_received
                triplet_dict_per_person[pid][img_name]["triplet_dropped"] = triplet_dropped

        return triplet_dict_per_person

    def _score_func(self, scores, accurate_match=True):
        ret_score = np.max(scores)

        if accurate_match:
            ret_score = 1 if ret_score > 0.99 else 0

        return ret_score

    def receive(self, received_output):
        self.triplet_dict_per_person = self._reformat_received_output(received_output)

    def fit(self):
        assert self.triplet_dict_per_person, "No message received"

        output = {}
        for pid in range(self.num_persons):

            personal_preds = self.triplet_dict_per_person[pid]

            query_text = self.query_text[pid]
            output[pid] = {"query_text": query_text}

            for img_name, item in personal_preds.items():
                triplet_received = item["triplet_received"]
                triplet_dropped = item["triplet_dropped"]

                triplet_scores = []
                scores = self.mf.predict(query_text)
                for triplet, score in zip(self.triplet_dict.values(), scores):
                    if triplet in triplet_received:
                        triplet_scores.append((triplet, score))

                for triplet in triplet_dropped:
                    triplet_scores.append((triplet, 0.))

                output[pid][img_name] = {
                    "scores": triplet_scores,
                    "dropped": triplet_dropped}

        return output

    def eval(self, match_scores):
        for pid in range(self.num_persons):
            personal_scores = match_scores[pid]
            img_names = []

            if pid not in self.output:
                self.output[pid] = {}

            for img_name, item in personal_scores.items():
                if img_name == "query_text":
                    continue
                img_names.append(img_name)

                scores = [score for triplet, score in item["scores"]]

                if img_name not in self.output[pid]:
                    self.output[pid][img_name] = {}

                self.output[pid][img_name]["max_score"] = round(
                    (self.output[pid][img_name].get("max_score", 0.)
                     * self.repeat_counter + self._score_func(scores)) \
                    / (self.repeat_counter + 1), 5)

            self.output[pid]["mean_max_scores"] = np.round(np.mean(
                [self.output[pid][img_name]["max_score"]
                 for img_name in img_names]), 5)

        self.repeat_counter += 1
