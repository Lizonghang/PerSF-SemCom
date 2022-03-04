import numpy as np
from .textmatch.models.text_embedding.model_factory_sklearn import ModelFactory


class TextMatcher:
    def __init__(self, original_output):
        self.num_persons = 7
        assert self.num_persons <= 7, "Maximum number of persons: 7"
        self.mf = ModelFactory()
        self.triplet_dict = self._collect_all_words(original_output)
        self.triplet_dict_per_person = None
        self.query_text = self._automatically_generate_query_text(original_output)
        self.mf.init(words_dict=self.triplet_dict, update=True)
        self.output = {}
        self.repeat_counter = 0

    def _collect_all_words(self, semsal_output):
        triplet_list_all = []

        for img_name in semsal_output.keys():
            merged_output_ = semsal_output[img_name]
            query_ids = merged_output_["queries"]

            for pid in range(self.num_persons):
                triplet_list = [merged_output_["reltr_output"][qid.item()]['semantic']
                                for qid in query_ids]
                for tri in triplet_list:
                    if tri not in triplet_list_all:
                        triplet_list_all.append(tri)

        triplet_dict_all = {}
        for idx in range(len(triplet_list_all)):
            triplet_dict_all[idx] = triplet_list_all[idx]
        return triplet_dict_all

    def _automatically_generate_query_text(self, semsal_output):
        query_text = {}
        for pid in range(self.num_persons):
            query_text[pid] = {}

        for img_name in semsal_output.keys():
            merged_output_ = semsal_output[img_name]
            query_ids = merged_output_["queries"]

            for pid in range(self.num_persons):
                priority = np.array(merged_output_[pid]["priority"])
                query_ids_sorted = query_ids[np.argsort(-priority)]
                qid = query_ids_sorted[0].item()
                query_text[pid][img_name] = merged_output_["reltr_output"][qid]["semantic"]

        return query_text

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

    def receive(self, received_output):
        self.triplet_dict_per_person = self._reformat_received_output(received_output)

    def fit(self):
        assert self.triplet_dict_per_person, "No message received"

        output = {}
        for pid in range(self.num_persons):
            personal_preds = self.triplet_dict_per_person[pid]
            output[pid] = {}

            for img_name, item in personal_preds.items():
                query_text = self.query_text[pid][img_name]
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
                    "query_text": query_text,
                    "scores": triplet_scores,
                    "dropped": triplet_dropped}

        return output

    def eval(self, match_scores, score_func=np.max):
        for pid in range(self.num_persons):
            personal_scores = match_scores[pid]

            if pid not in self.output:
                self.output[pid] = {}

            for img_name, item in personal_scores.items():
                scores = [score for triplet, score in item["scores"]]

                if img_name not in self.output[pid]:
                    self.output[pid][img_name] = {}

                self.output[pid][img_name]["match_score"] = round(
                    (self.output[pid][img_name].get("match_score", 0.)
                     * self.repeat_counter + score_func(scores)) \
                    / (self.repeat_counter + 1), 5)

        self.repeat_counter += 1