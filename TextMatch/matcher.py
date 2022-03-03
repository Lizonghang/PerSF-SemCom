import numpy as np
import torch

from .textmatch.models.text_embedding.model_factory_sklearn import ModelFactory
from .textmatch.config.constant import setup_seed

setup_seed(2)


class TextMatcher:
    def __init__(self, semsal_output, drop_mode):
        self.num_persons = 7
        assert self.num_persons <= 7, "Maximum number of persons: 7"
        self.mf = ModelFactory()
        self.drop_mode = drop_mode
        self.triplet_dict = self._collect_all_words(semsal_output)
        self.triplet_dict_per_person = self._semantic_comm_with_packet_loss(semsal_output)
        self.query_text = self._automatically_generate_query_text(semsal_output)
        self.mf.init(words_dict=self.triplet_dict, update=True)
        self.semsal_output = semsal_output
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

    def _no_packet_drop(self, query_ids):
        return query_ids, torch.Tensor()

    def _drop_packet_with_random_probability(self, query_ids, prob=0.2):
        """Random sampling, no sorting."""
        select_vars = torch.rand(query_ids.size()) > prob

        query_ids_received = query_ids[select_vars]
        query_ids_dropped = query_ids[torch.where(~select_vars)]

        return query_ids_received, query_ids_dropped

    def _drop_packet_with_power_scheduler(self, query_ids, priority, prob=0.2):
        """Sort according to attention priority and strategically dropping."""
        query_ids_sorted = query_ids[np.argsort(-priority)]
        return query_ids_sorted, 0

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

    def _semantic_comm_with_packet_loss(self, semsal_output):
        triplet_dict_per_person = {}
        for pid in range(self.num_persons):
            triplet_dict_per_person[pid] = {}

        for img_name in semsal_output.keys():
            merged_output_ = semsal_output[img_name]
            query_ids = merged_output_["queries"]

            for pid in range(self.num_persons):
                if self.drop_mode == "random_drop":
                    # Drop packets with random probability
                    query_ids_received, query_ids_dropped = self._drop_packet_with_random_probability(query_ids)
                elif self.drop_mode == "schedule":
                    priority = np.array(merged_output_[pid]["priority"])
                    query_ids_received, query_ids_dropped = self._drop_packet_with_power_scheduler(query_ids, priority)
                else:
                    query_ids_received, query_ids_dropped = self._no_packet_drop(query_ids)

                triplet_received = [merged_output_["reltr_output"][qid.item()]['semantic']
                                    for qid in query_ids_received]
                triplet_dropped = [merged_output_["reltr_output"][qid.item()]['semantic']
                                   for qid in query_ids_dropped]

                triplet_dict_per_person[pid][img_name] = {}
                triplet_dict_per_person[pid][img_name]["triplet_received"] = triplet_received
                triplet_dict_per_person[pid][img_name]["triplet_dropped"] = triplet_dropped

        return triplet_dict_per_person

    def reset(self):
        self.triplet_dict_per_person = self._semantic_comm_with_packet_loss(self.semsal_output)

    def fit(self):
        output = {}
        for pid in range(self.num_persons):
            personal_preds = self.triplet_dict_per_person[pid]
            output[pid] = {}

            for img_name, item in personal_preds.items():
                triplet_received = item["triplet_received"]
                triplet_dropped = item["triplet_dropped"]
                query_text = self.query_text[pid][img_name]
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
