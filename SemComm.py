import torch
import numpy as np
from pkg_drop import PDPget
from TextMatch.textmatch.config.constant import setup_seed

setup_seed(2)


class SemComm:
    def __init__(self, args):
        self.args = args
        self.num_persons = args.num_persons
        self.drop_mode = args.drop_mode

    def _no_packet_drop(self, query_ids):
        return query_ids, torch.Tensor([])

    def _drop_packet_with_random_probability(self, query_ids, exp_iter):
        """Random sampling, no sorting."""
        if len(query_ids) == 1:
            return query_ids, torch.Tensor([])

        drop_prob = PDPget(self.args.power, np.ones(query_ids.shape))

        torch.manual_seed(exp_iter)
        select_vars = torch.rand(query_ids.size()) > torch.Tensor(drop_prob)

        query_ids_received = query_ids[select_vars]
        query_ids_dropped = query_ids[torch.where(~select_vars)]

        return query_ids_received, query_ids_dropped

    def _drop_packet_with_power_scheduler(self, query_ids, priority, exp_iter):
        """Drop by power scheduler and sort by saliency priority."""
        if len(query_ids) == 1:
            return query_ids, torch.Tensor([])

        drop_prob = PDPget(self.args.power, priority)

        torch.manual_seed(exp_iter)
        select_vars = torch.rand(query_ids.size()) > torch.Tensor(drop_prob)

        query_ids_received = query_ids[select_vars]
        priority_received = priority[select_vars]
        query_ids_sorted = query_ids_received[np.argsort(-priority_received)]

        query_ids_dropped = query_ids[torch.where(~select_vars)]

        return query_ids_sorted, query_ids_dropped

    def send(self, semsal_output, exp_iter):
        triplet_dict_per_person = {}

        for img_name in semsal_output.keys():
            merged_output_ = semsal_output[img_name]
            triplet_dict_per_person[img_name] = {}

            for pid in range(self.num_persons):
                query_ids = merged_output_[pid]["queries"]

                if self.drop_mode == "random_drop":
                    # Drop packets with random probability
                    query_ids_received, query_ids_dropped = self._drop_packet_with_random_probability(
                        query_ids, exp_iter)
                elif self.drop_mode == "schedule":
                    # Drop packets by power scheduler, according to saliency priority
                    priority = np.array(merged_output_[pid]["priority"])
                    query_ids_received, query_ids_dropped = self._drop_packet_with_power_scheduler(
                        query_ids, priority, exp_iter)
                else:
                    # Drop packets randomly, without considering saliency
                    query_ids_received, query_ids_dropped = self._no_packet_drop(query_ids)

                triplet_received = [merged_output_["reltr_output"][qid.item()]['semantic']
                                    for qid in query_ids_received]
                triplet_dropped = [merged_output_["reltr_output"][qid.item()]['semantic']
                                   for qid in query_ids_dropped]

                triplet_dict_per_person[img_name][pid] = {}
                triplet_dict_per_person[img_name][pid]["triplet_received"] = triplet_received
                triplet_dict_per_person[img_name][pid]["triplet_dropped"] = triplet_dropped

                if pid == 0 and img_name == semsal_output.keys()[0]:
                    print(f"received: {triplet_received}")
                    print(f"drop: {triplet_dropped}")

        return triplet_dict_per_person
