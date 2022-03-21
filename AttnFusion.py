import os
import gc
import math
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
from skimage.transform import resize
from PIL import Image


class AttnFusion:
    def __init__(self, RelTR, Saliency, args):
        self.num_persons = args.num_persons
        self.args = args
        self.RelTR = RelTR
        self.Saliency = Saliency

    def _run_reltr(self, img_path):
        if not hasattr(self, "reltr"):
            self.reltr = self.RelTR(self.args)

        output = self.reltr.fit(img_path)
        return output

    def _run_multi_saliency(self, img_path):
        if not hasattr(self, "saliency_list"):
            self.saliency_list = [self.Saliency(pid, self.args)
                                  for pid in range(self.num_persons)]

        output = []
        for pid in range(self.num_persons):
            # clear session to avoid graph conflicts
            tf.keras.backend.clear_session()
            # run inference
            output.append(self.saliency_list[pid].fit(img_path))

        return output

    def _save_pkl(self, objs, names, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for obj, name in zip(objs, names):
            with open(os.path.join(output_dir, name), "wb") as fp:
                pickle.dump(obj, fp)
        print(f">> Outputs are saved to path {output_dir}")

    def _load_pkl(self, names, output_dir):
        output = []
        for name in names:
            with open(os.path.join(output_dir, name), "rb") as fp:
                output.append(pickle.load(fp))
        # print(">> Pkl outputs are loaded")
        return output

    def _merge_function(self, mapA, mapB, alpha=-1):
        merge_mode = self.args.merge_mode
        if merge_mode == "weighted_sum":
            alpha_ = self.args.alpha if alpha == -1 else alpha
            merged_ = alpha_ * mapA + (1 - alpha_) * mapB
        elif merge_mode == "matrix_mul":
            merged_ = mapA * mapB
        else:
            merged_ = mapA + mapB
        return merged_

    def _normalize_attention_heatmap(self, reltr_output):
        max_scale_ = 0.
        query_ids = reltr_output["queries"]

        # obtain max_scale_ across all attention heatmaps
        for qid in query_ids:
            item = reltr_output[qid.item()]
            attn_sub_max_ = item["attn_weights_sub"].max()
            attn_obj_max_ = item["attn_weights_obj"].max()
            scale_ = max(attn_sub_max_, attn_obj_max_)
            max_scale_ = max(max_scale_, scale_)

        # normalize attention heatmap using max_scale_
        for qid in query_ids:
            item = reltr_output[qid.item()]
            item["attn_weights_sub"] /= max_scale_
            item["attn_weights_obj"] /= max_scale_

    def _normalize_saliency_heatmap(self, saliency_output, attn_size):
        max_scale_ = 0.
        num_persons = len(saliency_output)

        # obtain max_scale_ across all personal saliency heatmaps
        for pid in range(num_persons):
            sal_attn = saliency_output[pid]["attn"]
            # resize saliency heatmaps
            sal_attn = resize(sal_attn, attn_size)
            saliency_output[pid]["attn"] = sal_attn
            # count maximum scale factor
            scale_ = sal_attn.max()
            max_scale_ = max(max_scale_, scale_)

        # normalize saliency heatmap using max_scale_
        for pid in range(num_persons):
            sal_attn = saliency_output[pid]["attn"]
            sal_attn /= max_scale_

    def _merge_outputs(self, reltr_output, saliency_output):
        query_ids = reltr_output["queries"]
        num_persons = len(saliency_output)

        merged_output = {
            "reltr_output": reltr_output,
            "saliency_output": saliency_output,
            "num_persons": num_persons
        }

        for pid in range(num_persons):
            merged_output[pid] = {"priority": None}

            # rescale saliency attention map
            sal_attn = saliency_output[pid]["attn"]
            merged_output[pid]["sal_attn"] = sal_attn

            priority_arr = np.array([])
            # merge json info
            for qid in query_ids:
                qid = qid.item()
                merged_output[pid][qid] = {}
                item = reltr_output[qid]
                reltr_attn_sub = item["attn_weights_sub"]
                reltr_attn_obj = item["attn_weights_obj"]

                # merge attention map of reltr and saliency model
                merged_attn_sub = self._merge_function(reltr_attn_sub, sal_attn)
                merged_attn_obj = self._merge_function(reltr_attn_obj, sal_attn)

                # crop attention area according to bboxes
                [[xmin, ymin], [xmax, ymax]] = item["attn_bbox_sub"]
                xmin, ymin = max(xmin, 0), max(ymin, 0)
                cropped_attn_sub = merged_attn_sub[ymin:ymax+1, xmin:xmax+1]
                [[xmin, ymin], [xmax, ymax]] = item["attn_bbox_obj"]
                xmin, ymin = max(xmin, 0), max(ymin, 0)
                cropped_attn_obj = merged_attn_obj[ymin:ymax+1, xmin:xmax+1]

                # calculate maximum attention weight
                max_attn_sub = cropped_attn_sub.max()
                max_attn_obj = cropped_attn_obj.max()
                rel_confidence = item["rel_confidence"]
                # priority = max_attn_sub * max_attn_obj * rel_confidence
                priority = max_attn_sub * max_attn_obj

                merged_output[pid][qid]["merged_attn_sub"] = merged_attn_sub
                merged_output[pid][qid]["merged_attn_obj"] = merged_attn_obj
                merged_output[pid][qid]["cropped_attn_sub"] = cropped_attn_sub
                merged_output[pid][qid]["cropped_attn_obj"] = cropped_attn_obj
                merged_output[pid][qid]["max_attn_sub"] = max_attn_sub
                merged_output[pid][qid]["max_attn_obj"] = max_attn_obj
                merged_output[pid][qid]["rel_confidence"] = rel_confidence
                priority_arr = np.append(priority_arr, priority)

            # normalize priority
            priority_arr = priority_arr / priority_arr.sum()
            merged_output[pid]["priority"] = priority_arr

        return merged_output

    def _remove_duplicates(self, output):
        query_ids = output["reltr_output"]["queries"]

        for pid in range(self.num_persons):
            per_sal = output[pid]
            priority = per_sal["priority"]
            check_dict = {}

            for idx, qid in enumerate(query_ids):
                qid = qid.item()
                semantic = output["reltr_output"][qid]["semantic"]
                if semantic not in check_dict \
                        or priority[idx] > check_dict[semantic]["priority"]:
                    check_dict[semantic] = {"qid": qid, "priority": priority[idx]}

            new_query_ids = []
            new_priority = []
            for semantic, item in check_dict.items():
                qid, pri = item.values()
                new_query_ids.append(qid)
                new_priority.append(pri)
            per_sal["queries"] = torch.Tensor(new_query_ids).int()
            per_sal["priority"] = np.array(new_priority)
            per_sal["priority"] /= per_sal["priority"].sum()

            for qid in query_ids:
                if qid not in output[pid]["queries"]:
                    per_sal.pop(qid.item())

        return output

    def _collect_query_text_candidate(self, reltr_output, saliency_output):
        triplet_priorities = {}
        query_ids = reltr_output["queries"]
        num_persons = len(saliency_output)

        for pid in range(num_persons):
            triplet_priorities[pid] = {}

            # rescale saliency attention map
            sal_attn = saliency_output[pid]["attn"]

            # merge json info
            for qid in query_ids:
                qid = qid.item()
                item = reltr_output[qid]
                reltr_attn_sub = item["attn_weights_sub"]
                reltr_attn_obj = item["attn_weights_obj"]

                # merge attention map of reltr and saliency model
                merged_attn_sub = self._merge_function(reltr_attn_sub, sal_attn, alpha=0)
                merged_attn_obj = self._merge_function(reltr_attn_obj, sal_attn, alpha=0)

                # crop attention area according to bboxes
                [[xmin, ymin], [xmax, ymax]] = item["attn_bbox_sub"]
                xmin, ymin = max(xmin, 0), max(ymin, 0)
                cropped_attn_sub = merged_attn_sub[ymin:ymax + 1, xmin:xmax + 1]
                [[xmin, ymin], [xmax, ymax]] = item["attn_bbox_obj"]
                xmin, ymin = max(xmin, 0), max(ymin, 0)
                cropped_attn_obj = merged_attn_obj[ymin:ymax + 1, xmin:xmax + 1]

                # calculate maximum attention weight
                max_attn_sub = cropped_attn_sub.max()
                max_attn_obj = cropped_attn_obj.max()
                rel_confidence = item["rel_confidence"]
                # priority = max_attn_sub * max_attn_obj * rel_confidence
                priority = max_attn_sub * max_attn_obj

                semantic = reltr_output[qid]["semantic"]
                triplet_priorities[pid][semantic] = priority

        return triplet_priorities

    def _get_query_text(self, priorities_list):
        counter = {}
        priority_sum = {}
        for pid in range(self.num_persons):
            counter[pid] = {}
            priority_sum[pid] = {}

        # number of semantics occurs
        for item_per_image in priorities_list:
            for pid in range(self.num_persons):
                for triplet, priority in item_per_image[pid].items():
                    curr_num_ = counter[pid].get(triplet, 0)
                    counter[pid][triplet] = curr_num_ + 1
                    curr_pri_ = priority_sum[pid].get(triplet, 0)
                    priority_sum[pid][triplet] = curr_pri_ + priority

        # calculate average priorities for each triplet
        num_images = len(priorities_list)
        for pid in range(self.num_persons):
            for triplet in priority_sum[pid].keys():
                num_ = counter[pid][triplet]
                priority_sum[pid][triplet] /= num_
                priority_sum[pid][triplet] *= math.log(num_, num_images)

        # use the triplet with maximum priority as the query text
        suggest_query_text = {}
        for pid in range(self.num_persons):
            suggest_query_text[pid] = list(priority_sum[pid].keys())[
                np.argmax(list(priority_sum[pid].values()))]

        return suggest_query_text

    def visualize(self, output):
        for pid in range(output["num_persons"]):
            priority = output[pid]["priority"]
            order = np.argsort(np.argsort(-priority))
            query_ids = output[pid]["queries"]
            num_queries = len(query_ids)

            fig, axs = plt.subplots(ncols=num_queries, nrows=6, figsize=(3*num_queries, 13))
            axs = axs.T
            for idx in range(num_queries):
                axs_ = axs[idx] if num_queries > 1 else axs

                query_id = query_ids[idx].item()
                semantic = output["reltr_output"][query_id]["semantic"]
                semantic = semantic.replace(" ", "\ ")

                # show reltr attention map (subject)
                ax = axs_[0]
                ax.imshow(output["reltr_output"][query_id]["attn_weights_sub"])
                ax.axis("off")

                sub_confidence_ = round(output['reltr_output'][query_id]['sub_confidence'], 3)
                ax.set_title(f"subject confidence: {sub_confidence_}")

                # show reltr attention map (object)
                ax = axs_[1]
                ax.imshow(output["reltr_output"][query_id]["attn_weights_obj"])
                ax.axis("off")

                obj_confidence_ = round(output['reltr_output'][query_id]['obj_confidence'], 3)
                ax.set_title(f"object confidence: {obj_confidence_}")

                # show reltr bboxes
                ax = axs_[2]
                img = Image.open(output["saliency_output"][pid]["img_path"])
                ax.imshow(img)
                ((sxmin, symin), (sxmax, symax)) = output["reltr_output"][query_id]["img_bbox_sub"]
                ((oxmin, oymin), (oxmax, oymax)) = output["reltr_output"][query_id]["img_bbox_obj"]
                ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                           fill=False, color="blue", linewidth=2.5))
                ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                           fill=False, color="orange", linewidth=2.5))
                ax.axis("off")

                rel_confidence_ = round(output['reltr_output'][query_id]['rel_confidence'], 3)
                overall_confidence_ = round(sub_confidence_ * obj_confidence_ * rel_confidence_, 3)
                ax.set_title(f"relation confidence: {rel_confidence_}\n"
                             f"overall confidence: {overall_confidence_}\n"
                             f"query id: {query_id}\n"
                             r"semantic: $\bf{" + semantic + "}$")

                # show saliency map
                ax = axs_[3]
                ax.imshow(output["saliency_output"][pid]["attn"])
                ax.axis("off")
                ax.set_title("saliency map")

                # show merged attention map (subject)
                qid = query_ids[idx].item()
                ax = axs_[4]
                ax.imshow(output[pid][qid]["merged_attn_sub"])
                ax.axis("off")
                ax.set_title(f"subject merged map")

                # show merged attention map (object)
                ax = axs_[5]
                ax.imshow(output[pid][qid]["merged_attn_obj"])
                ax.axis("off")
                ax.set_title(f"object merged map\n"
                             r"priority: $\bf{"+str(round(priority[idx], 5))+"}$\n"
                             r"order: $\bf{"+str(order[idx])+"}$")

            fig.tight_layout()

            # save figure to specified path
            img_name = os.path.basename(output["reltr_output"]["img_path"]).split(".")[0]
            output_img_dir = os.path.join(self.args.output_dir, img_name)
            output_img_name = os.path.join(output_img_dir, f"person_{pid}.png")
            # create dirs if not exist
            os.makedirs(self.args.output_dir, exist_ok=True)
            os.makedirs(output_img_dir, exist_ok=True)
            plt.savefig(output_img_name)

    def _run_fusion(self, reltr_output, saliency_output, visualize=False):
        # normalize attention heatmaps globally
        self._normalize_attention_heatmap(reltr_output)
        # normalize saliency heatmaps globally
        attn_size = reltr_output["attn_size"]
        self._normalize_saliency_heatmap(saliency_output, attn_size)

        # merge heatmaps
        merged_output = self._merge_outputs(reltr_output, saliency_output)
        merged_output = self._remove_duplicates(merged_output)
        priorities = self._collect_query_text_candidate(reltr_output, saliency_output)
        if visualize: self.visualize(merged_output)
        return merged_output, priorities

    def _save_to_text(self, output):
        img_name = os.path.basename(output["reltr_output"]["img_path"]).split(".")[0]
        output_img_dir = os.path.join(self.args.output_dir, img_name)
        output_txt_name = os.path.join(output_img_dir, "output.txt")
        # create dirs if not exist
        os.makedirs(self.args.output_dir, exist_ok=True)
        os.makedirs(output_img_dir, exist_ok=True)

        fp = open(output_txt_name, "w")

        fp.write(output["reltr_output"]["img_path"])
        fp.write("\n\n")

        for pid in range(output["num_persons"]):
            fp.write(f"Person {pid}:\n")
            query_ids = output[pid]["queries"]
            for idx, qid in enumerate(query_ids):
                qid = qid.item()
                fp.write(output["reltr_output"][qid]["semantic"]+": ")
                fp.write(str(round(output[pid]["priority"][idx], 5)))
                fp.write("\n")
            fp.write("\n")

    def fit(self, resume_pkl=False, save_pkl=False, save_txt=False, visualize=False):
        input_dir = self.args.input_dir
        output = {}
        priority_list = []

        imgs_to_process = [self.args.img_name] if self.args.img_name \
            else sorted(os.listdir(input_dir))

        for img_name in imgs_to_process:
            # print(f">> Processing image: {img_name}")

            img_path = os.path.join(input_dir, img_name)
            img_idx = img_name.split('.')[0]
            reltr_name = f"reltr_output_{img_idx}.pkl"
            saliency_name = f"saliency_output_{img_idx}.pkl"

            if resume_pkl:
                try:
                    reltr_output, saliency_output = self._load_pkl(
                        [reltr_name, saliency_name],
                        self.args.output_dir)
                except Exception:
                    print(f">> Image {img_name} not loaded")
                    continue
            else:
                # RelTR inference
                reltr_output = self._run_reltr(img_path)
                if not reltr_output: continue
                # Saliency inference (7 persons)
                saliency_output = self._run_multi_saliency(img_path)
                # save outputs
                if save_pkl:
                    self._save_pkl(
                        [reltr_output, saliency_output],
                        [reltr_name, saliency_name],
                        self.args.output_dir)

            merged_output, priorities = self._run_fusion(
                reltr_output, saliency_output, visualize=visualize)

            output[img_name] = merged_output
            if save_txt: self._save_to_text(merged_output)
            priority_list.append(priorities)

            del reltr_output, saliency_output
            gc.collect()

        return output, self._get_query_text(priority_list)
