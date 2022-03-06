import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from .models import build_model
from .config import CLASSES, REL_CLASSES


class RelTR:
    def __init__(self, args):
        self.args = args
        device = torch.device(args.device_reltr)

        # data transformer
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # build model and load weights
        self.model, _, _ = build_model(args)
        ckpt = torch.load(args.resume, map_location=device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def _rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self._box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def _img_to_attn_size(self, img_size, attn_size, img_bbox):
        scale_rate_h = img_size[0] / attn_size[0]
        scale_rate_w = img_size[1] / attn_size[1]
        ((xmin, ymin), (xmax, ymax)) = img_bbox

        xmin_attn_ = int(round(xmin / scale_rate_w, 0))
        ymin_attn_ = int(round(ymin / scale_rate_h, 0))
        xmax_attn_ = int(round(xmax / scale_rate_w, 0))
        ymax_attn_ = int(round(ymax / scale_rate_h, 0))

        attn_bbox = ((xmin_attn_, ymin_attn_), (xmax_attn_, ymax_attn_))
        return attn_bbox

    def _remove_duplicates(self, output):
        exist_semantics_ = {}
        qid_to_remove_ = []

        query_ids = output["queries"]
        for qid in query_ids:
            qid_ = qid.item()
            semantic_ = output[qid_]["semantic"]
            if semantic_ in exist_semantics_:
                qid_to_remove_.append(qid)
                output.pop(qid_)
            else:
                exist_semantics_[semantic_] = qid

        new_query_ids = output["queries"].tolist()
        [new_query_ids.remove(qid_) for qid_ in qid_to_remove_]
        output["queries"] = torch.Tensor(new_query_ids).type(torch.IntTensor)
        return output

    def fit(self, img_path):
        img = Image.open(img_path)
        img_size = img.size

        # print(">> Start testing with RelTR model...")

        # mean-std normalize the input image (batch-size: 1)
        input = self.transform(img).unsqueeze(0)
        outputs = self.model(input)

        probas = outputs["rel_logits"].softmax(-1)[0, :, :-1]
        probas_sub = outputs["sub_logits"].softmax(-1)[0, :, :-1]
        probas_obj = outputs["obj_logits"].softmax(-1)[0, :, :-1]
        # keep only predictions with 0.3+ confidence
        keep = torch.logical_and(
            probas.max(-1).values > 0.3,
            torch.logical_and(
                probas_sub.max(-1).values > 0.3,
                probas_obj.max(-1).values > 0.3)
        )

        # convert bboxes from [0; 1] to image scales
        sub_bboxes_scaled = self._rescale_bboxes(outputs["sub_boxes"][0, keep], img_size)
        obj_bboxes_scaled = self._rescale_bboxes(outputs["obj_boxes"][0, keep], img_size)

        # sort queries by their overall confidence
        keep_queries = torch.nonzero(keep, as_tuple=True)[0]

        # no predictions satisfy
        if len(keep_queries) == 0:
            return None

        confidence_ = -probas[keep_queries].max(-1)[0] \
                      * probas_sub[keep_queries].max(-1)[0] \
                      * probas_obj[keep_queries].max(-1)[0]
        indices = torch.argsort(confidence_)[:self.args.topk]
        keep_queries = keep_queries[indices]

        # use lists to temporarily store the outputs via up-values
        conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []

        hooks = [
            self.model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            self.model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
                lambda self, input, output: dec_attn_weights_sub.append(output[1])
            ),
            self.model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
                lambda self, input, output: dec_attn_weights_obj.append(output[1])
            )
        ]

        with torch.no_grad():
            self.model(input)

            for hook in hooks:
                hook.remove()

            conv_features = conv_features[0]
            dec_attn_weights_sub = dec_attn_weights_sub[0]
            dec_attn_weights_obj = dec_attn_weights_obj[0]

            # get feature map shape (h,w) and original image shape (img_h, img_w)
            h, w = conv_features["0"].tensors.shape[-2:]
            img_w, img_h = img_size

            results = {
                "img_path": img_path,
                "queries": keep_queries,
                "img_size": (img_h, img_w),
                "attn_size": (h, w)
            }

            for idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                    zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
                # record info for each item
                results[idx.item()] = {}

                # extract info
                attn_weights_sub_ = dec_attn_weights_sub[0, idx].view(h, w)
                attn_weights_obj_ = dec_attn_weights_obj[0, idx].view(h, w)
                img_bbox_sub_ = ((int(sxmin.item()), int(symin.item())),
                                 (int(sxmax.item()), int(symax.item())))
                img_bbox_obj_ = ((int(oxmin.item()), int(oymin.item())),
                                 (int(oxmax.item()), int(oymax.item())))
                attn_bbox_sub_ = self._img_to_attn_size((img_h, img_w), (h, w), img_bbox_sub_)
                attn_bbox_obj_ = self._img_to_attn_size((img_h, img_w), (h, w), img_bbox_obj_)
                sub_name_ = CLASSES[probas_sub[idx].argmax()]
                obj_name_ = CLASSES[probas_obj[idx].argmax()]
                rel_name_ = REL_CLASSES[probas[idx].argmax()]
                semantic_ = sub_name_ + " " + rel_name_ + " " + obj_name_
                sub_confidence_ = probas_sub[idx].max().item()
                obj_confidence_ = probas_obj[idx].max().item()
                rel_confidence_ = probas[idx].max().item()

                results[idx.item()]["attn_weights_sub"] = attn_weights_sub_
                results[idx.item()]["attn_weights_obj"] = attn_weights_obj_
                results[idx.item()]["img_bbox_sub"] = img_bbox_sub_
                results[idx.item()]["img_bbox_obj"] = img_bbox_obj_
                results[idx.item()]["attn_bbox_sub"] = attn_bbox_sub_
                results[idx.item()]["attn_bbox_obj"] = attn_bbox_obj_
                results[idx.item()]["sub_name"] = sub_name_
                results[idx.item()]["obj_name"] = obj_name_
                results[idx.item()]["rel_name"] = rel_name_
                results[idx.item()]["sub_confidence"] = sub_confidence_
                results[idx.item()]["obj_confidence"] = obj_confidence_
                results[idx.item()]["rel_confidence"] = rel_confidence_
                results[idx.item()]["semantic"] = semantic_

        return results
        # return self._remove_duplicates(results)

    def visualize(self, output_json):
        queries = output_json["queries"]
        num_queries = len(queries)

        fig, axs = plt.subplots(ncols=num_queries, nrows=3, figsize=(22, 7))
        axs = axs.T
        for idx in range(num_queries):
            query_id = queries[idx].item()

            ax = axs[idx][0]
            ax.imshow(output_json[query_id]["attn_weights_sub"])
            ax.axis("off")
            ax.set_title(f"query id: {query_id}")

            ax = axs[idx][1]
            ax.imshow(output_json[query_id]["attn_weights_obj"])
            ax.axis("off")

            ax = axs[idx][2]
            ax.imshow(output_json["img"])

            ((sxmin, symin), (sxmax, symax)) = output_json[query_id]["img_bbox_sub"]
            ((oxmin, oymin), (oxmax, oymax)) = output_json[query_id]["img_bbox_obj"]
            ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                       fill=False, color="blue", linewidth=2.5))
            ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                       fill=False, color="orange", linewidth=2.5))
            ax.axis("off")

            ax.set_title(output_json[query_id]["semantic"], fontsize=10)

        fig.tight_layout()

        # save figure to specified path
        os.makedirs(self.args.output_dir, exist_ok=True)
        img_name = os.path.basename(output_json["img_path"]).split(".")[0]
        plt.savefig(os.path.join(self.args.output_dir, f"reltr_{img_name}.png"))
