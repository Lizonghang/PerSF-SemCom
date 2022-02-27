import matplotlib.pyplot as plt
from PIL import Image

from .data import *
from .download import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)


class Saliency:
    """The main class for executing network testing. It loads the specified
    dataset iterator and optimized saliency model. By default, when no model
    checkpoint is found locally, the pretrained weights will be downloaded.
    Testing only works for models trained on the same device as specified in
    the config file.
    """
    def __init__(self, pid, args):
        self.person_id = pid
        self.args = args
        self.device = args.device_saliency
        datasets_list = [
            "salicon", "mit1003", "cat2000",
            "dutomron", "pascals", "osie", "fiwi"]
        self.dataset = datasets_list[pid]
        self.graph_def = self._load_tf_graph(self.dataset)

    def _load_tf_graph(self, dataset):
        model_name = "model_%s_%s.pb" % (dataset, self.device)
        current_path = os.path.dirname(os.path.realpath(__file__))
        ckpt_path = os.path.join(current_path, "ckpt")
        ckpt_file = os.path.join(ckpt_path, model_name)

        if not os.path.isfile(ckpt_file):
            download_pretrained_weights(ckpt_path, model_name[:-3])

        graph_def = tf.GraphDef()
        with tf.gfile.Open(ckpt_file, "rb") as file:
            graph_def.ParseFromString(file.read())

        return graph_def

    def fit(self, img_path):
        iterator = get_dataset_iterator("test", self.dataset, img_path)
        next_element, init_op = iterator
        input_images, original_shape, file_path = next_element

        [predicted_maps] = tf.import_graph_def(
            self.graph_def,
            input_map={"input": input_images},
            return_elements=["output:0"]
        )

        saliency_map = postprocess_saliency_map(predicted_maps[0], original_shape[0])

        print(">> Start testing with %s %s model..." % (self.dataset.upper(), self.device))

        with tf.Session() as sess:
            sess.run([init_op])

            while True:
                try:
                    attn, path = sess.run([saliency_map, file_path])
                except tf.errors.OutOfRangeError:
                    break

                # convert attention map to matrix of size (h, w)
                # and normalize the attention values
                attn = np.moveaxis(attn, -1, 0)[0, :]
                attn = attn / attn.sum()

        results = {
            "img_path": str(path[0][0], encoding="utf-8"),
            "attn": attn
        }
        return results

    def visualize(self, output):
        num_persons = len(output)
        fig, axs = plt.subplots(ncols=num_persons, nrows=2, figsize=(22, 7))
        axs = axs.T

        for idx in range(num_persons):
            axs_ = axs[idx] if num_persons > 1 else axs

            ax = axs_[0]
            img = Image.open(output[idx]["img_path"])
            ax.imshow(img)
            ax.axis("off")

            ax = axs_[1]
            ax.imshow(output[idx]["attn"])
            ax.axis("off")

        fig.tight_layout()

        # save figure to specified path
        os.makedirs(self.args.output_dir, exist_ok=True)
        img_name = os.path.basename(output[0]["img_path"]).split(".")[0]
        plt.savefig(os.path.join(self.args.output_dir, f"saliency_{img_name}.png"))
