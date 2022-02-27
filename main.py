import argparse
from RelTR import RelTR
from Saliency import Saliency
from SemSal import SemSal


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # common
    parser.add_argument('--input_dir', type=str, default='data/',
                        help="Directory of inference images")
    parser.add_argument('--output_dir', type=str, default='output/',
                        help="Directory of output files (e.g., images, logs)")
    parser.add_argument('--device_reltr', default='cuda:0',
                        help='device to use in reltr inference')
    parser.add_argument('--device_saliency', default='cpu',
                        help='device to use in saliency inference')

    # reltr args
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--topk', default=100, type=int,
                        help="Number of output queries")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--resume', default='RelTR/ckpt/checkpoint0149.pth',
                        help='resume from checkpoint')
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    semsal = SemSal(RelTR, Saliency, args)
    semsal.fit(resume_pkl=False, save_pkl=True, visualize=True)
