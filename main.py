import argparse

from RelTR import RelTR
from Saliency import Saliency
from SemSal import SemSal
from TextMatch import TextMatcher
from SemComm import SemComm


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # common
    parser.add_argument('--input_dir', type=str, default='data/',
                        help="directory of inference images")
    parser.add_argument('--img_name', type=str, default='',
                        help='only the specified image file will be processed')
    parser.add_argument('--output_dir', type=str, default='output/',
                        help="directory of output files (e.g., images, logs)")
    parser.add_argument('--device_reltr', type=str, default='cuda:0',
                        help='device to use in reltr inference')
    parser.add_argument('--device_saliency', default='cpu',
                        help='device to use in saliency inference')
    parser.add_argument('--resume_pkl', type=int, default=0,
                        help='whether to use saved pickle files, to save execution time')

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
                        help="return the fpn if there is the tag")

    # for semsal merge
    parser.add_argument('--merge_mode', choices=['weighted_sum', 'matrix_mul'], default='weighted_sum',
                        help='function to merge two attention maps, default to be simply sum')
    parser.add_argument('--alpha', default=0., type=float,
                        help='weight to merge RelTR and saliency map')
    parser.add_argument('--num_persons', type=int, default=7,
                        help='number of persons to simulate, only 7 is supported currently')

    # for semantic communication
    parser.add_argument('--drop_mode', choices=['no_drop', 'random_drop', 'schedule'],
                        default='schedule', help='whether and how to drop packets')
    parser.add_argument('--power', type=int, default=6000,
                        help='transfer power of sender')

    # for text matcher
    parser.add_argument('--repeat_exp', default=50, type=int,
                        help='number of repeat experiments to evaluate match score')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    assert args.num_persons == 7, \
        "Only 7 persons are supported in this version"

    # Merge subject and object saliency
    semsal = SemSal(RelTR, Saliency, args)

    if args.resume_pkl:
        # run with saved pickle files
        semsal_output, suggest_query_text = semsal.fit(
            resume_pkl=True, save_pkl=False, save_txt=False, visualize=False)
    else:
        # first run
        semsal_output, suggest_query_text = semsal.fit(
            resume_pkl=False, save_pkl=True, save_txt=True, visualize=True)

    sem_comm = SemComm(args)
    text_matcher = TextMatcher(semsal_output, suggest_query_text, args)
    print("Personalized query text:", suggest_query_text)

    for exp_iter in range(args.repeat_exp):
        print(f">> Repeat {exp_iter} times ...")

        # Send packets through loseless semantic comm network
        sent = sem_comm.send(semsal_output, exp_iter)

        # Evaluate match score on the user side
        text_matcher.receive(sent)
        match_scores = text_matcher.fit()
        text_matcher.eval(match_scores)

    result = text_matcher.output
    print([result[pid]["mean_max_scores"]
           for pid in result.keys()])
