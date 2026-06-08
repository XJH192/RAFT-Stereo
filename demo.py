import sys
sys.path.append('core')

import argparse
import glob
import logging
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def load_checkpoint(model, restore_ckpt, use_dense_frontend):
    checkpoint = torch.load(restore_ckpt)
    strict = not use_dense_frontend
    result = model.load_state_dict(checkpoint, strict=strict)
    if strict:
        return

    missing = [key for key in result.missing_keys if 'dense_matcher' not in key]
    unexpected = [key for key in result.unexpected_keys if 'dense_matcher' not in key]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing: {missing}. Unexpected: {unexpected}")

    if result.missing_keys:
        logging.warning("Dense frontend keys are not stored in the checkpoint; using the module's current weights.")


def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    load_checkpoint(model, args.restore_ckpt, args.use_dense_frontend)

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

            file_stem = imfile1.split('/')[-2]
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--use_dense_frontend', action='store_true', help='enable transformer dense matcher before RAFT refinement')
    parser.add_argument('--dense_frontend_type', choices=['row_transformer', 'loftr', 'roma'], default='row_transformer', help='frontend matcher used before RAFT refinement')
    parser.add_argument('--dense_frontend_pretrained', type=str, default='outdoor', help='pretrained preset for the selected frontend matcher')
    parser.add_argument('--dense_frontend_confidence_thresh', type=float, default=0.2, help='minimum confidence kept from LoFTR/RoMa matches')
    parser.add_argument('--dense_frontend_max_vertical_offset', type=float, default=2.0, help='maximum allowed vertical mismatch in pixels for LoFTR/RoMa matches')
    parser.add_argument('--dense_frontend_blur_kernel', type=int, default=7, help='odd kernel size used to spread sparse matches onto the RAFT grid')
    parser.add_argument('--dense_frontend_blur_std', type=float, default=2.0, help='gaussian std used to spread sparse matches onto the RAFT grid')
    parser.add_argument('--dense_frontend_trainable', action='store_true', help='allow gradients through the selected external frontend when supported')
    parser.add_argument('--dense_matcher_layers', type=int, default=2, help='number of transformer layers in the row transformer matcher')
    parser.add_argument('--dense_matcher_heads', type=int, default=8, help='number of attention heads in the row transformer matcher')
    parser.add_argument('--dense_matcher_ffn_dim', type=int, default=512, help='feedforward width in the row transformer matcher')
    
    args = parser.parse_args()

    demo(args)
