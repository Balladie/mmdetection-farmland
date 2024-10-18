import argparse
import os
import shutil
from tqdm import tqdm
from mmdet.apis import DetInferencer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="out_dirs")
    # parser.add_argument("--img-out-dir", type=str, default="vis")
    # parser.add_argument("--pred-out-dir", type=str, default="preds")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    inferencer = DetInferencer(model=args.config, weights=args.ckpt)

    img_fns = []

    for fn in tqdm(os.listdir(args.input_dir)):
        img_path = os.path.join(args.input_dir, fn)
        inferencer(img_path, out_dir=args.out_dir, no_save_pred=False)
    #     img_fns.append(fn)

    # pred_dir = os.path.join(args.out_dir, args.pred_out_dir)

    # for pred_fn, img_fn in zip(sorted(os.listdir(pred_dir)), img_fns):
    #     src_path = os.path.join(pred_dir, pred_fn)
    #     tgt_path = os.path.join(pred_dir, img_fn)
    #     shutil.move(src_path, tgt_path)