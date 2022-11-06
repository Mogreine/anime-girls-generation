import argparse

from src.models.models import BaseDiffusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_ckpt', "-i", type=str, required=True)
    parser.add_argument('--out_ckpt', "-o", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model = BaseDiffusion.load_from_checkpoint(args.in_ckpt)
    model.encoder.save_pretrained(args.out_ckpt)
