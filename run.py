import argparse
import os
import sys

from functools import partial
import config
from utils import readlines_txt, read_json
from Textgenerator import Textgenerator, transform_img, rotate_text


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic text data")

    parser.add_argument(
        "-txt",
        "--text",
        type=str,
        nargs="?",
        help="text used for generating image",
        default=None,
    )

    parser.add_argument(
        "-i",
        "--text_file",
        type=str,
        nargs="?",
        help="a specified text file as source for the text",
        default=None,
    )

    parser.add_argument(
        "-ft",
        "--font",
        type=str,
        nargs="?",
        help="Define font to be used",
        default=None
    )

    parser.add_argument(
        "-tc",
        "--text_color_file",
        type=str,
        nargs="?",
        help="a specified file as source for a set of text color",
        default="./dataset/text_colors/text_color.json",
    )

    parser.add_argument(
        "-n",
        "--n_image",
        type=int,
        nargs="?",
        help="the number of output images",
        default=1,
    )

    parser.add_argument(
        "-nj",
        "--n_job",
        type=int,
        nargs="?",
        help="the number of cpu core",
        default=1,
    )

    parser.add_argument(
        "-ht",
        "--img_height",
        type=int,
        nargs="?",
        help="the height of image",
        default=32,
    )

    parser.add_argument(
        "-sk",
        "--skew_angle",
        type=int,
        nargs="?",
        help="rotate image between (-skew_angle , skew_angle) degree",
        default=0,
    )

    parser.add_argument(
        "-pd",
        "--padding",
        type=int,
        nargs="?",
        help="defince background space around text",
        default=10,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default='./output/',
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    generator = Textgenerator(
        transform=transform_img,
        rotate_text_f= partial(rotate_text, limit=args.skew_angle),
        saved_dir=args.output_dir,
        img_height=args.img_height,
        padding=args.padding,
        n_job=args.n_job
    )

    generator.run(
        n_img=args.n_image,
        txt=args.text,
        txt_file=args.text_file,
        txt_color_file=args.text_color_file, 
        font=args.font,
        font_dir=config.FONT_FOLDER, 
        bg_dir=config.BG_FODLER
    )

if __name__ == "__main__":
    main()