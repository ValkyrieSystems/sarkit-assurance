import argparse
import math
import sys
import textwrap

import numpy as np
import sarkit.sidd as sksidd
from PIL import Image

from . import _cli

try:
    from smart_open import open
except ImportError:
    pass


def product_image_thumb(sidd_reader, image_number, thumbnail_file, output_size):
    """Produce a thumbnail for the SIDD product image channel identified by image_number"""
    arr = sidd_reader.read_image(image_number)
    img_meta = sidd_reader.metadata.images[image_number]
    px_type = img_meta.xmltree.findtext("{*}Display/{*}PixelType")
    if px_type in ("MONO8I", "MONO16I"):
        arr = arr.astype(
            arr.dtype.newbyteorder("="), copy=False
        )  # PIL doesn't like >u2
        img = Image.fromarray(arr)
    elif px_type == "MONO8LU":
        lut = img_meta.lookup_table
        img = Image.fromarray(lut[arr])
    elif px_type == "RGB24I":
        img = Image.fromarray(arr[..., np.newaxis].view(np.uint8))
    elif px_type == "RGB8LU":
        # PIL thumbnail for P (lookup) modes only use nearest resampling, which we want to avoid
        lut = img_meta.lookup_table
        img = Image.fromarray(lut[arr][..., np.newaxis].view(np.uint8))
    else:
        raise RuntimeError(f"Unrecognized PixelType: {px_type}")

    factor = math.sqrt(arr.size / output_size)
    (width, height) = (img.width // factor, img.height // factor)
    if factor > 2 and img.mode.startswith("I;16"):
        # first stage of PIL thumbnail broken for "special modes"; manual resize to try to get I16 to work
        img = img.resize(
            (img.width // int(factor), img.height // int(factor)),
            resample=Image.Resampling.BOX,
            reducing_gap=None,
        )

    img.thumbnail((width, height))
    img.save(thumbnail_file)


class _SiddThumbSubcommand(_cli.Subcommand):
    def get_argument_parser_kwargs(self):
        return dict(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent("""\
                Create thumbnails from SIDD product images. No attempt is made at SIPS processing.
                Each thumbnail's image mode is dependent on the PixelType:
                    MONO8I  -> 8-bit, grayscale
                    MONO8LU -> 8-bit or 16-bit, grayscale (depending on lookup table)
                    MONO16I -> 16-bit, grayscale
                    RGB8LU  -> 3x8-bit, true color
                    RGB24I  -> 3x8-bit, true color
                """),
        )

    def add_arguments(self, parser):
        parser.add_argument("sidd_file", help="path to SIDD file")
        parser.add_argument(
            "thumbnail_file",
            help=(
                "path to output thumbnail(s). The string '{num}' will be replaced with the image number. "
                "The file format is determined from the filename extension."
            ),
        )
        parser.add_argument(
            "--image-number",
            action="append",
            type=int,
            help=(
                "0-based index of product image to read. May be specified more than once. "
                "If unspecified, a thumbnail for each product image is generated."
            ),
        )
        parser.add_argument(
            "--num-mebipixels",
            default=1.0,
            type=float,
            help="maximum number of mebipixels to output (default: 1.0 MiB)",
        )

    def run_command(self, config):
        output_size = config.num_mebipixels * 2**20

        with open(config.sidd_file, "rb") as f, sksidd.NitfReader(f) as r:
            actual_img_nums = range(len(r.metadata.images))
            img_nums = set(config.image_number or actual_img_nums)
            bad_image_numbers = img_nums.difference(actual_img_nums)
            if bad_image_numbers:
                raise ValueError(f"{bad_image_numbers=}")

            thumbnames = {
                num: config.thumbnail_file.format(num=num) for num in img_nums
            }
            if len(set(thumbnames.values())) != len(thumbnames):
                raise RuntimeError("Duplicate output filenames detected")

            for img_num, thumbname in thumbnames.items():
                product_image_thumb(r, img_num, thumbname, output_size)
        return 0


main = _SiddThumbSubcommand().as_callable()

if __name__ == "__main__":
    sys.exit(main())
