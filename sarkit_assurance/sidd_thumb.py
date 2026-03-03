import argparse
import math
import pathlib
import textwrap

import numpy as np
import sarkit.sidd as sksidd
from PIL import Image

try:
    from smart_open import open
except ImportError:
    pass


def main(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Create thumbnails from a SIDD product image. No attempt is made at SIPS processing.
            The thumbnail's image mode is dependent on the PixelType:
                MONO8I  -> 8-bit, grayscale
                MONO8LU -> 8-bit or 16-bit, grayscale (depending on lookup table)
                MONO16I -> 16-bit, grayscale
                RGB8LU  -> 3x8-bit, true color
                RGB24I  -> 3x8-bit, true color
            """),
    )
    parser.add_argument("sidd_file", help="Path to input SIDD file")
    parser.add_argument(
        "thumbnail_file",
        type=pathlib.Path,
        help="Path to output thumbnail. The format to use is determined from the filename extension.",
    )
    parser.add_argument(
        "--image-number",
        required=True,
        type=int,
        help="0-based index of SIDD Product image to read",
    )
    parser.add_argument(
        "--num-mebipixels",
        default=1.0,
        type=float,
        help="Maximum number of mebipixels to output",
    )
    config = parser.parse_args(args)

    output_size = config.num_mebipixels * 2**20

    with open(config.sidd_file, "rb") as f, sksidd.NitfReader(f) as r:
        arr = r.read_image(config.image_number)

    img_meta = r.metadata.images[config.image_number]
    px_type = img_meta.xmltree.findtext("{*}Display/{*}PixelType")
    if px_type in ("MONO8I", "MONO16I"):
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
    img.save(config.thumbnail_file)


if __name__ == "__main__":
    main()
