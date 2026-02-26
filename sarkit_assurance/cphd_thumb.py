import argparse
import math
import pathlib

import numpy as np
import sarkit.cphd as skcphd
from PIL import Image

try:
    from smart_open import open
except ImportError:
    pass


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("cphd_file", help="Path to input CPHD file")
    parser.add_argument(
        "thumbnail_file", type=pathlib.Path, help="Path to output thumbnail"
    )
    parser.add_argument("--channel-id", required=True, help="CPHD channel to use")
    parser.add_argument(
        "--num-mebipixels",
        default=1.0,
        type=float,
        help="Maximum number of mebipixels to output",
    )
    config = parser.parse_args(args)

    output_size = config.num_mebipixels * 2**20

    with open(config.cphd_file, "rb") as f, skcphd.Reader(f) as r:
        sig, pvps = r.read_channel(config.channel_id)

    if sig.dtype.names is None:
        assert sig.dtype.newbyteorder("=") == np.dtype("c8")
        sig = sig.real**2 + sig.imag**2
    else:
        sig = sig["real"].astype(np.float32) ** 2 + sig["imag"].astype(np.float32) ** 2
    sig = np.sqrt(sig)
    if "AmpSF" in pvps.dtype.names:
        sig *= np.abs(pvps["AmpSF"][:, np.newaxis])
    cutoff_vals = np.quantile(sig, [0.0, 0.99])
    sig -= cutoff_vals[0]
    sig *= 256 / (cutoff_vals[1] - cutoff_vals[0])
    sig_remapped = np.clip(np.floor(sig), a_min=0, a_max=255).astype(np.uint8)

    img = Image.fromarray(sig_remapped)

    factor = math.sqrt(sig.size / output_size)
    (width, height) = (img.width // factor, img.height // factor)
    img.thumbnail((width, height))
    img.save(config.thumbnail_file)


if __name__ == "__main__":
    main()
