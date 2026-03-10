import argparse
import math

import numpy as np
import sarkit.cphd as skcphd
from PIL import Image

from . import names

try:
    from smart_open import open
except ImportError:
    pass


def channel_thumb(cphd_reader, channel_id, thumbnail_file, output_size):
    """Produce a thumbnail for the CPHD channel identified by channel_id"""
    sig, pvps = cphd_reader.read_channel(channel_id)

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
    img.save(thumbnail_file)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Create thumbnails from CPHD signal arrays"
    )
    parser.add_argument("cphd_file", help="Path to input CPHD file")
    parser.add_argument(
        "thumbnail_file",
        help="Path to output thumbnail(s). The string '{ch_id}' will be replaced with channel identifier.",
    )
    parser.add_argument(
        "--channel-id",
        action="append",
        help=(
            "Identifier of channel to use. May be specified more than once. "
            "If unspecified, a thumbnail for each channel is generated."
        ),
    )
    parser.add_argument(
        "--num-mebipixels",
        default=1.0,
        type=float,
        help="Maximum number of mebipixels to output",
    )
    config = parser.parse_args(args)

    output_size = config.num_mebipixels * 2**20

    with open(config.cphd_file, "rb") as f, skcphd.Reader(f) as r:
        actual_ch_ids = [
            x.text
            for x in r.metadata.xmltree.findall("{*}Data/{*}Channel/{*}Identifier")
        ]
        ch_ids = set(config.channel_id or actual_ch_ids)
        bad_channel_ids = ch_ids.difference(actual_ch_ids)
        if bad_channel_ids:
            raise ValueError(f"{bad_channel_ids=}")

        thumbnames = {
            ch_id: config.thumbnail_file.format(ch_id=names.sanitize_name(ch_id))
            for ch_id in ch_ids
        }
        if len(set(thumbnames.values())) != len(thumbnames):
            raise RuntimeError("Duplicate output filenames detected")

        for ch_id, thumbname in thumbnames.items():
            channel_thumb(r, ch_id, thumbname, output_size)


if __name__ == "__main__":
    main()
