import numpy as np
import numpy.typing as npt
import sarkit.sicd as sksicd


def scale_to_byte(data):
    min_val = data.min()
    max_val = data.max()
    return ((data - min_val) * 256 / (max_val - min_val)).clip(0, 255).astype(np.uint8)


def simple_log_remap(data, cut_low_frac=0.05, cut_high_frac=0.0, min_low_relative=1e-6):
    """Log encode an array to 8 bits based upon percentile saturation and maximum dynamic range

    Parameters
    ----------
    data : np.ndarray
        data to encode
    cut_low_frac : float, optional
        The fraction of the samples to consider below threshold.
        (Default: ``0.05``)
    cut_high_frac : float, optional
        The fraction of the samples to consider above threshold.
        (Default: ``0.00``)
    min_low_relative : float, optional
        If not zero, clip ``data`` to ``min_low_relative  * max(data)`` before statistics.
        (Default: ``1e-6``)

    Returns
    -------
    np.ndarray
        8-bit encoding
    """

    low_cutoff = np.maximum(
        np.quantile(data, cut_low_frac), data.max() * min_low_relative
    )
    high_cutoff = np.quantile(data, 1.0 - cut_high_frac)

    data = np.log10(data.clip(low_cutoff, high_cutoff))
    data = scale_to_byte(data)
    return data


def simple_sicd_remap(image: npt.NDArray) -> npt.NDArray:
    """Convert a SICD pixel array into an viewable 8-bit image

    Parameters
    ----------
    image : np.ndarray
        SICD pixels

    Returns
    -------
    np.ndarray
        8-bit image
    """

    def is_pixel_type(sicd_pixel_enum):
        image_dtype_native = image.dtype.newbyteorder("=")
        sicd_dtype_native = sksicd.PIXEL_TYPES[sicd_pixel_enum]["dtype"].newbyteorder(
            "="
        )
        return image_dtype_native == sicd_dtype_native

    if is_pixel_type("RE16I_IM16I"):
        img = (
            image["real"].astype(np.float32) ** 2
            + image["imag"].astype(np.float32) ** 2
        )
    elif is_pixel_type("AMP8I_PHS8I"):
        img = image["amp"].astype(np.float32) ** 2
    else:
        img = image.real**2 + image.imag**2

    return simple_log_remap(img)
