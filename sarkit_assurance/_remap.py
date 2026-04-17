import numpy as np
import numpy.typing as npt
import sarkit.sicd as sksicd


def _scale_to_byte(data):
    min_val = data.min()
    max_val = data.max()
    return ((data - min_val) * 256 / (max_val - min_val)).clip(0, 255).astype(np.uint8)


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

    img = np.log10(img.clip(img.max() / 1e6, None))
    img = _scale_to_byte(img)
    return img
