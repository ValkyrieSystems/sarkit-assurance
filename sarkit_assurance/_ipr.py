"""Common utilities for IPR tools"""

import json

import numpy as np
import scipy.fft

UPSAMPLE_RATIO = 16


def _qcap(vals):
    """Do a quadratic cap sub-sample max interpolation."""
    ndx = np.argmax(vals)
    if ndx in (0, len(vals) - 1):
        return float(ndx), vals[ndx]
    a, b, c = vals[(ndx - 1) : (ndx + 2)]
    p = (a - c) / (2 * (a - 2 * b + c))
    y_p = b - (a - c) * (p / 4)
    return ndx + p, y_p


def analyze_complex_ipr(chip, offset_rc, search_dist=None):
    """TODO: docstring (signature may change with plotting/measurements)"""
    if search_dist is None:
        search_dist = min(chip.shape) // 2 - 1

    # Upsample
    chip_upsampled = _upsample_1d(
        _upsample_1d(chip, UPSAMPLE_RATIO, pixel_shift=offset_rc[1]).T,
        UPSAMPLE_RATIO,
        pixel_shift=offset_rc[0],
    ).T

    # Find brightest pixel in upsampled chip
    sub_chip_start = (
        int(chip_upsampled.shape[0] // 2 - search_dist * UPSAMPLE_RATIO),
        int(chip_upsampled.shape[1] // 2 - search_dist * UPSAMPLE_RATIO),
    )
    sub_chip = chip_upsampled[
        sub_chip_start[0] : int(
            chip_upsampled.shape[0] // 2 + search_dist * UPSAMPLE_RATIO + 1
        ),
        sub_chip_start[1] : int(
            chip_upsampled.shape[1] // 2 + search_dist * UPSAMPLE_RATIO + 1
        ),
    ]
    max_pixel = np.unravel_index(
        np.argmax(np.abs(sub_chip)), sub_chip.shape
    ) + np.array(sub_chip_start)

    # Upsample slices (again) to measure peak location
    peak_phase = np.angle(chip_upsampled[tuple(max_pixel)])
    chip_upsampled *= np.exp(-1j * peak_phase)
    tgt_vs_rc = [
        _upsample_1d(chip_upsampled[:, max_pixel[1]], UPSAMPLE_RATIO),
        _upsample_1d(chip_upsampled[max_pixel[0], :], UPSAMPLE_RATIO),
    ]

    tgt_pixel = [x.size // 2 for x in tgt_vs_rc]
    offset_rc = np.full((2,), np.nan)
    peak_rc = np.full((2,), np.nan)
    for dim in range(2):
        this_offset_upsampled, this_peak = _qcap(
            np.abs(
                tgt_vs_rc[dim][
                    (max_pixel[dim] - 1) * UPSAMPLE_RATIO : (max_pixel[dim] + 1)
                    * UPSAMPLE_RATIO
                ]
            )
        )
        offset_rc[dim] = (
            this_offset_upsampled
            - (tgt_pixel[dim] - (max_pixel[dim] - 1) * UPSAMPLE_RATIO)
        ) / float(UPSAMPLE_RATIO**2)
        peak_rc[dim] = this_peak

    retval = {"offset_rc": offset_rc, "peak_power": max(peak_rc) ** 2}
    return retval


def _upsample_1d(image, factor, pixel_shift=0.0):
    image = _fft_ops(
        image.shape[-1], image.shape[-1], 1.0 / image.shape[-1], -1, True, True, image
    )
    image *= np.exp(
        1j
        * 2.0
        * np.pi
        * pixel_shift
        * (np.arange(image.shape[-1]) - image.shape[-1] // 2)
        / image.shape[-1]
    )
    image = _fft_ops(
        int(image.shape[-1] * factor),
        int(image.shape[-1] * factor),
        1.0,
        1,
        True,
        True,
        image,
    )
    return image


def _fft_ops(n_fft, n_out, scale, sign, fold_in, fold_out, image):
    """Performs 1D FFT or IFFT of an array, along with scaling, sizing,
        and fold-in/fold-out operations using the scipy fft pack as the
        FFT "engine".

    Parameters
    ----------
    n_fft : int
        The size of FFT to perform.
    n_out : int
        The number of output samples.
    scale : float
        The scaling to be applied to resulting FFT.
    sign : int
        The sign of the exponent to use in the FFT.
    fold_in : bool
        If ``True``, the input is folded.

        If ``False``, the input remains the same.
    fold_out : bool
        If ``True``, the output is folded.

        If ``False``, the output remains the same.
    image : array-like
        The input data to FFT.

    Returns
    -------
    ndarray
        Transformed ``image``
    """
    n_in = image.shape[-1]
    if n_fft < n_in or n_fft < n_out:
        raise ValueError("Invalid FFT size")

    tmp = np.zeros(image.shape[:-1] + (n_fft,), dtype=image.dtype)

    if fold_in:
        center = n_in // 2
        remain = n_in - center
        tmp[..., 0:remain] = image[..., center:]
        tmp[..., -center:] = image[..., 0:center]
    else:
        tmp[..., 0:n_in] = image[..., 0:]

    if sign < 0:
        tmp = scipy.fft.fft(tmp, n_fft) * scale
    else:
        # Additional scale by n_fft to remove ifft's 1/n_fft scaling
        tmp = scipy.fft.ifft(tmp, n_fft) * (n_fft * scale)

    output = np.zeros(tmp.shape[:-1] + (n_out,), dtype=tmp.dtype)

    if fold_out:
        center = n_out // 2
        remain = n_out - center
        output[..., 0:center] = tmp[..., -center:]
        output[..., center:] = tmp[..., 0:remain]
    else:
        output[..., :] = tmp[..., :n_out]

    return output


class NdArrJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
