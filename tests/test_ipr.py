import numpy as np
import pytest
import scipy.fft
import scipy.signal

from sarkit_assurance import _ipr


@pytest.mark.parametrize("upsampled_offset", [(0, 0), (1, 0), (-2, 3), (0, -1)])
def test_estimate_peak(upsampled_offset):
    upsampled_offset = np.asarray(upsampled_offset)
    upsampled_edge = 1024
    resamp_factor = 4

    chip_edge = 200

    kspace_edge = int(upsampled_edge * 0.8)

    kctr = np.array((upsampled_edge // 2, upsampled_edge // 2))
    k_data = np.zeros((upsampled_edge, upsampled_edge), dtype=np.complex64)
    k_data[
        (kctr[0] - kspace_edge // 2) : (kctr[0] + kspace_edge // 2),
        (kctr[1] - kspace_edge // 2) : (kctr[1] + kspace_edge // 2),
    ] = 1
    z = scipy.fft.fftshift(scipy.fft.fft2(k_data))
    z_us = z[
        (
            upsampled_edge // 2 + upsampled_offset[0] - (chip_edge * resamp_factor) // 2
        ) : (
            upsampled_edge // 2 + upsampled_offset[0] + (chip_edge * resamp_factor) // 2
        ),
        (
            upsampled_edge // 2 + upsampled_offset[1] - (chip_edge * resamp_factor) // 2
        ) : (
            upsampled_edge // 2 + upsampled_offset[1] + (chip_edge * resamp_factor) // 2
        ),
    ]
    z_ds = scipy.signal.resample(scipy.signal.resample(z_us, chip_edge).T, chip_edge).T
    z_ds /= np.abs(z_ds).max()

    offset = -upsampled_offset // resamp_factor

    def assert_exact_peak(pk):
        assert pk == pytest.approx(1.0)

    def assert_interp_peak(pk):
        assert pk > 1.0

    if not np.any(upsampled_offset):
        # if no offset, peak should match
        assert_peak = assert_exact_peak
    else:
        # otherwise, peak should be higher than observed due to quadcap
        assert_peak = assert_interp_peak

    offset0, pk0, _ = _ipr.estimate_peak(z_ds, offset_rc=(0.0, 0.0))
    assert offset0 == pytest.approx(offset, abs=_ipr.UPSAMPLE_RATIO**2)
    assert_peak(pk0)

    offset_off, pk_off, _ = _ipr.estimate_peak(z_ds, offset_rc=offset)
    assert offset_off == pytest.approx(0.0, abs=_ipr.UPSAMPLE_RATIO**2)
    assert_peak(pk_off)

    # shifting search window far away and limiting distance doesn't find peak
    _, pk_bad, _ = _ipr.estimate_peak(z_ds, offset_rc=(24.0, 24.0), search_dist=2)
    assert pk_bad < 0.5
