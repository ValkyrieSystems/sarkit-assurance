import numpy as np
import sarkit.sicd as sksicd

from sarkit_assurance import _remap


def test_simple_sicd_remap_re32f_im32f():
    rng = np.random.default_rng()
    cpx = np.empty(shape=(100, 200), dtype=sksicd.PIXEL_TYPES["RE32F_IM32F"]["dtype"])
    cpx.real = rng.uniform(-1000, 1000, cpx.shape).astype(dtype=cpx.real.dtype)
    cpx.imag = rng.uniform(-1000, 1000, cpx.shape).astype(dtype=cpx.imag.dtype)

    img = _remap.simple_sicd_remap(cpx)
    assert img.dtype == np.uint8
    assert img.shape == cpx.shape
    amp = np.abs(cpx)
    minloc = np.unravel_index(amp.argmin(), amp.shape)
    maxloc = np.unravel_index(amp.argmax(), amp.shape)
    assert img[minloc] == 0
    assert img[maxloc] == 255


def test_simple_sicd_remap_re16i_im16i():
    rng = np.random.default_rng()
    cpx = np.empty(shape=(100, 200), dtype=sksicd.PIXEL_TYPES["RE16I_IM16I"]["dtype"])
    low = np.iinfo(cpx["real"].dtype).min
    high = np.iinfo(cpx["real"].dtype).max + 1
    cpx["real"] = rng.integers(low, high, size=cpx.shape, dtype=cpx["real"].dtype)
    cpx["imag"] = rng.integers(low, high, size=cpx.shape, dtype=cpx["imag"].dtype)

    img = _remap.simple_sicd_remap(cpx)
    assert img.dtype == np.uint8
    assert img.shape == cpx.shape
    amp = np.sqrt(cpx["real"] ** 2.0 + cpx["imag"] ** 2.0)
    minloc = np.unravel_index(amp.argmin(), amp.shape)
    maxloc = np.unravel_index(amp.argmax(), amp.shape)
    assert img[minloc] == 0
    assert img[maxloc] == 255


def test_simple_sicd_remap_amp8i_phs8i():
    rng = np.random.default_rng()
    cpx = np.empty(shape=(100, 200), dtype=sksicd.PIXEL_TYPES["AMP8I_PHS8I"]["dtype"])
    low = np.iinfo(cpx["amp"].dtype).min
    high = np.iinfo(cpx["phase"].dtype).max + 1
    cpx["amp"] = rng.integers(low, high, size=cpx.shape, dtype=cpx["amp"].dtype)
    cpx["phase"] = rng.integers(low, high, size=cpx.shape, dtype=cpx["phase"].dtype)

    img = _remap.simple_sicd_remap(cpx)
    assert img.dtype == np.uint8
    assert img.shape == cpx.shape
    amp = cpx["amp"]
    minloc = np.unravel_index(amp.argmin(), amp.shape)
    maxloc = np.unravel_index(amp.argmax(), amp.shape)
    assert img[minloc] == 0
    assert img[maxloc] == 255
