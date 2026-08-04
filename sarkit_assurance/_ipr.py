"""Common utilities for IPR tools"""

import dataclasses
import json
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
import scipy.fft
import scipy.signal

from . import _remap

UPSAMPLE_RATIO = 16
NEAR_PK_NYQ_SAMPLES_KEEP = 10
FIG_WIDTH_PX = 350
FIG_HEIGHT_PX = 325


@dataclasses.dataclass
class Axis:
    x0: float
    xss: float
    name: str = ""
    units: str = ""

    def get_label(self) -> str:
        if not self.units:
            return self.name
        return f"{self.name} ({self.units})"

    def rescale(self, sf: float, *, name=None, units=None):
        self.x0 *= sf
        self.xss *= sf
        if name is not None:
            self.name = name
        if units is not None:
            self.units = units


@dataclasses.dataclass
class Cut:
    data: np.ndarray
    domain: Axis


@dataclasses.dataclass
class Chip:
    data: np.ndarray
    row: Axis
    col: Axis


@dataclasses.dataclass
class IprParts:
    chip: Chip
    vs_row: Cut
    vs_col: Cut


def _qcap(vals):
    """Do a quadratic cap sub-sample max interpolation."""
    ndx = np.argmax(vals)
    if ndx in (0, len(vals) - 1):
        return float(ndx), vals[ndx]
    a, b, c = vals[(ndx - 1) : (ndx + 2)]
    p = (a - c) / (2 * (a - 2 * b + c))
    y_p = b - (a - c) * (p / 4)
    return ndx + p, y_p


def estimate_peak(chip, *, offset_rc=(0.0, 0.0), search_dist=None):
    """TODO: docstring"""
    if search_dist is None:
        search_dist = min(chip.shape) // 2 - 1

    # Upsample and recenter
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
    sub_chip_upsampled = chip_upsampled[
        sub_chip_start[0] : int(
            chip_upsampled.shape[0] // 2 + search_dist * UPSAMPLE_RATIO + 1
        ),
        sub_chip_start[1] : int(
            chip_upsampled.shape[1] // 2 + search_dist * UPSAMPLE_RATIO + 1
        ),
    ]
    est_peak_rc_upsampled = np.unravel_index(
        np.argmax(np.abs(sub_chip_upsampled)), sub_chip_upsampled.shape
    ) + np.array(sub_chip_start)

    # Upsample slices (again) to measure peak location
    peak_phase = np.angle(chip_upsampled[tuple(est_peak_rc_upsampled)])
    chip_upsampled *= np.exp(-1j * peak_phase)
    tgt_vs_rc = [
        _upsample_1d(chip_upsampled[:, est_peak_rc_upsampled[1]], UPSAMPLE_RATIO),
        _upsample_1d(chip_upsampled[est_peak_rc_upsampled[0], :], UPSAMPLE_RATIO),
    ]

    est_peak_rc = np.full((2,), np.nan)
    peak_amp = np.full((2,), np.nan)
    for dim in range(2):
        near_pk_slice_upsampled2 = slice(
            (est_peak_rc_upsampled[dim] - 1) * UPSAMPLE_RATIO,
            (est_peak_rc_upsampled[dim] + 1) * UPSAMPLE_RATIO,
        )
        this_offset_upsampled2, this_peak = _qcap(
            np.abs(tgt_vs_rc[dim][near_pk_slice_upsampled2])
        )
        est_peak_rc[dim] = (near_pk_slice_upsampled2.start + this_offset_upsampled2) / (
            UPSAMPLE_RATIO**2
        )
        peak_amp[dim] = this_peak

    tgt_peak_rc = np.array([x.size // 2 for x in tgt_vs_rc]) / (UPSAMPLE_RATIO**2)
    est_offset_rc = est_peak_rc - tgt_peak_rc
    peak_power = max(peak_amp) ** 2
    iprparts = IprParts(
        chip=Chip(
            data=chip_upsampled,
            row=Axis(-tgt_peak_rc[0], UPSAMPLE_RATIO ** (-1), "rows from chip center"),
            col=Axis(-tgt_peak_rc[1], UPSAMPLE_RATIO ** (-1), "cols from chip center"),
        ),
        vs_row=Cut(
            data=tgt_vs_rc[0] / peak_amp[0],
            domain=Axis(
                -est_peak_rc[0], UPSAMPLE_RATIO ** (-2), "rows from estimated peak"
            ),
        ),
        vs_col=Cut(
            data=tgt_vs_rc[1] / peak_amp[1],
            domain=Axis(
                -est_peak_rc[1], UPSAMPLE_RATIO ** (-2), "cols from estimated peak"
            ),
        ),
    )
    return est_offset_rc, peak_power, iprparts


def downsample_chip(chip: Chip):
    """Undo the upsampling from estimate_peak"""
    data_ds = _downsample_1d(
        _downsample_1d(chip.data, UPSAMPLE_RATIO).T, UPSAMPLE_RATIO
    ).T
    chip_ds = dataclasses.replace(chip, data=data_ds)
    for n, rc in enumerate(("row", "col")):
        dim: Axis = getattr(chip_ds, rc)
        dim.xss *= UPSAMPLE_RATIO
        dim.x0 = -dim.xss * (data_ds.shape[n] // 2)
    return chip_ds


def create_spectral_chip(
    iprparts: IprParts,
    sgn: float,
) -> IprParts:

    to_sf_func = scipy.fft.fft2 if sgn < 0 else scipy.fft.ifft2
    kchip_data = scipy.fft.fftshift(to_sf_func(scipy.fft.ifftshift(iprparts.chip.data)))
    kss = 1.0 / np.array(kchip_data.shape)
    kchip = Chip(
        data=kchip_data,
        row=Axis(x0=-kss[0] * (kchip_data.shape[0] // 2), xss=kss[0], name="ΔKrow"),
        col=Axis(x0=-kss[1] * (kchip_data.shape[1] // 2), xss=kss[1], name="ΔKcol"),
    )

    def get_spatfreq_cut(x, n_keep, rc):
        k_x = _fft_ops(x.size, n_keep, 1.0 / x.size, sgn, True, True, x)
        k_xss = 1 / (n_keep)
        k_x0 = -k_xss * (n_keep // 2)
        return Cut(data=k_x, domain=Axis(k_x0, k_xss, f"ΔK{rc}"))

    vs_krow = get_spatfreq_cut(iprparts.vs_row.data, kchip_data.shape[0], "row")
    vs_kcol = get_spatfreq_cut(iprparts.vs_col.data, kchip_data.shape[1], "col")
    return IprParts(
        chip=kchip,
        vs_row=vs_krow,
        vs_col=vs_kcol,
    )


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


def _downsample_1d(image, factor):
    image = _fft_ops(
        image.shape[-1],
        image.shape[-1] // factor,
        1.0 / image.shape[-1],
        -1,
        True,
        True,
        image,
    )
    image = _fft_ops(image.shape[-1], image.shape[-1], 1.0, 1, True, True, image)
    return image


def _fft_ops(n_fft, n_out, scale, sign, fold_in, fold_out, image):
    """Performs 1D FFT or IFFT of an array, along with scaling, sizing,
        and fold-in/fold-out operations using the scipy fft as the
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


def linear_range_slice(x0, xss, min, max):
    low_index = int(np.ceil((min - x0) / xss))
    past_high_index = int(np.ceil((max - x0) / xss))
    return slice(low_index, past_high_index)


def _get_power_db(sig):
    amp = np.abs(sig)
    return np.where(amp > 0, 20.0 * np.log10(amp), np.nan)


def create_tir_trace(chip: Chip):
    remapped_chip = _remap.simple_log_remap(
        np.abs(chip.data) ** 2,
        cut_low_frac=0.0,
        cut_high_frac=0.0,
        min_low_relative=1e-10,
    )
    # for plotly, x is columns (horizontal), y is rows (vertical)
    im_trace = go.Heatmap(
        z=remapped_chip,
        x0=chip.col.x0,
        dx=chip.col.xss,
        y0=chip.row.x0,
        dy=chip.row.xss,
        showscale=False,
        colorscale="Greys",
        reversescale=True,
    )
    return im_trace


def create_sf_amp_phase_traces(kchip: Chip):
    kchip_amp = _remap.scale_to_byte(np.abs(kchip.data))
    kchip_phase = _remap.scale_to_byte(np.angle(kchip.data))

    # for plotly, x is columns (horizontal), y is rows (vertical)
    traces = (
        go.Heatmap(
            z=z,
            x0=kchip.col.x0,
            dx=kchip.col.xss,
            y0=kchip.row.x0,
            dy=kchip.row.xss,
            showscale=False,
            colorscale="Viridis",
        )
        for z in (kchip_amp, kchip_phase)
    )
    return traces


def create_sf_power_trace(iprslice: Cut, bw: float):
    pwr_db = _get_power_db(iprslice.data)
    in_band_med = np.median(
        pwr_db[
            linear_range_slice(iprslice.domain.x0, iprslice.domain.xss, -bw / 2, bw / 2)
        ]
    )
    return go.Scatter(
        y=pwr_db - in_band_med,
        x0=iprslice.domain.x0,
        dx=iprslice.domain.xss,
        line_color="blue",
    )


def create_sf_phase_trace(iprslice: Cut, bw: float):
    phase_deg = np.full(iprslice.data.shape, np.nan)
    inband = linear_range_slice(
        iprslice.domain.x0, iprslice.domain.xss, -bw / 2, bw / 2
    )
    phase_deg[inband] = scipy.signal.detrend(
        np.rad2deg(np.angle(iprslice.data[inband]))
    )
    return go.Scatter(
        y=phase_deg, x0=iprslice.domain.x0, dx=iprslice.domain.xss, line_color="blue"
    )


def plot_ipr(
    iprparts: IprParts,
    k_iprparts: IprParts,
    spacing_rc: Sequence[float],
    bw_rc: Sequence[float],
) -> go.Figure:
    # TODO: docstring/interface - may change when more measurements are added (e.g. pass spatial freq cuts in)

    # downselect row/col cut - only keep portion near peak, convert to nyquist samples
    def get_nyqsamples_near_pk(iprcut: Cut, ss: float, bw: float) -> Cut:
        xss_nyqsamp = iprcut.domain.xss * ss * bw
        x0_nyqsamp = iprcut.domain.x0 * ss * bw
        keep_slice = linear_range_slice(
            x0_nyqsamp,
            xss_nyqsamp,
            -NEAR_PK_NYQ_SAMPLES_KEEP / 2,
            NEAR_PK_NYQ_SAMPLES_KEEP / 2,
        )
        x0_nyqsamp_near_pk = x0_nyqsamp + keep_slice.start * xss_nyqsamp
        new_domain = dataclasses.replace(
            iprcut.domain,
            x0=x0_nyqsamp_near_pk,
            xss=xss_nyqsamp,
            units="Nyquist Samples",
        )
        return Cut(
            data=iprcut.data[keep_slice],
            domain=new_domain,
        )

    vs_row_near_pk = get_nyqsamples_near_pk(iprparts.vs_row, spacing_rc[0], bw_rc[0])
    vs_col_near_pk = get_nyqsamples_near_pk(iprparts.vs_col, spacing_rc[1], bw_rc[1])

    plot_titles: tuple[str, ...] = (
        "Target Image Response (TIR)<br>Relative to Projected Location",
        "",
        "Spatial Frequency (SF) Amplitude",
        "Spatial Frequency (SF) Phase",
    )
    for arrow in ("\N{LEFT RIGHT ARROW}", "\N{UP DOWN ARROW}"):
        plot_titles += (
            f"TIR {arrow} Power Slice<br>Relative to Estimated Location",
            f"TIR {arrow} Phase Slice<br>Relative to Estimated Location",
            f"SF {arrow} Power Slice",
            f"SF {arrow} Phase Slice",
        )

    rows = 3
    cols = 4
    fig = go.Figure().set_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=plot_titles,
        horizontal_spacing=0.4 / cols,
    )
    fig.update_layout(
        showlegend=False,
        width=cols * FIG_WIDTH_PX,
        height=rows * FIG_HEIGHT_PX,
    )

    def get_rc_phase(sig):
        return np.rad2deg(np.angle(sig * np.exp(-0.5j * np.pi))) + 90

    fig.add_trace(trace=create_tir_trace(iprparts.chip), row=1, col=1)
    # TODO: add table trace
    kchip_amp, kchip_phase = create_sf_amp_phase_traces(k_iprparts.chip)
    fig.add_trace(trace=kchip_amp, row=1, col=3)
    fig.add_trace(trace=kchip_phase, row=1, col=4)

    # rc/krc
    for cut, k_cut, bw, subplot_row in zip(
        (vs_row_near_pk, vs_col_near_pk),
        (k_iprparts.vs_row, k_iprparts.vs_col),
        bw_rc,
        (3, 2),
    ):
        fig.add_scatter(
            y=_get_power_db(cut.data),
            x0=cut.domain.x0,
            dx=cut.domain.xss,
            line_color="blue",
            row=subplot_row,
            col=1,
        )
        fig.add_scatter(
            y=get_rc_phase(cut.data),
            x0=cut.domain.x0,
            dx=cut.domain.xss,
            line_color="blue",
            row=subplot_row,
            col=2,
        )

        fig.add_trace(trace=create_sf_power_trace(k_cut, bw), row=subplot_row, col=3)
        fig.add_vline(x=-bw / 2, line_color="red", row=subplot_row, col=3)
        fig.add_vline(x=bw / 2, line_color="red", row=subplot_row, col=3)

        fig.add_trace(trace=create_sf_phase_trace(k_cut, bw), row=subplot_row, col=4)
        fig.add_vline(x=-bw / 2, line_color="red", row=subplot_row, col=4)
        fig.add_vline(x=bw / 2, line_color="red", row=subplot_row, col=4)

    # Customize axes
    fig.update_xaxes(title_text=iprparts.chip.col.get_label(), row=1, col=1)
    fig.update_yaxes(
        title_text=iprparts.chip.row.get_label(), autorange="reversed", row=1, col=1
    )

    for col in (3, 4):
        fig.update_xaxes(title_text=k_iprparts.chip.col.get_label(), row=1, col=col)
        fig.update_yaxes(
            title_text=k_iprparts.chip.row.get_label(),
            autorange="reversed",
            row=1,
            col=col,
        )

    # Update x-axes
    fig.update_xaxes(
        title_text=vs_col_near_pk.domain.get_label(), range=[-4, 4], row=2, col=1
    )
    fig.update_xaxes(
        title_text=vs_col_near_pk.domain.get_label(), range=[-4, 4], row=2, col=2
    )
    fig.update_xaxes(
        title_text=k_iprparts.vs_col.domain.get_label(),
        range=np.array([-0.5, 0.5]) / spacing_rc[1],
        row=2,
        col=3,
    )
    fig.update_xaxes(
        title_text=k_iprparts.vs_col.domain.get_label(),
        range=np.array([-0.5, 0.5]) / spacing_rc[1],
        row=2,
        col=4,
    )

    fig.update_xaxes(
        title_text=vs_row_near_pk.domain.get_label(), range=[-4, 4], row=3, col=1
    )
    fig.update_xaxes(
        title_text=vs_row_near_pk.domain.get_label(), range=[-4, 4], row=3, col=2
    )
    fig.update_xaxes(
        title_text=k_iprparts.vs_row.domain.get_label(),
        range=np.array([-0.5, 0.5]) / spacing_rc[0],
        row=3,
        col=3,
    )
    fig.update_xaxes(
        title_text=k_iprparts.vs_row.domain.get_label(),
        range=np.array([-0.5, 0.5]) / spacing_rc[0],
        row=3,
        col=4,
    )

    # Update y-axes
    for subplot_row in (2, 3):
        fig.update_yaxes(
            title_text="Power (dB)", range=[-30, 3], row=subplot_row, col=1
        )
        fig.update_yaxes(
            title_text="Phase (deg)", range=[-20, 200], row=subplot_row, col=2
        )
        fig.update_yaxes(title_text="Power (dB)", range=[-5, 5], row=subplot_row, col=3)
        fig.update_yaxes(
            title_text="Phase (deg)", range=[-200, 200], row=subplot_row, col=4
        )

    return fig
