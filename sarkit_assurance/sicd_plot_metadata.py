"""Utilities for generating plots of SICD metadata"""

import argparse
import html
import itertools
import pathlib

import lxml.etree
import numpy as np
import numpy.polynomial.polynomial as npp
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psp
import sarkit.sicd as sksicd
import shapely.affinity
import shapely.geometry
from scipy import constants

from . import _plot_metadata

try:
    from smart_open import open
except ImportError:
    pass


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _scale_to_byte(data):
    min_val = data.min()
    max_val = data.max()
    return ((data - min_val) * 256 / (max_val - min_val)).clip(0, 255).astype(np.uint8)


def valid_data_polygon(sicd_ew):
    """Generate the ValidData polygon object"""
    vertices = sicd_ew["ImageData"].get("ValidData", None)
    if vertices is None:
        # Use edges of full image
        nrows = sicd_ew["ImageData"]["FullImage"]["NumRows"]
        ncols = sicd_ew["ImageData"]["FullImage"]["NumCols"]

        vertices = [(0, 0), (0, ncols - 1), (nrows - 1, ncols - 1), (nrows - 1, 0)]

    return shapely.geometry.Polygon(vertices)


def pfa_phd_polygon(sicd_ew):
    """Polygon describing the PHD available for PFA."""
    tstart = sicd_ew["ImageFormation"]["TStartProc"]
    tend = sicd_ew["ImageFormation"]["TEndProc"]
    times = np.linspace(tstart, tend, 101)

    pa_poly = sicd_ew["PFA"]["PolarAngPoly"]
    angles = npp.polyval(times, pa_poly)

    def fx_pa_to_krg_kaz(fx, polar_angle):
        sf_poly = sicd_ew["PFA"]["SpatialFreqSFPoly"]
        ksf = npp.polyval(polar_angle, sf_poly)
        kap = fx * 2.0 / constants.speed_of_light * ksf
        krg = kap * np.cos(polar_angle)
        kaz = kap * np.sin(polar_angle)
        return kaz, krg

    min_tx = sicd_ew["RadarCollection"]["TxFrequency"]["Min"]
    max_tx = sicd_ew["RadarCollection"]["TxFrequency"]["Max"]
    lower_kaz, lower_krg = fx_pa_to_krg_kaz(min_tx, angles)
    upper_kaz, upper_krg = fx_pa_to_krg_kaz(max_tx, angles)

    return shapely.geometry.Polygon(
        zip(
            np.concatenate((lower_kaz, upper_kaz[::-1])),
            np.concatenate((lower_krg, upper_krg[::-1])),
        )
    )


def pfa_rect_aperture_polygon(sicd_ew):
    """Polygon describing the PFA rectangular aperture"""
    krg1 = sicd_ew["PFA"]["Krg1"]
    krg2 = sicd_ew["PFA"]["Krg2"]
    kaz1 = sicd_ew["PFA"]["Kaz1"]
    kaz2 = sicd_ew["PFA"]["Kaz2"]

    return shapely.geometry.Polygon(
        [(kaz1, krg1), (kaz1, krg2), (kaz2, krg2), (kaz2, krg1)]
    )


def _plot_polygon(fig, poly, swap_xy=False, **kwargs):
    x_vals, y_vals = zip(*poly.exterior.coords[:])
    if swap_xy:
        x_vals, y_vals = y_vals, x_vals
    fig.add_scatter(x=x_vals, y=y_vals, **kwargs)


def _plot_box(fig, min_x, min_y, max_x, max_y, **kwargs):
    _plot_polygon(fig, shapely.geometry.box(min_x, min_y, max_x, max_y), **kwargs)


class Plotter(_plot_metadata.Plotter):
    """A SICD metadata plotter class."""

    def __init__(self, file, title, *, use_sample_data=False):
        # TODO: harmonize interface with CPHD (accommodate XML-only?)
        sample_data = None
        try:
            self.xml = lxml.etree.parse(file)
        except lxml.etree.ParseError:
            file.seek(0)
            with sksicd.NitfReader(file) as r:
                self.xml = r.metadata.xmltree
                if use_sample_data:
                    sample_data = r.read_image()
        self.ew = sksicd.ElementWrapper(self.xml.getroot())
        self.sample_data = sample_data
        if self.sample_data is not None:
            if self.ew["ImageData"]["PixelType"] == "RE16I_IM16I":
                self.sample_data = self.sample_data["real"].astype(
                    np.float32
                ) + 1j * self.sample_data["imag"].astype(np.float32)
            elif self.ew["ImageData"]["PixelType"] != "RE32F_IM32F":
                raise NotImplementedError()
            self.downsample_factor = max(
                1, int(np.floor(np.sqrt(np.prod(self.sample_data.shape) / (1 << 19))))
            )

        self.nominal_width = 1280
        self.nominal_height = 800

        super().__init__(title)

    def _grid_spacing(self):
        return np.array(
            [
                self.ew["Grid"]["Row"]["SS"],
                self.ew["Grid"]["Col"]["SS"],
            ]
        )

    def _pxl_to_xy(self, pixel):
        return (pixel - self.ew["ImageData"]["SCPPixel"]) * self._grid_spacing()

    def _xy_to_pxl(self, xy):
        return xy / self._grid_spacing() + self.ew["ImageData"]["SCPPixel"]

    def _kcoa_polys(self):
        return (self.ew["Grid"][x].get("DeltaKCOAPoly", [[0]]) for x in ("Row", "Col"))

    def plot_polar_format(self):
        """Plot the PFA aperture."""
        if self.xml.findtext("{*}ImageFormation/{*}ImageFormAlgo") != "PFA":
            return []

        fig = go.Figure()

        # center line
        fig.add_vline(x=0, line_color="gray", line_dash="dash")

        # reported PHD
        _plot_polygon(
            fig,
            pfa_phd_polygon(self.ew),
            fill="toself",
            name="Reported PHD",
            line=dict(color="green"),
        )

        # rect_aperture
        rect_aperture = pfa_rect_aperture_polygon(self.ew)
        _plot_polygon(
            fig,
            rect_aperture,
            fill="toself",
            name="Rectangular Aperture",
            line=dict(color="blue"),
        )

        # SCP BW
        imprespbw_row = self.ew["Grid"]["Row"]["ImpRespBW"]
        imprespbw_col = self.ew["Grid"]["Col"]["ImpRespBW"]
        k_c = rect_aperture.centroid.coords[0][-1]
        _plot_box(
            fig,
            -imprespbw_col / 2,
            k_c - imprespbw_row / 2,
            imprespbw_col / 2,
            k_c + imprespbw_row / 2,
            fill="toself",
            name="SCP ImpRespBW",
            line=dict(color="red"),
        )

        fig.update_xaxes(title_text="Kaz (cyc/m)", side="top")
        fig.update_yaxes(
            title_text="Krg (cyc/m)",
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
        )
        seg_id = self.xml.findtext("{*}ImageFormation/{*}SegmentIdentifier")
        fig.update_layout(
            title_text=self.format_title(f"PFA:{seg_id}"),
            title_y=0.97,
            height=self.nominal_height,
            width=self.nominal_width,
            meta="pfa",
        )
        return [fig]

    def plot_sampled_spatial_frequency_support(self):
        """Plot the spectral support."""
        if self.sample_data is None:
            return []

        fig = psp.make_subplots(
            rows=3,
            cols=4,
            specs=[
                [{"rowspan": 3}, {}, {}, {}],
                [None, {}, {}, {}],
                [None, {}, {}, {}],
            ],
            column_widths=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=["Chip Center Locations"] + ["NO VALID PIXELS"] * 9,
        )
        fig.update_annotations(
            yshift=20,  # move annotations up so that subplot titles don't collide with axes
            font_size=10,
        )
        fig.update_layout(
            showlegend=False,
            coloraxis={"colorscale": "gray", "showscale": False},
            height=self.nominal_height,
            width=self.nominal_width,
            meta="sampled_spatial_frequency_support",
        )
        valid_fig = self.plot_image_data(include_thumb=True, mark_scp=False)[0]
        for trace in valid_fig["data"]:
            fig.add_trace(trace, row=1, col=1)
        fig.update_xaxes(
            title_text="Columns (px)", side="top", constrain="domain", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Rows (px)",
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            row=1,
            col=1,
        )

        chip_size = 512
        left = chip_size // 2
        right = chip_size - left
        image_shape = (
            self.ew["ImageData"]["NumRows"],
            self.ew["ImageData"]["NumCols"],
        )

        first = (
            self.ew["ImageData"]["FirstRow"],
            self.ew["ImageData"]["FirstCol"],
        )
        valid_data = valid_data_polygon(self.ew).intersection(
            shapely.geometry.box(
                first[0], first[1], image_shape[0] + first[0], image_shape[1] + first[1]
            )
        )
        if valid_data.area == 0:
            return [fig]
        spacing = self._grid_spacing()
        k_ss = 1.0 / (np.asarray(spacing) * chip_size)
        delta_k = k_ss[:, np.newaxis] * (-left - 0.5, right - 0.5)
        krow_poly, kcol_poly = self._kcoa_polys()
        width = np.asarray(
            (
                self.ew["Grid"]["Row"]["ImpRespBW"],
                self.ew["Grid"]["Col"]["ImpRespBW"],
            )
        )

        def get_envelope_lengths(polygon):
            return np.ptp(np.array(polygon.envelope.bounds).reshape((2, 2)), axis=0)

        buffer_dist = min(
            chip_size // 2,
            *get_envelope_lengths(shapely.affinity.scale(valid_data, 1 / 4, 1 / 4)),
        )
        pixel_boundary = valid_data.buffer(-buffer_dist)
        if pixel_boundary.area == 0 or not hasattr(pixel_boundary, "exterior"):
            pixel_boundary = valid_data
        oriented_pixel_boundary = shapely.geometry.polygon.orient(
            pixel_boundary, sign=-1
        )  # orient clockwise
        envelope_lengths = get_envelope_lengths(oriented_pixel_boundary)
        scaled_pixel_boundary = shapely.affinity.scale(
            oriented_pixel_boundary, *np.reciprocal(envelope_lengths)
        )
        scaled_vertices = [
            scaled_pixel_boundary.exterior.interpolate(d, normalized=True).coords[0]
            for d in np.linspace(0, 1, 8, endpoint=False)
        ]
        vertex_directions = unit(
            np.array(scaled_vertices) - scaled_pixel_boundary.centroid.coords[0]
        )
        top_leftish_vertex = np.argmax(np.dot(vertex_directions, [-1, -1]))
        scaled_border = shapely.geometry.Polygon(
            scaled_vertices[top_leftish_vertex:] + scaled_vertices[:top_leftish_vertex]
        )
        border = list(
            shapely.affinity.scale(scaled_border, *envelope_lengths).exterior.coords
        )[:-1]
        if pixel_boundary.intersects(
            shapely.geometry.Point(tuple(self.ew["ImageData"]["SCPPixel"]))
        ):
            sample_pixels = border + [tuple(self.ew["ImageData"]["SCPPixel"])]
        else:
            sample_pixels = border + [tuple(shapely.centroid(pixel_boundary).coords[0])]
        sample_pixels = [
            [int(p) for p in sample_pixels[x]] for x in [0, 1, 2, 7, 8, 3, 6, 5, 4]
        ]  # round & rearrange border around scp
        sample_coords = self._pxl_to_xy(sample_pixels)
        delta_k_ctrs = [
            (npp.polyval2d(*xy, krow_poly), npp.polyval2d(*xy, kcol_poly))
            for xy in sample_coords
        ]

        titles = ["Chip Center Locations"] + [
            f"XY = ({x:.0f}, {y:.0f}): KCTR = ({kx:.3f}, {ky:.3f})"
            for (x, y), (kx, ky) in zip(sample_coords, delta_k_ctrs)
        ]
        fig.add_scatter(
            x=[x[1] for x in sample_pixels],
            y=[x[0] for x in sample_pixels],
            mode="markers",
            marker=dict(size=12, color="cyan"),
            row=1,
            col=1,
        )

        for index, (pixel, delta_k_ctr) in enumerate(zip(sample_pixels, delta_k_ctrs)):
            # Pad the chip to CHIP_SIZE in case it's eroded
            rchip = self.sample_data[
                max(pixel[0] - left, 0) : pixel[0] + right,
                max(pixel[1] - left, 0) : pixel[1] + right,
            ]
            chip = np.zeros((chip_size, chip_size), rchip.dtype)
            chip[: rchip.shape[0], : rchip.shape[1]] = rchip
            fchip = np.abs(np.fft.fftshift(np.fft.fft2(chip))) ** 2
            fchip = np.log(fchip.clip(fchip.max() / 1e5, None))
            fchip = _scale_to_byte(fchip)
            rc = dict(row=(index // 3 + 1), col=(index % 3 + 2))
            fig.add_heatmap(
                z=fchip,
                x0=delta_k[1][0],
                y0=delta_k[0][0],
                dx=k_ss[1],
                dy=k_ss[0],
                showscale=False,
                colorscale="gray",
                **rc,
            )

            delta_k_min = (
                np.remainder(
                    delta_k_ctr - width / 2.0 - delta_k[:, 0], 1.0 / np.asarray(spacing)
                )
                + delta_k[:, 0]
            )
            delta_k_max = (
                np.remainder(
                    delta_k_ctr + width / 2.0 - delta_k[:, 0], 1.0 / np.asarray(spacing)
                )
                + delta_k[:, 0]
            )
            fig.add_vline(x=delta_k_min[1], line=dict(color="blue", width=2), **rc)
            fig.add_vline(x=delta_k_max[1], line=dict(color="blue", width=2), **rc)
            fig.add_hline(y=delta_k_min[0], line=dict(color="blue", width=2), **rc)
            fig.add_hline(y=delta_k_max[0], line=dict(color="blue", width=2), **rc)
            fig.update_xaxes(ticks="outside", side="top", **rc)
            fig.update_yaxes(ticks="outside", autorange="reversed", **rc)
        for ann, title in zip(fig.layout.annotations, titles):
            ann.text = title
        return [fig]

    def plot_segments(self):
        """Plot showing the size and location of the segments."""
        segments = self.ew["RadarCollection"]["Area"]["Plane"]["SegmentList"]["Segment"]
        if not segments:
            return []

        full_area_shape = (
            self.ew["RadarCollection"]["Area"]["Plane"]["XDir"]["NumLines"],
            self.ew["RadarCollection"]["Area"]["Plane"]["YDir"]["NumSamples"],
        )

        fig = go.Figure()
        full_area = shapely.geometry.box(
            0, 0, full_area_shape[1] - 1, full_area_shape[0] - 1
        )
        x_vals, y_vals = zip(*full_area.exterior.coords[:])
        fig.add_scatter(
            x=x_vals, y=y_vals, name="Radar Collection Area", line=dict(color="black")
        )
        colors_set = itertools.cycle(["blue", "green", "yellow", "gray", "red"])
        for seg in segments:
            start_line = seg["StartLine"]
            end_line = seg["EndLine"]
            start_samp = seg["StartSample"]
            end_samp = seg["EndSample"]
            seg_id = seg["Identifier"]

            this_color = next(colors_set)
            _plot_box(
                fig,
                start_samp,
                start_line,
                end_samp,
                end_line,
                fill="toself",
                line=dict(color=this_color, dash="dash"),
                mode="lines",
                name=seg_id,
                legendgroup=seg_id,
            )

            fig.add_scatter(
                x=[(start_samp + end_samp) / 2.0],
                y=[(start_line + end_line) / 2.0],
                text=seg_id,
                mode="text",
                legendgroup=seg_id,
                showlegend=False,
            )
        fig.update_xaxes(title_text="Samples", side="top")
        fig.update_yaxes(
            title_text="Lines", autorange="reversed", scaleanchor="x", scaleratio=1
        )
        fig.update_layout(
            title_text=self.format_title("SegmentList"),
            title_y=0.97,
            height=self.nominal_height,
            width=self.nominal_width,
            meta="segments",
        )
        return [fig]

    def plot_image_data(self, include_thumb=False, mark_scp=True):
        """Plot showing ImageData."""
        full_image_shape = (
            self.ew["ImageData"]["FullImage"]["NumRows"],
            self.ew["ImageData"]["FullImage"]["NumCols"],
        )

        image_shape = (
            self.ew["ImageData"]["NumRows"],
            self.ew["ImageData"]["NumCols"],
        )

        first = (
            self.ew["ImageData"]["FirstRow"],
            self.ew["ImageData"]["FirstCol"],
        )

        scp_pixel = self.ew["ImageData"]["SCPPixel"]

        fig = go.Figure()

        if self.sample_data is not None and include_thumb:
            tmp = self.sample_data.real**2 + self.sample_data.imag**2
            tmp = np.log10(tmp.clip(tmp.max() / 1e6, None))
            tmp = downsample_all_dims(tmp, self.downsample_factor)
            tmp = _scale_to_byte(tmp)
            downsamp_offset = (self.downsample_factor - 1) / 2.0
            fig.add_heatmap(
                z=tmp,
                x0=(downsamp_offset + first[1]),
                y0=(downsamp_offset + first[0]),
                dx=self.downsample_factor,
                dy=self.downsample_factor,
                showscale=False,
                colorscale="gray",
            )

        _plot_box(
            fig,
            0,
            0,
            *full_image_shape[::-1],
            name="FullImage",
            line_color="green",
            mode="lines",
        )
        _plot_box(
            fig,
            *first[::-1],
            *image_shape[::-1],
            name="Image",
            line_color="blue",
            mode="lines",
        )
        _plot_polygon(
            fig,
            valid_data_polygon(self.ew),
            swap_xy=True,
            name="ValidData",
            line_color="red",
            mode="lines",
        )
        if mark_scp:
            fig.add_scatter(
                x=[scp_pixel[1]],
                y=[scp_pixel[0]],
                name="SCPPixel",
                mode="markers",
                marker_color="cyan",
            )
        fig.update_xaxes(
            title_text="Columns (px)",
            range=full_image_shape[1] * np.array([-0.05, 1.05]),
            side="top",
            constrain="domain",
        )
        fig.update_yaxes(
            title_text="Rows (px)",
            range=full_image_shape[0] * np.array([1.05, -0.05]),
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        )
        fig.update_layout(
            title_text=self.format_title("ImageData"),
            title_y=0.97,
            coloraxis_showscale=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
            height=self.nominal_height,
            width=self.nominal_width,
            meta="image_data",
        )
        return [fig]

    def plot_tcoa(self):
        """Plot of TimeCOAPoly."""
        tcoapoly = self.ew["Grid"]["TimeCOAPoly"]

        row_vals = np.linspace(0, self.ew["ImageData"]["FullImage"]["NumRows"], 100)
        col_vals = np.linspace(0, self.ew["ImageData"]["FullImage"]["NumCols"], 100)
        xrow, ycol = self._pxl_to_xy(np.stack([row_vals, col_vals], axis=-1)).T
        times = npp.polygrid2d(xrow, ycol, tcoapoly)

        fig = px.imshow(
            times, y=row_vals, x=col_vals, title=self.format_title("TimeCOAPoly (s)")
        )
        _plot_polygon(
            fig,
            valid_data_polygon(self.ew),
            swap_xy=True,
            name="ValidData",
            line=dict(color="black", dash="dash"),
        )
        fig.update_xaxes(title_text="Columns (px)", side="top")
        fig.update_yaxes(title_text="Rows (px)")
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
            height=self.nominal_height,
            width=self.nominal_width,
            meta="tcoa",
        )
        return [fig]

    def plot_spatial_frequency_support(self):
        """Plot the claimed spatial frequency support."""
        scp_pxl = self.ew["ImageData"]["SCPPixel"]
        spacing = self._grid_spacing()

        delta_krow = (
            self.ew["Grid"]["Row"]["DeltaK1"],
            self.ew["Grid"]["Row"]["DeltaK2"],
        )
        delta_kcol = (
            self.ew["Grid"]["Col"]["DeltaK1"],
            self.ew["Grid"]["Col"]["DeltaK2"],
        )

        delta_k_coa_poly_row, delta_k_coa_poly_col = self._kcoa_polys()

        k_iprbw = np.array(
            [
                self.ew["Grid"]["Row"]["ImpRespBW"],
                self.ew["Grid"]["Col"]["ImpRespBW"],
            ]
        )

        x_vals = np.linspace(0, self.ew["ImageData"]["FullImage"]["NumRows"], 100)
        y_vals = np.linspace(0, self.ew["ImageData"]["FullImage"]["NumCols"], 100)
        xv, yv = np.meshgrid(
            (x_vals - scp_pxl[0]) * spacing[0], (y_vals - scp_pxl[1]) * spacing[1]
        )
        delta_k_coa_tgt_row = npp.polyval2d(xv, yv, delta_k_coa_poly_row)
        delta_k_coa_tgt_col = npp.polyval2d(xv, yv, delta_k_coa_poly_col)

        delta_k_coa_tgt_row = (delta_k_coa_tgt_row + 0.5 / spacing[0]) % (
            1.0 / spacing[0]
        ) - 0.5 / spacing[0]
        delta_k_coa_tgt_col = (delta_k_coa_tgt_col + 0.5 / spacing[1]) % (
            1.0 / spacing[1]
        ) - 0.5 / spacing[1]

        scp_kcoa = [
            npp.polyval2d(0, 0, delta_k_coa_poly_row),
            npp.polyval2d(0, 0, delta_k_coa_poly_col),
        ]

        fig = go.Figure()
        _plot_box(
            fig,
            delta_kcol[0],
            delta_krow[0],
            delta_kcol[1],
            delta_krow[1],
            name="DeltaK",
            fill="toself",
            mode="lines",
            line_color="green",
        )
        fig.add_scatter(
            x=delta_k_coa_tgt_col.flatten(),
            y=delta_k_coa_tgt_row.flatten(),
            name="TgtDeltaK",
            mode="markers",
        )
        _plot_box(
            fig,
            *(-0.5 / spacing[::-1]),
            *(0.5 / spacing[::-1]),
            name="Nyquist",
            line_color="red",
            mode="lines",
        )
        _plot_box(
            fig,
            *((-k_iprbw / 2.0 + scp_kcoa)[::-1]),
            *((+k_iprbw / 2.0 + scp_kcoa)[::-1]),
            name="SCP Support",
            fill="toself",
            mode="lines",
            line_color="blue",
        )
        max_extent = np.maximum(k_iprbw[1] / 2 + np.abs(scp_kcoa), 0.5 / spacing) * 1.1
        fig.update_xaxes(
            title_text="Kcol (cyc/m)", side="top", range=(-max_extent[1], max_extent[1])
        )
        fig.update_yaxes(
            title_text="Krow (cyc/m)",
            range=(max_extent[0], -max_extent[0]),
            scaleanchor="x",
            scaleratio=1,
        )
        fig.update_layout(
            title_text=self.format_title("Spatial Frequency Support"),
            title_y=0.97,
            height=self.nominal_height,
            width=self.nominal_width,
            meta="spatial_frequency_support",
        )
        return [fig]

    def plot_timeline(self):
        """Plot the collection and processing timeline information."""
        collect_start = self.ew["Timeline"]["CollectStart"]
        collect_duration = self.ew["Timeline"]["CollectDuration"]
        t_start_proc = self.ew["ImageFormation"]["TStartProc"]
        t_end_proc = self.ew["ImageFormation"]["TEndProc"]
        scp_time = self.ew["SCPCOA"]["SCPTime"]
        tcoapoly = self.ew["Grid"]["TimeCOAPoly"]

        exterior = valid_data_polygon(self.ew).exterior.coords[:]
        points = []
        for start, stop in zip(exterior[:-1], exterior[1:]):
            xvals = np.linspace(start[0], stop[0], 4, endpoint=False)
            yvals = np.linspace(start[1], stop[1], 4, endpoint=False)
            points = points + list(zip(xvals, yvals))

        points = self._pxl_to_xy(points)

        x_vals, y_vals = zip(*points)
        times = npp.polyval2d(x_vals, y_vals, tcoapoly).flatten()

        fig = go.Figure()
        fig.add_scatter(
            x=[t_start_proc, t_end_proc],
            y=[0, 0],
            line=dict(color="blue", width=1.5),
            marker_size=10,
            name=f"Processing Time: {t_end_proc - t_start_proc:0.3f} s",
        )
        fig.add_scatter(
            x=[scp_time],
            y=[0],
            marker=dict(symbol="x", size=16),
            line=dict(color="green", width=3),
            name=f"SCP Time: {scp_time:0.3f} s",
            mode="markers",
        )
        fig.add_vline(
            x=0,
            line=dict(color="gray", dash="dot"),
            annotation_font_color="gray",
            annotation_text="Collect Start",
            annotation_position="top right",
        )
        fig.add_vline(
            x=collect_duration,
            line=dict(color="gray", dash="dot"),
            annotation_font_color="gray",
            annotation_text=f"Duration: {collect_duration:0.3f} s",
            annotation_position="top left",
        )
        _plot_box(
            fig,
            times.min(),
            -20,
            times.max(),
            20,
            name=f"COA span: {np.ptp(times):0.3f} s",
            fill="toself",
            line_color="cyan",
            mode="lines",
        )
        fig.update_xaxes(
            title_text="Time (s)",
            range=(-collect_duration * 0.05, collect_duration * 1.05),
        )
        fig.update_yaxes(visible=False, range=(-1, 1))
        fig.update_layout(
            title_text=self.format_title(f"Timeline - {collect_start}"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="left", x=0),
            height=300,
            width=self.nominal_width,
            meta="timeline",
        )
        return [fig] + self._plot_timeline_ipp()

    def _eval_ipp_sets(self):
        ipp_sets = []
        for set_node in self.ew["Timeline"]["IPP"]["Set"]:
            ipp_set = set_node.to_dict()

            ipp_set["times"] = np.linspace(ipp_set["TStart"], ipp_set["TEnd"], 100)
            ipp_set["ipps"] = npp.polyval(ipp_set["times"], ipp_set["IPPPoly"])
            ipp_set["prfs"] = npp.polyval(
                ipp_set["times"], npp.polyder(ipp_set["IPPPoly"])
            )

            ipp_sets.append(ipp_set)

        return ipp_sets

    def _plot_timeline_ipp(self):
        ipp_sets = self._eval_ipp_sets()
        if not ipp_sets:
            return []

        min_tstart = min(ipp_set["TStart"] for ipp_set in ipp_sets)
        max_tend = max(ipp_set["TEnd"] for ipp_set in ipp_sets)
        extra_time_range = (max_tend - min_tstart) * 0.05

        # IPP Indices
        min_ipp = min(min(computed["ipps"]) for computed in ipp_sets)
        max_ipp = max(max(computed["ipps"]) for computed in ipp_sets)
        extra_ipp_range = max((max_ipp - min_ipp) * 0.05, 1)

        ipp_fig = go.Figure()
        colors = itertools.cycle(plotly.colors.qualitative.D3)
        for ipp_set in ipp_sets:
            ipp_fig.add_scatter(
                x=ipp_set["times"],
                y=ipp_set["ipps"],
                marker_color=next(colors),
                mode="markers",
            )
        ipp_fig.update_xaxes(
            title_text="Time (s)",
            range=(min_tstart - extra_time_range, max_tend + extra_time_range),
        )
        ipp_fig.update_yaxes(
            title_text="IPP Index",
            range=(min_ipp - extra_ipp_range, max_ipp + extra_ipp_range),
        )
        ipp_fig.update_layout(
            title_text=self.format_title(
                f"Timeline - Inter-Pulse Period ({len(ipp_sets)} Sets)"
            ),
            height=300,
            width=self.nominal_width,
            meta="timeline_ipp",
        )

        # PRF
        min_prf = min(min(computed["prfs"]) for computed in ipp_sets)
        max_prf = max(max(computed["prfs"]) for computed in ipp_sets)
        extra_prf_range = max((max_prf - min_prf) * 0.05, 1)

        title_text = self.format_title(
            f"Timeline - Pulse Repetition Frequency ({len(ipp_sets)} Sets)"
        )
        prfsf = self.ew["ImageFormation"]["RcvChanProc"].get("PRFScaleFactor", None)
        if prfsf is not None:
            title_text += f"<br>PRFScaleFactor: {prfsf}"

        prf_fig = go.Figure()
        colors = itertools.cycle(plotly.colors.qualitative.D3)
        for ipp_set in ipp_sets:
            prf_fig.add_scatter(
                x=ipp_set["times"],
                y=ipp_set["prfs"],
                marker_color=next(colors),
                mode="markers",
            )
        prf_fig.update_xaxes(
            title_text="Time (s)",
            range=(min_tstart - extra_time_range, max_tend + extra_time_range),
        )
        prf_fig.update_yaxes(
            title_text="PRF (Hz)",
            range=(min_prf - extra_prf_range, max_prf + extra_prf_range),
        )
        prf_fig.update_layout(
            title_text=title_text,
            height=300,
            width=self.nominal_width,
            meta="timeline_prf",
        )
        return [ipp_fig, prf_fig]

    def plot_scp(self):
        """Plot the location of the SCP on a map"""
        scp_lat, scp_lon, _ = self.ew["GeoData"]["SCP"]["LLH"]

        fig = go.Figure(go.Scattergeo())
        fig.update_geos(
            projection_type="orthographic",
            showcountries=True,
            lataxis_showgrid=True,
            lonaxis_showgrid=True,
            projection={"rotation": {"lat": scp_lat, "lon": scp_lon}, "scale": 2},
        )
        fig.update_layout(
            height=300,
            width=300,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            meta="scp",
        )
        fig.add_trace(
            go.Scattergeo(
                lon=[scp_lon],
                lat=[scp_lat],
                mode="markers",
                marker={"size": 10, "symbol": "diamond"},
            )
        )
        return [fig]

    def plot_weights(self):
        """Plot the Row and Col Wgts"""
        figures = []
        for direction in ("Row", "Col"):
            weights = self.ew["Grid"][direction].get("WgtFunct", None)
            if weights is None:
                continue
            window_name = self.ew["Grid"][direction]["WgtType"].get("WindowName", None)
            title = f"{direction} WgtFunct"
            if window_name:
                title += f" ({window_name})"

            miny = min(weights.min(), 0) - 0.1
            maxy = max(weights.max(), 1) + 0.1

            fig = go.Figure()
            fig.add_hline(y=0, line=dict(color="silver", dash="solid"))
            fig.add_hline(y=1, line=dict(color="silver", dash="dash"))
            fig.add_scatter(x=np.arange(weights.size) + 1, y=weights, mode="markers")
            fig.update_xaxes(title="Index")
            fig.update_yaxes(title="Wgt", range=[miny, maxy])
            fig.update_layout(
                title=title,
                height=self.nominal_height / 2,
                meta="weights",
            )
            figures.append(fig)
        return figures


def downsample_last_dim(data, factor):
    """Downsamples the last dimension of a multidimension `numpy.ndarray`.

    Args
    ----
    data: `numpy.ndarray`
        Multidimensional data to be downsampled.
    factor: int
        Downsample factor.

    Returns
    -------
    result: `numpy.ndarray`
        The downsampled data.

    """
    remove = data.shape[-1] % factor
    if remove == 0:
        return data.reshape(data.shape[:-1] + (-1, factor)).mean(axis=-1)
    else:
        return np.concatenate(
            [
                data[..., : data.shape[-1] - remove]
                .reshape(data.shape[:-1] + (-1, factor))
                .mean(axis=-1),
                data[..., np.newaxis, data.shape[-1] - remove :].mean(axis=-1),
            ],
            axis=-1,
        )


def downsample_all_dims(data, factor):
    """Function to downsample all dimensions of a multidimension `numpy.ndarray`.

    Args
    ----
    data: `numpy.ndarray`
        Multidimensional data to be downsampled.
    factor: int
        Downsample factor.

    Returns
    -------
    result: `numpy.ndarray`
        The downsampled data.

    """
    working = data
    for neg_dim in range(1, data.ndim + 1):
        working = np.moveaxis(
            downsample_last_dim(np.moveaxis(working, -neg_dim, -1), factor),
            -1,
            -neg_dim,
        )
    return working


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Produce various plots of information contained in a SICD"
    )
    parser.add_argument("sicd_nitf_or_xml")
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="directory where output plot(s) will be placed",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        help="prefix used in output filenames (Default: {sicd_nitf_or_xml.stem}_)",
    )
    parser.add_argument(
        "-s",
        "--sample-data",
        action="store_true",
        help="include plots that use the SICD data",
    )
    parser.add_argument(
        "-c",
        "--concatenate",
        action="store_true",
        help="concatenate plots into single HTML",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_false",
        dest="auto_open",
        help="don't open plots after creation",
    )
    config = parser.parse_args(args)

    with open(config.sicd_nitf_or_xml, "rb") as f:
        plotter = Plotter(
            f, html.escape(config.sicd_nitf_or_xml), use_sample_data=config.sample_data
        )
    save_func = plotter.save_combined if config.concatenate else plotter.save_separate
    prefix = (
        pathlib.PurePath(config.sicd_nitf_or_xml).stem + "_"
        if config.prefix is None
        else config.prefix
    )
    save_func(config.output_dir, prefix=prefix, auto_open=config.auto_open)


if __name__ == "__main__":  # pragma: no cover
    main()
