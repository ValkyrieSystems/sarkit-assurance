"""Utilities for generating plots of CRSD metadata"""

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
import sarkit.crsd as skcrsd
import sarkit.wgs84
import scipy.constants
import shapely
import shapely.geometry as shg

from . import _plot_metadata, cphd_plot_metadata, names, utils

try:
    from smart_open import open
except ImportError:
    pass


def diff_intfrac(a, b):
    return (a["Int"] - b["Int"]) + (a["Frac"] - b["Frac"])


def from_intfrac(a):
    return a["Int"] + a["Frac"]


class Plotter(_plot_metadata.Plotter):
    """A CRSD metadata plotter class."""

    def __init__(
        self,
        file,
        title,
        *,
        channels,
        sequences,
        include_fixed_pxps=False,
    ):
        with skcrsd.Reader(file) as r:
            self.xml = r.metadata.xmltree
            self.ew = skcrsd.ElementWrapper(self.xml.getroot())
            all_channels = [
                x.text
                for x in self.xml.findall("{*}Channel/{*}Parameters/{*}Identifier")
            ]
            all_sequences = [
                x.text
                for x in self.xml.findall("{*}TxSequence/{*}Parameters/{*}Identifier")
            ]

            if not set(sequences) <= set(all_sequences):
                raise ValueError(
                    (
                        f"Unrecognized sequence(s): {set(sequences) - set(all_sequences)}; "
                        f"Must be from: {all_sequences}"
                    )
                )
            if not set(channels) <= set(all_channels):
                raise ValueError(
                    (
                        f"Unrecognized channel(s): {set(channels) - set(all_channels)}; "
                        f"Must be from: {all_channels}"
                    )
                )
            self.channels = list(set(channels))
            self.sequences = list(set(sequences))
            self.pvps = {ch_id: r.read_pvps(ch_id) for ch_id in self.channels}
            self.ppps = {tx_id: r.read_ppps(tx_id) for tx_id in self.sequences}
            self.support_arrays = {
                x.text: r.read_support_array(x.text)
                for x in self.xml.findall("{*}SupportArray/*/{*}Identifier")
                if x.getparent().tag.endswith(("GainPhaseArray", "DwellTimeArray"))
            }

        self.include_fixed_pxps = include_fixed_pxps
        super().__init__(title)

    def get_ap_delta_ap(self, txrcv, pxpi, target_ecefs):
        apc_pos = pxpi[f"{txrcv}Pos"]
        acx = pxpi[f"{txrcv}ACX"]
        acy = pxpi[f"{txrcv}ACY"]
        eb = pxpi[f"{txrcv}EB"]
        ulos = utils.unit(target_ecefs[..., np.newaxis, :] - apc_pos)
        ap = np.stack([np.vecdot(ulos, acx), np.vecdot(ulos, acy)], axis=-1)
        # TODO: adjust eb and delta_ap for frequency?
        delta_ap = ap - eb
        return ap, delta_ap

    def plot_antenna(self):
        figs = []
        if not self.xml.getroot().tag.endswith("CRSDsar"):
            return figs
        info_by_gpid = {}
        for ch_id in self.channels:
            chanparam_ew = self.ew["Channel"].find("Parameters", Identifier=ch_id)
            rcvapat_id = chanparam_ew["RcvAPATId"]

            txid = chanparam_ew["SARImage"]["TxId"]
            txapat_id = self.ew["TxSequence"].find("Parameters", Identifier=txid)[
                "TxAPATId"
            ]

            target_times, target_ecef, target_iac = self.get_valid_target_dwell(
                chanparam_ew
            )
            pvp = self.pvps[ch_id]
            init_index = (
                np.digitize(target_times, from_intfrac(pvp["RcvStart"])).clip(
                    1, pvp.size
                )
                - 1
            )
            prop_time = (
                np.linalg.norm(
                    pvp["RcvPos"][init_index] - target_ecef[..., np.newaxis, :], axis=-1
                )
                / scipy.constants.speed_of_light
            )
            rcv_index = (
                np.digitize(
                    target_times + prop_time, from_intfrac(pvp["RcvStart"])
                ).clip(1, pvp.size)
                - 1
            )
            pvpi = pvp[rcv_index]
            tx_index = pvpi["TxPulseIndex"]
            pppi = self.ppps[txid][tx_index]

            for txrcv, name, apat_id, pxpi in [
                ("Rcv", ch_id, rcvapat_id, pvpi),
                ("Tx", txid, txapat_id, pppi),
            ]:
                apat_ew = self.ew["Antenna"].find("AntPattern", Identifier=apat_id)
                for gpid_type in ("Array", "Elem"):
                    gpid = apat_ew[f"{gpid_type}GPId"]
                    ap, delta_ap = self.get_ap_delta_ap(txrcv, pxpi, target_ecef)
                    info_by_gpid.setdefault((gpid, gpid_type), []).append(
                        {
                            "name": name,
                            "txrcv": txrcv,
                            "apat": apat_ew,
                            "target_times": target_times,
                            "target_ecef": target_ecef,
                            "target_iac": target_iac,
                            "target_dcxy": ap if gpid_type == "Elem" else delta_ap,
                        }
                    )
            for (gpid, gpid_type), infos in info_by_gpid.items():
                samples = self.support_arrays[gpid]
                sa_ew = self.ew["SupportArray"].find("GainPhaseArray", Identifier=gpid)
                this_fig = psp.make_subplots(rows=1, cols=2, horizontal_spacing=0.2)
                this_fig.add_heatmap(
                    z=samples["Gain"],
                    x0=sa_ew["X0"],
                    dx=sa_ew["XSS"],
                    y0=sa_ew["Y0"],
                    dy=sa_ew["YSS"],
                    transpose=True,
                    row=1,
                    col=1,
                    colorbar_title="Gain [dB]",
                    colorbar_x=0.42,
                    name="Gain",
                )
                this_fig.add_heatmap(
                    z=samples["Phase"],
                    x0=sa_ew["X0"],
                    dx=sa_ew["XSS"],
                    y0=sa_ew["Y0"],
                    dy=sa_ew["YSS"],
                    transpose=True,
                    row=1,
                    col=2,
                    colorbar_title="Phase [cycles]",
                    name="Phase",
                )
                for info in infos:
                    txt = f"{info['txrcv']}: {names.sanitize_name(info['name'])} <{names.sanitize_name(info['apat']['Identifier'])}>"
                    actual_dcs = go.Scatter(
                        x=info["target_dcxy"][..., 0].flatten(),
                        y=info["target_dcxy"][..., 1].flatten(),
                        name=txt,
                        mode="markers",
                        marker=dict(color="rgba(128, 128, 128, 0.1)"),
                        hoverinfo="skip",
                        legendgroup=txt,
                    )
                    this_fig.add_trace(actual_dcs, row=1, col=1)
                    this_fig.add_trace(actual_dcs, row=1, col=2)
                    this_fig.update_traces(row=1, col=2, showlegend=False)

                this_fig.update_layout(
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": -0.2,
                        "xanchor": "left",
                        "x": 0,
                    },
                    title_text=(
                        f"{self.format_title('Antenna Gain/Phase')}<br>"
                        f"GPId: {names.sanitize_name(gpid)} ({gpid_type})"
                    ),
                    height=self.nominal_height,
                    meta=f"antenna_{gpid}_{gpid_type}",
                )
                label_prefix = "ΔDC" if gpid_type == "Array" else "DC"
                for col in range(2):
                    this_fig.update_xaxes(
                        title_text=f"{label_prefix}X", row=1, col=1 + col
                    )
                    this_fig.update_yaxes(
                        title_text=f"{label_prefix}Y", row=1, col=1 + col
                    )
                figs.append(this_fig)
        return figs

    def plot_reference_time_freq(self):
        figs = []
        if not self.xml.getroot().tag.endswith("CRSDsar"):
            return figs
        if self.ew["TxSequence"]["TxWFType"] != "LFM":
            # not implemented yet
            return figs

        for ch_id in self.channels:
            chanparam_ew = self.ew["Channel"].find("Parameters", Identifier=ch_id)
            v_ch_ref = chanparam_ew["RefVectorIndex"]
            ref_pvp = self.pvps[ch_id][v_ch_ref]
            txid = chanparam_ew["SARImage"]["TxId"]
            p_ch_ref = ref_pvp["TxPulseIndex"]
            ref_ppp = self.ppps[txid][p_ch_ref]
            rcv_fs = chanparam_ew["Fs"]
            rcv_bw_inst = chanparam_ew["BWInst"]
            num_samps = self.ew["Data"]["Receive"].find("Channel", ChId=ch_id)[
                "NumSamples"
            ]
            rcv_duration = (num_samps - 1) / rcv_fs

            dfic_end = ref_pvp["DFIC0"] + rcv_duration * ref_pvp["FICRate"]
            fic0 = ref_pvp["RefFreq"] + ref_pvp["DFIC0"]
            fic_end = ref_pvp["RefFreq"] + dfic_end

            # TODO: replace with some sort of RTT
            propagation_time = (
                diff_intfrac(ref_pvp["RcvStart"], ref_ppp["TxTime"]) + rcv_duration / 2
            )

            fig = self.new_fig(f"Reference Time/Freq - {ch_id}", f"ref_tf_{ch_id}")
            figs.append(fig)
            fig.update_xaxes(
                title_text="Fast Time (s)",
            )
            fig.update_yaxes(
                exponentformat="SI",
                title_text="Frequency at RF (Hz)",
            )
            fig.add_scatter(
                x=[0, rcv_duration, rcv_duration, 0, 0],
                y=[
                    fic0 - rcv_bw_inst / 2,
                    fic_end - rcv_bw_inst / 2,
                    fic_end + rcv_bw_inst / 2,
                    fic0 + rcv_bw_inst / 2,
                    fic0 - rcv_bw_inst / 2,
                ],
                name="Receive Passband",
                legendgroup="rcv",
                legendgrouptitle_text=f"Receive Parameters - v_CH_REF={v_ch_ref}",
                mode="lines",
                fill="toself",
            )
            nyquist_line1 = np.round((ref_pvp["FRCV1"] - ref_pvp["RefFreq"]) / rcv_fs)
            nyquist_line2 = np.round((ref_pvp["FRCV2"] - ref_pvp["RefFreq"]) / rcv_fs)
            nyquist_lines = np.arange(nyquist_line1, nyquist_line2 + 2) - 0.5
            fig.add_scatter(
                x=[0, rcv_duration, np.nan] * nyquist_lines.size,
                y=(
                    nyquist_lines[:, np.newaxis] * rcv_fs
                    + ref_pvp["RefFreq"]
                    + [0, 0, 0]
                ).flatten(),
                name="Receive Nyquist",
                legendgroup="rcv",
                mode="lines",
                line_dash="dot",
                line_color="rgba(200, 16, 16, 0.25)",
            )
            fig.add_scatter(
                x=[0, rcv_duration],
                y=[fic0, fic_end],
                name="f_IC",
                legendgroup="rcv",
            )
            for param in ("FRCV1", "FRCV2", "RefFreq"):
                fig.add_scatter(
                    x=[0, rcv_duration],
                    y=[ref_pvp[param], ref_pvp[param]],
                    name=param,
                    legendgroup="rcv",
                    mode="lines",
                )

            # Transmit
            # echostart = txtime + proptime - txmt/2 - rcvstart
            echo_start = (
                diff_intfrac(ref_ppp["TxTime"], ref_pvp["RcvStart"])
                + propagation_time
                - ref_ppp["TXmt"] / 2
            )
            echo_end = echo_start + ref_ppp["TXmt"]
            fx_start = ref_ppp["FxFreq0"] - ref_ppp["FxRate"] * ref_ppp["TXmt"] / 2
            fx_end = ref_ppp["FxFreq0"] + ref_ppp["FxRate"] * ref_ppp["TXmt"] / 2
            fig.add_scatter(
                x=[echo_start, echo_end],
                y=[fx_start, fx_end],
                name="Transmit LFM Echo",
                legendgroup="xmt",
                legendgrouptitle_text=f"TxSequence Parameters - p_CH_REF={p_ch_ref}",
                mode="lines",
            )
            fig.update_layout(legend=dict(groupclick="toggleitem"))
        return figs

    def plot_dwell(self):
        figs = []
        if not self.xml.getroot().tag.endswith("CRSDsar"):
            return figs

        cod_poly_info = {}
        dwell_poly_info = {}
        dta_info = {}
        for chan in self.channels:
            chan_param_ew = self.ew["Channel"].find("Parameters", Identifier=chan)
            dt_ew = chan_param_ew["SARImage"]["DwellTimes"]
            chan_imgarea = shapely.Polygon(
                skcrsd.get_channel_image_area(self.xml, chan)
            )
            if "Polynomials" in dt_ew:
                cod_id = dt_ew["Polynomials"]["CODId"]
                dwell_id = dt_ew["Polynomials"]["DwellId"]
                cod_poly_info.setdefault(cod_id, {})[chan] = chan_imgarea
                dwell_poly_info.setdefault(dwell_id, {})[chan] = chan_imgarea
            else:
                dta_id = dt_ew["Array"]["DTAId"]
                dta_info.setdefault(dta_id, {})[chan] = chan_imgarea
        channel_colors = dict(
            zip(self.channels, itertools.cycle(plotly.colors.qualitative.Plotly))
        )

        def plot_dt_poly(dt_type, dt_id, chan_info):
            poly = self.ew["DwellPolynomials"].find(f"{dt_type}Time", Identifier=dt_id)[
                f"{dt_type}TimePoly"
            ]
            valid_areas = shg.MultiPolygon(list(chan_info.values()))
            valid_area_pad = pad_geom(valid_areas, 0.05)

            iax_samples = np.linspace(
                valid_area_pad.bounds[0], valid_area_pad.bounds[2], 129
            )
            iay_samples = np.linspace(
                valid_area_pad.bounds[1], valid_area_pad.bounds[3], 128
            )

            iax_grid, iay_grid = np.meshgrid(iax_samples, iay_samples, indexing="ij")
            grid_points = shg.MultiPoint(
                np.stack([iax_grid, iay_grid], -1).reshape(-1, 2)
            )
            mask = np.array(
                [pt.within(valid_area_pad) for pt in grid_points.geoms]
            ).reshape(iax_grid.shape)

            sampled_times = npp.polygrid2d(iax_samples, iay_samples, poly)
            sampled_times[~mask] = np.nan

            fig = px.imshow(
                sampled_times,
                x=iay_samples,
                y=iax_samples,
                aspect="auto",
                origin="lower",
                color_continuous_scale="gray",
                labels={"x": "IAY [m]", "y": "IAX [m]", "color": f"{dt_type} [s]"},
                title=f"{self.format_title('DwellPolynomial')}<br>{dt_type} : {dt_id}",
            )
            fig.update_layout(meta=f"dwell_{dt_type}_{dt_id}")
            fig.update_yaxes(autorange="reversed")
            for ch_id, poly in chan_info.items():
                poly_vertices = shapely.get_coordinates(poly.exterior)
                poly_vertices = np.concatenate([poly_vertices, poly_vertices[:1]])
                fig.add_scatter(
                    x=poly_vertices[:, 1],
                    y=poly_vertices[:, 0],
                    name=ch_id,
                    fill=None,
                    marker_color=channel_colors[ch_id],
                )
            fig.update_layout(
                showlegend=True,
                legend={
                    "orientation": "h",
                    "xanchor": "right",
                    "yanchor": "bottom",
                    "x": 1,
                    "y": 1.02,
                },
            )
            return fig

        def plot_dta(dt_id, chan_info):
            samples = self.support_arrays[dt_id]
            sa_ew = self.ew["SupportArray"].find("DwellTimeArray", Identifier=dt_id)
            this_fig = psp.make_subplots(rows=1, cols=2, horizontal_spacing=0.2)
            this_fig.add_heatmap(
                z=samples["COD"],
                x0=sa_ew["X0"],
                dx=sa_ew["XSS"],
                y0=sa_ew["Y0"],
                dy=sa_ew["YSS"],
                transpose=True,
                row=1,
                col=1,
                colorbar_title="COD [s]",
                colorbar_x=0.42,
                name="COD",
            )
            this_fig.add_heatmap(
                z=samples["DT"],
                x0=sa_ew["X0"],
                dx=sa_ew["XSS"],
                y0=sa_ew["Y0"],
                dy=sa_ew["YSS"],
                transpose=True,
                row=1,
                col=2,
                colorbar_title="Dwell [s]",
                name="Dwell",
            )
            this_fig.update_layout(
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": -0.2,
                    "xanchor": "left",
                    "x": 0,
                },
                title_text=(
                    f"{self.format_title('DwellTimeArray')}<br>"
                    f"DTAId: {names.sanitize_name(dt_id)}"
                ),
                height=self.nominal_height,
                meta=f"dwelltimearray_{dt_id}",
            )
            for col in range(2):
                this_fig.update_xaxes(title_text="IAX", row=1, col=1 + col)
                this_fig.update_yaxes(title_text="IAY", row=1, col=1 + col)
            for ch_id, poly in chan_info.items():
                poly_vertices = shapely.get_coordinates(poly.exterior)
                poly_vertices = np.concatenate([poly_vertices, poly_vertices[:1]])
                poly_trace = go.Scatter(
                    x=poly_vertices[:, 1],
                    y=poly_vertices[:, 0],
                    name=ch_id,
                    fill=None,
                    marker_color=channel_colors[ch_id],
                    legendgroup=ch_id,
                )
                this_fig.add_trace(poly_trace, row=1, col=1)
                this_fig.add_trace(poly_trace, row=1, col=2)
                this_fig.update_traces(row=1, col=2, showlegend=False)
            return this_fig

        figs = [
            plot_dt_poly("Dwell", dt_id, chan_info)
            for dt_id, chan_info in dwell_poly_info.items()
        ]
        figs.extend(
            plot_dt_poly("COD", dt_id, chan_info)
            for dt_id, chan_info in cod_poly_info.items()
        )
        figs.extend(plot_dta(dt_id, chan_info) for dt_id, chan_info in dta_info.items())
        return figs

    def plot_image_area(self):
        if not self.xml.getroot().tag.endswith("CRSDsar"):
            return []
        fig = go.Figure()
        color_set = itertools.cycle(
            zip(plotly.colors.qualitative.Pastel2, plotly.colors.qualitative.Set2)
        )

        im_rect, im_poly = _make_image_area(
            self.ew["SceneCoordinates"]["ImageArea"],
            name="Scene",
            colors=next(color_set),
        )

        def make_ll_string(ll_node):
            return "<br>".join(
                f"{lxml.etree.QName(x).localname}: {x.text}°"
                for x in ll_node.getchildren()
            )

        iacp_labels = [
            make_ll_string(x)
            for x in self.xml.find("{*}SceneCoordinates/{*}ImageAreaCornerPoints")
        ]
        for label, ptx, pty, yshift in zip(
            iacp_labels, im_rect["x"], im_rect["y"], [20, 20, -20, -20]
        ):
            fig.add_annotation(
                x=ptx, y=pty, text=label, showarrow=False, xshift=0, yshift=yshift
            )

        fig.add_trace(im_rect)
        if im_poly:
            fig.add_trace(im_poly)

        if (
            extended_area_ew := self.ew["SceneCoordinates"].get("ExtendedArea", None)
        ) is not None:
            ext_rect, ext_poly = _make_image_area(
                extended_area_ew, name="Extended", colors=next(color_set)
            )
            fig.add_trace(ext_rect)
            if ext_poly is not None:
                fig.add_trace(ext_poly)

        channel_colors = dict(zip(self.channels + self.sequences, color_set))

        for channel_ia_element in self.xml.findall(
            "{*}Channel/{*}Parameters/{*}SARImage/{*}ImageArea"
        ):
            chan_id = channel_ia_element.getparent().findtext("{*}Identifier")
            if chan_id in self.channels:
                fig.add_traces(
                    [
                        t
                        for t in _make_image_area(
                            channel_ia_element,
                            name=f"Channel: {chan_id}",
                            colors=channel_colors[chan_id],
                        )
                        if t
                    ]
                )

        antenna_aiming = self._antenna_aiming_in_image_area()
        for txrcv, txrcv_aiming in antenna_aiming.items():
            symbol = {"Tx": "y-down", "Rcv": "y-up"}[txrcv]
            for name, aiming in txrcv_aiming.items():
                boresights = aiming["boresights"]

                def add_boresight_trace(points, name, color):
                    intersects = np.isfinite(np.linalg.norm(points, axis=-1))
                    if intersects.any():
                        fig.add_trace(
                            go.Scatter(
                                x=points[:, 1],
                                y=points[:, 0],
                                name=name,
                                legendgroup=name,
                                showlegend=False,
                                mode="lines+markers",
                                marker=dict(
                                    symbol=symbol,
                                    line_color=color,
                                    color=color,
                                    line_width=2,
                                ),
                            )
                        )
                        first_point = points[intersects][0]
                    else:
                        first_point = (np.nan, np.nan)
                        name = name + " (no intersection)"
                    fig.add_trace(
                        go.Scatter(
                            x=[first_point[1]],
                            y=[first_point[0]],
                            name=name,
                            legendgroup=name,
                            showlegend=True,
                            mode="lines+markers",
                            marker=dict(
                                symbol=symbol,
                                size=15,
                                line_color=color,
                                color=color,
                                line_width=2,
                            ),
                        )
                    )

                add_boresight_trace(
                    boresights["mechanical"],
                    name=f"{txrcv}: {name} MB",
                    color=channel_colors[name][0],
                )
                if "electrical" in boresights:
                    add_boresight_trace(
                        boresights["electrical"],
                        name=f"{txrcv}: {name} EB ",
                        color=channel_colors[name][-1],
                    )

        fig.update_layout(
            xaxis_title="IAY [m]",
            yaxis_title="IAX [m]",
            title_text=self.format_title("Image Area"),
            meta="image_area",
        )
        fig.update_yaxes(autorange="reversed")
        return [fig]

    def _antenna_aiming_in_image_area(self):
        results = {
            "Tx": {},
            "Rcv": {},
        }

        if self.xml.find("{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar") is None:
            # Only Planar is handled
            return results

        iarp = self.ew["SceneCoordinates"]["IARP"]["ECF"]
        iax = self.ew["SceneCoordinates"]["ReferenceSurface"]["Planar"]["uIAX"]
        iay = self.ew["SceneCoordinates"]["ReferenceSurface"]["Planar"]["uIAY"]
        iaz = np.cross(iax, iay)

        def _intersect_boresight_with_image_area(apc_positions, uacx, uacy, ebx, eby):
            uacz = np.cross(uacx, uacy)
            ebz = (1 - ebx**2 - eby**2) ** 0.5

            along = (
                uacx * np.asarray(ebx)[..., np.newaxis]
                + uacy * np.asarray(eby)[..., np.newaxis]
                + uacz * np.asarray(ebz)[..., np.newaxis]
            )

            distance = -np.vecdot(apc_positions - iarp, iaz) / np.vecdot(along, iaz)
            plane_points_ecf = apc_positions + distance[:, np.newaxis] * along
            plane_points_x = np.vecdot(plane_points_ecf - iarp, iax)
            plane_points_y = np.vecdot(plane_points_ecf - iarp, iay)
            return np.stack((plane_points_x, plane_points_y)).T

        for txrcv, pxps in [("Tx", self.ppps), ("Rcv", self.pvps)]:
            for name, pxp_data in pxps.items():
                indices = np.rint(
                    np.linspace(0, len(pxp_data) - 1, 51, endpoint=True)
                ).astype(int)
                results[txrcv][name] = {}
                uacx = pxp_data[f"{txrcv}ACX"][indices]
                uacy = pxp_data[f"{txrcv}ACY"][indices]
                eb = pxp_data[f"{txrcv}EB"][indices]
                pos = pxp_data[f"{txrcv}Pos"][indices]
                results[txrcv][name] = {
                    "boresights": {
                        "mechanical": _intersect_boresight_with_image_area(
                            pos, uacx, uacy, 0, 0
                        ),
                        "electrical": _intersect_boresight_with_image_area(
                            pos, uacx, uacy, eb[:, 0], eb[:, 1]
                        ),
                    }
                }
        return results

    def get_target_dwelltimes(self, target_ia_coords, chan_param_ew):
        if "Polynomials" in chan_param_ew["SARImage"]["DwellTimes"]:
            return skcrsd.compute_dwelltimes_using_poly(
                chan_param_ew["Identifier"],
                target_ia_coords[..., 0],
                target_ia_coords[..., 1],
                self.xml,
            )
        dta_id = chan_param_ew["SARImage"]["DwellTimes"]["Array"]["DTAId"]
        return skcrsd.compute_dwelltimes_using_dta(
            chan_param_ew["Identifier"],
            target_ia_coords[..., 0],
            target_ia_coords[..., 1],
            self.xml,
            self.support_arrays[dta_id],
        )

    def get_valid_target_dwell(
        self, chan_param_ew, target_grid_size=11, dwell_grid_size=11
    ):
        """Return a set of targets spanning a channel's image area along with times spanning their dwell"""
        ia_poly = shapely.Polygon(
            skcrsd.get_channel_image_area(self.xml, chan_param_ew["Identifier"])
        )
        target_ia_coords = utils.get_samples_in_poly(
            ia_poly, grid_size=target_grid_size
        )
        cod_times, dwell_times = self.get_target_dwelltimes(
            target_ia_coords, chan_param_ew
        )

        target_ecef_coords = skcrsd.iac_to_ecf(self.xml, target_ia_coords)

        target_times = cod_times[..., np.newaxis] + dwell_times[
            ..., np.newaxis
        ] * np.linspace(-0.5, 0.5, dwell_grid_size)
        return target_times, target_ecef_coords, target_ia_coords

    def plot_map(self):
        """Plot some locations on a map"""
        iarp_lat, iarp_lon, _ = self.ew["SceneCoordinates"]["IARP"]["LLH"]
        iacps = self.ew["SceneCoordinates"]["ImageAreaCornerPoints"]
        # repeat start to close polygon
        iacps = np.concatenate([iacps, iacps[:1]], axis=0)

        fig = go.Figure(go.Scattergeo())
        fig.update_geos(
            projection_type="orthographic",
            showcountries=True,
            lataxis_showgrid=True,
            lonaxis_showgrid=True,
            projection={"rotation": {"lat": iarp_lat, "lon": iarp_lon}, "scale": 1},
        )
        fig.update_layout(
            height=700,
            title_text=self.format_title("Map"),
            meta="map",
        )
        fig.add_trace(
            go.Scattergeo(
                lon=[iarp_lon],
                lat=[iarp_lat],
                mode="markers",
                marker={"size": 10, "symbol": "diamond"},
                text="IARP",
                name="IARP",
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lon=iacps[:, 1],
                lat=iacps[:, 0],
                mode="lines+markers",
                text=[f"IACP({x % 4 + 1})" for x in range(5)],
                name="IACPs",
            )
        )
        refgeom_refpt_llh = sarkit.wgs84.cartesian_to_geodetic(
            self.ew["ReferenceGeometry"]["RefPoint"]["ECF"]
        )
        fig.add_trace(
            go.Scattergeo(
                lon=[refgeom_refpt_llh[1]],
                lat=[refgeom_refpt_llh[0]],
                mode="markers",
                text="ReferenceGeometry.RefPoint",
                name="ReferenceGeometry.RefPoint",
            )
        )
        tx_refpoints = [
            self.ew["TxSequence"].find("Parameters", Identifier=txid)["TxRefPoint"][
                "ECF"
            ]
            for txid in self.sequences
        ]
        if tx_refpoints:
            tx_refpoints_llh = sarkit.wgs84.cartesian_to_geodetic(tx_refpoints)
            fig.add_trace(
                go.Scattergeo(
                    lon=tx_refpoints_llh[:, 1],
                    lat=tx_refpoints_llh[:, 0],
                    mode="markers",
                    text=[names.sanitize_name(txid) for txid in self.sequences],
                    name="TxRefPoint(s)",
                    marker_symbol="triangle-down-open",
                )
            )
        rcv_refpoints = [
            self.ew["Channel"].find("Parameters", Identifier=chid)["RcvRefPoint"]["ECF"]
            for chid in self.channels
        ]
        if rcv_refpoints:
            rcv_refpoints_llh = sarkit.wgs84.cartesian_to_geodetic(rcv_refpoints)
            fig.add_trace(
                go.Scattergeo(
                    lon=rcv_refpoints_llh[:, 1],
                    lat=rcv_refpoints_llh[:, 0],
                    mode="markers",
                    text=[names.sanitize_name(chid) for chid in self.channels],
                    name="RcvRefPoint(s)",
                    marker_symbol="triangle-up-open",
                )
            )
        return [fig]

    def plot_pvps(self):
        figs = {}
        for channel in self.channels:
            pvps = self.pvps[channel]
            pvp_data = {}
            for name in pvps.dtype.names:
                if pvps[name].dtype == np.dtype([("Int", ">i8"), ("Frac", ">f8")]):
                    pvp_data[name] = from_intfrac(pvps[name])
                else:
                    pvp_data[name] = pvps[name]
            fixed_pvps = {
                k: fixed_v
                for k, v in pvp_data.items()
                if (fixed_v := np.unique(v, axis=0)).shape[0] == 1
            }
            if fixed_pvps:
                figs[(channel, "Fixed-PVPs")] = _plot_metadata.plot_pvp_table(
                    fixed_pvps
                )
            for key, value in pvp_data.items():
                if not self.include_fixed_pxps and key in fixed_pvps:
                    continue
                if value.ndim == 1:
                    fig = _plot_metadata.plot_one_dim(value)
                elif value.ndim == 2 and value.shape[1] == 2:
                    fig = _plot_metadata.plot_two_dim(*value.T)
                elif value.ndim == 2 and value.shape[1] == 3:
                    fig = _plot_metadata.plot_three_dim(*value.T)
                figs[(channel, key)] = fig

        for (chan, key), fig in figs.items():
            fig.update_layout(
                title_text=f"<b>{key}</b> -  <i>{self.title} (channel: {chan})</i>",
                meta=f"pvp_{chan}_{key}",
            )
        return list(figs.values())

    def plot_ppps(self):
        figs = {}
        for sequence in self.sequences:
            ppps = self.ppps[sequence]
            ppp_data = {}
            for name in ppps.dtype.names:
                if ppps[name].dtype == np.dtype([("Int", ">i8"), ("Frac", ">f8")]):
                    ppp_data[name] = from_intfrac(ppps[name])
                else:
                    ppp_data[name] = ppps[name]
            fixed_ppps = {
                k: fixed_v
                for k, v in ppp_data.items()
                if (fixed_v := np.unique(v, axis=0)).shape[0] == 1
            }
            if fixed_ppps:
                figs[(sequence, "Fixed-PPPs")] = _plot_metadata.plot_pvp_table(
                    fixed_ppps
                )
            for key, value in ppp_data.items():
                if not self.include_fixed_pxps and key in fixed_ppps:
                    continue
                if value.ndim == 1:
                    fig = _plot_metadata.plot_one_dim(value)
                elif value.ndim == 2 and value.shape[1] == 2:
                    fig = _plot_metadata.plot_two_dim(*value.T)
                elif value.ndim == 2 and value.shape[1] == 3:
                    fig = _plot_metadata.plot_three_dim(*value.T)
                figs[(sequence, key)] = fig

        for (seq, key), fig in figs.items():
            fig.update_layout(
                title_text=f"<b>{key}</b> -  <i>{self.title} (sequence: {seq})</i>",
                meta=f"ppp_{seq}_{key}",
            )
        return list(figs.values())


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Produce various plots of information contained in a CRSD"
    )
    parser.add_argument("crsd_file", help="CRSD file to analyze")
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="directory where output plot(s) will be placed",
    )

    channel_group = parser.add_argument_group(
        title="Channel Selection",
        description=(
            "If these arguments are omitted, all channels are used. CRSDsar channels also select the relevant "
            "transmit pulse sequences."
        ),
    )
    channel_group.add_argument(
        "--ref-chan", action="store_true", help="include the reference channel"
    )
    channel_group.add_argument(
        "--chan",
        nargs="+",
        help="channel identifier(s) to include",
    )

    sequence_group = parser.add_argument_group(
        title="Transmit Sequence Selection",
        description="If these arguments are omitted, all sequences are used.",
    )
    sequence_group.add_argument(
        "--ref-seq", action="store_true", help="include the reference transmit sequence"
    )
    sequence_group.add_argument(
        "--seq",
        nargs="+",
        help="transmit sequence identifier(s) to include",
    )

    parser.add_argument(
        "-p",
        "--prefix",
        help="prefix used in output filenames (Default: {crsd_file.stem}_)",
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
    parser.add_argument("--plot-fixed", action="store_true", help="plot fixed PXPs")
    config = parser.parse_args(args)

    with open(config.crsd_file, "rb") as f, skcrsd.Reader(f) as r:
        xmltree = r.metadata.xmltree
        crsd_type = lxml.etree.QName(xmltree.getroot()).localname

        # channel selection
        ch_ids = set()
        if config.chan:
            ch_ids.update(config.chan)
        if config.ref_chan:
            ref_ch_id = xmltree.findtext("{*}Channel/{*}RefChId")
            if ref_ch_id is None:
                raise ValueError("Does not have a RefChId")
            ch_ids.add(ref_ch_id)

        all_ch_ids = [
            x.text for x in xmltree.findall("{*}Channel/{*}Parameters/{*}Identifier")
        ]
        if not ch_ids:
            ch_ids = sorted(all_ch_ids)
        else:
            unrecognized = ch_ids.difference(all_ch_ids)
            if unrecognized:
                raise ValueError(f"Unrecognized channel(s): {unrecognized}")
            ch_ids = sorted(ch_ids)

        # tx sequence selection
        tx_ids = set()
        if config.seq:
            tx_ids.update(config.seq)
        if config.ref_seq:
            ref_tx_id = xmltree.findtext("{*}TxSequence/{*}RefTxId")
            if ref_tx_id is None:
                raise ValueError("Does not have a RefTxId")
            tx_ids.add(ref_tx_id)
        if crsd_type == "CRSDsar":
            tx_ids.update(
                xmltree.findtext(
                    f"{{*}}Channel/{{*}}Parameters[{{*}}Identifier='{c}']/{{*}}SARImage/{{*}}TxId"
                )
                for c in ch_ids
            )

        all_tx_ids = [
            x.text for x in xmltree.findall("{*}TxSequence/{*}Parameters/{*}Identifier")
        ]
        if not tx_ids:
            tx_ids = sorted(all_tx_ids)
        else:
            unrecognized = tx_ids.difference(all_tx_ids)
            if unrecognized:
                raise ValueError(f"Unrecognized transmit sequence(s): {unrecognized}")
            tx_ids = sorted(tx_ids)

        f.seek(0)
        plotter = Plotter(
            f,
            html.escape(config.crsd_file),
            channels=ch_ids,
            sequences=tx_ids,
            include_fixed_pxps=config.plot_fixed,
        )
    save_func = plotter.save_combined if config.concatenate else plotter.save_separate
    prefix = (
        pathlib.PurePath(config.crsd_file).stem + "_"
        if config.prefix is None
        else config.prefix
    )
    save_func(config.output_dir, prefix=prefix, auto_open=config.auto_open)


pad_geom = cphd_plot_metadata.pad_geom
_make_image_area = cphd_plot_metadata._make_image_area


if __name__ == "__main__":
    main()  # pragma: no cover
