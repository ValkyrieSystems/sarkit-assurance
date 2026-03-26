"""Utilities for generating plots of CPHD metadata"""

import argparse
import html
import itertools
import pathlib

import lxml.etree
import numpy as np
import numpy.linalg as npl
import numpy.polynomial.polynomial as npp
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psp
import sarkit.cphd as skcphd
import scipy.constants
import shapely
import shapely.geometry as shg

from . import _plot_metadata

try:
    from smart_open import open
except ImportError:
    pass


class Plotter(_plot_metadata.Plotter):
    """A CPHD metadata plotter class."""

    def __init__(
        self,
        file,
        title,
        *,
        channels=None,
        include_fixed_pvps=False,
        include_all_support_arrays=False,
    ):
        with skcphd.Reader(file) as r:
            self.xml = r.metadata.xmltree
            self.ew = skcphd.ElementWrapper(self.xml.getroot())
            all_channels = [chan["Identifier"] for chan in self.ew["Data"]["Channel"]]
            self.pvps = {ch_id: r.read_pvps(ch_id) for ch_id in all_channels}

            if not channels:
                self.channels = all_channels
            else:
                if not set(channels) <= set(all_channels):
                    raise ValueError(
                        (
                            f"Unrecognized channel(s): {set(channels) - set(all_channels)}; "
                            f"Must be from: {all_channels}"
                        )
                    )
                self.channels = channels
            if include_all_support_arrays:
                sa_ids = [
                    x.text
                    for x in self.xml.findall("{*}Data/{*}SupportArray/{*}Identifier")
                ]
            else:
                sa_id_elems = []
                for ch_id in self.channels:
                    ant_pats = self.xml.findall(
                        f'{{*}}Channel/{{*}}Parameters[{{*}}Identifier="{ch_id}"]/{{*}}Antenna/{{*}}TxAPATId'
                    ) + self.xml.findall(
                        f'{{*}}Channel/{{*}}Parameters[{{*}}Identifier="{ch_id}"]/{{*}}Antenna/{{*}}RcvAPATId'
                    )
                    for ant_pat in ant_pats:
                        apat_id = ant_pat.text
                        sa_id_elems.extend(
                            self.xml.findall(
                                f'{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier="{apat_id}"]/*/{{*}}AntGPId'
                            )
                            + self.xml.findall(
                                f'{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier="{apat_id}"]/{{*}}GainPhaseArray/{{*}}ArrayId'
                            )
                            + self.xml.findall(
                                f'{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier="{apat_id}"]/{{*}}GainPhaseArray/{{*}}ElementId'
                            )
                        )
                    sa_id_elems.extend(
                        self.xml.findall(
                            f'{{*}}Channel/{{*}}Parameters[{{*}}Identifier="{ch_id}"]/{{*}}DwellTimes/{{*}}DTAId'
                        )
                    )
                sa_ids = {x.text for x in sa_id_elems}

            self.support_arrays = {x: r.read_support_array(x) for x in sa_ids}
        self.include_fixed_pvps = include_fixed_pvps
        super().__init__(title)

    # TODO: antenna stuff; does some of this belong in SARkit?
    def plot_antenna(self):
        # APATId -> list of antenna info
        apat_info = {}
        for chan in self.channels:
            chan_param = self.xml.find(
                f"{{*}}Channel/{{*}}Parameters[{{*}}Identifier='{chan}']"
            )
            chan_param_ew = skcphd.ElementWrapper(chan_param)
            if "Antenna" not in chan_param_ew:
                continue

            for side in get_antenna_info(self.xml, chan_param_ew, self.pvps[chan]):
                apat_info.setdefault(side["apat_id"], []).append(side)

        figs = []
        if not apat_info:
            return figs

        for apat_id, infos in apat_info.items():
            for pattern_type, poly_arg in {
                "Array": "delta_dcs",
                "Element": "dcs",
            }.items():
                this_fig = psp.make_subplots(rows=1, cols=2, horizontal_spacing=0.2)
                domain = np.concatenate([x[poly_arg] for x in infos], axis=0)
                x, y, samples = sample_antenna_polys_near_points(
                    infos[0]["antenna_pattern"][pattern_type], domain
                )
                # Gain
                this_fig.add_heatmap(
                    z=samples["GainPoly"],
                    x=x,
                    y=y,
                    transpose=True,
                    row=1,
                    col=1,
                    colorbar_title="Gain [dB]",
                    colorbar_x=0.42,
                    name="Gain",
                )

                # Phase
                this_fig.add_heatmap(
                    z=samples["PhasePoly"],
                    x=x,
                    y=y,
                    transpose=True,
                    row=1,
                    col=2,
                    colorbar_title="Phase [cycles]",
                    name="Phase",
                )
                for info in infos:
                    this_domain = info[poly_arg]
                    actual_dcs = go.Scatter(
                        x=this_domain[:, 0],
                        y=this_domain[:, 1],
                        name=f"{info['channel']} <{info['side']}>",
                        mode="markers",
                        marker=dict(color="rgba(128, 128, 128, 0.1)"),
                        hoverinfo="skip",
                        legendgroup=f"{info['channel']} <{info['side']}>",
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
                        f"{self.format_title('Antenna Gain/Phase Polynomials')}<br>"
                        f"{pattern_type.title()} - APATId: {apat_id}"
                    ),
                    height=700,
                    meta=f"antenna_polynomial_{pattern_type}_{apat_id}",
                )
                label_prefix = "ΔDC" if pattern_type == "Array" else "DC"
                for col in range(2):
                    this_fig.update_xaxes(
                        title_text=f"{label_prefix}X", row=1, col=1 + col
                    )
                    this_fig.update_yaxes(
                        title_text=f"{label_prefix}Y", row=1, col=1 + col
                    )
                figs.append(this_fig)
        return figs

    def plot_timeline(self):
        fig = go.Figure(
            go.Bar(
                base=[self.pvps[chan]["TxTime"].min() for chan in self.channels],
                x=[np.ptp(self.pvps[chan]["TxTime"]) for chan in self.channels],
                y=list(self.pvps.keys()),
                orientation="h",
            )
        )
        timeline = self.xml.find("{*}Global/{*}Timeline")
        fig.add_vline(
            x=float(timeline.findtext("{*}TxTime1")), annotation_text="TxTime1"
        )
        fig.add_vline(
            x=float(timeline.findtext("{*}TxTime2")), annotation_text="TxTime2"
        )
        fig.update_layout(
            xaxis_title=f"Time since CollectionStart:{timeline.findtext('{*}CollectionStart')} [s]",
            title_text=self.format_title("Timeline"),
            meta="timeline",
        )
        return [fig]

    def plot_image_area(self):
        fig = go.Figure()
        color_set = itertools.cycle(
            zip(plotly.colors.qualitative.Pastel2, plotly.colors.qualitative.Set2)
        )

        im_rect, im_poly = _make_image_area(
            self.xml.find("{*}SceneCoordinates/{*}ImageArea"),
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
            extended_area_element := self.xml.find(
                "{*}SceneCoordinates/{*}ExtendedArea"
            )
        ) is not None:
            ext_rect, ext_poly = _make_image_area(
                extended_area_element, name="Extended", colors=next(color_set)
            )
            fig.add_trace(ext_rect)
            if ext_poly is not None:
                fig.add_trace(ext_poly)

        channel_colors = dict(zip(self.channels, color_set))

        for channel_ia_element in self.xml.findall(
            "{*}Channel/{*}Parameters/{*}ImageArea"
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
        for channel, aiming in antenna_aiming.items():
            for txrcv, symbol in zip(
                ["Tx", "TxPVP", "Rcv", "RcvPVP"],
                ("triangle-down-open", "y-down", "triangle-up-open", "y-up"),
            ):
                if txrcv not in aiming:
                    continue
                boresights = aiming[txrcv]["boresights"]
                apcid = aiming[txrcv].get("APCId")

                def add_boresight_trace(points, name, color):
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
                    first_point = points[np.isfinite(points[:, 0])][0]
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
                    name=f"Channel: {channel} {txrcv} MB "
                    + (f"({apcid})" if apcid is not None else "PVP"),
                    color=channel_colors[channel][0],
                )
                if "electrical" in boresights:
                    add_boresight_trace(
                        boresights["electrical"],
                        name=f"Channel: {channel} {txrcv} EB "
                        + (f"({apcid})" if apcid is not None else "PVP"),
                        color=channel_colors[channel][-1],
                    )

        fig.update_layout(
            xaxis_title="IAY [m]",
            yaxis_title="IAX [m]",
            title_text=self.format_title("Image Area"),
            meta="image_area",
        )
        fig.update_yaxes(autorange="reversed")
        return [fig]

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
        return [fig]

    def _antenna_aiming_in_image_area(self):
        results = {}

        if self.xml.find("{*}SceneCoordinates/{*}ReferenceSurface/{*}Planar") is None:
            # Only Planar is handled
            return results

        apcs = {}
        for apc_node in self.xml.findall("{*}Antenna/{*}AntPhaseCenter"):
            apc_id = apc_node.findtext("{*}Identifier")
            apcs[apc_id] = {
                "acf_id": apc_node.findtext("{*}ACFId"),
            }

        acfs = {}
        for acf_node in self.ew["Antenna"]["AntCoordFrame"]:
            acf_id = acf_node["Identifier"]
            acfs[acf_id] = {
                "xaxis_poly": acf_node["XAxisPoly"],
                "yaxis_poly": acf_node["YAxisPoly"],
            }

        patterns = {}
        for antpat_node in self.ew["Antenna"]["AntPattern"]:
            pat_id = antpat_node["Identifier"]
            patterns[pat_id] = {
                "dcx_poly": antpat_node["EB"]["DCXPoly"],
                "dcy_poly": antpat_node["EB"]["DCYPoly"],
            }

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

        def _compute_boresights(apc_id, antpat_id, times, apc_positions):
            acf_id = apcs[apc_id]["acf_id"]
            uacx = npp.polyval(times, acfs[acf_id]["xaxis_poly"]).T
            uacy = npp.polyval(times, acfs[acf_id]["yaxis_poly"]).T

            boresights = {
                "mechanical": _intersect_boresight_with_image_area(
                    apc_positions, uacx, uacy, 0, 0
                )
            }

            if any(patterns[antpat_id]["dcx_poly"]) or any(
                patterns[antpat_id]["dcy_poly"]
            ):
                eb_dcx = npp.polyval(times, patterns[antpat_id]["dcx_poly"])
                eb_dcy = npp.polyval(times, patterns[antpat_id]["dcy_poly"])

                boresights["electrical"] = _intersect_boresight_with_image_area(
                    apc_positions, uacx, uacy, eb_dcx, eb_dcy
                )

            return boresights

        for channel in self.channels:
            chan_ant_node = self.xml.find(
                f'{{*}}Channel/{{*}}Parameters[{{*}}Identifier="{channel}"]/{{*}}Antenna'
            )
            if chan_ant_node is None:
                continue

            pvp_data = self.pvps[channel]
            indices = np.rint(
                np.linspace(0, len(pvp_data["TxTime"]) - 1, 51, endpoint=True)
            ).astype(int)
            results[channel] = {}
            tx_apc_id = chan_ant_node.findtext("{*}TxAPCId")
            results[channel]["Tx"] = {
                "APCId": tx_apc_id,
                "boresights": _compute_boresights(
                    tx_apc_id,
                    chan_ant_node.findtext("{*}TxAPATId"),
                    pvp_data["TxTime"][indices],
                    pvp_data["TxPos"][indices],
                ),
            }

            rcv_apc_id = chan_ant_node.findtext("{*}RcvAPCId")
            results[channel]["Rcv"] = {
                "APCId": rcv_apc_id,
                "boresights": _compute_boresights(
                    rcv_apc_id,
                    chan_ant_node.findtext("{*}RcvAPATId"),
                    pvp_data["RcvTime"][indices],
                    pvp_data["RcvPos"][indices],
                ),
            }
            pvp_sides = [
                side for side in ["Tx", "Rcv"] if f"{side}ACX" in pvp_data.dtype.fields
            ]
            for side in pvp_sides:
                uacx = pvp_data[f"{side}ACX"][indices]
                uacy = pvp_data[f"{side}ACY"][indices]
                eb = pvp_data[f"{side}EB"][indices]
                pos = pvp_data[f"{side}Pos"][indices]
                results[channel][f"{side}PVP"] = {
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

    def plot_image_grid(self):
        grid = self.ew["SceneCoordinates"].get("ImageGrid", None)
        if grid is None:
            return []
        grid_size = [grid["IAXExtent"]["NumLines"], grid["IAYExtent"]["NumSamples"]]
        grid_spacing = [
            grid["IAXExtent"]["LineSpacing"],
            grid["IAYExtent"]["SampleSpacing"],
        ]
        grid_image_area = grid_size * np.array(grid_spacing)
        grid_title = (
            f"Image Display Grid<br>{grid_size[0]} Lines x {grid_size[1]} Samples"
        )
        coord_title = (
            f"Grid in Image Area Coordinates: {grid_image_area[0]:.2f} m x {grid_image_area[1]:.2f} m<br>"
            f"Spacing [m]: {grid_spacing}"
        )
        fig = psp.make_subplots(cols=2, subplot_titles=(grid_title, coord_title))
        grid_first = [grid["IAXExtent"]["FirstLine"], grid["IAYExtent"]["FirstSample"]]
        grid_indices = (
            np.array(
                [
                    [0, 0],
                    [0, grid_size[1]],
                    [grid_size[0], grid_size[1]],
                    [grid_size[0], 0],
                    [0, 0],
                ]
            )
            - 0.5
        )
        iarp_ls = grid["IARPLocation"]
        grid_coords = (grid_indices + grid_first - iarp_ls) * grid_spacing
        iarp_coord = [0, 0]
        fig.add_trace(
            go.Scatter(
                x=grid_indices[:, 1],
                y=grid_indices[:, 0],
                fill="toself",
                name="Indices",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=[iarp_ls[1]], y=[iarp_ls[0]], name="IARP"), row=1, col=1
        )
        fig.update_xaxes(title_text="Samples", row=1, col=1)
        fig.update_yaxes(
            title_text="Lines",
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=grid_coords[:, 1],
                y=grid_coords[:, 0],
                fill="toself",
                name="Coordinates",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=[iarp_coord[1]], y=[iarp_coord[0]], name="IARP"), row=1, col=2
        )

        for segment in grid["SegmentList"].get("Segment", []):
            name = segment["Identifier"]
            seg_indices = np.array(
                segment["SegmentPolygon"].tolist() + [segment["SegmentPolygon"][0]]
            )
            seg_coords = (seg_indices + grid_first - iarp_ls) * grid_spacing
            fig.add_trace(
                go.Scatter(
                    x=seg_indices[:, 1],
                    y=seg_indices[:, 0],
                    name=name,
                    line={"dash": "dot"},
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=seg_coords[:, 1],
                    y=seg_coords[:, 0],
                    name=name,
                    line={"dash": "dot"},
                ),
                row=1,
                col=2,
            )

        fig.update_xaxes(title_text="IAY [m]", row=1, col=2)
        fig.update_yaxes(
            title_text="IAX [m]",
            autorange="reversed",
            scaleanchor="x2",
            scaleratio=1,
            row=1,
            col=2,
        )
        fig.update_layout(
            title_text=self.format_title("Image Grid"), height=700, meta="image_grid"
        )

        im_rect, im_poly = _make_image_area(
            self.xml.find("{*}SceneCoordinates/{*}ImageArea"),
            name="Scene",
        )
        if im_poly is not None:
            ia_coords = np.stack((im_poly.y, im_poly.x), axis=-1)
        else:
            ia_coords = np.stack((im_rect.y, im_rect.x), axis=-1)
        ia_indices = ia_coords / grid_spacing + iarp_ls - grid_first
        fig.add_trace(
            go.Scatter(
                x=ia_indices[:, 1],
                y=ia_indices[:, 0],
                name="ImageArea Indices",
                line={"dash": "dash"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ia_coords[:, 1],
                y=ia_coords[:, 0],
                name="ImageArea Coordinates",
                line={"dash": "dash"},
            ),
            row=1,
            col=2,
        )
        return [fig]

    def _plot_support_array(self, sa_id):
        data = self.support_arrays[sa_id]
        sa_elem = self.xml.find(f'{{*}}SupportArray/*[{{*}}Identifier="{sa_id}"]')
        sa_ew = skcphd.ElementWrapper(sa_elem)
        if sa_elem.tag.endswith("IAZArray"):
            axis_labels = ["IAX [m]", "IAY [m]"]
        elif sa_elem.tag.endswith("AntGainPhase"):
            axis_labels = ["DCX", "DCY"]
        elif sa_elem.tag.endswith("DwellTimeArray"):
            axis_labels = ["IAX [m]", "IAY [m]"]
        elif sa_elem.tag.endswith("AddedSupportArray"):
            axis_labels = [f"X [{sa_ew['XUnits']}]", f"Y [{sa_ew['YUnits']}]"]
        else:
            raise RuntimeError(f"Unrecognized support array tag: {sa_elem.tag}")

        domains = [
            np.arange(data.shape[0]) * sa_ew["XSS"] + sa_ew["X0"],
            np.arange(data.shape[1]) * sa_ew["YSS"] + sa_ew["Y0"],
        ]

        def make_fig(img, title=None):
            if np.ma.is_masked(img):
                fimg = np.array(
                    img, dtype=np.float64 if img.dtype.itemsize > 4 else np.float32
                )
                np.copyto(fimg, np.nan, where=img.mask)
                img = fimg
            fig = px.imshow(
                img.T,
                x=domains[0],
                y=domains[1],
                aspect="auto",
                title=title,
                origin="lower",
                labels=dict(zip(("x", "y", "color"), axis_labels)),
            )
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=domains[0].min() - sa_ew["XSS"] / 2,
                y0=domains[1].min() - sa_ew["YSS"] / 2,
                x1=domains[0].max() + sa_ew["XSS"] / 2,
                y1=domains[1].max() + sa_ew["YSS"] / 2,
            )
            fig.update_layout(meta=title)
            return fig

        def make_all_figs(data, title=None):
            figs = []
            if data.dtype.names is None:
                if np.iscomplexobj(data):
                    figs.append(
                        make_fig(data.real),
                        "real" if title is None else f"real({title})",
                    )
                    figs.append(
                        make_fig(data.imag),
                        "imag" if title is None else f"imag({title})",
                    )
                elif data.dtype.str.startswith("S"):
                    return figs
                else:
                    figs.append(make_fig(data, title))
            else:
                for name in data.dtype.names:
                    figs.extend(
                        make_all_figs(
                            data[name], name if title is None else f"{title}.{name}"
                        )
                    )
            return figs

        return make_all_figs(data)

    def plot_support_arrays(self):
        figs = []
        for name in self.support_arrays:
            new_figs = self._plot_support_array(name)
            for fig in new_figs:
                new_title = self.format_title(f"Support Array: {name}")
                if (existing_title := fig["layout"]["title"]["text"]) is not None:
                    new_title = f"{new_title}<br>{existing_title}"
                fig.update_layout(
                    title_text=new_title,
                    meta=f"support_array_{name}.{fig['layout']['meta']}",
                )
            figs.extend(new_figs)
        return figs

    def plot_dwell(self):
        # cod/dwell id -> dict of (channel_id -> valid polygon)
        cod_poly_info = {}
        dwell_poly_info = {}
        for chan in self.channels:
            chan_param = self.xml.find(
                f"{{*}}Channel/{{*}}Parameters[{{*}}Identifier='{chan}']"
            )
            chan_param_ew = skcphd.ElementWrapper(chan_param)
            cod_id = chan_param_ew["DwellTimes"]["CODId"]
            dwell_id = chan_param_ew["DwellTimes"]["DwellId"]
            cod_poly_info.setdefault(cod_id, {})[chan] = get_valid_area(
                self.xml, chan_param_ew
            )
            dwell_poly_info.setdefault(dwell_id, {})[chan] = get_valid_area(
                self.xml, chan_param_ew
            )

        def plot_dt_poly(dt_type, dt_id, chan_info):
            poly_elem = self.xml.find(
                f"{{*}}Dwell/{{*}}{dt_type}Time[{{*}}Identifier='{dt_id}']/{{*}}{dt_type}TimePoly"
            )
            poly = skcphd.Poly2dType().parse_elem(poly_elem)
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
                title=f"{self.format_title('Dwell')}<br>{dt_type} : {dt_id}",
            )
            fig.update_layout(meta=f"dwell_{dt_type}_{dt_id}")
            fig.update_yaxes(autorange="reversed")
            for ch_id, poly in chan_info.items():
                poly_vertices = shapely.get_coordinates(poly.exterior)
                poly_vertices = np.concatenate([poly_vertices, poly_vertices[:1]])
                fig.add_scatter(
                    x=poly_vertices[:, 1],
                    y=poly_vertices[:, 0],
                    fill=None,
                    name=ch_id,
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

        figs = [
            plot_dt_poly("Dwell", dt_id, chan_info)
            for dt_id, chan_info in dwell_poly_info.items()
        ]
        figs += [
            plot_dt_poly("COD", dt_id, chan_info)
            for dt_id, chan_info in cod_poly_info.items()
        ]
        return figs

    def plot_pvps(self):
        figs = {}
        for channel in self.channels:
            pvps = self.pvps[channel]
            pvp_data = {name: pvps[name] for name in pvps.dtype.names}
            fixed_pvps = {
                k: fixed_v
                for k, v in pvp_data.items()
                if (fixed_v := np.unique(v, axis=0)).shape[0] == 1
            }
            if fixed_pvps:
                figs[(channel, "Fixed PVPs")] = plot_fixed_pvp_table(fixed_pvps)

            for key, value in pvp_data.items():
                if not self.include_fixed_pvps and key in fixed_pvps:
                    continue
                if value.ndim == 1:
                    fig = plot_one_dim(value)
                elif value.ndim == 2 and value.shape[1] == 2:
                    fig = plot_two_dim(*value.T)
                elif value.ndim == 2 and value.shape[1] == 3:
                    fig = plot_three_dim(*value.T)
                figs[(channel, key)] = fig

        for (chan, key), fig in figs.items():
            fig.update_layout(
                title_text=f"<b>{key}</b> -  <i>{self.title} (channel: {chan})</i>",
                meta=f"pvp_{chan}_{key}",
            )
        return list(figs.values())

    def plot_toa_toarate(self):
        figs = []
        for channel_id in self.channels:
            pvps = self.pvps[channel_id]
            chan_param = self.xml.find(
                f"{{*}}Channel/{{*}}Parameters[{{*}}Identifier='{channel_id}']"
            )
            chan_param_ew = skcphd.ElementWrapper(chan_param)
            target_times, target_ecef_coords, target_image_coords = (
                get_valid_target_dwell(self.xml, chan_param_ew)
            )

            approx_ref_time = calc_tref(pvps)

            time_from_ref_time = target_times[..., np.newaxis] - np.expand_dims(
                approx_ref_time, (0, 1)
            )
            closest_vector = np.nanargmin(np.abs(time_from_ref_time), axis=-1)
            srps = pvps["SRPPos"][closest_vector]

            delta_toas, delta_toa_rates = geom_to_dtoa_and_dtoa_rate(
                target_ecef_coords[..., np.newaxis, :],
                pvps["TxPos"][closest_vector],
                pvps["TxVel"][closest_vector],
                pvps["RcvPos"][closest_vector],
                pvps["RcvVel"][closest_vector],
                srps,
            )
            prev_vec = np.maximum(closest_vector - 1, 0)
            next_vec = np.minimum(closest_vector + 1, pvps.shape[0] - 1)
            pris = (pvps["TxTime"][next_vec] - pvps["TxTime"][prev_vec]) / (
                next_vec - prev_vec
            )
            fcs = (pvps["FX1"][closest_vector] + pvps["FX2"][closest_vector]) / 2.0

            broadcast_coords = np.tile(
                target_image_coords[:, np.newaxis, ...], [1, delta_toas.shape[-1], 1]
            )
            fig = psp.make_subplots(
                rows=1, cols=3, subplot_titles=("Image Area", "ΔTOA", "ΔTOA Rate")
            )
            # Targets on Image Area
            image_area_traces = [
                t
                for t in _make_image_area(
                    self.xml.find("{*}SceneCoordinates/{*}ImageArea"),
                    name="Scene",
                    colors=["DimGray", "Gray"],
                )
                if t
            ]
            for trace in image_area_traces:
                trace.fillcolor = trace["line"]["color"]
                trace.opacity = 0.2
                trace.mode = None
            fig.add_traces(image_area_traces)
            fig.add_scatter(
                x=broadcast_coords[..., 1].ravel(),
                y=broadcast_coords[..., 0].ravel(),
                meta="link",
                mode="markers",
                name="Targets",
                row=1,
                col=1,
            )
            fig.update_xaxes(title_text="IAY [m]", row=1, col=1)
            fig.update_yaxes(title_text="IAX [m]", autorange="reversed", row=1, col=1)

            # Delta TOAs
            fig.add_scatter(
                x=target_times.ravel(),
                y=delta_toas.ravel(),
                meta="link",
                mode="markers",
                name="Target ΔTOAs",
                row=1,
                col=2,
            )
            for name, symbol in {
                "TOA2": "triangle-down",
                "TOA1": "triangle-up",
            }.items():
                fig.add_scatter(
                    x=target_times.ravel(),
                    y=pvps[name][closest_vector].ravel(),
                    marker_color="cyan",
                    marker_symbol=symbol,
                    mode="markers",
                    meta="link",
                    name=f"Δ{name}",
                    legendgroup="toa1_toa2",
                    row=1,
                    col=2,
                )
            if set(("TOAE1", "TOAE2")) < set(pvps.dtype.names):
                for name, symbol in {
                    "TOAE2": "triangle-down",
                    "TOAE1": "triangle-up",
                }.items():
                    fig.add_scatter(
                        x=target_times.ravel(),
                        y=pvps[name][closest_vector].ravel(),
                        marker_color="magenta",
                        marker_symbol=symbol,
                        mode="markers",
                        meta="link",
                        name=f"Δ{name}",
                        legendgroup="toae1_toae2",
                        row=1,
                        col=2,
                    )
            fig.update_xaxes(title_text="slow time [s]", row=1, col=2)
            fig.update_yaxes(title_text="ΔTOA [s]", row=1, col=2)

            # Delta TOA Rates
            fig.add_scatter(
                x=target_times.ravel(),
                y=delta_toa_rates.ravel(),
                meta="link",
                mode="markers",
                name="Target ΔTOA Rates",
                row=1,
                col=3,
            )
            for sgn, symbol in {1: "triangle-down", -1: "triangle-up"}.items():
                fig.add_scatter(
                    x=target_times.ravel(),
                    y=(sgn / pris / fcs / 2).ravel(),
                    marker_color="gray",
                    marker_symbol=symbol,
                    mode="markers",
                    meta="link",
                    name="PRF Folding Lines",
                    legendgroup="PRF Folding Lines",
                    row=1,
                    col=3,
                )
            fig.update_xaxes(title_text="slow time [s]", row=1, col=3)
            fig.update_yaxes(title_text="ΔTOA Rate [s/s]", row=1, col=3)
            fig.update_layout(
                title_text=(f"{self.format_title('ΔTOA & ΔTOA Rate')}<br>{channel_id}"),
                height=700,
                meta=f"delta_toa_delta_toa_rate_{channel_id}",
            )
            figs.append(fig)
        return figs


def _make_image_area(area_element, name=None, colors=None):
    area_ew = skcphd.ElementWrapper(area_element)
    x1, y1 = area_ew["X1Y1"]
    x2, y2 = area_ew["X2Y2"]
    # swap x/y in trace so x=rows/vertical, y=cols/horizontal
    rect = go.Scatter(
        x=[y1, y2, y2, y1, y1],
        y=[x1, x1, x2, x2, x1],
        fill="toself",
        name=f"{name + ' ' if name is not None else ''}Rectangle",
    )
    if colors:
        rect["line"]["color"] = colors[0]
    if (vertices := area_ew.get("Polygon", None)) is not None:
        vertices = np.concatenate([vertices, vertices[:1]])
        poly = go.Scatter(
            x=vertices[:, 1],
            y=vertices[:, 0],
            fill="toself",
            name=f"{name + ' ' if name is not None else ''}Polygon",
            line={"color": rect["line"]["color"], "dash": "dot", "width": 1},
        )
        if colors:
            poly["line"]["color"] = colors[-1]
    else:
        poly = None
    return rect, poly


def plot_one_dim(ydata):
    """Plot a single parameter versus index

    Args
    ----
    ydata: `numpy.ndarray`
        The y-axis data values.

    """
    fig = go.Figure(
        data=go.Scatter(x=np.arange(0, len(ydata)), y=ydata, mode="markers")
    )
    fig.update_layout(xaxis_title="Vector #")
    return fig


def plot_two_dim(xdata, ydata):
    """Plot a 2D parametric curve

    Args
    ----
    xdata, ydata: `numpy.ndarray`
        The x-axis and y-axis data values.

    """
    fig = go.Figure(
        data=go.Scatter(
            x=xdata,
            y=ydata,
            mode="markers",
            marker={
                "size": 2,
                "color": np.arange(xdata.size),
                "colorbar": {
                    "title": "Vector #",
                },
            },
        )
    )
    return fig


def plot_three_dim(xdata, ydata, zdata):
    """Plot a 3D parametric curve

    Args
    ----
    xdata, ydata, zdata: `numpy.ndarray`
        The x-axis, y-axis, and z-axis data values.

    """
    return go.Figure(
        data=[
            go.Scatter3d(
                x=xdata,
                y=ydata,
                z=zdata,
                mode="markers",
                marker={
                    "size": 2,
                    "color": np.arange(xdata.size),
                    "colorbar": {
                        "title": "Vector #",
                    },
                },
            )
        ]
    )


def plot_fixed_pvp_table(fixed_pvps):
    """Create a table displaying PVPs whose values are fixed."""
    align = ["left", "center"]
    return go.Figure(
        data=[
            go.Table(
                header={"values": ["Parameter", "Value"], "align": align},
                cells={
                    "values": [list(fixed_pvps.keys()), list(fixed_pvps.values())],
                    "align": align,
                },
            )
        ]
    )


def get_valid_area(xmltree, chan_param_ew):
    ew = skcphd.ElementWrapper(xmltree.getroot())

    def get_imagearea_poly(imgarea_ew):
        region = shg.box(*imgarea_ew["X1Y1"], *imgarea_ew["X2Y2"])
        if (poly := imgarea_ew.get("Polygon", None)) is not None:
            region = region.intersection(shg.Polygon(poly))
        return region

    ia_poly = get_imagearea_poly(ew["SceneCoordinates"]["ImageArea"])
    if "ImageArea" in chan_param_ew:
        ia_poly = get_imagearea_poly(chan_param_ew["ImageArea"])
    return ia_poly


def get_valid_targets(xmltree, chan_param_ew, grid_size=11):
    ia_poly = get_valid_area(xmltree, chan_param_ew)

    # grid of points and intersect
    bounds = np.asarray(ia_poly.bounds).reshape(2, 2)  # [[xmin, ymin], [xmax, ymax]]
    mesh = np.stack(
        np.meshgrid(
            np.linspace(bounds[0, 0], bounds[1, 0], grid_size),
            np.linspace(bounds[0, 1], bounds[1, 1], grid_size),
        ),
        axis=-1,
    )
    coords = shg.MultiPoint(
        np.concatenate(
            [mesh.reshape(-1, 2), np.asarray(ia_poly.exterior.coords)[:-1, :]],
            axis=0,
        )
    )

    return shapely.get_coordinates(ia_poly.intersection(coords))


def get_target_dwelltimes(target_ia_coords, xmltree, chan_param_ew):
    dwell_id = chan_param_ew["DwellTimes"]["DwellId"]
    cod_id = chan_param_ew["DwellTimes"]["CODId"]

    dwell_elem = xmltree.find(
        f'{{*}}Dwell/{{*}}DwellTime[{{*}}Identifier="{dwell_id}"]/{{*}}DwellTimePoly'
    )
    cod_elem = xmltree.find(
        f'{{*}}Dwell/{{*}}CODTime[{{*}}Identifier="{cod_id}"]/{{*}}CODTimePoly'
    )

    dwell_poly = skcphd.Poly2dType().parse_elem(dwell_elem)
    cod_poly = skcphd.Poly2dType().parse_elem(cod_elem)

    dwell_times = npp.polyval2d(
        target_ia_coords[..., 0], target_ia_coords[..., 1], dwell_poly
    )
    cod_times = npp.polyval2d(
        target_ia_coords[..., 0], target_ia_coords[..., 1], cod_poly
    )
    return cod_times, dwell_times


def get_valid_target_dwell(
    xmltree, chan_param_ew, target_grid_size=11, dwell_grid_size=11
):
    """Return a set of targets spanning a channel's image area along with times spanning their dwell"""
    target_ia_coords = get_valid_targets(xmltree, chan_param_ew, target_grid_size)
    cod_times, dwell_times = get_target_dwelltimes(
        target_ia_coords, xmltree, chan_param_ew
    )

    target_ecef_coords = iac_to_ecef(xmltree, target_ia_coords)

    target_times = cod_times[..., np.newaxis] + dwell_times[
        ..., np.newaxis
    ] * np.linspace(-0.5, 0.5, dwell_grid_size)
    return target_times, target_ecef_coords, target_ia_coords


def iac_to_ecef(xmltree, ia_coords):
    """Converts ImageAreaCoordinates to ECF"""
    # TODO: should be in SARkit
    sc_ew = skcphd.ElementWrapper(xmltree.find("{*}SceneCoordinates"))
    if "Planar" in sc_ew["ReferenceSurface"]:
        iarp = sc_ew["IARP"]["ECF"]
        uiax = sc_ew["ReferenceSurface"]["Planar"]["uIAX"]
        uiay = sc_ew["ReferenceSurface"]["Planar"]["uIAY"]
        # TODO: in SARkit, should have a conditional uiaz
        return iarp + uiax * ia_coords[..., :1] + uiay * ia_coords[..., 1:]
    else:
        raise NotImplementedError("HAE surface not supported")


def geom_to_dtoa_and_dtoa_rate(
    ecef_point, tx_apc_pos, tx_apc_vel, rcv_apc_pos, rcv_apc_vel, srp
):
    """Calculate delta time of arrival and delta time of arrival rate."""

    def one_dir(apc_pos, apc_vel, pt):
        los_to_apc = apc_pos - pt
        r_apc = npl.norm(los_to_apc, axis=-1)
        toa = r_apc / scipy.constants.speed_of_light
        toa_rate = (
            np.vecdot(los_to_apc / r_apc[..., np.newaxis], apc_vel)
            / scipy.constants.speed_of_light
        )
        return toa, toa_rate

    def one_dir_rel(apc_pos, apc_vel):
        pt_toa, pt_rate = one_dir(apc_pos, apc_vel, ecef_point)
        srp_toa, srp_rate = one_dir(apc_pos, apc_vel, srp)
        return pt_toa - srp_toa, pt_rate - srp_rate

    tx_toa, tx_toa_rate = one_dir_rel(tx_apc_pos, tx_apc_vel)
    rcv_toa, rcv_toa_rate = one_dir_rel(rcv_apc_pos, rcv_apc_vel)
    return tx_toa + rcv_toa, tx_toa_rate + rcv_toa_rate


def calc_tref(pvps):
    # TODO: should be in SARkit
    r_xmt_srp = np.linalg.norm(pvps["TxPos"] - pvps["SRPPos"], axis=-1)
    r_rcv_srp = np.linalg.norm(pvps["RcvPos"] - pvps["SRPPos"], axis=-1)
    return pvps["TxTime"] + (r_xmt_srp / (r_xmt_srp + r_rcv_srp)) * (
        pvps["RcvTime"] - pvps["TxTime"]
    )


def get_onesided_info(
    xmltree,
    apc_id,
    apat_id,
    target_times,
    target_ecef_coords,
    apc_pos,
    apc_times,
    ref_times,
):
    apat = xmltree.find(f"{{*}}Antenna/{{*}}AntPattern[{{*}}Identifier='{apat_id}']")
    acf_id = xmltree.findtext(
        f"{{*}}Antenna/{{*}}AntPhaseCenter[{{*}}Identifier='{apc_id}']/{{*}}ACFId"
    )
    acf = xmltree.find(f"{{*}}Antenna/{{*}}AntCoordFrame[{{*}}Identifier='{acf_id}']")

    acf_ew = skcphd.ElementWrapper(acf)
    apat_ew = skcphd.ElementWrapper(apat)

    time_from_ref_time = target_times[..., np.newaxis] - np.expand_dims(
        ref_times, (0, 1)
    )
    closest_vector = np.nanargmin(np.abs(time_from_ref_time), axis=-1)

    los = target_ecef_coords - apc_pos[closest_vector]
    ulos = los / npl.norm(los, axis=-1, keepdims=True)

    t = apc_times[closest_vector]
    acx = np.moveaxis(npp.polyval(t, acf_ew["XAxisPoly"]), 0, -1)
    acy = np.moveaxis(npp.polyval(t, acf_ew["YAxisPoly"]), 0, -1)

    def unit(v):
        return v / npl.norm(v, axis=-1, keepdims=True)

    uacz = unit(np.cross(acx, acy))
    u = unit(unit(acx) + unit(acy))
    v = np.cross(uacz, u)
    acx_norm = unit(u - v)
    acy_norm = unit(u + v)

    dcx = np.vecdot(ulos, acx_norm)
    dcy = np.vecdot(ulos, acy_norm)
    dcs = np.stack((dcx, dcy), axis=-1).reshape(-1, 2)

    eb_dcx = npp.polyval(t, apat_ew["EB"]["DCXPoly"])
    eb_dcy = npp.polyval(t, apat_ew["EB"]["DCYPoly"])
    eb_dcs = np.stack((eb_dcx, eb_dcy), axis=-1).reshape(-1, 2)

    return {
        "apat_id": apat_id,
        "antenna_pattern": apat_ew,
        "dcs": dcs,
        "eb_dcs": eb_dcs,
        "delta_dcs": dcs - eb_dcs,
    }


def get_antenna_info(xmltree, chan_param_ew, chan_pvps):
    """Return the direction cosines for a set of targets spanning the scene polygon over times spanning their dwell."""
    target_times, target_ecef_coords, _ = get_valid_target_dwell(xmltree, chan_param_ew)
    tref = calc_tref(chan_pvps)
    retval = []
    for side in ("Tx", "Rcv"):
        this_info = get_onesided_info(
            xmltree,
            chan_param_ew["Antenna"][f"{side}APCId"],
            chan_param_ew["Antenna"][f"{side}APATId"],
            target_times,
            target_ecef_coords[..., np.newaxis, :],
            chan_pvps[f"{side}Pos"],
            chan_pvps[f"{side}Time"],
            tref,
        )
        this_info["channel"] = chan_param_ew["Identifier"]
        this_info["side"] = side
        retval.append(this_info)
    return retval


def pad_geom(geom, frac):
    buffer_len = frac * np.max(np.diff(np.array(geom.bounds).reshape((-1, 2)), axis=0))
    return geom.buffer(buffer_len)


def sample_antenna_polys_near_points(apat_gp_ew, dcs, scale=1.05):
    """Sample antenna polynomials on a grid that encompasses a set of direction cosines."""
    dcs_polygon = shg.MultiPoint(dcs).convex_hull
    dcs_expanded_polygon = pad_geom(dcs_polygon, scale - 1.0).intersection(
        shg.box(-1, -1, 1, 1)
    )

    x = np.linspace(*dcs_expanded_polygon.envelope.bounds[0::2], 63)
    y = np.linspace(*dcs_expanded_polygon.envelope.bounds[1::2], 65)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    grid_points = shg.MultiPoint(np.stack([x_grid, y_grid], -1).reshape(-1, 2))
    mask = np.array(
        [pt.within(dcs_expanded_polygon) for pt in grid_points.geoms]
    ).reshape(x_grid.shape)

    samples = {}
    for name in ("GainPoly", "PhasePoly"):
        poly = apat_gp_ew[name]
        these_samples = npp.polygrid2d(x, y, poly)
        these_samples[~mask] = np.nan
        samples[name] = these_samples

    return x, y, samples


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Produce various plots of information contained in a CPHD"
    )
    parser.add_argument("cphd_file", help="CPHD file to analyze")
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
        help="prefix used in output filenames (Default: {cphd_file.stem}_)",
    )
    parser.add_argument(
        "-c",
        "--concatenate",
        action="store_true",
        help="concatenate plots into single HTML",
    )
    parser.add_argument("--plot-fixed", action="store_true", help="plot fixed PVPs")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_false",
        dest="auto_open",
        help="don't open plots after creation",
    )
    parser.add_argument(
        "--all-support-arrays",
        action="store_true",
        help="plot all support arrays. Default is to plot only support arrays referenced by a plotted channel",
    )
    channel_group = parser.add_mutually_exclusive_group()
    channel_group.add_argument(
        "--ref-chan", action="store_true", help="only use the reference channel's PVPs"
    )
    channel_group.add_argument(
        "--chan",
        action="extend",
        nargs="+",
        help="use the specified channels' PVPs (default: all)",
    )
    config = parser.parse_args(args)

    with open(config.cphd_file, "rb") as f, skcphd.Reader(f) as r:
        if config.ref_chan:
            channels = [r.metadata.xmltree.findtext("{*}Channel/{*}RefChId")]
        else:
            channels = config.chan

        f.seek(0)
        plotter = Plotter(
            f,
            html.escape(config.cphd_file),
            channels=channels,
            include_fixed_pvps=config.plot_fixed,
            include_all_support_arrays=config.all_support_arrays,
        )
    save_func = plotter.save_combined if config.concatenate else plotter.save_separate
    prefix = (
        pathlib.PurePath(config.cphd_file).stem + "_"
        if config.prefix is None
        else config.prefix
    )
    save_func(config.output_dir, prefix=prefix, auto_open=config.auto_open)


if __name__ == "__main__":
    main()  # pragma: no cover
