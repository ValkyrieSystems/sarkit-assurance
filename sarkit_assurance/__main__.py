import argparse
import sys

from . import (
    _cli,
    cphd_ipr,
    cphd_plot_metadata,
    cphd_thumb,
    crsd_plot_metadata,
    joint_chip_to_html,
    sicd_chip_to_html,
    sicd_ipr,
    sicd_plot_metadata,
    sidd_chip_to_html,
    sidd_thumb,
)


def main(args=None):
    parser = argparse.ArgumentParser(description="sarkit-assurance tools")
    subcommands = parser.add_subparsers(
        title="subcommands", required=True, dest="command"
    )

    def add_subcommand(sc: _cli.Subcommand, name, **sc_kwargs) -> None:
        command = subcommands.add_parser(
            name, **sc.get_argument_parser_kwargs(), **sc_kwargs
        )
        sc.add_arguments(command)
        command.set_defaults(command_handler=sc.run_command)

    add_subcommand(
        cphd_ipr._CphdIprSubcommand(), "cphd_ipr", help="analyze target IPRs in a CPHD"
    )
    add_subcommand(
        cphd_plot_metadata._CphdPlotMetadataSubcommand(),
        "cphd_plot_metadata",
        help="produce various plots of information contained in a CPHD",
    )
    add_subcommand(
        cphd_thumb._CphdThumbSubcommand(),
        "cphd_thumb",
        help="create thumbnails from CPHD signal arrays",
    )
    add_subcommand(
        crsd_plot_metadata._CrsdPlotMetadataSubcommand(),
        "crsd_plot_metadata",
        help="produce various plots of information contained in a CRSD",
    )
    add_subcommand(
        joint_chip_to_html._JointChipToHtmlSubcommand(),
        "joint_chip_to_html",
        help="chip brightest pixel supported in both a SICD and SIDD",
    )
    add_subcommand(
        sicd_chip_to_html._SicdChipToHtmlSubcommand(),
        "sicd_chip_to_html",
        help="plot GeoJSON Features",
    )
    add_subcommand(
        sicd_ipr._SicdIprSubcommand(),
        "sicd_ipr",
        help="analyze target IPRs in a SICD",
    )
    add_subcommand(
        sicd_plot_metadata._SicdPlotMetadataSubcommand(),
        "sicd_plot_metadata",
        help="produce various plots of information contained in a SICD",
    )
    add_subcommand(
        sidd_chip_to_html._SiddChipToHtmlSubcommand(),
        "sidd_chip_to_html",
        help="plot GeoJSON Features",
    )
    add_subcommand(
        sidd_thumb._SiddThumbSubcommand(),
        "sidd_thumb",
        help="create thumbnails from SIDD product images",
    )

    config = parser.parse_args(args)
    return config.command_handler(config)


if __name__ == "__main__":
    sys.exit(main())
