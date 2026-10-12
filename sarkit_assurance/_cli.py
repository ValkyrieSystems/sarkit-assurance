import argparse
import typing


def add_cphd_chan_arg_group(parser):
    """Add a CPHD "Channel Selection" argument group to an ArgumentParser

    Intended for use with `selected_cphd_channels`
    """
    channel_group = parser.add_argument_group(
        title="channel selection",
        description="If these arguments are omitted, all channels are used.",
    )
    channel_group.add_argument(
        "--ref-chan", action="store_true", help="include the reference channel"
    )
    channel_group.add_argument(
        "--chan",
        action="extend",
        nargs="+",
        help="channel identifier(s) to include",
    )
    return channel_group


def selected_cphd_channels(cphd_xmltree, args) -> list[str]:
    """Return a sorted, deduplicated list of requested CPHD channel identifiers

    Intended for use with `add_cphd_chan_arg_group`
    """
    ch_ids = set()
    if args.chan:
        ch_ids.update(args.chan)
    if args.ref_chan:
        ch_ids.add(cphd_xmltree.findtext("{*}Channel/{*}RefChId"))

    all_ch_ids = [
        x.text for x in cphd_xmltree.findall("{*}Channel/{*}Parameters/{*}Identifier")
    ]
    if not ch_ids:
        return sorted(all_ch_ids)
    unrecognized = ch_ids.difference(all_ch_ids)
    if unrecognized:
        raise ValueError(f"Unrecognized channel(s): {unrecognized}")
    return sorted(ch_ids)


class Subcommand:
    """Class describing a CLI subcommand"""

    def get_argument_parser_kwargs(self) -> dict[str, typing.Any]:
        """ArgumentParser constructor arguments

        Returns
        -------
        dict
            dictionary of ArgumentParser arguments
        """
        raise NotImplementedError()

    def add_arguments(self, parser: argparse.ArgumentParser):
        """Add arguments to a parser

        Parameters
        ----------
        parser : argparse.ArgumentParser

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def run_command(self, config: argparse.Namespace) -> int:
        """Run the subcommand

        Parameters
        ----------
        config : argparse.Namespace

        Returns
        -------
        int
            return code
        """
        raise NotImplementedError()

    def as_callable(self):
        parser = argparse.ArgumentParser(**self.get_argument_parser_kwargs())
        self.add_arguments(parser)

        def wrap(args=None):
            config = parser.parse_args(args)
            return self.run_command(config)

        return wrap
