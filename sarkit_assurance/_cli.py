def add_cphd_chan_arg_group(parser):
    """Add a CPHD "Channel Selection" argument group to an ArgumentParser

    Intended for use with `selected_cphd_channels`
    """
    channel_group = parser.add_argument_group(
        title="Channel Selection",
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
