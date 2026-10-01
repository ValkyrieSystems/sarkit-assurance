import itertools
import subprocess
import sys

EXPECTED_SUBCOMMANDS = [
    "cphd_ipr",
    "cphd_plot_metadata",
    "cphd_thumb",
    "crsd_plot_metadata",
    "joint_chip_to_html",
    "sicd_chip_to_html",
    "sicd_ipr",
    "sicd_plot_metadata",
    "sidd_chip_to_html",
    "sidd_thumb",
]


def ep_callable(epname):
    def func(args):
        subprocess.check_output([epname] + args)

    return func


def run_module(args):
    subprocess.check_output([sys.executable, "-m", "sarkit_assurance"] + args)


def test_main():
    callers = [ep_callable("sarkit-assurance"), ep_callable("ska"), run_module]
    help_flags = ["-h", "--help"]
    for subcmd, caller, help_flag in zip(
        EXPECTED_SUBCOMMANDS, itertools.cycle(callers), itertools.cycle(help_flags)
    ):
        caller([subcmd, help_flag])
