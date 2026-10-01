import pkgutil
import subprocess
import sys

import pytest

import sarkit_assurance

RUNNABLE_MODULES = [
    "sarkit_assurance.__main__",
    "sarkit_assurance.cphd_ipr",
    "sarkit_assurance.cphd_plot_metadata",
    "sarkit_assurance.cphd_thumb",
    "sarkit_assurance.crsd_plot_metadata",
    "sarkit_assurance.joint_chip_to_html",
    "sarkit_assurance.sicd_chip_to_html",
    "sarkit_assurance.sicd_ipr",
    "sarkit_assurance.sicd_plot_metadata",
    "sarkit_assurance.sidd_chip_to_html",
    "sarkit_assurance.sidd_thumb",
]


@pytest.mark.parametrize(
    "modinfo",
    list(
        pkgutil.walk_packages(
            sarkit_assurance.__path__, sarkit_assurance.__name__ + "."
        )
    ),
)
def test_module_execution_as_script(modinfo):
    out = subprocess.check_output([sys.executable, "-m", modinfo.name, "--help"])
    assert (len(out) > 0) == (modinfo.name in RUNNABLE_MODULES)
