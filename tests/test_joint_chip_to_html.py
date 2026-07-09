import sarkit_assurance.joint_chip_to_html


def test_joint_chip_to_html_smoke(tmp_path, example_sicd, example_sidd):
    sarkit_assurance.joint_chip_to_html.main(
        [str(example_sicd), str(example_sidd), str(tmp_path)]
    )

    assert len(list(tmp_path.glob("*.json"))) == 1
    assert len(list(tmp_path.glob("*.html"))) == 2
