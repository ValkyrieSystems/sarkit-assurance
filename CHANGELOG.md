# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `joint_chip_to_html`
- `crsd_plot_metadata`


## [0.2.1] - 2026-06-29

### Fixed
- computation of `delta_row_col` in `sidd_chip_to_html._proj_ecef_to_image`
- made log remap configurable so that it can be used for chipping of 16-bit SIDDs


## [0.2.0] - 2026-06-04

### Added
- `sicd_chip_to_html`
- `sidd_chip_to_html`


## [0.1.1] - 2026-05-05

### Fixed
- `cphd_plot_metadata.Plotter.plot_image_grid` for Segments without a SegmentPolygon


## [0.1.0] - 2026-04-16

### Added
- `cphd_plot_metadata`
- `cphd_thumb`
- `sicd_plot_metadata`
- `sidd_thumb`

[unreleased]: https://github.com/ValkyrieSystems/sarkit-assurance/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/ValkyrieSystems/sarkit-assurance/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ValkyrieSystems/sarkit-assurance/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ValkyrieSystems/sarkit-assurance/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ValkyrieSystems/sarkit-assurance/releases/tag/v0.1.0
