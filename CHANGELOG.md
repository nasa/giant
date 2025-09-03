# Changelog

All notable changes to this project will be documented in this file.
This project loosely adheres to Semantic Versioning.

## [2.0.0] - 2025-09-03

Note:
- This release is most assuredly not backwards compatible with GIANT version 1.0
- There are additional improvements and fixes not listed here.

### Added
- New functionality in the coverage and photometry subpackages.
- New examples folder.
- New constraint_matching relative navigation capability leveraging opportunistic features.

### Changed
- Updated the build system to current Python standards.
- Significant internal refactoring to improve readability and maintainability.
  - Split the former image_processing module into a package.
    - Replaced the ImageProcessing class with focused components (e.g., PointSourceFinder, ImageFlattener, ...).
  - Split several other modules into individual packages (with less backwards compatibility breaking)
    - rotations
    - stellar_opnav.estimators
    - stellar_opnav.visualizers
    - calibration.estimators
    - calibration.visualizers
- Renamed “catalogue” to “catalog” throughout the codebase and documentation. Also renamed a few other less consequential classes and functions
- Updated license to Apache 2.0.
- Improved type hinting across the codebase.
- Updated most interfaces to use dataclass-based options instances instead of keyword arguments.

### Removed
- Removed GIANTCatalog; use Gaia instead.

### Fixed
- Numerous bugs and edge cases.
