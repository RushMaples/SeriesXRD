# Demo data attribution

The SeriesXRD Ti-6Al-4V demo uses a small exposure-time subset of the
following dataset:

> Daniel, C. S., Zeng, X., Michalik, Š., & Quinta da Fonseca, J. (2022).
> *Synchrotron X-ray Diffraction Dataset - Measuring Bulk Crystallographic
> Texture from Differently-Orientated Ti-6Al-4V Samples* (v1) [Data set].
> Zenodo. <https://doi.org/10.5281/zenodo.7270710>

The source is licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
SeriesXRD downloads only `rawdata_1.zip`, `rawdata_calibration.zip`, and
`sxrd_metadata_1.yaml`. It extracts 12 Ti-6Al-4V CBF frames and one CeO2
calibration CBF, gives the CBF files descriptive names, and records the
original archive path and a SHA-256 checksum for every extracted file. The
scientific pixel data are not modified.

The starting geometry and analysis context were checked against:

> Daniel, C. S., Zeng, X., Hunt, S., & Quinta da Fonseca, J. (2022).
> *Synchrotron X-ray Diffraction Analysis - Measuring Bulk Crystallographic
> Texture from Differently-Orientated Ti-6Al-4V Samples* (v1) [Data set].
> Zenodo. <https://doi.org/10.5281/zenodo.7310550>

## Derived calibration file

`geometry/Ti64_908mm_refined.poni` is a SeriesXRD demo artifact derived from
CeO2 run 103764. It was refined with pyFAI using the published wavelength
(0.12423 A), a PilatusCdTe2M detector, and the nominal 908 mm distance. The
median absolute mismatch between the first 15 observed and reference CeO2
rings was approximately 0.0010 degrees 2-theta (maximum 0.0021 degrees).

The CBF headers contain placeholder wavelength and detector-distance values.
For this demo, the published YAML metadata and the supplied PONI file are the
authoritative sources for geometry.

## BibTeX

```bibtex
@dataset{daniel_2022_ti64_sxrd,
  author    = {Daniel, Christopher Stuart and Zeng, Xiaohan and
               Michalik, {\v{S}}tefan and Quinta da Fonseca, Jo{\~a}o},
  title     = {Synchrotron X-ray Diffraction Dataset - Measuring Bulk
               Crystallographic Texture from Differently-Orientated
               Ti-6Al-4V Samples},
  year      = {2022},
  publisher = {Zenodo},
  version   = {v1},
  doi       = {10.5281/zenodo.7270710},
  url       = {https://doi.org/10.5281/zenodo.7270710}
}
```
