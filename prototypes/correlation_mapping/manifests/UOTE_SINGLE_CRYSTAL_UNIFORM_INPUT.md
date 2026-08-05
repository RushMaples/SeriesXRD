# UOTe single-crystal uniform-correlation input contract

This manifest adapts the reduced single-crystal XY files to the same generic
`frame, scan, pressure, channel, file_path` contract used by
`uniform-correlation-v2.1`.  It does not change any peak-detection, fitting,
tracking, similarity, or support threshold.

The official compression series contains one frame for each
`orientation_0deg × pressure` and `orientation_10deg × pressure` pair.  The
lowest acquisition number is selected deterministically when an orientation
has repeated exposure at the same pressure.  Excluded source rows remain in
the manifest with an explicit reason:

- the isolated 5-degree exposure has no pressure series;
- the two 2.4 GPa `noNe` frames are a separate decompression branch with only
  one pressure level;
- later 9.8 GPa repeated exposures would violate the required unique
  `scan × pressure` input key.

The source XY headers declare wavelength `0.413300 Å` and channel
`spots_masked`.  The runner must still receive the wavelength explicitly and
must verify it against every included header.
