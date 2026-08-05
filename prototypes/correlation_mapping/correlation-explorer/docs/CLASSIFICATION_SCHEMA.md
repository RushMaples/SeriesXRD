# Classification schema

The explorer keeps logical scientific records separate from physical image
assets. This preserves run and display-profile semantics even when images are
hardlinks or have identical SHA-256 values.

## Primary entities

### PlotRecord

One searchable scientific context. Important fields include result status,
validation status, sample, run transform, transform dependency, correlation
family, visualization type, display-profile domain, anchor/window metadata,
companions, classification evidence, and warnings.

### Asset

One physical PNG identified by device/inode and SHA-256. Multiple PlotRecords
may reference the same Asset. Asset deduplication never removes logical
records.

### DataCompanion

An allowlisted CSV, compressed CSV, JSON, or manifest related to a PlotRecord.
The media server can serve only these indexed paths.

## Curated gallery allowlist

The default gallery contains:

```text
formal correlation plots, Log²                         987
Log-denoised transformed-profile powder waterfalls     280
pre-denoise XY-derived powder waterfalls, Log²         280
total                                                  1547
```

The formal root's `_sources/` tree is metadata/provenance input and is not a
gallery source.

## Semantic constraints

- `result_status` and `validation_status` are independent.
- `correlation_transform` and `display_profile_domain` are independent.
- Log² shaded waterfalls are available in two explicit display domains.
  `correlation_transform` shows the Log-denoised profile used by the ROI
  calculation. `original_positive` keeps the same Log² correlation color
  while reconstructing height from source spots-channel XY signal before the
  nonlinear transform.
- `original_positive` is not an untouched representative raw scan. It is the
  previously approved pressure-level composite: positive-clipped and
  measurement-normalized source XY components are summed within frame,
  averaged across distinct frames per peak, and the 12–22 formal peaks are
  summed at each pressure. This preserves one curve per GPa and one visible
  support for every formal correlation cell.
- Location plots remain part of the single Log² formal package.
- Powder c=0.75 applies to powder ROI/waterfall support, not single-crystal
  ellipses or integer window products.
- A pressure-level point has no single natural scalar q-width. Store support
  bounds and, when available, observation q-width min/median/max; otherwise use
  null rather than inventing a value.
- Powder point UIDs and single-crystal track IDs are different namespaces.
- Formal counts are checked on PlotRecords; storage deduplication is checked on
  Assets.

## Classification evidence priority

1. Master/companion CSV, JSON, manifest, and validation reports.
2. Allowlisted directory structure.
3. Filename parsing.
4. Image title/OCR only as a recorded fallback.
