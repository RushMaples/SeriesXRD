# XRD Correlation Atlas

Local, read-only search and comparison UI for curated UOTe correlation plots.
The application never writes to the selected correlation results directory.

The audited gallery currently exposes **1,942 Log²-only plots** backed by
**1,942 image assets**. It includes powder formal maps, 275-anchor
single-crystal all-peak maps, fixed-window correlations, and Log² correlation
colors shaded on original-positive XY-derived waterfall profiles.

## Run

```bash
cd prototypes/correlation_mapping/correlation-explorer
npm install
CORRELATION_RESULTS_ROOT=/path/to/correlation/results npm run index
CORRELATION_RESULTS_ROOT=/path/to/correlation/results npm run dev
```

Open `http://127.0.0.1:4311`. For a production build:

```bash
npm run build
CORRELATION_RESULTS_ROOT=/path/to/correlation/results npm start
```

Then open `http://127.0.0.1:4312`.

Run the regression suite with:

```bash
npm test
```

The tests cover audited counts, the five scientific natural-language search
examples, facets and scientific sorts, indexed image/companion delivery, and
the read-only API boundary.

## Data boundary

- `scripts/build-index.mjs` reads only the curated Log² formal and waterfall
  allowlist.
- `CORRELATION_RESULTS_ROOT` may point to a results directory outside the Git
  checkout. If it is omitted, the explorer looks for `../results`.
- The gallery excludes obsolete transformed-profile waterfalls and the old
  75-track single-crystal maps. All waterfalls use original-positive display
  profiles while retaining Log² correlation colors.
- `data/plot-index.json` and `data/classification-audit.json` are reproducible
  local artifacts and are ignored by Git because they contain local paths.
- The API serves media only by indexed `plot_id`; arbitrary filesystem paths
  and non-GET methods are rejected.
- Favorites and collections live in browser local storage.

See [classification schema](docs/CLASSIFICATION_SCHEMA.md) and
[design system](design/DESIGN_SYSTEM.md). The generated concepts, accepted
implementation screenshots, and the point-by-point comparison are documented
in [the fidelity ledger](design/FIDELITY_LEDGER.md).
