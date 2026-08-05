# XRD Correlation Atlas

Local, read-only search and comparison UI for curated UOTe correlation plots.
The application never writes to the selected correlation results directory.

The audited gallery currently exposes **1,547 Log²-only plots** backed by
**1,547 image assets**. It includes formal Log² correlation plots, Log²
denoised transformed-profile waterfalls, and Log² correlations shaded on
pre-denoise XY-derived composites.

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
- The gallery includes both Log-denoised transformed-profile waterfalls and
  the approved pre-denoise XY-derived display suite.
- `data/plot-index.json` and `data/classification-audit.json` are reproducible
  local artifacts and are ignored by Git because they contain local paths.
- The API serves media only by indexed `plot_id`; arbitrary filesystem paths
  and non-GET methods are rejected.
- Favorites and collections live in browser local storage.

See [classification schema](docs/CLASSIFICATION_SCHEMA.md) and
[design system](design/DESIGN_SYSTEM.md). The generated concepts, accepted
implementation screenshots, and the point-by-point comparison are documented
in [the fidelity ledger](design/FIDELITY_LEDGER.md).
