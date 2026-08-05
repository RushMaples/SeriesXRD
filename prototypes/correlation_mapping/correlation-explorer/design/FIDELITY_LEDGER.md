# Design fidelity and QA ledger

Date: 2026-08-04

Visual source of truth:

- `xrd-correlation-atlas-main-concept.png`
- `xrd-correlation-atlas-compare-concept.png`

Accepted implementation captures:

- `qa-library-search.png` — current 1568×1003 Log waterfall state; only the
  pre-denoise XY-derived display remains.

Historical layout captures retained for visual-regression reference (their
embedded index counts predate the Log waterfall policy update):

- `qa-library.png` — desktop library layout
- `qa-compare.png` — comparison layout
- `qa-mobile.png` — narrow-screen layout

## Point-by-point comparison

| Area | Concept intent | Accepted implementation | Result |
|---|---|---|---|
| Global shell | Compact white app bar, scientific mark, large search, index summary | Same four-part hierarchy; the audit icon opens the real classification audit | Match |
| Library composition | Filter rail, two-column media results, metadata inspector | 292 px filters, flexible two-column cards, 354 px inspector at 1568 px | Match |
| Scientific card hierarchy | Plot first; sample/transform/status and anchor details second | Real indexed PNG, then sample, transform, PASS, family, point UID, pressure, local peak, 2θ and q support | Match |
| Facets | Status, sample, transform, family, visualization, display domain, pressure | All requested dimensions plus signal channel, method and aggregation; across/within frame definitions remain separate | Improved fidelity |
| Semantic legend | Numeric zero must differ from missing/omitted | Separate keys for measured zero, missing slot, omitted anchor row and strict-triangle omission | Match |
| Inspector | Preview, metadata and companions in a persistent right rail | Preview/Metadata/Files tabs, q-width summary, result run, asset identity, indexed companion links | Match |
| Compare tray | Persistent 2–4 item selection with clear/compare actions | Same behavior; four-item limit is enforced with a visible notice | Match |
| Compare workspace | Equal plot columns, synchronized zoom, metadata differences | Supports 2–4 plots, synchronized zoom, only-differences control, companion actions and semantics notice | Match |
| Narrow screen | Preserve search and scientific browsing without horizontal page overflow | 390 px body width equals viewport; facets scroll horizontally; detail opens only on explicit selection | Intent preserved |

## Above-the-fold copy diff

The generated concept intentionally used illustrative copy. The implementation
replaces it with authoritative index data:

| Concept copy | Implementation copy | Reason |
|---|---|---|
| `Index 1,248,732 plots` | `Index 2,538 plots` | Exact audited gallery count after excluding Log-denoised shaded waterfalls |
| `280 matching plots` | `2 matching plots` for the exact sample query | Exact peak-number search: Log ROI heatmap plus pre-denoise XY-derived waterfall |
| Generic `Transform` | `Correlation calculation` | Prevents confusion with the curve shown in waterfalls |
| Generic `Display profile domain` | `Curve displayed` | Plain-language distinction between transformed and original-positive curves |
| Illustrative run/file names | Actual indexed run IDs and companion paths | No invented provenance |

No generated concept copy was copied into metadata when it contradicted the
index. The implementation keeps `Log²` correlation calculation separate from
`Original XY-derived (pre-denoise)` curve display everywhere.

## Intentional deviations

- The concept showed three comparison columns; the QA capture uses two to
  exercise the minimum valid comparison. The interface supports two through
  four.
- The current curated gallery has no unshaded waterfall records, so that facet
  is shown disabled rather than populated with invented content.
- Result lifecycle and validation are separate facets rather than the concept's
  combined status wording.
- Favorites and collections are browser-local; scientific result files and the
  index source remain read-only.
- On narrow screens, facets use a horizontal strip instead of a modal drawer so
  every classification remains inspectable without introducing hidden state.

## Verification record

- Index build: PASS — 1,942 Log² PlotRecords, 1,942 Assets, 0 errors, 0 warnings.
- Production build: PASS.
- Automated tests: 15/15 PASS.
- Browser console: 0 warnings or errors.
- Desktop width: 1568 px viewport and 1568 px document width.
- Mobile width: 390 px viewport and 390 px document width.
- Search examples checked: powder/3.75 GPa/peak 16/Log/ROI; single-crystal
  Log/within-frame; original-profile Log waterfall colors; q-width 0.75; and
  0–5° across-frame window.
- Interactions checked: search, URL state, favorite, two-item compare tray,
  compare navigation, synchronized 150% zoom, only-differences metadata,
  mobile detail open/close, images and companion links.
