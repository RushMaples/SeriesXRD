# XRD Correlation Atlas design system

The two PNG files in this directory are the visual source of truth for the
Library and Compare states. The UI is a scientific archive, not a metrics
dashboard.

## Layout

- Desktop app bar: 64 px.
- Library: 264 px filter rail, flexible results canvas, 360 px inspector.
- Compare tray: fixed to the bottom when it contains plots.
- Compare view: two to four equal plot columns above a metadata-difference
  table.
- At narrow widths, facets become a horizontally scrollable filter strip and
  metadata opens as a full-width overlay only after explicit plot selection;
  the results canvas remains the primary surface.

## Tokens

```text
background             #FFFFFF
chrome                  #F7F9FC
surface-muted           #F2F5F8
text                    #172033
text-muted              #5D6879
border                  #DDE3EA
border-strong           #C6CFDA
primary                 #0B5FCC
primary-soft            #EAF2FF
pass                    #2F8F55
pass-soft               #E7F5EB
warning                 #B7791F
warning-soft            #FFF5D9
danger                  #B84A4A
```

Viridis/plasma/coolwarm colors belong to the scientific images and legends;
they are not reused as selection, favorite, or warning colors.

## Typography and geometry

- UI font: Inter, `SF Pro Text`, `Segoe UI`, sans-serif.
- Scientific/numeric fallback: `IBM Plex Sans`, `SFMono-Regular`, monospace.
- App title 20/26 semibold; page heading 24/30 semibold.
- Controls 13/18 medium; metadata and facets 12/18; captions 11/16.
- Radii 6 or 8 px; no decorative large-radius cards.
- One-pixel dividers, minimal shadows, true-white plot surfaces.

## Component families

- App bar, global search, status summary.
- Facet group, checkbox row, numeric range.
- Result media row/card, table row, pagination.
- Inspector tabs, metadata row, companion link, semantics legend.
- Compare tray item, compare column, metadata-difference table.
- Audit summary and warning list.

## Exact terminology

- `Correlation calculation: Log²` and
  `Curve displayed: Original XY-derived (pre-denoise)`
  are separate rows.
- `Across frames — same 2θ window` and
  `Within one frame — different window pairs` are never collapsed.
- Powder anchor plots use `local peak`, explicitly not track ID.
- A zero is numeric data. Structural missing, omitted anchor rows, strict
  triangle omissions, and not-computed cells are separate states.
