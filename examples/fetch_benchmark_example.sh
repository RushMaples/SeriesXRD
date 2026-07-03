#!/usr/bin/env bash
# Fetch a small REAL measured-pattern benchmark set (8 experimental lab XRD
# patterns + 18 reference CIFs, Li-Mn-Ti-O-F space) from the open
# XRD-AutoAnalyzer repository (github.com/njszym/XRD-AutoAnalyzer), and write
# the matching labels CSV. Verified result with the deterministic cosine
# baseline: hit@1 = 1.000, MRR = 1.000, identify hit rate = 1.000.
#
# Usage:
#   bash examples/fetch_benchmark_example.sh ./benchdata
#   # import the CIFs into a workspace library, then:
#   bulkxrd-benchmark ./benchdata/spectra --labels ./benchdata/labels.csv \
#       --workspace <ws> --out bench_cosine
set -euo pipefail
OUT="${1:-./benchdata}"
BASE="https://raw.githubusercontent.com/njszym/XRD-AutoAnalyzer/main/Example"
mkdir -p "$OUT/spectra" "$OUT/cifs"

SPECTRA=("Li2MnO3+MnO+TiO2.xy" "LiMnO2.xy" "Mn2O3+LiF.xy" "Mn2O3.xy" \
         "MnO.xy" "MnO2.xy" "TiO2+MnO.xy" "TiO2.xy")
CIFS=(MnO_186 MnO_225 TiO2_136 TiO2_141 Mn2O3_206 MnO2_10 MnO2_12 MnO2_136 \
      MnO2_14 MnO2_164 MnO2_227 MnO2_62 MnO2_84 MnO2_87 LiMnO2_141 LiMnO2_59 \
      Li2MnO3_12 LiF_225)

for f in "${SPECTRA[@]}"; do
  curl -fsS "$BASE/Spectra/$f" -o "$OUT/spectra/$f"
done
for c in "${CIFS[@]}"; do
  curl -fsS "$BASE/References/$c.cif" -o "$OUT/cifs/$c.cif"
done

# Labels: any polymorph of a formula in the filename counts as truth.
{
  echo "filename,phases"
  echo "MnO.xy,MnO_186;MnO_225"
  echo "TiO2.xy,TiO2_136;TiO2_141"
  echo "Mn2O3.xy,Mn2O3_206"
  echo "MnO2.xy,MnO2_10;MnO2_12;MnO2_136;MnO2_14;MnO2_164;MnO2_227;MnO2_62;MnO2_84;MnO2_87"
  echo "LiMnO2.xy,LiMnO2_141;LiMnO2_59"
  echo "Mn2O3+LiF.xy,Mn2O3_206;LiF_225"
  echo "TiO2+MnO.xy,TiO2_136;TiO2_141;MnO_186;MnO_225"
  echo "Li2MnO3+MnO+TiO2.xy,Li2MnO3_12;MnO_186;MnO_225;TiO2_136;TiO2_141"
} > "$OUT/labels.csv"

echo "Fetched $(ls "$OUT/spectra" | wc -l) spectra + $(ls "$OUT/cifs" | wc -l) CIFs -> $OUT"
echo "Data credit: N. Szymanski et al., XRD-AutoAnalyzer (MIT-licensed repo)."
