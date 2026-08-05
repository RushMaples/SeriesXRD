import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const index = JSON.parse(
  await readFile(new URL("../data/plot-index.json", import.meta.url), "utf8"),
);
const audit = JSON.parse(
  await readFile(new URL("../data/classification-audit.json", import.meta.url), "utf8"),
);

test("curated gallery has the audited logical and physical counts", () => {
  assert.equal(index.records.length, 2_538);
  assert.equal(index.assets.length, 2_140);
  assert.equal(index.summary.plot_records, 2_538);
  assert.equal(index.summary.assets, 2_140);
  assert.equal(audit.status, "PASS");
  assert.equal(audit.errors.length, 0);
  assert.equal(audit.warnings.length, 0);
});

test("formal, waterfall, and exploratory composition stays exact", () => {
  const counts = index.records.reduce((result, record) => {
    const kind = record.result_status === "exploratory"
      ? "exploratory"
      : record.visualization_type === "waterfall_shaded"
        ? record.display_profile_domain === "original_positive" ? "originalWaterfall" : "transformedWaterfall"
        : "formal";
    result[kind] = (result[kind] || 0) + 1;
    return result;
  }, {});
  assert.deepEqual(counts, {
    formal: 1_974,
    exploratory: 4,
    transformedWaterfall: 280,
    originalWaterfall: 280,
  });
});

test("Log-squared shading is available only on the pre-denoise XY-derived display", () => {
  const logWaterfalls = index.records.filter(
    (record) => record.visualization_type === "waterfall_shaded" && record.correlation_transform === "log_squared",
  );
  assert.equal(logWaterfalls.length, 280);
  assert.ok(logWaterfalls.every((record) => record.display_profile_domain === "original_positive"));
  assert.equal(
    index.records.some(
      (record) => record.visualization_type === "waterfall_shaded"
        && record.correlation_transform === "log_squared"
        && record.display_profile_domain === "correlation_transform",
    ),
    false,
  );

  const expTransformed = index.records.filter(
    (record) => record.visualization_type === "waterfall_shaded"
      && record.correlation_transform === "exp_squared"
      && record.display_profile_domain === "correlation_transform",
  );
  assert.equal(expTransformed.length, 280);
});

test("gallery excludes provenance-only source images and keeps companion targets", () => {
  assert.equal(index.records.some((record) => record.image_path_relative.includes("/_sources/")), false);
  for (const record of index.records) {
    assert.ok(record.id);
    assert.ok(record.asset_id);
    assert.ok(record.image_path.startsWith(index.results_root));
    assert.ok(Array.isArray(record.companion_paths));
  }
});
