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
  assert.equal(index.records.length, 1_942);
  assert.equal(index.assets.length, 1_942);
  assert.equal(index.summary.plot_records, 1_942);
  assert.equal(index.summary.assets, 1_942);
  assert.equal(audit.status, "PASS");
  assert.equal(audit.errors.length, 0);
  assert.equal(audit.warnings.length, 0);
});

test("formal and waterfall composition stays exact", () => {
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
    formal: 1387,
    originalWaterfall: 555,
  });
});

test("Log-squared waterfalls use original-profile displays only", () => {
  const logWaterfalls = index.records.filter(
    (record) => record.visualization_type === "waterfall_shaded" && record.correlation_transform === "log_squared",
  );
  assert.equal(logWaterfalls.length, 555);
  assert.equal(
    logWaterfalls.filter((record) => record.display_profile_domain === "correlation_transform").length,
    0,
  );
  assert.equal(
    logWaterfalls.filter((record) => record.display_profile_domain === "original_positive").length,
    555,
  );
  assert.ok(index.records.every((record) => record.correlation_transform === "log_squared"));
});

test("single-crystal peak maps include all 275 independent anchors", () => {
  for (const family of ["roi_area", "location"]) {
    const rows = index.records.filter(
      (record) => record.sample === "single_crystal"
        && record.visualization_type === "heatmap"
        && record.correlation_family === family,
    );
    assert.equal(rows.length, 275);
    assert.ok(rows.every((record) => record.entity_type === "single_crystal_frame_local_peak_anchor"));
    assert.ok(rows.every((record) => record.track === null));
  }
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
