import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import test, { after, before } from "node:test";

const port = 4327;
const base = `http://127.0.0.1:${port}`;
let server;

before(async () => {
  server = spawn(process.execPath, ["server/index.mjs"], {
    cwd: new URL("..", import.meta.url),
    env: { ...process.env, PORT: String(port), HOST: "127.0.0.1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const ready = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("API server did not become ready")), 10_000);
    server.stdout.setEncoding("utf8");
    server.stdout.on("data", (chunk) => {
      if (chunk.includes("XRD Correlation Atlas API")) {
        clearTimeout(timer);
        resolve();
      }
    });
    server.once("exit", (code) => reject(new Error(`API server exited early (${code})`)));
  });
  await ready;
});

after(async () => {
  if (!server || server.exitCode !== null) return;
  server.kill("SIGTERM");
  await once(server, "exit");
});

async function json(path, options) {
  const response = await fetch(`${base}${path}`, options);
  return { response, body: await response.json() };
}

test("metadata and default page expose the full audited index", async () => {
  const meta = await json("/api/meta");
  assert.equal(meta.response.status, 200);
  assert.equal(meta.body.summary.plot_records, 2_538);
  assert.equal(meta.body.summary.assets, 2_140);
  assert.equal(meta.body.audit_status, "PASS");

  const plots = await json("/api/plots?page_size=2");
  assert.equal(plots.response.status, 200);
  assert.equal(plots.body.total, 2_538);
  assert.equal(plots.body.index_total, 2_538);
  assert.equal(plots.body.items.length, 2);
  assert.ok(plots.body.items[0].image_url.startsWith("/api/plots/"));
});

const searches = [
  ["powder 3.75 GPa peak 16 log ROI", (item) => item.sample === "powder" && item.anchor_pressure_gpa === 3.75 && item.anchor_peak_number === 16],
  ["single crystal exp within frame", (item) => item.sample === "single_crystal" && item.correlation_transform === "exp_squared" && item.correlation_family === "window_to_window_within_same_frame"],
  ["original xy waterfall log colors", (item) => item.display_profile_domain === "original_positive" && item.visualization_type === "waterfall_shaded"],
  ["qwidth 0.75", (item) => item.half_width_factor === 0.75],
  ["window 0-5 across frames", (item) => item.window_start_deg === 0 && item.window_end_deg === 5 && item.correlation_family === "window_to_window_across_frames"],
];

for (const [query, predicate] of searches) {
  test(`scientific search resolves: ${query}`, async () => {
    const result = await json(`/api/plots?page_size=100&q=${encodeURIComponent(query)}`);
    assert.equal(result.response.status, 200);
    assert.ok(result.body.total > 0, `No match for ${query}`);
    assert.ok(result.body.items.every(predicate), `Unexpected match for ${query}`);
  });
}

test("facet filters and scientific sorts operate on indexed fields", async () => {
  const filtered = await json("/api/plots?sample=powder&correlation_transform=log_squared&correlation_family=roi_area&page_size=4&sort=peak_asc");
  assert.equal(filtered.response.status, 200);
  assert.ok(filtered.body.total > 0);
  assert.ok(filtered.body.items.every((item) => item.sample === "powder" && item.correlation_transform === "log_squared" && item.correlation_family === "roi_area"));
  const peaks = filtered.body.items.map((item) => item.anchor_peak_number).filter((value) => value !== null);
  assert.deepEqual(peaks, [...peaks].sort((a, b) => a - b));
});

test("Log waterfall API exposes only original-domain shading", async () => {
  const result = await json(
    "/api/plots?correlation_transform=log_squared&visualization_type=waterfall_shaded&page_size=100",
  );
  assert.equal(result.response.status, 200);
  assert.equal(result.body.total, 280);
  assert.ok(result.body.items.every((item) => item.display_profile_domain === "original_positive"));
});

test("image and companion routes serve only indexed files", async () => {
  const page = await json("/api/plots?q=window%200-5%20across%20frames&page_size=1");
  const item = page.body.items[0];
  const image = await fetch(`${base}${item.image_url}`);
  assert.equal(image.status, 200);
  assert.equal(image.headers.get("content-type"), "image/png");
  const signature = new Uint8Array((await image.arrayBuffer()).slice(0, 8));
  assert.deepEqual([...signature], [137, 80, 78, 71, 13, 10, 26, 10]);

  assert.ok(item.companions.length > 0);
  const companion = await fetch(`${base}${item.companions[0].url}`);
  assert.equal(companion.status, 200);
  assert.match(companion.headers.get("content-type") || "", /csv|json|gzip/);

  const arbitrary = await fetch(`${base}/api/file?path=${encodeURIComponent("/etc/passwd")}`);
  assert.equal(arbitrary.status, 404);
});

test("API remains read-only and sanitizes malformed pagination", async () => {
  const writeAttempt = await json("/api/plots", { method: "POST" });
  assert.equal(writeAttempt.response.status, 405);
  assert.match(writeAttempt.body.error, /read-only/i);

  const malformed = await json("/api/plots?page=not-a-number&page_size=not-a-number");
  assert.equal(malformed.response.status, 200);
  assert.equal(malformed.body.page, 1);
  assert.equal(malformed.body.page_size, 24);
});
