#!/usr/bin/env node

import { existsSync, mkdirSync, renameSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  listFilesRecursive,
  median,
  numberOrNull,
  pngMetadata,
  posixRelative,
  readCsv,
  readJson,
  sha256File,
} from './lib/index-io.mjs';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const EXPLORER_ROOT = path.resolve(SCRIPT_DIR, '..');
const CORRELATIONS_ROOT = path.resolve(EXPLORER_ROOT, '..');
const RESULTS_ROOT = path.resolve(
  process.env.CORRELATION_RESULTS_ROOT || path.join(CORRELATIONS_ROOT, 'results'),
);
const DATA_ROOT = path.join(EXPLORER_ROOT, 'data');

const FORMAL_RUN_ID = 'uote_nonlinear_squared_qwidth075_comparison_20260803';
const FORMAL_ROOT = path.join(RESULTS_ROOT, FORMAL_RUN_ID);
const ORIGINAL_WATERFALL_RUN_ID = 'waterfall_log_correlation_on_original_profiles_qwidth075_20260804';
const ORIGINAL_WATERFALL_ROOT = path.join(FORMAL_ROOT, ORIGINAL_WATERFALL_RUN_ID);
const SINGLE_RUN_ID = 'uote_single_crystal_all_peak_log_squared_20260805';
const SINGLE_ROOT = path.join(RESULTS_ROOT, SINGLE_RUN_ID);
const SINGLE_ANALYSIS_ROOT = path.join(SINGLE_ROOT, 'single_crystal', 'all_peak_log_squared');
const SINGLE_WATERFALL_RUN_ID = 'single_crystal_all_peak_original_xy_waterfalls';
const SINGLE_WATERFALL_ROOT = path.join(SINGLE_ROOT, 'waterfall_original_xy');
const MODES = ['log_squared'];
const SAMPLES = ['powder', 'single_crystal'];
const FAMILIES = [
  'location',
  'roi_area',
  'window_to_window_across_frames',
  'window_to_window_within_same_frame',
];

const EXPECTED = {
  curated_total: 1942,
  formal_main_total: 1387,
  powder_original_profile_waterfalls: 280,
  original_profile_waterfalls: 555,
  per_transform: {
    powder: {
      location: 280,
      roi_area: 280,
      window_to_window_across_frames: 168,
      window_to_window_within_same_frame: 40,
    },
    single_crystal: {
      location: 275,
      roi_area: 275,
      window_to_window_across_frames: 57,
      window_to_window_within_same_frame: 12,
    },
  },
};

const COMPANION_EXTENSIONS = new Map([
  ['matrix_csv', '.csv'],
  ['peak_color_mapping_csv_gz', '.csv.gz'],
  ['collection_index_csv', '.csv'],
  ['validation_json', '.json'],
]);

const records = [];
const recordIds = new Set();
const assetsBySha = new Map();
const errors = [];
const warnings = [];

function invariant(condition, message) {
  if (!condition) errors.push(message);
  return condition;
}

function asAbsolute(filePath) {
  return path.resolve(filePath);
}

function relativeToCorrelations(filePath) {
  return posixRelative(CORRELATIONS_ROOT, filePath);
}

function under(parent, candidate) {
  const relative = path.relative(parent, candidate);
  return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function addCompanion(kind, filePath) {
  const expectedExtension = COMPANION_EXTENSIONS.get(kind);
  const absolute = asAbsolute(filePath);
  if (!expectedExtension) {
    errors.push(`Companion kind is not whitelisted: ${kind}`);
  }
  if (!absolute.endsWith(expectedExtension ?? '__invalid__')) {
    errors.push(`Companion extension mismatch for ${kind}: ${absolute}`);
  }
  if (!under(RESULTS_ROOT, absolute)) {
    errors.push(`Companion escapes results root: ${absolute}`);
  }
  if (!existsSync(absolute)) {
    errors.push(`Missing companion: ${absolute}`);
  }
  return {
    kind,
    path: absolute,
    path_relative: relativeToCorrelations(absolute),
    size_bytes: existsSync(absolute) ? statSync(absolute).size : null,
  };
}

function addAsset(imagePath, hints = {}) {
  const absolute = asAbsolute(imagePath);
  if (!existsSync(absolute)) {
    errors.push(`Missing image: ${absolute}`);
    return null;
  }
  if (!under(RESULTS_ROOT, absolute)) {
    errors.push(`Image escapes results root: ${absolute}`);
  }
  const metadata = pngMetadata(absolute);
  const sha256 = hints.sha256 || sha256File(absolute);
  const existing = assetsBySha.get(sha256);
  if (existing) {
    if (existing.size_bytes !== metadata.size_bytes || existing.width !== metadata.width || existing.height !== metadata.height) {
      errors.push(`SHA-256 collision or inconsistent metadata for ${sha256}: ${absolute}`);
    }
    existing.aliases.add(absolute);
    existing.alias_paths_relative.add(relativeToCorrelations(absolute));
    existing.inode_keys.add(`${metadata.device}:${metadata.inode}`);
    existing.maximum_hardlink_count = Math.max(existing.maximum_hardlink_count, metadata.hardlink_count);
    return existing.id;
  }

  const asset = {
    id: `sha256:${sha256}`,
    sha256,
    mime_type: 'image/png',
    size_bytes: metadata.size_bytes,
    width: metadata.width,
    height: metadata.height,
    hash_source: hints.hash_source || 'computed',
    aliases: new Set([absolute]),
    alias_paths_relative: new Set([relativeToCorrelations(absolute)]),
    inode_keys: new Set([`${metadata.device}:${metadata.inode}`]),
    maximum_hardlink_count: metadata.hardlink_count,
  };
  assetsBySha.set(sha256, asset);
  return asset.id;
}

function addRecord(record) {
  if (recordIds.has(record.id)) {
    errors.push(`Duplicate logical plot id: ${record.id}`);
    return;
  }
  recordIds.add(record.id);
  records.push(record);
}

function transformLabel(mode) {
  return mode === 'log_squared' ? 'Log-squared' : 'Unsupported transform';
}

function sampleLabel(sample) {
  return sample === 'single_crystal' ? 'Single crystal' : sample === 'powder' ? 'Powder' : 'Powder + single crystal';
}

function channelLabel(channel) {
  return channel === 'fit_control' ? 'fit control' : channel === 'spots' ? 'spots' : channel;
}

function makeTags(values) {
  return [...new Set(values.filter((value) => value !== null && value !== undefined && value !== '').map(String))];
}

function finaliseAssets() {
  return [...assetsBySha.values()]
    .map((asset) => {
      const aliases = [...asset.aliases].sort();
      const relativeAliases = [...asset.alias_paths_relative].sort();
      return {
        id: asset.id,
        sha256: asset.sha256,
        mime_type: asset.mime_type,
        size_bytes: asset.size_bytes,
        width: asset.width,
        height: asset.height,
        hash_source: asset.hash_source,
        canonical_path: aliases[0],
        canonical_path_relative: relativeAliases[0],
        aliases,
        alias_paths_relative: relativeAliases,
        inode_keys: [...asset.inode_keys].sort(),
        maximum_hardlink_count: asset.maximum_hardlink_count,
      };
    })
    .sort((a, b) => a.id.localeCompare(b.id));
}

function loadFormalManifest(mode) {
  const manifestPath = path.join(FORMAL_ROOT, mode, 'SOURCE_MANIFEST.csv');
  const rows = readCsv(manifestPath);
  return {
    rows,
    byDestination: new Map(rows.map((row) => [row.destination_relative, row])),
  };
}

function loadPowderMetadata(mode) {
  const root = path.join(FORMAL_ROOT, '_sources', mode, 'powder_roi');
  const anchors = readCsv(path.join(root, 'anchor_map_index.csv'));
  const registry = readCsv(path.join(root, 'point_registry.csv'));
  const observations = readCsv(path.join(root, 'observation_spots_absolute_profile_audit.csv'));
  const byPointUid = new Map(registry.map((row) => [row.point_uid, row]));
  const anchorByIndex = new Map(anchors.map((row) => [Number(row.anchor_index_0based), row]));
  const anchorByPointUid = new Map(anchors.map((row) => [row.point_uid, row]));
  const qWidths = new Map();
  for (const observation of observations) {
    const values = qWidths.get(observation.point_uid) || [];
    const value = numberOrNull(observation.q_width);
    if (value !== null) values.push(value);
    qWidths.set(observation.point_uid, values);
  }
  return { anchorByIndex, anchorByPointUid, byPointUid, qWidths };
}

function parsePowderAnchorToken(token, mode, metadata) {
  const match = token.match(/^anchor_(\d{3})_P(.+?)_peak(\d{2})_([TU]\d+_P.+)$/);
  if (!match) {
    warnings.push(`Could not parse powder anchor token: ${token}`);
    return null;
  }
  const sequence = Number(match[1]);
  const filenameLocalPeak = Number(match[3]);
  const filenamePointUid = match[4];
  const anchorRow = metadata.anchorByIndex.get(sequence) || metadata.anchorByPointUid.get(filenamePointUid);
  const registryRow = metadata.byPointUid.get(filenamePointUid);
  if (!anchorRow) warnings.push(`Missing anchor_map_index row for ${mode}/${token}`);
  if (!registryRow) warnings.push(`Missing point_registry row for ${mode}/${token}`);
  if (anchorRow && (anchorRow.point_uid !== filenamePointUid || Number(anchorRow.local_peak_index) !== filenameLocalPeak)) {
    errors.push(`Anchor filename/registry mismatch for ${mode}/${token}`);
  }
  const qWidthValues = metadata.qWidths.get(filenamePointUid) || [];
  return {
    token,
    sequence_0based: sequence,
    point_uid: filenamePointUid,
    pressure_gpa: numberOrNull(anchorRow?.pressure_gpa),
    local_peak_index: filenameLocalPeak,
    q: numberOrNull(anchorRow?.q ?? registryRow?.q),
    two_theta_deg: numberOrNull(anchorRow?.two_theta_deg ?? registryRow?.two_theta_deg),
    source_table: registryRow?.source_table || null,
    source_track: numberOrNull(registryRow?.track),
    tracked_source_point: filenamePointUid.startsWith('T'),
    n_observations: numberOrNull(anchorRow?.n_observations ?? registryRow?.n_observations),
    distinct_frames: numberOrNull(anchorRow?.distinct_frames ?? registryRow?.distinct_frames),
    support_component_count: numberOrNull(registryRow?.support_component_count),
    support_q_min: numberOrNull(registryRow?.support_q_min),
    support_q_max: numberOrNull(registryRow?.support_q_max),
    observation_q_width_summary: qWidthValues.length
      ? {
          count: qWidthValues.length,
          min: Math.min(...qWidthValues),
          median: median(qWidthValues),
          max: Math.max(...qWidthValues),
          note: 'q_width is observation-level; this summary must not be treated as one exact anchor width.',
        }
      : null,
  };
}

function singlePeakObject(row) {
  return {
    id: row.peak_id,
    frame: Number(row.frame),
    local_peak_index: Number(row.local_peak_index),
    pressure_gpa: numberOrNull(row.pressure_GPa),
    two_theta_deg: numberOrNull(row.two_theta_deg),
    q: numberOrNull(row['q_A^-1']),
    azimuth_deg: numberOrNull(row.azim_deg),
    source_track_provenance_only: numberOrNull(row.track),
    observation_row: numberOrNull(row.obs_row),
    transformed_roi_area: numberOrNull(row.integrated_area),
    area_unit: row.area_unit,
  };
}

function mainCompanion(category, categoryRoot, imagePath) {
  const relative = posixRelative(categoryRoot, imagePath);
  const parts = relative.split('/');
  if (category === 'roi_area' || category === 'location') {
    invariant(parts[0] === 'heatmaps', `Unexpected ${category} image path: ${relative}`);
    return path.join(categoryRoot, 'matrices', path.basename(imagePath, '.png') + '.csv');
  }
  if (category === 'window_to_window_across_frames') {
    invariant(parts.length === 4 && parts[2] === 'heatmaps', `Unexpected across-frame image path: ${relative}`);
    return path.join(categoryRoot, parts[0], parts[1], 'matrices', path.basename(imagePath, '.png') + '.csv');
  }
  if (category === 'window_to_window_within_same_frame') {
    if (parts.length === 3 && parts[1] === 'aggregate' && parts[2] === 'heatmap.png') {
      return path.join(categoryRoot, parts[0], 'aggregate', 'matrix.csv');
    }
    invariant(parts.length === 4 && parts[1] === 'by_pressure' && parts[2] === 'heatmaps', `Unexpected within-frame image path: ${relative}`);
    return path.join(categoryRoot, parts[0], 'by_pressure', 'matrices', path.basename(imagePath, '.png') + '.csv');
  }
  throw new Error(`Unsupported category: ${category}`);
}

function classifyWindow(category, sample, categoryRoot, imagePath) {
  const relative = posixRelative(categoryRoot, imagePath);
  const parts = relative.split('/');
  const windowCount = sample === 'powder' ? 28 : 19;
  const lastEnd = sample === 'powder' ? 32 : 23;
  if (category === 'window_to_window_across_frames') {
    const match = path.basename(imagePath, '.png').match(/^window_(\d{2})_(\d+)_(\d+)$/);
    invariant(Boolean(match), `Could not parse window filename: ${relative}`);
    return {
      channel: parts[0],
      method: parts[1],
      scope: 'across_frames',
      pressure_gpa: null,
      window: match
        ? {
            index_0based: Number(match[1]),
            label: `${Number(match[2])}–${Number(match[3])}°`,
            start_deg: Number(match[2]),
            end_deg: Number(match[3]),
            width_deg: Number(match[3]) - Number(match[2]),
          }
        : null,
    };
  }
  const aggregate = parts[1] === 'aggregate';
  const pressureMatch = aggregate ? null : path.basename(imagePath, '.png').match(/^(.+)GPa$/);
  invariant(aggregate || Boolean(pressureMatch), `Could not parse within-frame pressure: ${relative}`);
  return {
    channel: parts[0],
    method: 'acf',
    scope: aggregate ? 'aggregate' : 'by_pressure',
    pressure_gpa: pressureMatch ? numberOrNull(pressureMatch[1]) : null,
    window: {
      kind: 'sliding_window_set',
      count: windowCount,
      first_label: '0–5°',
      last_label: `${lastEnd - 5}–${lastEnd}°`,
      width_deg: 5,
      step_deg: 1,
    },
  };
}

function buildFormalMain() {
  for (const mode of MODES) {
    const manifest = loadFormalManifest(mode);
    const powderMetadata = loadPowderMetadata(mode);
    const modeRoot = path.join(FORMAL_ROOT, mode);
    for (const sample of SAMPLES) {
      for (const category of FAMILIES) {
        // The former 75-track single-crystal peak maps are superseded by the
        // independent 275-peak run indexed in buildSingleAllPeak().
        if (sample === 'single_crystal' && (category === 'roi_area' || category === 'location')) continue;
        const categoryRoot = path.join(modeRoot, sample, category);
        const images = listFilesRecursive(categoryRoot, (filePath) => filePath.endsWith('.png'));
        for (const imagePath of images) {
          const destinationRelative = posixRelative(modeRoot, imagePath);
          const manifestRow = manifest.byDestination.get(destinationRelative);
          if (!manifestRow) warnings.push(`SOURCE_MANIFEST missing image row: ${mode}/${destinationRelative}`);
          const assetId = addAsset(imagePath, {
            sha256: manifestRow?.sha256,
            hash_source: manifestRow ? 'source_manifest' : 'computed_fallback',
          });
          const companion = mainCompanion(category, categoryRoot, imagePath);
          const relativeImage = relativeToCorrelations(imagePath);
          const base = path.basename(imagePath, '.png');
          let anchor = null;
          let track = null;
          let windowMetadata = null;
          let channel = null;
          let method = null;
          let scope = null;
          let pressureGpa = null;
          let title;
          let entityType;
          let matrixSemantics;

          if ((category === 'roi_area' || category === 'location') && sample === 'powder') {
            anchor = parsePowderAnchorToken(base, mode, powderMetadata);
            pressureGpa = anchor?.pressure_gpa ?? null;
            entityType = 'powder_pressure_peak_anchor';
            matrixSemantics = 'anchor_to_all_other_pressures; anchor-pressure row and absent peak slots are blank; explicit 0 is computed zero';
            title = `${sampleLabel(sample)} · ${transformLabel(mode)} · ${category === 'roi_area' ? 'ROI area' : 'Location'} · ${pressureGpa} GPa peak ${anchor?.local_peak_index}`;
          } else {
            const classified = classifyWindow(category, sample, categoryRoot, imagePath);
            ({ channel, method, scope, pressure_gpa: pressureGpa, window: windowMetadata } = classified);
            entityType = category === 'window_to_window_across_frames' ? 'angle_window_across_pressures' : 'window_set_within_pressure';
            matrixSemantics = category === 'window_to_window_across_frames'
              ? 'strict lower pressure-pair triangle; diagonal and mirrored upper triangle are intentionally blank'
              : 'strict lower window-pair triangle; diagonal and mirrored upper triangle are intentionally blank';
            const detail = category === 'window_to_window_across_frames'
              ? `${windowMetadata?.label} · ${method}`
              : `${scope === 'aggregate' ? 'aggregate' : `${pressureGpa} GPa`} · ACF`;
            title = `${sampleLabel(sample)} · ${transformLabel(mode)} · ${channelLabel(channel)} · ${category === 'window_to_window_across_frames' ? 'Across frames' : 'Within frame'} · ${detail}`;
          }

          addRecord({
            id: `formal:${mode}:${destinationRelative.replace(/\.png$/, '')}`,
            title,
            asset_id: assetId,
            image_path: imagePath,
            image_path_relative: relativeImage,
            companion_paths: [addCompanion('matrix_csv', companion)],
            run_id: FORMAL_RUN_ID,
            result_status: 'current_formal',
            validation_status: 'PASS',
            sample,
            correlation_transform: mode,
            metric_transform_dependency: category !== 'location',
            correlation_family: category,
            visualization_type: 'heatmap',
            display_profile_domain: 'not_applicable',
            entity_type: entityType,
            anchor,
            track,
            window: windowMetadata,
            channel,
            method,
            scope,
            pressure_gpa: pressureGpa,
            half_width_factor: category === 'roi_area' && sample === 'powder' ? 0.75 : null,
            triangle_policy: category.startsWith('window_') ? 'strict_lower_no_diagonal' : 'not_applicable',
            matrix_semantics: matrixSemantics,
            tags: makeTags([
              sample,
              mode,
              category,
              entityType,
              channel,
              method,
              scope,
              pressureGpa !== null ? `${pressureGpa} GPa` : null,
              anchor?.point_uid,
              anchor ? `peak ${anchor.local_peak_index}` : null,
              track?.id,
              windowMetadata?.label,
              category === 'roi_area' && sample === 'powder' ? 'qwidth 0.75' : null,
            ]),
            classification_warnings: [],
          });
        }
      }
    }
  }
}

function loadSingleAllPeakContext() {
  const peaks = readCsv(path.join(SINGLE_ANALYSIS_ROOT, 'peak_registry.csv'));
  const anchors = readCsv(path.join(SINGLE_ANALYSIS_ROOT, 'per_anchor_peak_map_index.csv'));
  invariant(peaks.length === 275, `Single-crystal registry: expected 275 peaks, found ${peaks.length}`);
  invariant(anchors.length === 275, `Single-crystal index: expected 275 anchors, found ${anchors.length}`);
  const byFrameSlot = new Map(
    peaks.map((row) => [`${Number(row.frame)}:${Number(row.local_peak_index)}`, row]),
  );
  const byPeakId = new Map(peaks.map((row) => [row.peak_id, row]));
  return { peaks, anchors, byFrameSlot, byPeakId };
}

function buildSingleAllPeak(context) {
  for (const family of ['location', 'roi_area']) {
    const pngField = family === 'location' ? 'location_png' : 'area_png';
    const csvField = family === 'location' ? 'location_csv' : 'area_csv';
    for (const anchorRow of context.anchors) {
      const key = `${Number(anchorRow.anchor_frame)}:${Number(anchorRow.anchor_local_peak)}`;
      const peakRow = context.byFrameSlot.get(key);
      invariant(Boolean(peakRow), `Missing single-crystal peak metadata for ${key}`);
      const anchor = peakRow ? singlePeakObject(peakRow) : null;
      const imagePath = path.join(SINGLE_ANALYSIS_ROOT, anchorRow[pngField]);
      const matrixPath = path.join(SINGLE_ANALYSIS_ROOT, anchorRow[csvField]);
      const assetId = addAsset(imagePath);
      addRecord({
        id: `single-all-peak:log_squared:${family}:${anchorRow.anchor_peak_id}`,
        title: `Single crystal · Log-squared · ${family === 'roi_area' ? 'ROI area' : 'Location'} · ${anchor?.pressure_gpa} GPa frame ${anchor?.frame} peak ${anchor?.local_peak_index}`,
        asset_id: assetId,
        image_path: imagePath,
        image_path_relative: relativeToCorrelations(imagePath),
        companion_paths: [addCompanion('matrix_csv', matrixPath)],
        run_id: SINGLE_RUN_ID,
        result_status: 'current_formal',
        validation_status: 'PASS',
        sample: 'single_crystal',
        correlation_transform: 'log_squared',
        metric_transform_dependency: family !== 'location',
        correlation_family: family,
        visualization_type: 'heatmap',
        display_profile_domain: 'not_applicable',
        entity_type: 'single_crystal_frame_local_peak_anchor',
        anchor,
        track: null,
        window: null,
        channel: 'spots',
        method: family === 'roi_area' ? 'min_max_log_squared_roi_area' : 'radial_location_tolerance',
        scope: 'all_other_frames_all_local_peaks',
        pressure_gpa: anchor?.pressure_gpa ?? null,
        half_width_factor: null,
        triangle_policy: 'anchor_frame_row_blank',
        matrix_semantics: '12 registered frame rows × 35 frame-local peak slots; every other-frame peak is scored; absent slots and the anchor frame are blank; source track is provenance only.',
        tags: makeTags([
          'single_crystal',
          'log_squared',
          family,
          'all peaks',
          `${anchor?.pressure_gpa} GPa`,
          `frame ${anchor?.frame}`,
          `peak ${anchor?.local_peak_index}`,
          anchor?.id,
        ]),
        classification_warnings: [],
      });
    }
  }
}

function buildSingleOriginalWaterfalls(context) {
  const indexPath = path.join(SINGLE_WATERFALL_ROOT, 'WATERFALL_INDEX.csv');
  const validationPath = path.join(SINGLE_WATERFALL_ROOT, 'SUITE_VALIDATION.json');
  const rows = readCsv(indexPath);
  invariant(rows.length === 275, `Single-crystal waterfalls: expected 275, found ${rows.length}`);
  for (const row of rows) {
    const peakRow = context.byPeakId.get(row.anchor_peak_id);
    invariant(Boolean(peakRow), `Missing waterfall peak metadata for ${row.anchor_peak_id}`);
    const anchor = peakRow ? singlePeakObject(peakRow) : null;
    const imagePath = path.join(SINGLE_WATERFALL_ROOT, row.png);
    addRecord({
      id: `waterfall:${SINGLE_WATERFALL_RUN_ID}:log_squared:${row.anchor_peak_id}`,
      title: `Single crystal · Log-squared ROI waterfall · ${anchor?.pressure_gpa} GPa frame ${anchor?.frame} peak ${anchor?.local_peak_index} · original XY`,
      asset_id: addAsset(imagePath),
      image_path: imagePath,
      image_path_relative: relativeToCorrelations(imagePath),
      companion_paths: [
        addCompanion('matrix_csv', row.source_area_matrix),
        addCompanion('collection_index_csv', indexPath),
        addCompanion('validation_json', validationPath),
      ],
      run_id: SINGLE_WATERFALL_RUN_ID,
      parent_run_id: SINGLE_RUN_ID,
      result_status: 'current_formal',
      validation_status: 'PASS',
      sample: 'single_crystal',
      correlation_transform: 'log_squared',
      metric_transform_dependency: true,
      correlation_family: 'roi_area',
      visualization_type: 'waterfall_shaded',
      display_profile_domain: 'original_positive',
      entity_type: 'single_crystal_frame_local_peak_anchor',
      anchor,
      track: null,
      window: null,
      channel: 'spots',
      method: 'min_max_log_squared_roi_area',
      scope: 'all_registered_frames',
      pressure_gpa: anchor?.pressure_gpa ?? null,
      half_width_factor: null,
      trace_source: 'original_spot_masked_xy',
      display_profile_source: 'original_positive_spot_masked_xy_per_tiff_exposure',
      display_profile_construction: 'The source masked XY intensity is positive-clipped and divided by TIFF exposure. One shared 99.5% display cap sets waterfall height; no nonlinear transform changes the displayed curve.',
      waterfall: {
        cross_frame_colored_cells: numberOrNull(row.joined_cross_frame_peak_cells),
        pressure_rows: 12,
      },
      matrix_semantics: 'Colors come from the Log-squared all-peak ROI-area matrix; curve height comes from the original spot-masked XY file. Every projected peak also has a lossless support ribbon.',
      tags: makeTags([
        'single_crystal',
        'log_squared',
        'roi_area',
        'waterfall',
        'original xy',
        'all peaks',
        `${anchor?.pressure_gpa} GPa`,
        `peak ${anchor?.local_peak_index}`,
      ]),
      classification_warnings: [],
    });
  }
}

function readWaterfallCounts(suiteRoot, mode) {
  const filePath = path.join(suiteRoot, 'powder', mode, 'WATERFALL_INDEX.csv');
  return {
    filePath,
    byAnchor: new Map(readCsv(filePath).map((row) => [row.anchor_token, row])),
  };
}

function buildWaterfallSuite({ suiteRoot, suiteRunId, expected, displayProfileDomain, allowedModes = null }) {
  const masterPath = path.join(suiteRoot, 'MASTER_WATERFALL_INDEX.csv');
  const masterRows = readCsv(masterPath);
  const rows = allowedModes
    ? masterRows.filter((row) => allowedModes.includes(row.mode))
    : masterRows;
  invariant(rows.length === expected, `${suiteRunId}: expected ${expected} master rows, found ${rows.length}`);
  const modes = [...new Set(rows.map((row) => row.mode))];
  const modeContext = new Map();
  for (const mode of modes) {
    modeContext.set(mode, {
      powder: loadPowderMetadata(mode),
      counts: readWaterfallCounts(suiteRoot, mode),
      mappingPath: path.join(suiteRoot, 'powder', mode, 'PEAK_COLOR_MAPPING.csv.gz'),
      suiteValidationPath: path.join(suiteRoot, 'powder', mode, 'SUITE_VALIDATION.json'),
    });
  }

  for (const row of rows) {
    const context = modeContext.get(row.mode);
    const imagePath = asAbsolute(row.waterfall_png);
    invariant(under(suiteRoot, imagePath), `Waterfall image outside suite root: ${imagePath}`);
    const anchor = parsePowderAnchorToken(row.anchor, row.mode, context.powder);
    const countRow = context.counts.byAnchor.get(row.anchor);
    if (!countRow) warnings.push(`Missing WATERFALL_INDEX row for ${suiteRunId}/${row.anchor}`);
    const assetId = addAsset(imagePath, { sha256: row.png_sha256, hash_source: 'master_waterfall_index' });
    const companions = [
      addCompanion('matrix_csv', row.source_matrix),
      addCompanion('peak_color_mapping_csv_gz', context.mappingPath),
      addCompanion('collection_index_csv', context.counts.filePath),
      addCompanion('validation_json', context.suiteValidationPath),
    ];
    addRecord({
      id: `waterfall:${suiteRunId}:${row.mode}:${row.anchor}`,
      title: `${sampleLabel('powder')} · ${transformLabel(row.mode)} ROI waterfall · ${anchor?.pressure_gpa} GPa peak ${anchor?.local_peak_index} · ${displayProfileDomain === 'original_positive' ? 'pre-denoise XY-derived profile' : 'transformed profile'}`,
      asset_id: assetId,
      image_path: imagePath,
      image_path_relative: relativeToCorrelations(imagePath),
      companion_paths: companions,
      run_id: suiteRunId,
      parent_run_id: FORMAL_RUN_ID,
      result_status: 'current_formal',
      validation_status: row.validation_status || 'PASS',
      sample: 'powder',
      correlation_transform: row.mode,
      metric_transform_dependency: true,
      correlation_family: 'roi_area',
      visualization_type: 'waterfall_shaded',
      display_profile_domain: displayProfileDomain,
      entity_type: 'powder_pressure_peak_anchor',
      anchor,
      track: null,
      window: null,
      channel: 'spots',
      method: 'integrated_iou',
      scope: 'all_pressures',
      pressure_gpa: anchor?.pressure_gpa ?? null,
      half_width_factor: 0.75,
      trace_source: 'formal_composite',
      display_profile_source: displayProfileDomain === 'original_positive'
        ? 'source_spots_channel_xy_pre_nonlinear_transform'
        : 'formal_correlation_transform_profile',
      display_profile_construction: displayProfileDomain === 'original_positive'
        ? 'Source XY signal within the 519 formal observation supports; positive-clipped and measurement-normalized, summed within frame, averaged across distinct frames per peak, then summed across the 12–22 formal peaks at each pressure. No Log nonlinear transform is applied to the displayed curve.'
        : 'The exact nonlinear-transformed formal observation profiles used by ROI correlation, aggregated into one pressure-level composite.',
      waterfall: {
        cross_pressure_colored_cells: numberOrNull(countRow?.cross_pressure_colored_cells),
        positive_cross_pressure_cells: numberOrNull(countRow?.positive_cross_pressure_cells),
        zero_cross_pressure_cells: numberOrNull(countRow?.zero_cross_pressure_cells),
        mapping_rows: numberOrNull(countRow?.mapping_rows),
        pressure_rows: 19,
      },
      matrix_semantics: displayProfileDomain === 'original_positive'
        ? 'ROI colors come from the Log-squared source matrix; curve height comes from the pre-denoise XY-derived original-positive pressure composite.'
        : 'ROI values come from the source matrix; display profile domain is the correlation-transform pressure composite.',
      tags: makeTags([
        'powder',
        row.mode,
        'roi_area',
        'waterfall',
        displayProfileDomain,
        displayProfileDomain === 'original_positive' ? 'original xy pre-denoise' : 'denoised transformed profile',
        `${anchor?.pressure_gpa} GPa`,
        `peak ${anchor?.local_peak_index}`,
        anchor?.point_uid,
        'qwidth 0.75',
      ]),
      classification_warnings: [],
    });
  }
}

function nestedCounts(items, fields) {
  const output = {};
  for (const item of items) {
    let cursor = output;
    fields.forEach((field, index) => {
      const key = String(item[field] ?? 'null');
      if (index === fields.length - 1) cursor[key] = (cursor[key] || 0) + 1;
      else cursor = cursor[key] ||= {};
    });
  }
  return output;
}

function countFormalMain(mode, sample, family) {
  return records.filter(
    (record) => record.visualization_type === 'heatmap'
      && record.correlation_transform === mode
      && record.sample === sample
      && record.correlation_family === family
  ).length;
}

function buildAudit(assets) {
  const formalMain = records.filter((record) => record.visualization_type === 'heatmap').length;
  const originalWaterfalls = records.filter(
    (record) => record.visualization_type === 'waterfall_shaded'
      && record.display_profile_domain === 'original_positive',
  ).length;
  const exactCountChecks = {};
  for (const mode of MODES) {
    for (const sample of SAMPLES) {
      for (const family of FAMILIES) {
        const key = `${mode}/${sample}/${family}`;
        const expected = EXPECTED.per_transform[sample][family];
        const actual = countFormalMain(mode, sample, family);
        exactCountChecks[key] = { expected, actual, pass: actual === expected };
      }
    }
  }
  const imagePaths = records.map((record) => record.image_path);
  const sharedAssets = assets.filter((asset) => asset.aliases.length > 1);
  const sharedHardlinkAssets = sharedAssets.filter((asset) => asset.inode_keys.length < asset.aliases.length);
  const sharedCopyAssets = sharedAssets.filter((asset) => asset.inode_keys.length === asset.aliases.length);
  const recordsByAsset = new Map();
  for (const record of records) {
    const values = recordsByAsset.get(record.asset_id) || [];
    values.push(record.id);
    recordsByAsset.set(record.asset_id, values);
  }
  const sharedSemanticAssets = [...recordsByAsset.entries()]
    .filter(([, ids]) => ids.length > 1)
    .map(([assetId, ids]) => ({ asset_id: assetId, logical_record_count: ids.length, record_ids: ids }))
    .sort((a, b) => b.logical_record_count - a.logical_record_count || a.asset_id.localeCompare(b.asset_id));
  const checks = {
    curated_total: { expected: EXPECTED.curated_total, actual: records.length, pass: records.length === EXPECTED.curated_total },
    formal_main_total: { expected: EXPECTED.formal_main_total, actual: formalMain, pass: formalMain === EXPECTED.formal_main_total },
    original_profile_waterfalls: { expected: EXPECTED.original_profile_waterfalls, actual: originalWaterfalls, pass: originalWaterfalls === EXPECTED.original_profile_waterfalls },
    log_squared_only: {
      pass: records.every((record) => record.correlation_transform === 'log_squared'),
    },
    exact_formal_breakdown: { pass: Object.values(exactCountChecks).every((check) => check.pass), groups: exactCountChecks },
    unique_record_ids: { pass: recordIds.size === records.length, count: recordIds.size },
    unique_image_paths: { pass: new Set(imagePaths).size === imagePaths.length, count: new Set(imagePaths).size },
    images_exist: { pass: records.every((record) => existsSync(record.image_path)) },
    companions_exist: { pass: records.every((record) => record.companion_paths.every((companion) => existsSync(companion.path))) },
    companion_kinds_and_extensions_whitelisted: {
      pass: records.every((record) => record.companion_paths.every((companion) => {
        const expectedExtension = COMPANION_EXTENSIONS.get(companion.kind);
        return Boolean(expectedExtension) && companion.path.endsWith(expectedExtension);
      })),
    },
    all_validation_statuses_pass: { pass: records.every((record) => record.validation_status === 'PASS') },
    no_source_images_in_gallery: { pass: records.every((record) => !record.image_path_relative.includes('/_sources/')) },
    all_gallery_images_under_allowlist: {
      pass: records.every((record) =>
        under(path.join(FORMAL_ROOT, 'log_squared'), record.image_path)
        || under(ORIGINAL_WATERFALL_ROOT, record.image_path)
        || under(SINGLE_ROOT, record.image_path)),
    },
  };
  const allChecksPass = Object.values(checks).every((check) => check.pass);
  return {
    schema_version: '1.1.0',
    generated_at: new Date().toISOString(),
    status: allChecksPass && errors.length === 0 ? 'PASS' : 'FAIL',
    checks,
    summary: {
      logical_plot_records: records.length,
      unique_content_assets: assets.length,
      logical_records_sharing_an_asset: [...recordsByAsset.values()].filter((ids) => ids.length > 1).reduce((sum, ids) => sum + ids.length, 0),
      shared_content_asset_groups: sharedSemanticAssets.length,
      shared_hardlink_asset_groups: sharedHardlinkAssets.length,
      byte_identical_separate_file_asset_groups: sharedCopyAssets.length,
      classification_warning_count: warnings.length,
      error_count: errors.length,
    },
    duplicate_policy: {
      rule: 'Never collapse semantic PlotRecords by SHA-256. Assets are content-addressed; records retain transform/sample/family facets.',
      shared_semantic_asset_examples: sharedSemanticAssets.slice(0, 25),
    },
    counts_by_run_sample_transform_family: nestedCounts(records, ['run_id', 'sample', 'correlation_transform', 'correlation_family']),
    errors,
    warnings,
  };
}

function writeJsonAtomic(filePath, value) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.tmp-${process.pid}`;
  writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  renameSync(temporary, filePath);
}

function main() {
  for (const requiredRoot of [FORMAL_ROOT, ORIGINAL_WATERFALL_ROOT, SINGLE_ANALYSIS_ROOT, SINGLE_WATERFALL_ROOT]) {
    if (!existsSync(requiredRoot)) throw new Error(`Required result root does not exist: ${requiredRoot}`);
  }
  buildFormalMain();
  const singleContext = loadSingleAllPeakContext();
  buildSingleAllPeak(singleContext);
  buildWaterfallSuite({
    suiteRoot: ORIGINAL_WATERFALL_ROOT,
    suiteRunId: ORIGINAL_WATERFALL_RUN_ID,
    expected: EXPECTED.powder_original_profile_waterfalls,
    displayProfileDomain: 'original_positive',
  });
  buildSingleOriginalWaterfalls(singleContext);

  records.sort((a, b) => a.id.localeCompare(b.id));
  const assets = finaliseAssets();
  const audit = buildAudit(assets);
  const index = {
    schema_version: '1.1.0',
    generated_at: audit.generated_at,
    correlations_root: CORRELATIONS_ROOT,
    results_root: RESULTS_ROOT,
    result_roots: {
      current_formal: FORMAL_ROOT,
      original_profile_waterfalls: ORIGINAL_WATERFALL_ROOT,
      single_crystal_all_peak: SINGLE_ROOT,
    },
    scope: {
      policy: 'curated_allowlist',
      excluded_from_gallery: [
        'results/**/_sources/**',
        'validation files as standalone plots',
        'intermediate quicklooks',
      ],
      note: 'Only Log-squared correlation results are indexed. Waterfalls show Log-squared correlation colors on original-positive XY-derived profiles; obsolete transformed-profile and 75-track single-crystal plots are excluded.',
    },
    summary: {
      plot_records: records.length,
      assets: assets.length,
      audit_status: audit.status,
      counts_by_result_status: nestedCounts(records, ['result_status']),
      counts_by_sample: nestedCounts(records, ['sample']),
      counts_by_transform: nestedCounts(records, ['correlation_transform']),
      counts_by_family: nestedCounts(records, ['correlation_family']),
      counts_by_visualization: nestedCounts(records, ['visualization_type']),
    },
    records,
    assets,
  };

  writeJsonAtomic(path.join(DATA_ROOT, 'plot-index.json'), index);
  writeJsonAtomic(path.join(DATA_ROOT, 'classification-audit.json'), audit);
  console.log(JSON.stringify({
    status: audit.status,
    plot_records: records.length,
    assets: assets.length,
    errors: errors.length,
    warnings: warnings.length,
    index: path.join(DATA_ROOT, 'plot-index.json'),
    audit: path.join(DATA_ROOT, 'classification-audit.json'),
  }, null, 2));
  if (audit.status !== 'PASS') process.exitCode = 1;
}

main();
