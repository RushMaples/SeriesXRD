import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import path from 'node:path';

export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"') {
        if (text[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          quoted = false;
        }
      } else {
        field += character;
      }
      continue;
    }

    if (character === '"' && field.length === 0) {
      quoted = true;
    } else if (character === ',') {
      row.push(field);
      field = '';
    } else if (character === '\n') {
      row.push(field.replace(/\r$/, ''));
      rows.push(row);
      row = [];
      field = '';
    } else {
      field += character;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field.replace(/\r$/, ''));
    rows.push(row);
  }

  const nonemptyRows = rows.filter((cells) => cells.some((cell) => cell !== ''));
  if (nonemptyRows.length === 0) return [];
  const [headers, ...body] = nonemptyRows;
  return body.map((cells) => Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ''])));
}

export function readCsv(filePath) {
  const bytes = readFileSync(filePath);
  const text = filePath.endsWith('.gz') ? gunzipSync(bytes).toString('utf8') : bytes.toString('utf8');
  return parseCsv(text);
}

export function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

export function listFilesRecursive(root, predicate = () => true) {
  if (!existsSync(root)) return [];
  const output = [];
  const visit = (directory) => {
    for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(absolute);
      else if (entry.isFile() && predicate(absolute)) output.push(absolute);
    }
  };
  visit(root);
  return output;
}

export function sha256File(filePath) {
  return createHash('sha256').update(readFileSync(filePath)).digest('hex');
}

export function pngMetadata(filePath) {
  const header = readFileSync(filePath).subarray(0, 24);
  if (header.length < 24 || header.toString('ascii', 1, 4) !== 'PNG') {
    throw new Error(`Not a valid PNG: ${filePath}`);
  }
  const stats = statSync(filePath);
  return {
    width: header.readUInt32BE(16),
    height: header.readUInt32BE(20),
    size_bytes: stats.size,
    device: String(stats.dev),
    inode: String(stats.ino),
    hardlink_count: stats.nlink,
  };
}

export function median(numbers) {
  if (numbers.length === 0) return null;
  const ordered = [...numbers].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 1 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

export function numberOrNull(value) {
  if (value === '' || value === null || value === undefined) return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function posixRelative(root, filePath) {
  return path.relative(root, filePath).split(path.sep).join('/');
}

