import type { SavedLibrary } from "./types";

const STORAGE_KEY = "xrd-correlation-atlas.library.v1";
const EMPTY_LIBRARY: SavedLibrary = { version: 1, favorites: [], collections: {} };

export function loadSavedLibrary(): SavedLibrary {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as Partial<SavedLibrary> | null;
    if (!parsed || parsed.version !== 1) return EMPTY_LIBRARY;
    return {
      version: 1,
      favorites: Array.isArray(parsed.favorites) ? [...new Set(parsed.favorites)] : [],
      collections:
        parsed.collections && typeof parsed.collections === "object"
          ? Object.fromEntries(
              Object.entries(parsed.collections).map(([name, ids]) => [
                name,
                Array.isArray(ids) ? [...new Set(ids)] : [],
              ]),
            )
          : {},
    };
  } catch {
    return EMPTY_LIBRARY;
  }
}

export function persistSavedLibrary(library: SavedLibrary): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(library));
}

export function toggleFavorite(library: SavedLibrary, plotId: string): SavedLibrary {
  const favorites = new Set(library.favorites);
  favorites.has(plotId) ? favorites.delete(plotId) : favorites.add(plotId);
  return { ...library, favorites: [...favorites] };
}

export function addToCollection(library: SavedLibrary, name: string, plotId: string): SavedLibrary {
  const normalized = name.trim();
  if (!normalized) return library;
  const ids = new Set(library.collections[normalized] ?? []);
  ids.add(plotId);
  return { ...library, collections: { ...library.collections, [normalized]: [...ids] } };
}

export function removeFromCollection(library: SavedLibrary, name: string, plotId: string): SavedLibrary {
  const ids = new Set(library.collections[name] ?? []);
  ids.delete(plotId);
  return { ...library, collections: { ...library.collections, [name]: [...ids] } };
}
