"""Cache management for expensive computations."""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


class CacheManager:
    """Global LRU cache for expensive computations, stored in project root."""

    def __init__(self, config, cache_dir=None):
        cache_cfg = config.get("cache", {})
        self.enabled = cache_cfg.get("enabled", True)
        self.max_frames = cache_cfg.get("max_frames", 100)
        # Cache in project root by default
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent.parent / ".cache"
        self.metadata_path = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        self.config = config

    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._default_metadata()
        return self._default_metadata()

    def _default_metadata(self):
        """Return default metadata structure."""
        return {"version": "1.0", "config_hash": None, "frames": {}, "access_order": []}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _enforce_lru(self):
        """Remove oldest frames if over max_frames limit."""
        while len(self.metadata["access_order"]) > self.max_frames:
            oldest_key = self.metadata["access_order"].pop(0)
            cache_file = self.cache_dir / f"{oldest_key}.npz"
            if cache_file.exists():
                cache_file.unlink()
            self.metadata["frames"].pop(oldest_key, None)

    def get_config_hash(self):
        """Hash detection-related config keys that affect caching."""
        keys = [
            "cellpose_flow_threshold",
            "cellpose_cellprob_threshold",
            "min_droplet_diameter",
            "max_droplet_diameter",
        ]
        data = {k: self.config.get(k) for k in keys}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def _get_cache_key(self, source_filename):
        """Generate cache key from source filename (not full path)."""
        name = Path(source_filename).stem
        return hashlib.sha256(name.encode()).hexdigest()[:16]

    def is_valid(self, source_filename):
        """Check if cache is valid for frame by source filename."""
        if not self.enabled:
            return False
        current_hash = self.get_config_hash()
        if self.metadata.get("config_hash") != current_hash:
            return False
        cache_key = self._get_cache_key(source_filename)
        cache_file = self.cache_dir / f"{cache_key}.npz"
        return cache_file.exists()

    def load_frame(self, source_filename):
        """Load cached data by source filename and update access order."""
        cache_key = self._get_cache_key(source_filename)
        cache_file = self.cache_dir / f"{cache_key}.npz"
        data = np.load(cache_file, allow_pickle=True)

        # Update LRU order
        if cache_key in self.metadata["access_order"]:
            self.metadata["access_order"].remove(cache_key)
        self.metadata["access_order"].append(cache_key)
        self._save_metadata()

        return {
            "min_projection": data["min_projection"],
            "droplet_coords": list(data["droplet_coords"]),
        }

    def save_frame(self, source_filename, min_proj, droplet_coords):
        """Save frame data by source filename and enforce LRU limit."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._get_cache_key(source_filename)
        cache_file = self.cache_dir / f"{cache_key}.npz"

        np.savez(
            cache_file,
            min_projection=min_proj,
            droplet_coords=np.array(droplet_coords, dtype=object),
        )

        # Update metadata
        self.metadata["config_hash"] = self.get_config_hash()
        self.metadata["frames"][cache_key] = {
            "source": str(source_filename),
            "cached_at": datetime.now().isoformat(),
        }

        # Update LRU order
        if cache_key in self.metadata["access_order"]:
            self.metadata["access_order"].remove(cache_key)
        self.metadata["access_order"].append(cache_key)

        self._enforce_lru()
        self._save_metadata()

    def clear(self):
        """Clear entire cache."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.metadata = self._default_metadata()
        print("Cache cleared.")
