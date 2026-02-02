"""Tests for the DropDrop pipeline."""

import pytest


class TestDropletInclusionPipeline:
    """Tests for the main pipeline."""

    def test_parse_filename_with_z_index(self):
        """Test filename parsing with z-stack index."""
        from dropdrop.pipeline import DropletInclusionPipeline

        pipeline = DropletInclusionPipeline(use_cache=False)
        z_idx, frame_idx = pipeline.parse_filename("image_z01_a01f05d4.tif")
        assert z_idx == 1
        assert frame_idx == 5

    def test_parse_filename_without_z_index(self):
        """Test filename parsing without z-stack index."""
        from dropdrop.pipeline import DropletInclusionPipeline

        pipeline = DropletInclusionPipeline(use_cache=False)
        z_idx, frame_idx = pipeline.parse_filename("image_a01f10d4.tif")
        assert z_idx == 0  # Default for non-z-stack
        assert frame_idx == 10


class TestCLI:
    """Tests for CLI helpers."""

    def test_parse_settings_full(self):
        """Test parsing full settings string."""
        from dropdrop.cli import parse_settings

        settings = parse_settings("d=1000,p=on,c=6.5e5,l=test_label")
        assert settings["dilution"] == 1000
        assert settings["poisson"] is True
        assert settings["count"] == 6.5e5
        assert settings["label"] == "test_label"

    def test_parse_settings_short_keys(self):
        """Test parsing with short key names."""
        from dropdrop.cli import parse_settings

        settings = parse_settings("d=500,p=off")
        assert settings["dilution"] == 500
        assert settings["poisson"] is False

    def test_parse_settings_empty(self):
        """Test parsing empty/None settings returns defaults."""
        from dropdrop.cli import parse_settings

        settings = parse_settings(None)
        assert settings["dilution"] == 500
        assert settings["poisson"] is True
        assert settings["count"] == 6.5e5
        assert settings["label"] is None

    def test_generate_project_name_with_label(self):
        """Test project name generation with label."""
        from dropdrop.cli import generate_project_name

        settings = {"label": "experiment1"}
        name = generate_project_name(settings)
        assert "experiment1" in name
        assert len(name.split("_")[0]) == 8  # YYYYMMDD

    def test_generate_project_name_without_label(self):
        """Test project name generation without label."""
        from dropdrop.cli import generate_project_name

        settings = {"label": None}
        name = generate_project_name(settings)
        assert "_" not in name
        assert len(name) == 8  # YYYYMMDD


class TestConfig:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        from dropdrop.config import load_config

        config = load_config()
        assert "cellpose_flow_threshold" in config
        assert "min_droplet_diameter" in config
        assert "px_to_um" in config


class TestCache:
    """Tests for cache manager."""

    def test_cache_key_generation(self):
        """Test cache key generation from filename."""
        from dropdrop.cache import CacheManager
        from dropdrop.config import load_config

        config = load_config()
        cache = CacheManager(config)
        key1 = cache._get_cache_key("image_a01f01d4.tif")
        key2 = cache._get_cache_key("image_a01f01d4.tif")
        key3 = cache._get_cache_key("image_a01f02d4.tif")

        assert key1 == key2  # Same filename = same key
        assert key1 != key3  # Different filename = different key
        assert len(key1) == 16  # SHA256[:16]
