"""
Tests for configuration presets.

Tests preset loading, validation, and CLI integration.
"""

import pytest

from src.satellite_control.config.presets import (
    ConfigPreset,
    get_preset_description,
    list_presets,
    load_preset,
)


class TestConfigPreset:
    """Tests for ConfigPreset class."""

    def test_preset_names(self):
        """All preset names should be defined."""
        assert ConfigPreset.FAST == "fast"
        assert ConfigPreset.BALANCED == "balanced"
        assert ConfigPreset.STABLE == "stable"
        assert ConfigPreset.PRECISION == "precision"

    def test_all_presets(self):
        """all() should return all preset names."""
        presets = ConfigPreset.all()
        assert len(presets) == 4
        assert ConfigPreset.FAST in presets
        assert ConfigPreset.BALANCED in presets
        assert ConfigPreset.STABLE in presets
        assert ConfigPreset.PRECISION in presets


class TestLoadPreset:
    """Tests for load_preset function."""

    def test_load_fast_preset(self):
        """load_preset should load FAST preset."""
        config = load_preset(ConfigPreset.FAST)

        assert isinstance(config, dict)
        assert "mpc" in config
        assert config["mpc"]["q_position"] == 2000.0
        assert config["mpc"]["max_velocity"] == 0.8

    def test_load_balanced_preset(self):
        """load_preset should load BALANCED preset."""
        config = load_preset(ConfigPreset.BALANCED)

        assert isinstance(config, dict)
        assert "mpc" in config

    def test_load_stable_preset(self):
        """load_preset should load STABLE preset."""
        config = load_preset(ConfigPreset.STABLE)

        assert isinstance(config, dict)
        assert "mpc" in config
        assert config["mpc"]["q_velocity"] == 15000.0
        assert config["mpc"]["max_velocity"] == 0.3

    def test_load_precision_preset(self):
        """load_preset should load PRECISION preset."""
        config = load_preset(ConfigPreset.PRECISION)

        assert isinstance(config, dict)
        assert "mpc" in config
        assert config["mpc"]["q_position"] == 5000.0
        assert config["mpc"]["max_velocity"] == 0.2

    def test_invalid_preset(self):
        """load_preset should raise ValueError for invalid preset."""
        with pytest.raises(ValueError) as exc_info:
            load_preset("invalid_preset")

        assert "Invalid preset name" in str(exc_info.value)
        assert "invalid_preset" in str(exc_info.value)

    def test_case_insensitive(self):
        """load_preset should handle case-insensitive preset names."""
        config1 = load_preset("FAST")
        config2 = load_preset("fast")
        config3 = load_preset("Fast")

        # All should produce same config
        assert config1 == config2 == config3


class TestGetPresetDescription:
    """Tests for get_preset_description function."""

    def test_all_presets_have_descriptions(self):
        """All presets should have descriptions."""
        for preset in ConfigPreset.all():
            description = get_preset_description(preset)
            assert isinstance(description, str)
            assert len(description) > 0

    def test_fast_description(self):
        """FAST preset should have appropriate description."""
        description = get_preset_description(ConfigPreset.FAST)
        assert "fast" in description.lower() or "aggressive" in description.lower()

    def test_stable_description(self):
        """STABLE preset should have appropriate description."""
        description = get_preset_description(ConfigPreset.STABLE)
        assert "stable" in description.lower() or "smooth" in description.lower()

    def test_invalid_preset_description(self):
        """get_preset_description should raise ValueError for invalid preset."""
        with pytest.raises(ValueError):
            get_preset_description("invalid_preset")


class TestListPresets:
    """Tests for list_presets function."""

    def test_list_all_presets(self):
        """list_presets should return all presets with descriptions."""
        presets = list_presets()

        assert isinstance(presets, dict)
        assert len(presets) == 4
        assert ConfigPreset.FAST in presets
        assert ConfigPreset.BALANCED in presets
        assert ConfigPreset.STABLE in presets
        assert ConfigPreset.PRECISION in presets

        # All should have descriptions
        for preset, description in presets.items():
            assert isinstance(description, str)
            assert len(description) > 0


class TestPresetCharacteristics:
    """Tests for preset characteristics and validation."""

    def test_fast_preset_characteristics(self):
        """FAST preset should have aggressive characteristics."""
        config = load_preset(ConfigPreset.FAST)

        # Higher position weights
        assert config["mpc"]["q_position"] >= 2000.0
        # Lower velocity weights (allows faster movement)
        assert config["mpc"]["q_velocity"] <= 5000.0
        # Higher max velocities
        assert config["mpc"]["max_velocity"] >= 0.8

    def test_stable_preset_characteristics(self):
        """STABLE preset should have conservative characteristics."""
        config = load_preset(ConfigPreset.STABLE)

        # Higher velocity weights (smoother)
        assert config["mpc"]["q_velocity"] >= 15000.0
        # Lower max velocities
        assert config["mpc"]["max_velocity"] <= 0.3
        # Larger damping zone
        assert config["mpc"]["damping_zone"] >= 0.4

    def test_precision_preset_characteristics(self):
        """PRECISION preset should have very conservative characteristics."""
        config = load_preset(ConfigPreset.PRECISION)

        # Very high weights
        assert config["mpc"]["q_position"] >= 5000.0
        assert config["mpc"]["q_velocity"] >= 20000.0
        # Very low max velocities
        assert config["mpc"]["max_velocity"] <= 0.2
        # Very large damping zone
        assert config["mpc"]["damping_zone"] >= 0.5
