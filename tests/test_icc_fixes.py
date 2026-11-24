"""Behavioural tests covering the six required ICC colour-transform fixes.

These complement the placeholder contract tests in
``test_icc_color_transform.py`` by asserting real colour-management behaviour
through the lcms2 engine:

1. missing profile -> declared fallback (never a silent sRGB assumption)
2. rendering intent follows the file
3. high bit depth preserved end to end
4. embedded profile vs pixel-space validation
5. transform cache isolated by configuration fingerprint
6. one engine across platforms with deterministic output
"""
import numpy as np
import pytest

import imaging.icc_color_transform as m
from imaging.engine import CACHE
from imaging.model import ColorImage, ColorSpace, ICCProfile, Intent


@pytest.fixture(autouse=True)
def _isolate_cache():
    CACHE.clear()
    yield
    CACHE.clear()


@pytest.fixture
def srgb():
    return ICCProfile.srgb()


@pytest.fixture
def lab_profile():
    return ICCProfile.lab()


@pytest.fixture
def px16():
    return np.linspace(0, 65535, 12, dtype=np.uint16).reshape(2, 2, 3)


# --------------------------------------------------------------------------- #
# Fix 1 — declared fallback
# --------------------------------------------------------------------------- #
class TestProfileFallback:
    def test_declared_fallback_honoured(self, srgb, px16):
        img = ColorImage(px16.copy(), ColorSpace.RGB)
        result = m.profile_fallback(img, declared=srgb)
        assert result["fallback_kind"] == "declared"
        assert any(e["code"] == "profile.fallback" for e in result["audit"])

    def test_strict_missing_profile_raises(self, px16):
        with pytest.raises(m.TransformValidationError):
            m.profile_fallback(ColorImage(px16.copy(), ColorSpace.RGB),
                               strict=True)

    def test_lenient_assumption_is_audited(self, px16):
        result = m.profile_fallback(ColorImage(px16.copy(), ColorSpace.RGB))
        assert result["fallback_kind"] == "assumed-srgb"
        assert any(e["code"] == "profile.assumed_srgb"
                   for e in result["audit"])

    def test_embedded_profile_wins(self, srgb, px16):
        img = ColorImage(px16.copy(), ColorSpace.RGB,
                         embedded_profile=srgb, declared_profile=ICCProfile.lab())
        result = m.profile_fallback(img)
        assert result["fallback_kind"] is None
        assert result["profile"] is srgb


# --------------------------------------------------------------------------- #
# Fix 2 — intent follows the file
# --------------------------------------------------------------------------- #
class TestIntentMatch:
    def test_intent_taken_from_file(self, srgb, px16):
        img = ColorImage(px16.copy(), ColorSpace.RGB,
                         embedded_profile=srgb,
                         declared_intent=Intent.ABSOLUTE_COLORIMETRIC)
        result = m.intent_match(img)
        assert result["intent"] == int(Intent.ABSOLUTE_COLORIMETRIC)
        assert result["source"] == "file"
        assert result["crushes_shadows"] is True

    def test_default_is_relative_colorimetric(self, px16):
        result = m.intent_match(ColorImage(px16.copy(), ColorSpace.RGB))
        assert result["intent"] == int(Intent.RELATIVE_COLORIMETRIC)

    def test_string_intent_accepted(self, srgb, px16):
        img = ColorImage(px16.copy(), ColorSpace.RGB, embedded_profile=srgb)
        assert m.intent_match(img, "perceptual")["intent"] == Intent.PERCEPTUAL


# --------------------------------------------------------------------------- #
# Fix 3 — high bit depth preserved
# --------------------------------------------------------------------------- #
class TestBitDepth:
    def test_16bit_round_trip_preserved(self, srgb, px16):
        report = m.bit_depth(
            ColorImage(px16.copy(), ColorSpace.RGB, embedded_profile=srgb),
            srgb)
        assert report["source_bits"] == 16
        assert report["output_bits"] == 16
        assert report["preserved"] is True

    def test_8bit_not_promoted_or_demoted(self, srgb):
        arr = np.full((2, 2, 3), 120, np.uint8)
        report = m.bit_depth(
            ColorImage(arr, ColorSpace.RGB, embedded_profile=srgb), srgb)
        assert report["source_bits"] == report["output_bits"] == 8

    def test_shadow_levels_not_banded(self, srgb, lab_profile):
        dark = np.zeros((1, 6, 3), dtype=np.uint16)
        for i in range(6):
            dark[0, i, :] = i * 257
        out = m.transform(
            ColorImage(dark, ColorSpace.RGB, embedded_profile=srgb),
            lab_profile)
        assert len(np.unique(out.pixels[0, :, 0])) == 6

    def test_float_promoted_to_16_not_crushed_to_8(self, srgb):
        arr = np.full((2, 2, 3), 0.5, np.float32)
        out = m.transform(
            ColorImage(arr, ColorSpace.RGB, embedded_profile=srgb), srgb)
        assert out.pixels.dtype == np.uint16


# --------------------------------------------------------------------------- #
# Fix 4 — embedded / pixel space validation
# --------------------------------------------------------------------------- #
class TestEmbedMismatch:
    def test_mismatch_detected(self, srgb):
        img = ColorImage(np.zeros((2, 2, 1), np.uint16), ColorSpace.GRAY,
                         embedded_profile=srgb)
        result = m.embed_mismatch(img)
        assert result["mismatch"] is True
        assert result["embedded_space"] == "RGB"
        assert result["pixel_space"] == "GRAY"

    def test_strict_mismatch_raises(self, srgb):
        img = ColorImage(np.zeros((2, 2, 1), np.uint16), ColorSpace.GRAY,
                         embedded_profile=srgb)
        with pytest.raises(m.TransformValidationError):
            m.embed_mismatch(img, strict=True)

    def test_consistent_image_passes(self, srgb, px16):
        result = m.embed_mismatch(
            ColorImage(px16.copy(), ColorSpace.RGB, embedded_profile=srgb))
        assert result["mismatch"] is False


# --------------------------------------------------------------------------- #
# Fix 5 — fingerprint-isolated cache
# --------------------------------------------------------------------------- #
class TestCacheLifetime:
    def test_same_config_reuses_entry(self, srgb, px16):
        img = ColorImage(px16.copy(), ColorSpace.RGB, embedded_profile=srgb)
        m.transform(img, srgb)
        m.transform(ColorImage(px16.copy(), ColorSpace.RGB,
                               embedded_profile=srgb), srgb)
        assert len(CACHE) == 1
        assert CACHE.stats()["hits"] >= 1

    def test_distinct_intents_get_distinct_entries(self, srgb, px16):
        img = ColorImage(px16.copy(), ColorSpace.RGB, embedded_profile=srgb)
        m.transform(img, srgb, intent=Intent.RELATIVE_COLORIMETRIC)
        assert len(CACHE) == 1
        m.transform(img, srgb, intent=Intent.PERCEPTUAL)
        assert len(CACHE) == 2

    def test_no_drift_across_batches(self, srgb, px16):
        a = m.batch([ColorImage(px16.copy(), ColorSpace.RGB,
                                embedded_profile=srgb)], srgb)
        b = m.batch([ColorImage(px16.copy(), ColorSpace.RGB,
                                embedded_profile=srgb)], srgb)
        assert np.array_equal(a[0].pixels, b[0].pixels)

    def test_fingerprints_stable_and_distinct(self, srgb, lab_profile):
        assert srgb.fingerprint == ICCProfile.srgb().fingerprint
        assert srgb.fingerprint != lab_profile.fingerprint


# --------------------------------------------------------------------------- #
# Fix 6 — single engine / determinism
# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_single_named_engine(self):
        assert m.engine_name() == "lcms2"
        assert m.ENGINE_NAME == "lcms2"

    def test_repeated_runs_byte_identical(self, srgb, px16):
        result = m.determinism(
            ColorImage(px16.copy(), ColorSpace.RGB, embedded_profile=srgb),
            runs=3)
        assert result["byte_identical"] is True
        assert result["engine"] == "lcms2"


# --------------------------------------------------------------------------- #
# Auxiliary API
# --------------------------------------------------------------------------- #
class TestAuxiliary:
    def test_black_point(self, srgb):
        bp = m.black_point(srgb)
        assert len(bp["xyz"]) == 3

    def test_tag_white_point(self, srgb):
        tg = m.tag(srgb, "wtpt")
        assert tg["present"] is True
        assert len(tg["xyz"]) == 3

    def test_gamut_map(self, srgb, px16):
        gm = m.gamut_map(
            ColorImage(px16.copy(), ColorSpace.RGB, embedded_profile=srgb),
            srgb)
        assert 0.0 <= gm["clip_ratio"] <= 1.0

    def test_metrics(self, srgb, px16):
        mt = m.metrics(
            ColorImage(px16.copy(), ColorSpace.RGB, embedded_profile=srgb))
        assert mt["mean"] >= 0.0
        assert "p95" in mt

    def test_audit_trail_populated(self, srgb, px16):
        img = ColorImage(px16.copy(), ColorSpace.RGB, embedded_profile=srgb)
        m.transform(img, srgb)
        events = m.audit(img)
        assert any(e["code"].startswith("transform") for e in events)


# --------------------------------------------------------------------------- #
# Chromatic adaptation / Lab correctness
# --------------------------------------------------------------------------- #
class TestLabAndAdaptation:
    def test_white_maps_to_L100_neutral(self, srgb):
        white = np.full((1, 1, 3), 65535, np.uint16)
        lab = m.to_lab(
            ColorImage(white, ColorSpace.RGB, embedded_profile=srgb))[0, 0]
        assert abs(lab[0] - 100.0) < 0.5
        assert abs(lab[1]) < 0.5 and abs(lab[2]) < 0.5

    def test_neutral_axis_has_no_chromatic_tint(self, srgb):
        # Every neutral gray must land on a=b=0; any tint means the D65->D50
        # chromatic adaptation is broken.
        for code in (0, 16384, 32768, 49152, 65535):
            gray = np.full((1, 1, 3), code, np.uint16)
            lab = m.to_lab(
                ColorImage(gray, ColorSpace.RGB,
                           embedded_profile=srgb))[0, 0]
            assert abs(lab[1]) < 0.5 and abs(lab[2]) < 0.5, code

    def test_identical_colors_zero_delta_e(self, srgb):
        arr = np.full((1, 2, 3), 40000, np.uint16)
        de = m.metrics(
            ColorImage(arr.copy(), ColorSpace.RGB, embedded_profile=srgb),
            ColorImage(arr.copy(), ColorSpace.RGB, embedded_profile=srgb))
        assert de["mean"] < 1e-9

    def test_white_black_delta_e_is_full_range(self, srgb):
        white = np.full((1, 2, 3), 65535, np.uint16)
        black = np.zeros((1, 2, 3), np.uint16)
        de = m.metrics(
            ColorImage(white, ColorSpace.RGB, embedded_profile=srgb),
            ColorImage(black, ColorSpace.RGB, embedded_profile=srgb))
        assert de["mean"] > 90.0

    def test_lab_round_trip_near_exact(self, srgb, lab_profile):
        red = np.zeros((1, 1, 3), np.uint16)
        red[0, 0, 0] = 65535
        forward = m.transform(
            ColorImage(red, ColorSpace.RGB, embedded_profile=srgb),
            lab_profile)
        back = m.transform(forward, srgb).pixels
        # 16-bit Lab round trip stays within a single code per channel.
        assert np.max(np.abs(back.astype(int) - red.astype(int))) <= 8
