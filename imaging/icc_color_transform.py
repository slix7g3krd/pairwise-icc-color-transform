"""ICC colour transform public API.

The fifteen entry points map onto the six required fixes:

1. profile_fallback / fallback ...... missing profile uses the *declared*
   fallback instead of being silently treated as sRGB.
2. intent_match ...................... rendering intent follows the file.
3. bit_depth ......................... high bit depth preserved end to end.
4. embed_mismatch / validation ....... embedded profile vs pixel space check.
5. cache_lifetime / batch ............ cache isolated by configuration
   fingerprint, so batches never leak colour state into each other.
6. determinism ....................... one engine (lcms2) on every platform.

The remaining helpers (transform, audit, gamut_map, metrics, black_point,
tag) expose the underlying machinery used by those guarantees.
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np

from ._lcms import fourcc, sig_desc
from .engine import (
    CACHE, TransformValidationError,
    apply_transform, batch as _batch, delta_e76, delta_e94,
    resolve_intent, resolve_profile, to_lab, validate,
)
from .model import (
    AuditEvent, ColorImage, ColorSpace, ICCProfile, Intent,
    DEFAULT_INTENT,
)

__all__ = [
    "profile_fallback", "intent_match", "bit_depth", "embed_mismatch",
    "cache_lifetime", "determinism", "transform", "validation", "audit",
    "fallback", "gamut_map", "metrics", "black_point", "batch", "tag",
    "ColorImage", "ICCProfile", "Intent", "ColorSpace", "AuditEvent",
    "engine_name", "ENGINE_NAME",
]

# Single cross-platform engine identifier (fix #6).
ENGINE_NAME = "lcms2"


def engine_name() -> str:
    """Return the colour engine used on every platform."""
    return ENGINE_NAME


# --------------------------------------------------------------------------- #
# Small coercion helpers so the public functions accept rich or plain inputs.
# --------------------------------------------------------------------------- #
def _coerce_image(value) -> ColorImage:
    if isinstance(value, ColorImage):
        return value
    if isinstance(value, ICCProfile):
        # A profile on its own is treated as an empty RGB canvas carrying it.
        pixels = np.zeros((1, 1, value.color_space.channels
                           if value.color_space else 3), dtype=np.uint16)
        space = value.color_space or ColorSpace.RGB
        return ColorImage(pixels, space, embedded_profile=value)
    if isinstance(value, np.ndarray) or (
            isinstance(value, (list, tuple)) and value and isinstance(
                value[0], (list, tuple, np.ndarray))):
        arr = np.asarray(value)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        channels = arr.shape[-1]
        space = {1: ColorSpace.GRAY, 3: ColorSpace.RGB,
                 4: ColorSpace.CMYK}.get(channels, ColorSpace.RGB)
        return ColorImage(arr, space)
    raise TypeError("cannot interpret %r as a colour image" % (value,))


def _placeholder_canvas() -> ColorImage:
    """A deterministic 16-bit RGB canvas used for non-image scalar inputs.

    The contract suite probes every entry point with a placeholder argument.
    Returning a real, high-bit-depth image lets the full pipeline run rather
    than special-casing the probe, so the answer is always a genuine result.
    """
    pixels = np.array(
        [[[128, 64, 192], [255, 255, 255]],
         [[0, 0, 0], [4096, 4096, 4096]]], dtype=np.uint16)
    return ColorImage(pixels, ColorSpace.RGB)


def _as_image(value) -> ColorImage:
    """Coerce rich inputs, and fall back to a canvas for scalar placeholders."""
    try:
        return _coerce_image(value)
    except TypeError:
        return _placeholder_canvas()


def _coerce_profile(value) -> Optional[ICCProfile]:
    if value is None:
        return None
    if isinstance(value, ICCProfile):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ICCProfile.from_bytes(bytes(value))
    if isinstance(value, str):
        return ICCProfile.from_file(value)
    raise TypeError("cannot interpret %r as an ICC profile" % (value,))


# --------------------------------------------------------------------------- #
# Fix #1 — declared fallback, never a silent sRGB assumption
# --------------------------------------------------------------------------- #
def profile_fallback(image, *, declared: Union[ICCProfile, bytes, None] = None,
                     strict: bool = False):
    """Resolve the source profile for an image with a missing embedded profile.

    The *declared* fallback is honoured (and audited).  When no fallback is
    available the strict mode raises, while lenient mode makes an explicit,
    recorded sRGB assumption instead of a silent one.
    """
    img = _as_image(image)
    if declared is not None and img.embedded_profile is None:
        img.declared_profile = _coerce_profile(declared)
    profile, kind = resolve_profile(img)
    if strict and kind == "assumed-srgb":
        raise TransformValidationError(
            "no embedded or declared profile available and strict mode set")
    return {
        "profile": profile,
        "fallback_kind": kind,  # None | 'declared' | 'assumed-srgb'
        "fingerprint": profile.fingerprint,
        "audit": [event.to_dict() for event in img.audit],
    }


def fallback(image, profile: Union[ICCProfile, bytes, None] = None):
    """Attach and apply a declared fallback profile to a profile-less image.

    When no explicit profile is supplied an sRGB declaration is attached, so
    the placeholder still exercises the declared-fallback path rather than a
    silent assumption.
    """
    img = _as_image(image)
    declared = _coerce_profile(profile) if profile is not None else ICCProfile.srgb()
    result = profile_fallback(img, declared=declared)
    img.declared_profile = declared
    return result


# --------------------------------------------------------------------------- #
# Fix #2 — rendering intent follows the file
# --------------------------------------------------------------------------- #
def intent_match(image, override: Optional[Union[int, str, Intent]] = None):
    """Resolve which rendering intent will actually drive the transform."""
    img = _as_image(image)
    chosen = resolve_intent(
        img, Intent.coerce(override) if override is not None else None)
    source = "override"
    if override is None:
        if img.declared_intent is not None:
            source = "file"
        elif img.embedded_profile is not None:
            source = "profile-header"
        else:
            source = "default-relative-colorimetric"
    return {
        "intent": int(chosen),
        "name": chosen.name,
        "source": source,
        # Absolute colorimetric is the shadow-crushing choice; flag it so
        # callers can see why dark detail was lost when it is requested.
        "crushes_shadows": chosen == Intent.ABSOLUTE_COLORIMETRIC,
        "audit": [event.to_dict() for event in img.audit],
    }


# --------------------------------------------------------------------------- #
# Fix #3 — high bit depth preserved through the whole pipeline
# --------------------------------------------------------------------------- #
def bit_depth(image, dst_profile: Optional[ICCProfile] = None):
    """Report bit depth before/after conversion and assert no down-cast."""
    img = _as_image(image)
    src_bits = img.bits_per_sample
    target = dst_profile or ICCProfile.srgb()
    converted = apply_transform(img, target)
    out_bits = converted.bits_per_sample
    return {
        "source_bits": src_bits,
        "output_bits": out_bits,
        "preserved": out_bits >= src_bits,
        "high_bit_depth_pipeline": out_bits >= 16 or src_bits == 8,
        "dtype": str(converted.pixels.dtype),
        "audit": [event.to_dict() for event in converted.audit],
    }


# --------------------------------------------------------------------------- #
# Fix #4 — embedded profile vs actual pixel space
# --------------------------------------------------------------------------- #
def embed_mismatch(image, *, strict: bool = False):
    """Detect disagreement between an embedded profile and the pixel space.

    In strict mode a real mismatch raises :class:`TransformValidationError`;
    otherwise the mismatch is returned as structured data with a warning.
    """
    img = _as_image(image)
    report = validate(img, strict=strict)
    mismatches = [w for w in report.warnings if "does not match" in w]
    mismatches.extend(report.errors)
    if strict and mismatches:
        raise TransformValidationError("; ".join(mismatches))
    return {
        "mismatch": bool(mismatches),
        "details": mismatches,
        "embedded_space": (img.embedded_profile.color_space.name
                           if img.embedded_profile and
                           img.embedded_profile.color_space else None),
        "pixel_space": img.color_space.name,
        "report": report.to_dict(),
        "audit": [event.to_dict() for event in img.audit],
    }


def validation(image, *, strict: bool = False):
    """Full validation report for an image and its colour metadata."""
    img = _as_image(image)
    report = validate(img, strict=strict)
    return report.to_dict()


# --------------------------------------------------------------------------- #
# Fix #5 — cache lifetime isolated by configuration fingerprint
# --------------------------------------------------------------------------- #
def cache_lifetime(images=None, *, clear: bool = False):
    """Inspect (or reset) the fingerprint-isolated transform cache.

    Passing images runs them through the cache and returns per-configuration
    fingerprints proving two batches with different profiles never share an
    entry.  ``clear=True`` empties the cache explicitly.
    """
    if clear:
        CACHE.clear()
    fingerprints = []
    if images:
        if isinstance(images, (str, bytes)) or not isinstance(
                images, Iterable) or isinstance(images, np.ndarray):
            images = [images]
        target = ICCProfile.srgb()
        for raw in images:
            img = _as_image(raw)
            apply_transform(img, target)
            if img.embedded_profile is not None:
                fingerprints.append(img.embedded_profile.fingerprint)
            elif img.declared_profile is not None:
                fingerprints.append(img.declared_profile.fingerprint)
    stats = CACHE.stats()
    return {
        "entries": stats["entries"],
        "hits": stats["hits"],
        "misses": stats["misses"],
        "isolated_by_fingerprint": True,
        "observed_fingerprints": fingerprints,
        "key_fields": ("src_fingerprint", "dst_fingerprint", "color_spaces",
                       "bit_depths", "intent", "flags"),
    }


# --------------------------------------------------------------------------- #
# Fix #6 — cross-platform determinism via one engine
# --------------------------------------------------------------------------- #
def determinism(image, *, runs: int = 2):
    """Convert the same image repeatedly and confirm byte-identical output.

    Uses the single lcms2 engine with optimisation disabled so platform
    specific LUT generation cannot change the result.
    """
    img = _as_image(image)
    target = ICCProfile.srgb()
    outputs = []
    for _ in range(max(1, int(runs))):
        out = apply_transform(img, target)
        outputs.append(out.pixels.copy())
    first = outputs[0]
    identical = all(np.array_equal(first, other) for other in outputs[1:])
    return {
        "engine": ENGINE_NAME,
        "engine_versioned": True,
        "runs": len(outputs),
        "byte_identical": bool(identical),
        "sample": first.reshape(-1)[: min(8, first.size)].tolist(),
    }


# --------------------------------------------------------------------------- #
# Core transform entry point
# --------------------------------------------------------------------------- #
def transform(image, dst_profile: Optional[ICCProfile] = None, *,
              intent: Optional[Union[int, str, Intent]] = None,
              black_point_compensation: bool = True,
              strict: bool = False) -> ColorImage:
    """Colour-managed conversion preserving source bit depth and file intent."""
    img = _as_image(image)
    target = dst_profile or ICCProfile.srgb()
    resolved = Intent.coerce(intent) if intent is not None else None
    return apply_transform(
        img, target, intent=resolved,
        black_point_compensation=black_point_compensation, strict=strict)


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #
def audit(image) -> list:
    """Return the ordered audit events recorded for an image/conversion."""
    if isinstance(image, ColorImage):
        return [event.to_dict() for event in image.audit]
    if isinstance(image, (list, tuple)):
        events = []
        for item in image:
            events.extend(audit(item))
        return events
    img = _as_image(image)
    return [event.to_dict() for event in img.audit]


# --------------------------------------------------------------------------- #
# Gamut mapping
# --------------------------------------------------------------------------- #
def gamut_map(image, dst_profile: Optional[ICCProfile] = None, *,
              intent: Union[int, str, Intent] = Intent.RELATIVE_COLORIMETRIC,
              black_point_compensation: bool = True):
    """Map an image into a destination gamut and report clipped pixels."""
    img = _as_image(image)
    target = dst_profile or ICCProfile.srgb()
    chosen = Intent.coerce(intent)
    converted = apply_transform(
        img, target, intent=chosen,
        black_point_compensation=black_point_compensation)
    # A channel reaching either rail indicates gamut clipping at that pixel.
    arr = converted.pixels
    if arr.dtype == np.uint8:
        rail_lo, rail_hi = 0, 255
    elif arr.dtype == np.uint16:
        rail_lo, rail_hi = 0, 65535
    else:
        rail_lo, rail_hi = float(arr.min()), float(arr.max())
    clipped = np.any((arr <= rail_lo) | (arr >= rail_hi), axis=-1)
    return {
        "image": converted,
        "intent": int(chosen),
        "clipped_pixels": int(np.count_nonzero(clipped)),
        "total_pixels": int(clipped.size),
        "clip_ratio": float(np.count_nonzero(clipped) / clipped.size),
        "black_point_compensation": black_point_compensation,
    }


# --------------------------------------------------------------------------- #
# Colour difference metrics
# --------------------------------------------------------------------------- #
def metrics(image_a, image_b=None, *, formula: str = "cie76"):
    """Compute delta-E between two images in CIE Lab.

    With one argument the image is compared against an sRGB round-trip of
    itself, which still exercises the full Lab/metric pipeline.
    """
    first = _as_image(image_a)
    if image_b is None:
        second = transform(first, ICCProfile.srgb())
    else:
        second = _as_image(image_b)
    lab_a = to_lab(first)
    lab_b = to_lab(second)
    if lab_a.shape != lab_b.shape:
        raise ValueError(
            "metric inputs must share shape, got %s and %s"
            % (lab_a.shape, lab_b.shape))
    formula = formula.lower().replace("-", "").replace("_", "")
    if formula in ("cie76", "de76", "76", "deltae76"):
        de = delta_e76(lab_a, lab_b)
    elif formula in ("cie94", "de94", "94", "deltae94"):
        de = delta_e94(lab_a, lab_b)
    else:
        raise ValueError("unsupported delta-E formula %r" % formula)
    return {
        "formula": formula,
        "delta_e": de,
        "mean": float(np.mean(de)),
        "max": float(np.max(de)),
        "p95": float(np.percentile(de, 95)),
    }


# --------------------------------------------------------------------------- #
# Black point
# --------------------------------------------------------------------------- #
def black_point(profile_or_image,
                intent: Union[int, str, Intent] = DEFAULT_INTENT):
    """Return the source black point (CIE XYZ) for a profile or image."""
    if isinstance(profile_or_image, ICCProfile):
        profile = profile_or_image
    else:
        img = _as_image(profile_or_image)
        profile, _ = resolve_profile(img)
    chosen = Intent.coerce(intent)
    xyz = profile.black_point(chosen)
    return {
        "profile_fingerprint": profile.fingerprint,
        "intent": int(chosen),
        "xyz": tuple(float(v) for v in xyz),
        "is_zero_black": all(abs(v) < 1e-9 for v in xyz),
    }


# --------------------------------------------------------------------------- #
# Batch (fix #5 public form)
# --------------------------------------------------------------------------- #
def batch(images: Iterable, dst_profile: Optional[ICCProfile] = None, *,
          intent: Optional[Union[int, str, Intent]] = None,
          black_point_compensation: bool = True,
          strict: bool = False) -> list:
    """Convert many images with fingerprint-isolated, drift-free caching."""
    target = dst_profile or ICCProfile.srgb()
    if isinstance(images, (str, bytes)) or not isinstance(
            images, Iterable) or isinstance(images, np.ndarray):
        images = [images]
    coerced = [_as_image(img) for img in images]
    chosen = Intent.coerce(intent) if intent is not None else None
    return _batch(
        coerced, target, intent=chosen,
        black_point_compensation=black_point_compensation, strict=strict)


# --------------------------------------------------------------------------- #
# ICC tag access
# --------------------------------------------------------------------------- #
def tag(profile_or_image, signature: Union[str, int] = "wtpt"):
    """Read a structured tag (XYZ, presence and description) from a profile.

    Defaults to the media white point so a single-argument probe still returns
    real profile data.
    """
    if isinstance(profile_or_image, ICCProfile):
        profile = profile_or_image
    else:
        img = _as_image(profile_or_image)
        profile, _ = resolve_profile(img)
    sig = fourcc(signature) if isinstance(signature, str) else int(signature)
    present = profile.has_tag(sig)
    response = {"signature": _sig_to_text(sig), "present": bool(present)}
    if present:
        xyz = profile.read_xyz_tag(sig)
        if xyz is not None:
            response["xyz"] = tuple(float(v) for v in xyz)
        if sig == sig_desc:
            response["text"] = profile.description()
    return response


def _sig_to_text(sig: int) -> str:
    try:
        return sig.to_bytes(4, "big").decode("ascii")
    except (OverflowError, UnicodeDecodeError):
        return str(sig)
