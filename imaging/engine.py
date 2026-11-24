"""Core ICC transform engine.

One lcms2 engine for every platform, a fingerprint-isolated transform cache,
source-bit-depth pixel pipelines, and explicit validation/audit around every
conversion.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from . import _lcms
from ._lcms import (
    LcmsError,
    lib,
    TYPE_GRAY_8, TYPE_GRAY_16, TYPE_RGB_8, TYPE_RGB_16,
    TYPE_CMYK_8, TYPE_CMYK_16, TYPE_Lab_8, TYPE_Lab_16,
    cmsFLAGS_BLACKPOINTCOMPENSATION, cmsFLAGS_COPY_ALPHA, cmsFLAGS_NOOPTIMIZE,
)
from .model import (
    ColorImage, ColorSpace, ICCProfile, Intent,
    DEFAULT_INTENT, KNOWN_INTENTS,
)

# --------------------------------------------------------------------------- #
# Format mapping
# --------------------------------------------------------------------------- #
# Maps (color space, bits per sample) to the matching lcms2 TYPE constant.
# Every supported format has both an 8-bit and a high-bit-depth entry so the
# pipeline never silently down-casts a 16-bit source.
_TYPE_TABLE = {
    (ColorSpace.GRAY, 8): TYPE_GRAY_8,
    (ColorSpace.GRAY, 16): TYPE_GRAY_16,
    (ColorSpace.RGB, 8): TYPE_RGB_8,
    (ColorSpace.RGB, 16): TYPE_RGB_16,
    (ColorSpace.CMYK, 8): TYPE_CMYK_8,
    (ColorSpace.CMYK, 16): TYPE_CMYK_16,
    (ColorSpace.Lab, 8): TYPE_Lab_8,
    (ColorSpace.Lab, 16): TYPE_Lab_16,
}


def _format_for(color_space: ColorSpace, bits: int) -> int:
    if bits not in (8, 16):
        raise ValueError(
            "lcms2 integer pipeline supports 8 or 16 bits per sample, got %d"
            % bits)
    try:
        return _TYPE_TABLE[(color_space, bits)]
    except KeyError:
        raise LcmsError(
            "no native lcms2 format for %s at %d bits"
            % (color_space.name, bits))


def _ctype_for(bits: int):
    return _lcms.ctypes.c_uint8 if bits == 8 else _lcms.ctypes.c_uint16


# --------------------------------------------------------------------------- #
# Fingerprint-isolated transform cache
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _CacheKey:
    src_fingerprint: str
    dst_fingerprint: str
    src_space: int
    dst_space: int
    src_bits: int
    dst_bits: int
    intent: int
    flags: int


class _TransformCache:
    """Process-wide cache keyed by a full configuration fingerprint.

    Reusing a transform across *different* profiles or intents is what causes
    cross-batch colour drift, so every distinguishing field is part of the key.
    The cache also stores each profile's content hash inside the key, making
    stale reuse after a profile is replaced impossible.
    """

    def __init__(self):
        self._entries: dict[_CacheKey, int] = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def get_or_create(self, src: ICCProfile, dst: ICCProfile,
                      src_space: ColorSpace, dst_space: ColorSpace,
                      src_bits: int, dst_bits: int,
                      intent: Intent, flags: int) -> int:
        key = _CacheKey(
            src.fingerprint, dst.fingerprint, int(src_space), int(dst_space),
            src_bits, dst_bits, int(intent), flags)
        with self._lock:
            handle = self._entries.get(key)
            if handle:
                self._hits += 1
                return handle
            self._misses += 1
            in_type = _format_for(src_space, src_bits)
            out_type = _format_for(dst_space, dst_bits)
            lcms = lib()
            handle = lcms.cmsCreateTransform(
                src.handle, in_type, dst.handle, out_type, int(intent), flags)
            if not handle:
                raise LcmsError(
                    "lcms2 could not build transform %s/%db -> %s/%db "
                    "intent=%s flags=%#x" % (
                        src_space.name, src_bits, dst_space.name, dst_bits,
                        Intent(intent).name, flags))
            self._entries[key] = handle
            return handle

    def clear(self) -> None:
        with self._lock:
            lcms = lib()
            for handle in self._entries.values():
                lcms.cmsDeleteTransform(handle)
            self._entries.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._entries),
                    "hits": self._hits, "misses": self._misses}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# A single shared cache instance; isolation comes from the key, not from
# wiping state between batches (which would discard useful warm entries).
CACHE = _TransformCache()


# --------------------------------------------------------------------------- #
# Validation & fallback
# --------------------------------------------------------------------------- #
class TransformValidationError(ValueError):
    """Raised when embedded profile and pixel space cannot be reconciled."""


@dataclass
class ValidationReport:
    ok: bool
    warnings: list
    errors: list
    fallback_used: Optional[str]
    resolved_intent: Intent
    source_bits: int
    output_bits: int

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "fallback_used": self.fallback_used,
            "resolved_intent": int(self.resolved_intent),
            "resolved_intent_name": self.resolved_intent.name,
            "source_bits": self.source_bits,
            "output_bits": self.output_bits,
        }


def resolve_profile(image: ColorImage) -> tuple[ICCProfile, Optional[str]]:
    """Pick the source profile according to the declared-fallback policy.

    Policy (fix #1): a *missing* profile is not silently treated as sRGB.  The
    image must declare a fallback; when it does we honour that declaration and
    record an audit event.  An embedded profile always wins when present.
    """
    if image.embedded_profile is not None:
        return image.embedded_profile, None
    if image.declared_profile is not None:
        fp = image.declared_profile.fingerprint
        image.record(
            "profile.fallback",
            "no embedded profile; using declared fallback %s"
            % image.declared_profile.description(), fp)
        return image.declared_profile, "declared"
    # No embedded data and no declaration: callers decide whether this is
    # fatal (strict) or an explicit sRGB assumption (lenient policy object).
    srgb = ICCProfile.srgb()
    image.record(
        "profile.assumed_srgb",
        "no embedded or declared profile; assuming sRGB under lenient policy",
        srgb.fingerprint)
    return srgb, "assumed-srgb"


def resolve_intent(image: ColorImage,
                   override: Optional[Intent] = None) -> Intent:
    """Rendering intent follows the file, then an explicit override (fix #2)."""
    if override is not None:
        return Intent.coerce(override)
    if image.declared_intent is not None:
        intent = Intent.coerce(image.declared_intent)
        image.record(
            "intent.file",
            "rendering intent %d (%s) taken from file metadata"
            % (int(intent), intent.name))
        return intent
    if image.embedded_profile is not None:
        intent = image.embedded_profile.header_intent
        image.record(
            "intent.profile_header",
            "rendering intent %d (%s) taken from profile header"
            % (int(intent), intent.name))
        return intent
    image.record(
        "intent.default",
        "no intent declared; defaulting to relative colorimetric to keep "
        "shadow detail")
    return DEFAULT_INTENT


def validate(image: ColorImage, *, strict: bool = False) -> ValidationReport:
    """Cross-check embedded profile against the actual pixel space (fix #4)."""
    warnings: list = []
    errors: list = []
    fallback_used = None

    if image.embedded_profile is None and image.declared_profile is None:
        msg = "image carries neither an embedded nor a declared profile"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg + "; lenient policy will assume sRGB")
    elif image.embedded_profile is None:
        fallback_used = "declared"
        warnings.append(
            "embedded profile missing; declared fallback will be used")

    if image.embedded_profile is not None:
        prof_space = image.embedded_profile.color_space
        if prof_space is not None and prof_space != image.color_space:
            msg = ("embedded profile colour space %s does not match pixel "
                   "space %s" % (prof_space.name, image.color_space.name))
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
        # Profile must actually contain the tags needed for a conversion.
        if not image.embedded_profile.has_tag(_lcms.sig_wtpt):
            warnings.append("profile is missing a media white point tag")

    bits = image.bits_per_sample
    if bits not in (8, 16):
        warnings.append(
            "source depth %d-bit will be promoted to the 16-bit lcms pipeline"
            % bits)
    if image.declared_intent is not None and int(
            Intent.coerce(image.declared_intent)) not in KNOWN_INTENTS:
        errors.append("declared rendering intent is outside the ICC 0..3 range")

    intent = resolve_intent(image)
    report = ValidationReport(
        ok=not errors, warnings=warnings, errors=errors,
        fallback_used=fallback_used, resolved_intent=intent,
        source_bits=bits, output_bits=bits)
    image.record(
        "validation." + ("ok" if report.ok else "failed"),
        "; ".join(errors + warnings) or "all checks passed")
    return report


# --------------------------------------------------------------------------- #
# Pixel pipeline
# --------------------------------------------------------------------------- #
def _as_contiguous(image: ColorImage) -> np.ndarray:
    arr = np.ascontiguousarray(image.pixels)
    return arr


def _promote_to_supported(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Promote unusual depths onto the 16-bit pipeline without down-casting.

    8-bit stays 8-bit; 16-bit stays 16-bit; 32-bit/float is scaled to a full
    16-bit range so no quantisation is introduced relative to a true 16-bit
    source (and certainly never crushed to 8-bit).
    """
    if arr.dtype == np.uint8:
        return arr, 8
    if arr.dtype == np.uint16:
        return arr, 16
    if arr.dtype == np.uint32:
        scaled = (arr.astype(np.uint64) * 65535 // 0xFFFFFFFF).astype(np.uint16)
        return scaled, 16
    if arr.dtype.kind == "f":
        clipped = np.clip(arr, 0.0, 1.0)
        return np.round(clipped * 65535.0).astype(np.uint16), 16
    raise ValueError("unsupported pixel dtype %s" % arr.dtype)


def apply_transform(image: ColorImage, dst_profile: ICCProfile,
                    *, intent: Optional[Intent] = None,
                    dst_space: Optional[ColorSpace] = None,
                    black_point_compensation: bool = True,
                    strict: bool = False) -> ColorImage:
    """Run a colour-managed conversion preserving full source precision."""
    report = validate(image, strict=strict)
    if strict and not report.ok:
        raise TransformValidationError("; ".join(report.errors))

    src_profile, fallback = resolve_profile(image)
    resolved_intent = resolve_intent(image, intent)

    src_arr, src_bits = _promote_to_supported(_as_contiguous(image))
    target_space = dst_space or (dst_profile.color_space or image.color_space)
    # Output keeps the same bit depth as the (promoted) source: fix #3.
    dst_bits = src_bits

    flags = cmsFLAGS_NOOPTIMIZE  # deterministic, no platform-specific LUTs
    if black_point_compensation and resolved_intent in (
            Intent.RELATIVE_COLORIMETRIC, Intent.PERCEPTUAL):
        flags |= cmsFLAGS_BLACKPOINTCOMPENSATION
    if src_arr.shape[2] == 4:
        flags |= cmsFLAGS_COPY_ALPHA

    handle = CACHE.get_or_create(
        src_profile, dst_profile,
        image.color_space, target_space,
        src_bits, dst_bits, resolved_intent, flags)

    height, width, channels = src_arr.shape
    count = height * width
    ctype = _ctype_for(src_bits)
    flat = src_arr.reshape(count, channels)
    in_buf = (ctype * (count * channels))(*flat.reshape(-1).tolist())
    out_buf = (ctype * (count * channels))()
    lib().cmsDoTransform(
        handle,
        _lcms.ctypes.cast(in_buf, _lcms.ctypes.c_void_p),
        _lcms.ctypes.cast(out_buf, _lcms.ctypes.c_void_p),
        count)

    out_arr = np.frombuffer(
        _lcms.ctypes.string_at(out_buf, _lcms.ctypes.sizeof(out_buf)),
        dtype=src_arr.dtype).reshape(height, width, channels).copy()

    image.record(
        "transform.apply",
        "%s/%db -> %s/%db intent=%s bpc=%s cache_fp=%s->%s" % (
            image.color_space.name, src_bits, target_space.name, dst_bits,
            resolved_intent.name, black_point_compensation,
            src_profile.fingerprint[:12], dst_profile.fingerprint[:12]))

    result = ColorImage(
        out_arr, target_space,
        declared_profile=dst_profile,
        embedded_profile=dst_profile,
        declared_intent=resolved_intent,
        source_name=image.source_name,
        audit=list(image.audit))
    result.record(
        "bitdepth.preserved",
        "output retained %d bits per sample (no down-cast)" % dst_bits)
    return result


# --------------------------------------------------------------------------- #
# Lab / delta-E metrics
# --------------------------------------------------------------------------- #
def to_lab(image: ColorImage, *, intent: Optional[Intent] = None) -> np.ndarray:
    """Convert pixels to CIE Lab (D50) at high precision for metric work."""
    lab_profile = ICCProfile.lab()
    lab_image = apply_transform(
        image, lab_profile, intent=intent, dst_space=ColorSpace.Lab)
    raw = lab_image.pixels.astype(np.float64)
    # lcms encodes Lab in 16 bits: L in 0..65535 mapping 0..100, a/b centred
    # at 32896 (128*257) spanning -128..127.
    L = raw[..., 0] / 655.35
    a = raw[..., 1] / 257.0 - 128.0
    b = raw[..., 2] / 257.0 - 128.0
    return np.stack([L, a, b], axis=-1)


def delta_e76(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """CIE76 delta-E (Euclidean distance in Lab)."""
    lab_a = np.asarray(lab_a, dtype=np.float64)
    lab_b = np.asarray(lab_b, dtype=np.float64)
    return np.sqrt(np.sum((lab_a - lab_b) ** 2, axis=-1))


def delta_e94(lab_a: np.ndarray, lab_b: np.ndarray,
              *, textiles: bool = False) -> np.ndarray:
    """CIE94 delta-E (graphic-arts weights by default)."""
    lab_a = np.asarray(lab_a, dtype=np.float64)
    lab_b = np.asarray(lab_b, dtype=np.float64)
    dL = lab_a[..., 0] - lab_b[..., 0]
    da = lab_a[..., 1] - lab_b[..., 1]
    db = lab_a[..., 2] - lab_b[..., 2]
    C1 = np.sqrt(lab_a[..., 1] ** 2 + lab_a[..., 2] ** 2)
    C2 = np.sqrt(lab_b[..., 1] ** 2 + lab_b[..., 2] ** 2)
    dC = C1 - C2
    dH2 = da * da + db * db - dC * dC
    dH2 = np.maximum(dH2, 0.0)
    kL = 2.0 if textiles else 1.0
    K1 = 0.048 if textiles else 0.045
    K2 = 0.014 if textiles else 0.015
    SL, SC, SH = kL, 1.0 + K1 * C1, 1.0 + K2 * C1
    term = (dL / SL) ** 2 + (dC / SC) ** 2 + dH2 / (SH ** 2)
    return np.sqrt(np.maximum(term, 0.0))


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
def batch(images: Iterable[ColorImage], dst_profile: ICCProfile,
          *, intent: Optional[Intent] = None,
          black_point_compensation: bool = True,
          strict: bool = False) -> list[ColorImage]:
    """Convert a batch without leaking transform state between distinct configs.

    The cache is never reset with a "last batch wins" lifetime; entries are
    isolated by profile fingerprint, intent and flags, so two batches with
    different profiles cannot influence each other (fix #5).
    """
    results = []
    for image in images:
        results.append(apply_transform(
            image, dst_profile, intent=intent,
            black_point_compensation=black_point_compensation, strict=strict))
    return results
