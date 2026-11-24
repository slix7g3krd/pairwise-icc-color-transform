"""ICC colour transform domain model.

The model is deliberately plain (no third-party image objects) so the same
code runs unchanged on Linux, macOS and Windows.  Pixel data is always held in
a NumPy array at the source bit depth; lcms2 is the single conversion engine
on every platform.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from . import _lcms
from ._lcms import (
    LcmsError,
    lib,
    sig_RGB, sig_GRAY, sig_CMYK, sig_Lab, sig_XYZ,
    sig_desc, sig_wtpt,
)


class Intent(IntEnum):
    """ICC rendering intents (values mandated by the ICC specification)."""

    PERCEPTUAL = 0
    RELATIVE_COLORIMETRIC = 1
    SATURATION = 2
    ABSOLUTE_COLORIMETRIC = 3

    @classmethod
    def coerce(cls, value) -> "Intent":
        if isinstance(value, Intent):
            return value
        if value is None:
            return cls.RELATIVE_COLORIMETRIC
        if isinstance(value, str):
            key = value.strip().lower().replace("-", "_").replace(" ", "_")
            aliases = {
                "perceptual": cls.PERCEPTUAL,
                "photo": cls.PERCEPTUAL,
                "relative": cls.RELATIVE_COLORIMETRIC,
                "relative_colorimetric": cls.RELATIVE_COLORIMETRIC,
                "relcol": cls.RELATIVE_COLORIMETRIC,
                "saturation": cls.SATURATION,
                "graphic": cls.SATURATION,
                "absolute": cls.ABSOLUTE_COLORIMETRIC,
                "absolute_colorimetric": cls.ABSOLUTE_COLORIMETRIC,
                "abscol": cls.ABSOLUTE_COLORIMETRIC,
            }
            if key not in aliases:
                raise ValueError("unknown rendering intent: %r" % value)
            return aliases[key]
        ivalue = int(value)
        if ivalue not in cls._value2member_map_:
            raise ValueError("rendering intent must be 0..3, got %r" % value)
        return cls(ivalue)


# Intents the ICC spec reserves and lcms2 actually implements.
KNOWN_INTENTS = frozenset(int(i) for i in Intent)

# Default when a file declares nothing: relative colorimetric keeps the
# media white point mapped to the destination white, preserving shadow detail
# instead of crushing it the way absolute colorimetric does.
DEFAULT_INTENT = Intent.RELATIVE_COLORIMETRIC


class ColorSpace(IntEnum):
    GRAY = sig_GRAY
    RGB = sig_RGB
    CMYK = sig_CMYK
    Lab = sig_Lab
    XYZ = sig_XYZ

    @property
    def channels(self) -> int:
        return {ColorSpace.GRAY: 1, ColorSpace.RGB: 3, ColorSpace.CMYK: 4,
                ColorSpace.Lab: 3, ColorSpace.XYZ: 3}[self]

    @classmethod
    def from_signature(cls, sig: int) -> Optional["ColorSpace"]:
        for member in cls:
            if int(member) == sig:
                return member
        return None


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AuditEvent:
    """One immutable entry in a transform's audit trail."""

    code: str
    detail: str
    profile_fingerprint: Optional[str] = None

    def to_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail,
                "profile_fingerprint": self.profile_fingerprint}


# --------------------------------------------------------------------------- #
# Colour image
# --------------------------------------------------------------------------- #
@dataclass
class ColorImage:
    """Pixel container plus the colour metadata needed to interpret it.

    ``pixels`` layout is ``(height, width, channels)``.  The dtype records the
    real bit depth (uint8 / uint16 / uint32 / float) and is preserved through
    every transform.
    """

    pixels: np.ndarray
    color_space: ColorSpace
    declared_profile: Optional["ICCProfile"] = None
    embedded_profile: Optional["ICCProfile"] = None
    declared_intent: Optional[Intent] = None
    source_name: str = "<memory>"
    audit: list = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.pixels, np.ndarray):
            self.pixels = np.asarray(self.pixels)
        if self.pixels.ndim == 2:
            self.pixels = self.pixels[..., np.newaxis]
        if self.pixels.ndim != 3:
            raise ValueError("pixels must have shape (height, width, channels)")
        if not isinstance(self.color_space, ColorSpace):
            self.color_space = ColorSpace(self.color_space)
        if self.pixels.shape[2] != self.color_space.channels:
            raise ValueError(
                "pixel channel count %d does not match colour space %s (%d)"
                % (self.pixels.shape[2], self.color_space.name,
                   self.color_space.channels))
        if self.declared_intent is not None:
            self.declared_intent = Intent.coerce(self.declared_intent)

    # -- bit depth ---------------------------------------------------------- #
    @property
    def bits_per_sample(self) -> int:
        kind = self.pixels.dtype
        if kind == np.uint8:
            return 8
        if kind == np.uint16:
            return 16
        if kind == np.uint32:
            return 32
        if kind.kind == "f":
            return kind.itemsize * 8
        raise ValueError("unsupported pixel dtype %s" % kind)

    @property
    def is_high_bit_depth(self) -> bool:
        return self.bits_per_sample > 8

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def pixel_count(self) -> int:
        return self.height * self.width

    def record(self, code: str, detail: str,
               fingerprint: Optional[str] = None) -> None:
        self.audit.append(AuditEvent(code, detail, fingerprint))


# --------------------------------------------------------------------------- #
# ICC profile wrapper
# --------------------------------------------------------------------------- #
class ICCProfile:
    """Owning wrapper around an lcms2 profile handle and its raw ICC bytes."""

    def __init__(self, handle: int, data: bytes, *, owns_handle: bool = True,
                 keepalive: Optional[bytes] = None):
        self._handle = int(handle)
        self._data = bytes(data)
        self._owns = owns_handle
        # lcms opens profiles in place, so the source bytes must outlive the
        # handle.  We retain a reference here for the object's whole lifetime.
        self._keepalive = keepalive if keepalive is not None else data
        self._fingerprint: Optional[str] = None

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_bytes(cls, data: bytes) -> "ICCProfile":
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("ICC profile data must be bytes-like")
        raw = bytes(data)
        if len(raw) < 128 or raw[36:40] != b"acsp":
            raise ValueError("data is not a valid ICC profile (missing 'acsp')")
        lcms = lib()
        handle = lcms.cmsOpenProfileFromMem(raw, len(raw))
        if not handle:
            raise LcmsError("lcms2 rejected the ICC profile data")
        return cls(handle, raw)

    @classmethod
    def from_file(cls, path: str) -> "ICCProfile":
        with open(path, "rb") as fh:
            return cls.from_bytes(fh.read())

    @classmethod
    def srgb(cls) -> "ICCProfile":
        lcms = lib()
        handle = lcms.cmsCreate_sRGBProfile()
        if not handle:
            raise LcmsError("cmsCreate_sRGBProfile failed")
        return cls(handle, _serialize(handle), owns_handle=True)

    @classmethod
    def lab(cls) -> "ICCProfile":
        lcms = lib()
        handle = lcms.cmsCreateLab4Profile(None)
        if not handle:
            raise LcmsError("cmsCreateLab4Profile failed")
        return cls(handle, _serialize(handle), owns_handle=True)

    @classmethod
    def xyz(cls) -> "ICCProfile":
        lcms = lib()
        handle = lcms.cmsCreateXYZProfile()
        if not handle:
            raise LcmsError("cmsCreateXYZProfile failed")
        return cls(handle, _serialize(handle), owns_handle=True)

    # -- introspection ------------------------------------------------------ #
    @property
    def handle(self) -> int:
        return self._handle

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def color_space(self) -> Optional[ColorSpace]:
        return ColorSpace.from_signature(lib().cmsGetColorSpace(self._handle))

    @property
    def pcs(self) -> Optional[ColorSpace]:
        return ColorSpace.from_signature(lib().cmsGetPCS(self._handle))

    @property
    def device_class(self) -> int:
        return int(lib().cmsGetDeviceClass(self._handle))

    @property
    def header_intent(self) -> Intent:
        return Intent(lib().cmsGetHeaderRenderingIntent(self._handle))

    def has_tag(self, signature: int) -> bool:
        return bool(lib().cmsIsTag(self._handle, int(signature)))

    def read_xyz_tag(self, signature: int):
        ptr = lib().cmsReadTag(self._handle, int(signature))
        if not ptr:
            return None
        xyz = ctypes_cast_xyz(ptr)
        return tuple(xyz)

    def white_point(self):
        return self.read_xyz_tag(sig_wtpt)

    def description(self) -> str:
        # 'desc' is an mluc/mlu structure; parse the profile description via
        # the ASCII text embedded in the tag when available, else fall back to
        # a deterministic placeholder derived from the fingerprint.
        ptr = lib().cmsReadTag(self._handle, sig_desc)
        if ptr:
            text = _read_mlu_or_text(ptr)
            if text:
                return text
        return "profile-" + self.fingerprint[:12]

    def black_point(self, intent: Intent = DEFAULT_INTENT):
        lcms = lib()
        bp = _lcms.ctypes.c_double * 3
        out = bp()
        ok = lcms.cmsDetectBlackPoint(
            _lcms.ctypes.byref(out), self._handle, int(intent), 0)
        if not ok:
            return (0.0, 0.0, 0.0)
        return tuple(out)

    # -- identity ----------------------------------------------------------- #
    @property
    def fingerprint(self) -> str:
        """Stable content fingerprint used to isolate transform cache entries."""
        if self._fingerprint is None:
            digest = hashlib.sha256()
            # The header's creation-date bytes (24:36) and profile ID (84:100)
            # are excluded so byte-identical colour pipelines share a fingerprint
            # even when re-serialised with a fresh timestamp.
            data = self._data
            digest.update(data[:24])
            digest.update(b"\x00" * 12)          # blank date/time
            digest.update(data[36:84])
            digest.update(b"\x00" * 16)          # blank profile id
            digest.update(data[100:])
            self._fingerprint = digest.hexdigest()
        return self._fingerprint

    @property
    def profile_id(self) -> bytes:
        return self._data[84:100]

    def close(self) -> None:
        if self._owns and self._handle:
            lib().cmsCloseProfile(self._handle)
            self._handle = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        cs = self.color_space
        return "ICCProfile(cs=%s, size=%d, fp=%s)" % (
            cs.name if cs else "?", self.size, self.fingerprint[:10])


# --------------------------------------------------------------------------- #
# Helpers (module-private)
# --------------------------------------------------------------------------- #
def _serialize(handle: int) -> bytes:
    lcms = lib()
    size = _lcms.ctypes.c_size_t(0)
    if not lcms.cmsSaveProfileToMem(handle, None, _lcms.ctypes.byref(size)):
        raise LcmsError("cmsSaveProfileToMem size query failed")
    buf = _lcms.ctypes.create_string_buffer(size.value)
    if not lcms.cmsSaveProfileToMem(handle, buf, _lcms.ctypes.byref(size)):
        raise LcmsError("cmsSaveProfileToMem write failed")
    return bytes(buf.raw[: size.value])


def ctypes_cast_xyz(ptr: int):
    return _lcms.ctypes.cast(
        ptr, _lcms.ctypes.POINTER(_lcms.ctypes.c_double * 3)).contents


def _read_mlu_or_text(ptr: int) -> str:
    """Best-effort extraction of a description tag's first ASCII string.

    lcms returns a pointer to an ``cmsMLU`` for 'desc'.  Walking that struct
    through ctypes is fragile across versions, so we scan the tag's backing
    memory for a printable run instead.  The result is used only for human
    readable labels, never for colour math.
    """
    try:
        # cmsMLU first entries begin after a small header; scan up to 1 KiB.
        chunk = _lcms.ctypes.string_at(ptr, 1024)
    except Exception:
        return ""
    run = bytearray()
    for byte in chunk:
        if 32 <= byte < 127:
            run.append(byte)
        else:
            if len(run) >= 4:
                return run.decode("ascii", "ignore")
            run = bytearray()
    return run.decode("ascii", "ignore") if len(run) >= 4 else ""
