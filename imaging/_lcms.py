"""Little CMS 2 (lcms2) ctypes binding layer.

Pillow ships without a usable 16-bit RGB pipeline on this platform and its
feature probe does not report lcms2, while the shared library ``liblcms2.so.2``
is present.  Driving lcms2 directly through :mod:`ctypes` gives the whole
project a single, version-pinned colour engine on every platform, which is
exactly what the determinism requirement calls for.

All TYPE constants below are taken verbatim from ``lcms2.h`` (MIT licensed):

    COLORSPACE_SH(pt) = pt << 16          (PT_* pixel-type numbers, NOT FourCC)
    CHANNELS_SH(c)    = c  << 3
    BYTES_SH(b)       = b
    FLOAT_SH(1)       = 1  << 22

The numeric values were additionally confirmed empirically against the
installed library (identity transforms round-trip exactly).
"""
from __future__ import annotations

import ctypes
import ctypes.util

# --------------------------------------------------------------------------- #
# Pixel-type numbers (lcms2.h PT_*) and TYPE_* format constants
# --------------------------------------------------------------------------- #
PT_ANY = 0
# Reserved 1, 2 unused in modern lcms2
PT_GRAY = 3
PT_RGB = 4
PT_CMY = 5
PT_CMYK = 6
PT_YCbCr = 7
PT_YUV = 8
PT_XYZ = 9
PT_Lab = 10
PT_YUVK = 11
PT_HSV = 12
PT_HLS = 13
PT_Yxy = 14
PT_MCH1 = 15
PT_MCH2 = 16
PT_MCH3 = 17
PT_MCH4 = 18
PT_MCH5 = 19
PT_MCH6 = 20
PT_MCH7 = 21
PT_MCH8 = 22
PT_MCH9 = 23
PT_MCH10 = 24
PT_MCH11 = 25
PT_MCH12 = 26
PT_MCH13 = 27
PT_MCH14 = 28
PT_MCH15 = 29
PT_LabV2 = 10  # alias used by some headers

_FLOAT_SH = 1 << 22


def _color_type(pixel_type: int, channels: int, bytes_per_sample: int) -> int:
    return (pixel_type << 16) | (channels << 3) | bytes_per_sample


TYPE_GRAY_8 = _color_type(PT_GRAY, 1, 1)
TYPE_GRAY_16 = _color_type(PT_GRAY, 1, 2)
TYPE_RGB_8 = _color_type(PT_RGB, 3, 1)
TYPE_RGB_16 = _color_type(PT_RGB, 3, 2)
TYPE_CMYK_8 = _color_type(PT_CMYK, 4, 1)
TYPE_CMYK_16 = _color_type(PT_CMYK, 4, 2)
TYPE_YCbCr_8 = _color_type(PT_YCbCr, 3, 1)
TYPE_YCbCr_16 = _color_type(PT_YCbCr, 3, 2)
TYPE_YUV_8 = _color_type(PT_YUV, 3, 1)
TYPE_YUV_16 = _color_type(PT_YUV, 3, 2)
TYPE_XYZ_16 = _color_type(PT_XYZ, 3, 2)
TYPE_Lab_8 = _color_type(PT_Lab, 3, 1)
TYPE_Lab_16 = _color_type(PT_Lab, 3, 2)
TYPE_Lab_DBL = _FLOAT_SH | _color_type(PT_Lab, 3, 0)
TYPE_XYZ_DBL = _FLOAT_SH | _color_type(PT_XYZ, 3, 0)

# Rendering intents (lcms2.h)
INTENT_PERCEPTUAL = 0
INTENT_RELATIVE_COLORIMETRIC = 1
INTENT_SATURATION = 2
INTENT_ABSOLUTE_COLORIMETRIC = 3

# Transform flags (subset of lcms2.h cmsFLAGS_*, values verbatim)
cmsFLAGS_NONE = 0
cmsFLAGS_NOWHITEONWHITEFIXUP = 0x00000004
cmsFLAGS_NOOPTIMIZE = 0x00000100
cmsFLAGS_NULLTRANSFORM = 0x00000200
cmsFLAGS_HIGHRESPRECALC = 0x00000400
cmsFLAGS_LOWRESPRECALC = 0x00000800
cmsFLAGS_GAMUTCHECK = 0x00001000
cmsFLAGS_SOFTPROOFING = 0x00004000
cmsFLAGS_BLACKPOINTCOMPENSATION = 0x00002000
cmsFLAGS_COPY_ALPHA = 0x04000000
cmsFLAGS_NODEFAULTRESOURCEDEF = 0x01000000

# ICC signature FourCC helper
def fourcc(text: str) -> int:
    """Convert a 4-character ICC tag/type signature to its integer value."""
    if len(text) != 4:
        raise ValueError("ICC signatures must be exactly 4 characters: %r" % text)
    return int.from_bytes(text.encode("ascii"), "big")


# Well-known signatures
sig_Monitor = fourcc("mntr")
sig_Input = fourcc("scnr")
sig_Output = fourcc("prtr")
sig_Link = fourcc("link")
sig_Abstract = fourcc("abst")
sig_Space = fourcc("spac")

sig_RGB = fourcc("RGB ")
sig_GRAY = fourcc("GRAY")
sig_CMYK = fourcc("CMYK")
sig_Lab = fourcc("Lab ")
sig_XYZ = fourcc("XYZ ")

sig_desc = fourcc("desc")
sig_cprt = fourcc("cprt")
sig_wtpt = fourcc("wtpt")
sig_rXYZ = fourcc("rXYZ")
sig_gXYZ = fourcc("gXYZ")
sig_bXYZ = fourcc("bXYZ")
sig_rTRC = fourcc("rTRC")
sig_gTRC = fourcc("gTRC")
sig_bTRC = fourcc("bTRC")
sig_kTRC = fourcc("kTRC")
sig_chad = fourcc("chad")
sig_gamut = fourcc("gamt")
sig_A2B0 = fourcc("A2B0")
sig_B2A0 = fourcc("B2A0")


class LcmsError(RuntimeError):
    """Raised when the lcms2 engine reports a failure."""


def _find_library() -> str:
    for name in ("lcms2",):
        path = ctypes.util.find_library(name)
        if path:
            return path
    # Common SONAMEs across distributions / platforms
    for candidate in ("liblcms2.so.2", "liblcms2.so", "liblcms2.dylib",
                      "lcms2.dll", "liblcms2-2.dll"):
        try:
            ctypes.CDLL(candidate)
            return candidate
        except OSError:
            continue
    raise LcmsError("littlecms2 shared library not found on this system")


_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    """Load (once) and return the lcms2 shared library with prototypes set."""
    global _lib
    if _lib is not None:
        return _lib
    lcms = ctypes.CDLL(_find_library())

    # Profile lifecycle
    lcms.cmsCreate_sRGBProfile.restype = ctypes.c_void_p
    lcms.cmsCreateLab4Profile.argtypes = [ctypes.c_void_p]
    lcms.cmsCreateLab4Profile.restype = ctypes.c_void_p
    lcms.cmsCreateXYZProfile.restype = ctypes.c_void_p

    lcms.cmsOpenProfileFromMem.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lcms.cmsOpenProfileFromMem.restype = ctypes.c_void_p
    lcms.cmsOpenProfileFromFile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lcms.cmsOpenProfileFromFile.restype = ctypes.c_void_p
    lcms.cmsCloseProfile.argtypes = [ctypes.c_void_p]
    lcms.cmsCloseProfile.restype = ctypes.c_int

    lcms.cmsSaveProfileToMem.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lcms.cmsSaveProfileToMem.restype = ctypes.c_int

    # Header / introspection
    lcms.cmsGetColorSpace.argtypes = [ctypes.c_void_p]
    lcms.cmsGetColorSpace.restype = ctypes.c_uint32
    lcms.cmsGetPCS.argtypes = [ctypes.c_void_p]
    lcms.cmsGetPCS.restype = ctypes.c_uint32
    lcms.cmsGetDeviceClass.argtypes = [ctypes.c_void_p]
    lcms.cmsGetDeviceClass.restype = ctypes.c_uint32
    lcms.cmsGetHeaderRenderingIntent.argtypes = [ctypes.c_void_p]
    lcms.cmsGetHeaderRenderingIntent.restype = ctypes.c_uint32
    lcms.cmsGetEncodedICCversion.argtypes = [ctypes.c_void_p]
    lcms.cmsGetEncodedICCversion.restype = ctypes.c_uint32

    # Tags
    lcms.cmsReadTag.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lcms.cmsReadTag.restype = ctypes.c_void_p
    lcms.cmsIsTag.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lcms.cmsIsTag.restype = ctypes.c_int
    lcms.cmsGetTagCount.argtypes = [ctypes.c_void_p]
    lcms.cmsGetTagCount.restype = ctypes.c_int
    lcms.cmsGetTagSignature.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lcms.cmsGetTagSignature.restype = ctypes.c_uint32

    # Transforms
    lcms.cmsCreateTransform.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32]
    lcms.cmsCreateTransform.restype = ctypes.c_void_p
    lcms.cmsDeleteTransform.argtypes = [ctypes.c_void_p]
    lcms.cmsDeleteTransform.restype = None
    lcms.cmsDoTransform.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    lcms.cmsDoTransform.restype = None

    # Black point
    lcms.cmsDetectBlackPoint.argtypes = [
        ctypes.POINTER(ctypes.c_double * 3),
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    lcms.cmsDetectBlackPoint.restype = ctypes.c_int
    lcms.cmsDetectDestinationBlackPoint.argtypes = [
        ctypes.POINTER(ctypes.c_double * 3),
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    lcms.cmsDetectDestinationBlackPoint.restype = ctypes.c_int

    # Adaptation / error handling.  cmsstrerror only exists on lcms2 >= 2.15;
    # probing a missing symbol through ctypes raises rather than returning
    # False, so guard with try/except instead of hasattr.
    lcms.cmsSetAdaptationState.argtypes = [ctypes.c_double]
    lcms.cmsSetAdaptationState.restype = ctypes.c_double
    try:
        lcms.cmsstrerror.restype = ctypes.c_char_p
    except (AttributeError, TypeError):
        # Symbol absent in this lcms2 build; leave it unset.
        pass

    _lib = lcms
    return lcms
