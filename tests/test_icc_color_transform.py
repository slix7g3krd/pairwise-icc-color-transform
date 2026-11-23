import pytest
import imaging.icc_color_transform as mod

CASES = [
    ("profile_fallback", (1,)),
    ("intent_match", (1,)),
    ("bit_depth", (1,)),
    ("embed_mismatch", (1,)),
    ("cache_lifetime", (1,)),
    ("determinism", (1,)),
    ("transform", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("gamut_map", (1,)),
    ("metrics", (1,)),
    ("black_point", (1,)),
    ("batch", (1,)),
    ("tag", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
