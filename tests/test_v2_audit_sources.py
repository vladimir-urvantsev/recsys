"""Source attestation must distinguish binary metadata without hiding code."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_v2_results.py"
SPEC = importlib.util.spec_from_file_location("v2_audit_source_test", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_known_appledouble_does_not_change_v1_implementation(tmp_path):
    code = b"value = 1\n"
    (tmp_path / "data.py").write_bytes(code)
    (tmp_path / "._data.py").write_bytes(bytes.fromhex("0005160700020000") + b"metadata")
    implementation, metadata = AUDIT.v1_implementation_hashes(tmp_path)
    assert implementation == {"data.py": hashlib.sha256(code).hexdigest()}
    assert set(metadata) == {"._data.py"}


def test_filename_without_binary_metadata_header_cannot_hide_extra_code(tmp_path):
    (tmp_path / "data.py").write_text("value = 1\n")
    (tmp_path / "._extra.py").write_text("value = 2\n")
    implementation, metadata = AUDIT.v1_implementation_hashes(tmp_path)
    assert set(implementation) == {"data.py", "._extra.py"}
    assert not metadata
    with pytest.raises(AssertionError):
        AUDIT.assert_equal(implementation, {"data.py": implementation["data.py"]})


def test_real_source_change_still_fails_original_lock_comparison(tmp_path):
    path = tmp_path / "data.py"
    path.write_text("value = 1\n")
    original, _ = AUDIT.v1_implementation_hashes(tmp_path)
    path.write_text("value = 2\n")
    current, _ = AUDIT.v1_implementation_hashes(tmp_path)
    with pytest.raises(AssertionError):
        AUDIT.assert_equal(current, original)
