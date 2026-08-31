"""
Tests for scripts/48_archive_pre_test.py. Builds fake versions of every
file it archives, runs it, verifies: correct file count, matching
sha256 in the manifest, files/dir are actually read-only at the OS level,
a second run refuses (ArchiveAlreadyExistsError) rather than overwriting,
and a missing required file raises FileNotFoundError. No GPU, no torch.

Usage:
  python scripts/audits/audit_archive_pre_test_dry_run.py
"""
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from importlib import import_module

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
archive_mod = import_module('48_archive_pre_test')


def main():
    tmpdir = tempfile.mkdtemp()
    try:
        canonical_dir = os.path.join(tmpdir, 'canonical_v2')
        os.makedirs(canonical_dir, exist_ok=True)

        orig_models = dict(archive_mod.MODELS)
        archive_mod.MODELS.clear()
        archive_mod.MODELS[0] = ('FakeModel', '/fake')
        try:
            import argparse
            args = argparse.Namespace(output_path=tmpdir)

            for src in archive_mod.files_to_archive(canonical_dir):
                with open(src, 'w') as f:
                    json.dump({'fake': os.path.basename(src)}, f)

            archive_mod.main(args)
            archive_dir = os.path.join(canonical_dir, 'archive_pre_test')
            manifest_path = os.path.join(archive_dir, 'archive_manifest.json')
            with open(manifest_path) as f:
                manifest = json.load(f)

            expected_count = len(archive_mod.files_to_archive(canonical_dir))
            assert len(manifest['files']) == expected_count, (len(manifest['files']), expected_count)
            print(f"Test 1 PASSED: archived exactly {expected_count} files (matches files_to_archive for 1 model).")

            for entry in manifest['files']:
                path = os.path.join(archive_dir, entry['filename'])
                with open(path, 'rb') as f:
                    real_hash = hashlib.sha256(f.read()).hexdigest()
                assert real_hash == entry['sha256'], entry['filename']
                mode = stat.S_IMODE(os.stat(path).st_mode)
                assert mode == 0o444, f"{entry['filename']} mode={oct(mode)}, expected 0o444"
            dir_mode = stat.S_IMODE(os.stat(archive_dir).st_mode)
            assert dir_mode == 0o555, f"archive dir mode={oct(dir_mode)}, expected 0o555"
            print("Test 2 PASSED: every archived file's SHA-256 matches its manifest entry, "
                  "and every file (0o444) and the directory itself (0o555) are read-only at the OS level.")

            try:
                archive_mod.main(args)
                raise SystemExit("FAILED: expected ArchiveAlreadyExistsError on a second run")
            except archive_mod.ArchiveAlreadyExistsError as e:
                assert 'already exists' in str(e)
                print(f"Test 3 PASSED: a second run refuses to overwrite the existing archive: {str(e)[:70]}")
        finally:
            archive_mod.MODELS.clear()
            archive_mod.MODELS.update(orig_models)
            # restore write perms so tempdir cleanup doesn't fail
            archive_dir = os.path.join(canonical_dir, 'archive_pre_test')
            if os.path.isdir(archive_dir):
                os.chmod(archive_dir, 0o755)
                for fname in os.listdir(archive_dir):
                    os.chmod(os.path.join(archive_dir, fname), 0o644)

        # ---- Test 4: missing required file raises, in a fresh tmpdir ----
        tmpdir2 = tempfile.mkdtemp()
        try:
            canonical_dir2 = os.path.join(tmpdir2, 'canonical_v2')
            os.makedirs(canonical_dir2, exist_ok=True)
            archive_mod.MODELS.clear()
            archive_mod.MODELS[0] = ('FakeModel', '/fake')
            sources = archive_mod.files_to_archive(canonical_dir2)
            for src in sources[:-1]:  # deliberately skip the last file
                with open(src, 'w') as f:
                    json.dump({'fake': True}, f)
            try:
                archive_mod.main(argparse.Namespace(output_path=tmpdir2))
                raise SystemExit("FAILED: expected FileNotFoundError for a missing required file")
            except FileNotFoundError as e:
                assert 'missing' in str(e)
                print(f"Test 4 PASSED: a missing required input file raises before any copying: {str(e)[:70]}")
        finally:
            archive_mod.MODELS.clear()
            archive_mod.MODELS.update(orig_models)
            shutil.rmtree(tmpdir2, ignore_errors=True)

        print()
        print("ALL ARCHIVE-PRE-TEST TESTS PASSED.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
