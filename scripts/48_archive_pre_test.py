"""
Copies the validation report, the frozen test config, and their full
per-model provenance chain (no_defence_baseline, alpha_freeze, and for
fixed_wei/adaptive: join_audit + summary) into a read-only archive
directory BEFORE any test data is read. Computes SHA-256 of every
archived file and writes a manifest recording path + hash + size.

Refuses to run if the archive directory already exists and is non-empty
-- never silently overwrites a prior archive (a one-shot held-out test
phase should have exactly one archive snapshot). After copying, every
archived file is chmod'd read-only (0o444) and the directory itself
0o555, so a later accidental write attempt fails at the OS level too,
not just by convention.

Usage:
  python scripts/48_archive_pre_test.py --output_path output
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import _defence_metrics as dm  # torch-free

MODELS = dm.MODEL_PATHS
ALPHA_ELIGIBLE_METHODS = ('fixed_wei', 'adaptive')


class ArchiveAlreadyExistsError(RuntimeError):
    pass


def sha256_of_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def files_to_archive(canonical_dir):
    files = ['experiment3_validation_report.json', 'experiment3_defence_frozen_config.json']
    for model_idx in sorted(MODELS.keys()):
        model_alias, _ = MODELS[model_idx]
        files.append(f'experiment3_no_defence_baseline_{model_alias}.json')
        files.append(f'experiment3_alpha_freeze_{model_alias}.json')
        for method in ALPHA_ELIGIBLE_METHODS:
            files.append(f'experiment3_validation_join_audit_{model_alias}_{method}.json')
            files.append(f'experiment3_validation_summary_{model_alias}_{method}.json')
    return [os.path.join(canonical_dir, f) for f in files]


def main(args):
    canonical_dir = os.path.join(args.output_path, 'canonical_v2')
    archive_dir = os.path.join(canonical_dir, 'archive_pre_test')

    if os.path.exists(archive_dir) and os.listdir(archive_dir):
        raise ArchiveAlreadyExistsError(
            f"{archive_dir} already exists and is non-empty -- refusing to overwrite a prior "
            f"archive. A one-shot held-out test phase should have exactly one archive snapshot."
        )
    os.makedirs(archive_dir, exist_ok=True)

    sources = files_to_archive(canonical_dir)
    missing = [p for p in sources if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} required file(s) missing, cannot archive: {missing}")

    manifest = {'archived_utc': datetime.now(timezone.utc).isoformat(),
                'git_commit': dm.git_commit_hash(), 'files': []}
    for src in sources:
        fname = os.path.basename(src)
        dst = os.path.join(archive_dir, fname)
        shutil.copy2(src, dst)
        src_hash, dst_hash = sha256_of_file(src), sha256_of_file(dst)
        assert src_hash == dst_hash, f"copy corrupted: {fname}"
        manifest['files'].append({'filename': fname, 'sha256': dst_hash, 'size_bytes': os.path.getsize(dst)})
        os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 0o444, read-only

    manifest_path = os.path.join(archive_dir, 'archive_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    os.chmod(manifest_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 0o444, read-only

    os.chmod(archive_dir, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)  # 0o555

    print(f"Archived {len(manifest['files'])} files to {archive_dir} (read-only, 0o444 each; dir 0o555).")
    for entry in manifest['files']:
        print(f"  {entry['filename']}: sha256={entry['sha256'][:16]}...  ({entry['size_bytes']} bytes)")
    print(f"\nManifest: {manifest_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    args = parser.parse_args()
    main(args)
