"""
Shared, pure-Python (no torch/transformers, no subprocess side effects beyond
a single read-only `git show`) utilities for verifying that
templates/templates_context_v1.json's PROMPT CONTENT -- the text that
actually gets rendered into a model input -- has not silently drifted
between two points in the repo's history, independent of metadata-only
changes (status, notes, timestamps, reviewer fields) that never reach the
model.

Motivation: audit_context_templates_dry_run.py's check 19 pins the formal
token-length audit's OWN recorded source_template_sha256 to a fixed value
(the whole-file hash at audit time) -- that is a provenance pin on the
audit artifact, not a proof that today's 16 template texts are identical to
what was audited. This module provides the actual content-level proof.

CONTENT_HASH_SCHEMA_VERSION documents exactly what is (and is not) hashed,
so a future change to the schema is detectable rather than silently
comparing incompatible hashes.
"""
import hashlib
import json
import subprocess

CONTENT_HASH_SCHEMA_VERSION = 'context_template_content_v1'

INCLUDED_FIELDS = [
    'families.<name> (the 4 family names themselves)',
    'families.<name>.variants.<id> (v1/v2/v3 -> full template text)',
    'families.<name>.family_specific_neutral_control (full template text)',
]
EXCLUDED_FIELDS = [
    'status',
    'source_note',
    'scope_note',
    'notes (all sub-keys)',
    'taxonomy_version',
    'families.<name>.operational_definition',
    'families.<name>.n_positive_variants',
    'families.<name>.neutral_control_type',
    'any timestamp or reviewer/provenance field',
]


def canonical_template_content_payload(data):
    """Extracts ONLY the fields in INCLUDED_FIELDS from a parsed
    templates_context_v1.json dict. Deliberately does not sort or reorder
    anything itself -- json.dumps(..., sort_keys=True) at hash time is what
    makes key order irrelevant; this function only controls which VALUES
    are included."""
    return {
        fam_name: {
            'variants': dict(fam['variants']),
            'family_specific_neutral_control': fam['family_specific_neutral_control'],
        }
        for fam_name, fam in data['families'].items()
    }


def template_content_sha256(data):
    """SHA-256 of the canonical (sort_keys=True, compact separators, UTF-8,
    non-ASCII-escaped) JSON serialization of canonical_template_content_payload.
    This is deterministic regardless of key insertion order in the source
    file, and changes if and only if a family name, variant id, or the
    literal text of any of the 16 templates changes."""
    payload = canonical_template_content_payload(data)
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def diff_json_paths(old, new, prefix=''):
    """Recursively compares two parsed JSON values, returning a sorted list
    of dotted paths (with an ' (added)'/' (removed)' suffix where a key
    exists on only one side) at which the two differ. Lists are compared by
    equality only (not element-wise diffed) -- sufficient for this module's
    use (templates_context_v1.json has no lists among its content fields;
    any list-valued difference is reported as a single changed path)."""
    paths = []
    if isinstance(old, dict) and isinstance(new, dict):
        keys = set(old.keys()) | set(new.keys())
        for k in sorted(keys):
            op = f"{prefix}.{k}" if prefix else k
            if k not in old:
                paths.append(op + ' (added)')
            elif k not in new:
                paths.append(op + ' (removed)')
            else:
                paths.extend(diff_json_paths(old[k], new[k], op))
    else:
        if old != new:
            paths.append(prefix)
    return sorted(paths)


def get_git_show_text(commit, path, cwd):
    """Read-only `git show <commit>:<path>`. Never checks out, never
    modifies the working tree. Returns (text, None) on success or
    (None, error_message) on failure (bad commit, path not found at that
    commit, git not installed, not a git repo, etc.) -- callers MUST treat
    the error case as PROVENANCE_UNVERIFIED, never as an implicit pass."""
    try:
        result = subprocess.run(
            ['git', 'show', f'{commit}:{path}'],
            cwd=cwd, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"git show subprocess failed to run: {type(e).__name__}: {e}"
    if result.returncode != 0:
        return None, f"git show {commit}:{path} exited {result.returncode}: {result.stderr.strip()}"
    return result.stdout, None
