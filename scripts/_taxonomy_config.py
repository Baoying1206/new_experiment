"""
Minimal shared source of truth for Wei et al.'s operational CO/MG mapping.

Not yet imported by 19_taxonomy_robustness.py, 21_component_causal_decomposition.py,
or 27_dual_axis_diagnosis.py -- per explicit instruction, existing scripts are
NOT being refactored right now (each keeps its own inline copy). This module
exists so any NEW script (e.g. the future scripts/28_experiment3_profile_vs_behavior.py)
has one place to import from instead of adding a 4th independent copy, and so
that copy can be checked for consistency against the existing ones at import
time rather than assumed.

CANONICAL_CO_MECHS / CANONICAL_MG_MECHS are treated as authoritative here
because scripts/19_taxonomy_robustness.py is the oldest and most heavily-used
of the existing copies (per the user's explicit instruction to treat it as
canonical for now, not because it was independently re-derived).

REFERENCE_COPIES below are literal transcriptions of the mapping as it
currently appears in 19/21/27 (copied by hand, since those numbered scripts
can't be `import`ed normally -- their filenames start with a digit). If those
files' mappings are ever edited, these reference copies must be updated too,
or the assertions below will (correctly) start failing -- that's
intentional: it's a tripwire, not a redundant maintenance burden to ignore.
"""

CANONICAL_CO_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy']
CANONICAL_MG_MECHS = ['persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
CANONICAL_REAL_MECHS = CANONICAL_CO_MECHS + CANONICAL_MG_MECHS

WEI_LABEL = {m: 'CO' for m in CANONICAL_CO_MECHS}
WEI_LABEL.update({m: 'MG' for m in CANONICAL_MG_MECHS})

# Literal transcriptions of the mapping as currently written in each script,
# checked below -- see module docstring for why these are hand-copied
# instead of imported.
REFERENCE_COPIES = {
    '19_taxonomy_robustness.py': {
        'CO': ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy'],
        'MG': ['persona_roleplay', 'fictional_framing', 'encoding_obfuscation'],
    },
    '21_component_causal_decomposition.py (MECH_CATEGORY, full names)': {
        'CO': [m for m, cat in {
            'prefix_injection': 'competing_objectives', 'refusal_suppression': 'competing_objectives',
            'instruction_hierarchy': 'competing_objectives', 'persona_roleplay': 'mismatched_generalization',
            'fictional_framing': 'mismatched_generalization', 'encoding_obfuscation': 'mismatched_generalization',
        }.items() if cat == 'competing_objectives'],
        'MG': [m for m, cat in {
            'prefix_injection': 'competing_objectives', 'refusal_suppression': 'competing_objectives',
            'instruction_hierarchy': 'competing_objectives', 'persona_roleplay': 'mismatched_generalization',
            'fictional_framing': 'mismatched_generalization', 'encoding_obfuscation': 'mismatched_generalization',
        }.items() if cat == 'mismatched_generalization'],
    },
    '27_dual_axis_diagnosis.py': {
        'CO': ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy'],
        'MG': ['persona_roleplay', 'fictional_framing', 'encoding_obfuscation'],
    },
}


def _validate():
    all_mechs = CANONICAL_CO_MECHS + CANONICAL_MG_MECHS
    assert len(all_mechs) == 6, f"expected 6 mechanisms total, got {len(all_mechs)}"
    assert len(set(all_mechs)) == 6, f"expected 6 UNIQUE mechanisms, got duplicates: {all_mechs}"
    assert set(CANONICAL_CO_MECHS).isdisjoint(CANONICAL_MG_MECHS), "CO and MG overlap"

    for source_name, mapping in REFERENCE_COPIES.items():
        assert sorted(mapping['CO']) == sorted(CANONICAL_CO_MECHS), (
            f"{source_name}'s CO list {sorted(mapping['CO'])} does not match canonical "
            f"{sorted(CANONICAL_CO_MECHS)} -- the source file has been edited since this "
            f"reference copy was transcribed; update REFERENCE_COPIES in this file."
        )
        assert sorted(mapping['MG']) == sorted(CANONICAL_MG_MECHS), (
            f"{source_name}'s MG list {sorted(mapping['MG'])} does not match canonical "
            f"{sorted(CANONICAL_MG_MECHS)} -- the source file has been edited since this "
            f"reference copy was transcribed; update REFERENCE_COPIES in this file."
        )


_validate()  # runs at import time -- any script that imports this module gets the check for free
