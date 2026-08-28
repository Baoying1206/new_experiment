"""
Runtime dynamic loader for the canonical V2 taxonomy config, read fresh from
templates/templates_en.json every call -- NOT a hand-copied/hardcoded
Python constant like scripts/_taxonomy_config.py (which exists for the OLD
scripts 19/30/32 and is intentionally left as-is). Only scripts/33_
canonical_taxonomy_geometry.py (and any future V2 script) should import
this; 19/30/32 are not touched.

All the assertions below run every call (not just at import time), so a
templates_en.json edited between calls in the same process would still be
caught, not just checked once and cached.
"""
import json
import os

SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_TEMPLATES_PATH = os.path.join(SCRIPT_DIR, '..', 'templates', 'templates_en.json')


def load_taxonomy_v2(path=None):
    path = path or DEFAULT_TEMPLATES_PATH
    with open(path, encoding='utf-8') as f:
        t = json.load(f)

    taxonomy_version = t.get('taxonomy_version')
    assert taxonomy_version == 'wei_canonical_v2', (
        f"{path}'s taxonomy_version is {taxonomy_version!r}, expected 'wei_canonical_v2' -- "
        f"refusing to run the V2 canonical analysis against a config that isn't marked as such."
    )

    active_mechanisms = t.get('active_mechanisms')
    assert active_mechanisms is not None, f"{path} has no 'active_mechanisms' field"
    assert len(active_mechanisms) == 6, (
        f"{path}'s active_mechanisms has {len(active_mechanisms)} entries, expected exactly 6: "
        f"{active_mechanisms}"
    )
    assert len(set(active_mechanisms)) == 6, (
        f"{path}'s active_mechanisms has duplicate entries: {active_mechanisms}"
    )

    mech_cats = t.get('mechanism_categories')
    assert mech_cats is not None, f"{path} has no 'mechanism_categories' field"
    co_mechs = mech_cats.get('competing_objectives')
    mg_mechs = mech_cats.get('mismatched_generalization')
    assert co_mechs is not None and mg_mechs is not None, (
        f"{path}'s mechanism_categories is missing competing_objectives or "
        f"mismatched_generalization: {mech_cats}"
    )
    assert len(co_mechs) == 3, f"competing_objectives has {len(co_mechs)} entries, expected 3: {co_mechs}"
    assert len(mg_mechs) == 3, f"mismatched_generalization has {len(mg_mechs)} entries, expected 3: {mg_mechs}"
    assert set(co_mechs).isdisjoint(mg_mechs), (
        f"competing_objectives and mismatched_generalization overlap: "
        f"{set(co_mechs) & set(mg_mechs)}"
    )
    assert set(active_mechanisms) == set(co_mechs) | set(mg_mechs), (
        f"active_mechanisms {sorted(active_mechanisms)} != CO∪MG "
        f"{sorted(set(co_mechs) | set(mg_mechs))}"
    )

    templates = t.get('templates', {})
    mechanism_of = {}
    for m in active_mechanisms:
        assert m in templates, f"active_mechanisms lists {m!r} but it's not in templates: {path}"
        tpl_mech = templates[m].get('mechanism')
        expected = 'competing_objectives' if m in co_mechs else 'mismatched_generalization'
        assert tpl_mech == expected, (
            f"templates[{m!r}]['mechanism'] = {tpl_mech!r}, but mechanism_categories says it "
            f"should be {expected!r} -- template-level and category-mapping-level taxonomy "
            f"disagree, refusing to proceed with an internally inconsistent config."
        )
        mechanism_of[m] = tpl_mech

    for stale in ('instruction_hierarchy', 'fictional_framing'):
        assert stale not in active_mechanisms, (
            f"{stale!r} (pre-correction stand-in, not a real Wei et al. attack) is still in "
            f"active_mechanisms -- {path} was not correctly updated to the V2 taxonomy."
        )

    return {
        'taxonomy_version': taxonomy_version,
        'active_mechanisms': list(active_mechanisms),
        'CO_mechs': list(co_mechs),
        'MG_mechs': list(mg_mechs),
        'mechanism_of': mechanism_of,
        'config_path': os.path.abspath(path),
    }
