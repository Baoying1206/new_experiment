"""
Shared language-set constants. Import this instead of hardcoding language
lists in new scripts, so the confirmatory/pilot distinction stays in one
place.

CONFIRMATORY_LANGUAGES: the 6-language formal experiment (2 per resource
tier), introduced when the pilot's original 9-language cross-lingual scope
was judged too large for a master's thesis given the 572-instruction (up
from 75) restart. de/ko/sw are excluded from the confirmatory run but their
data (templates, xling generation inputs, and all original 9-language
75-instruction pilot results) are kept -- not deleted, not overwritten.

PILOT_LANGUAGES: the original 9-language set. Existing scripts that already
hardcode this 9-language list (04, 06, 07, 08, 09, 14, 16, and their slurm
counterparts) operate on the original 75-instruction completions_{lang}.json
files and are intentionally left untouched -- they remain valid as
pilot/appendix evidence and this rescoping does not affect them.

Any NEW script analyzing the confirmatory (572/200-instruction) data should
import CONFIRMATORY_LANGUAGES from here rather than hardcoding a list, and
should print/log which language set + which input suffix (_full572 for en,
_xling for the rest) it actually used -- do not silently glob whatever
completions_*.json files happen to exist in an output directory, since both
old-pilot and new-confirmatory files now coexist there.
"""

PILOT_LANGUAGES = ['en', 'zh', 'de', 'ko', 'ar', 'th', 'yo', 'sw', 'am']

CONFIRMATORY_LANGUAGES = ['en', 'zh', 'ar', 'th', 'yo', 'am']
CONFIRMATORY_XLING_LANGUAGES = ['zh', 'ar', 'th', 'yo', 'am']  # CONFIRMATORY_LANGUAGES minus 'en'

EXCLUDED_FROM_CONFIRMATORY = ['de', 'ko', 'sw']  # kept as pilot_only, not deleted

PILOT_RESOURCE_TIERS = {
    'high': ['en', 'zh', 'de'],
    'medium': ['ko', 'ar', 'th'],
    'low': ['yo', 'sw', 'am'],
}

CONFIRMATORY_RESOURCE_TIERS = {
    'high': ['en', 'zh'],
    'medium': ['ar', 'th'],
    'low': ['yo', 'am'],
}
