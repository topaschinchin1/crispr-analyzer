"""
Multi-nuclease support for all major CRISPR systems.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


IUPAC = {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
    'R': '[AG]', 'Y': '[CT]', 'S': '[GC]', 'W': '[AT]',
    'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]',
    'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]',
}


def pam_to_regex(pam: str) -> re.Pattern:
    """Convert an IUPAC PAM string to a compiled regex."""
    pattern = ''.join(IUPAC.get(c.upper(), c) for c in pam)
    return re.compile(pattern, re.IGNORECASE)


@dataclass
class NucleaseProfile:
    name: str
    family: str
    pam: str
    cut_offset: int  # relative to PAM-proximal end of protospacer
    pam_side: str = '3prime'  # '3prime' or '5prime'
    stagger: int = 0  # staggered cut (Cas12a = 4-5nt)
    editing_window: Optional[tuple] = None  # for base/prime editors
    max_insertion: Optional[int] = None  # for prime editors

    @property
    def pam_regex(self) -> re.Pattern:
        return pam_to_regex(self.pam)


# ---------------------------------------------------------------------------
# Built-in nuclease library
# ---------------------------------------------------------------------------

NUCLEASE_LIBRARY: dict[str, dict[str, NucleaseProfile]] = {
    'Cas9': {
        'SpCas9': NucleaseProfile(
            name='SpCas9', family='Cas9', pam='NGG',
            cut_offset=-3, pam_side='3prime',
        ),
        'SaCas9': NucleaseProfile(
            name='SaCas9', family='Cas9', pam='NNGRRT',
            cut_offset=-3, pam_side='3prime',
        ),
        'FnCas9': NucleaseProfile(
            name='FnCas9', family='Cas9', pam='NGG',
            cut_offset=-3, pam_side='3prime',
        ),
        'xCas9': NucleaseProfile(
            name='xCas9', family='Cas9', pam='NGN',
            cut_offset=-3, pam_side='3prime',
        ),
        'SpCas9-NG': NucleaseProfile(
            name='SpCas9-NG', family='Cas9', pam='NG',
            cut_offset=-3, pam_side='3prime',
        ),
    },
    'Cas12a': {
        'AsCas12a': NucleaseProfile(
            name='AsCas12a', family='Cas12a', pam='TTTV',
            cut_offset=18, pam_side='5prime', stagger=4,
        ),
        'LbCas12a': NucleaseProfile(
            name='LbCas12a', family='Cas12a', pam='TTTV',
            cut_offset=18, pam_side='5prime', stagger=4,
        ),
        'FnCas12a': NucleaseProfile(
            name='FnCas12a', family='Cas12a', pam='TTN',
            cut_offset=18, pam_side='5prime', stagger=4,
        ),
    },
    'Cas12f': {
        'Cas12f1': NucleaseProfile(
            name='Cas12f1', family='Cas12f', pam='TTTN',
            cut_offset=15, pam_side='5prime', stagger=4,
        ),
        'UnCas12f1': NucleaseProfile(
            name='UnCas12f1', family='Cas12f', pam='TTTN',
            cut_offset=15, pam_side='5prime', stagger=4,
        ),
    },
    'BaseEditors': {
        'BE3': NucleaseProfile(
            name='BE3', family='BaseEditors', pam='NGG',
            cut_offset=-3, pam_side='3prime',
            editing_window=(1, 20),
        ),
        'BE4': NucleaseProfile(
            name='BE4', family='BaseEditors', pam='NGG',
            cut_offset=-3, pam_side='3prime',
            editing_window=(1, 20),
        ),
        'ABE8e': NucleaseProfile(
            name='ABE8e', family='BaseEditors', pam='NGG',
            cut_offset=-3, pam_side='3prime',
            editing_window=(4, 8),
        ),
    },
    'PrimeEditors': {
        'PE2': NucleaseProfile(
            name='PE2', family='PrimeEditors', pam='NGG',
            cut_offset=-3, pam_side='3prime',
            max_insertion=50,
        ),
        'PE3': NucleaseProfile(
            name='PE3', family='PrimeEditors', pam='NGG',
            cut_offset=-3, pam_side='3prime',
            max_insertion=80,
        ),
    },
}


class NucleaseEngine:
    """Resolve and manage nuclease profiles."""

    def __init__(self):
        self.library = NUCLEASE_LIBRARY

    # ------------------------------------------------------------------
    def get(self, family: str, variant: Optional[str] = None) -> NucleaseProfile:
        """Return a NucleaseProfile by family and optional variant name."""
        fam = self.library.get(family)
        if fam is None:
            raise ValueError(
                f"Unknown nuclease family '{family}'. "
                f"Available: {list(self.library.keys())}"
            )
        if variant is None:
            variant = next(iter(fam))  # default to first variant
        profile = fam.get(variant)
        if profile is None:
            raise ValueError(
                f"Unknown variant '{variant}' in family '{family}'. "
                f"Available: {list(fam.keys())}"
            )
        return profile

    def custom(self, pam: str, cut_offset: int, pam_side: str = '3prime',
               stagger: int = 0, **kwargs) -> NucleaseProfile:
        """Create a custom nuclease profile."""
        return NucleaseProfile(
            name='Custom', family='Custom', pam=pam,
            cut_offset=cut_offset, pam_side=pam_side,
            stagger=stagger, **kwargs,
        )

    def find_cut_site(self, guide_seq: str, reference: str,
                      profile: NucleaseProfile) -> Optional[int]:
        """
        Locate the expected cut position on the reference sequence
        given a guide RNA and nuclease profile.
        Returns 0-based cut index on the reference, or None.
        """
        guide_dna = guide_seq.upper().replace('U', 'T')
        ref_upper = reference.upper()

        # Search both strands
        for strand, seq in [('fwd', ref_upper), ('rev', _revcomp(ref_upper))]:
            idx = seq.find(guide_dna)
            if idx == -1:
                continue
            # Compute cut position
            if profile.pam_side == '3prime':
                cut_pos = idx + len(guide_dna) + profile.cut_offset
            else:
                cut_pos = idx + profile.cut_offset
            # Convert rev-strand position back
            if strand == 'rev':
                cut_pos = len(reference) - cut_pos
            if 0 <= cut_pos <= len(reference):
                return cut_pos
        return None

    def list_all(self) -> list[str]:
        """Return flat list of 'Family/Variant' strings."""
        out = []
        for fam, variants in self.library.items():
            for var in variants:
                out.append(f"{fam}/{var}")
        return out


def _revcomp(seq: str) -> str:
    comp = str.maketrans('ACGTacgt', 'TGCAtgca')
    return seq.translate(comp)[::-1]
