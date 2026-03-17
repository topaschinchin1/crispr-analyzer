"""
Platform-specific configuration for Illumina, Nanopore, and PacBio.
"""

from dataclasses import dataclass


@dataclass
class PlatformConfig:
    name: str
    quality_encoding: str
    min_quality: int
    typical_read_length: tuple  # (min, max)
    error_model: str
    homopolymer_aware: bool


PLATFORMS: dict[str, PlatformConfig] = {
    'illumina': PlatformConfig(
        name='Illumina',
        quality_encoding='phred33',
        min_quality=20,
        typical_read_length=(75, 300),
        error_model='substitution_heavy',
        homopolymer_aware=False,
    ),
    'nanopore': PlatformConfig(
        name='Oxford Nanopore',
        quality_encoding='phred33',
        min_quality=7,
        typical_read_length=(200, 100_000),
        error_model='indel_heavy',
        homopolymer_aware=True,
    ),
    'pacbio': PlatformConfig(
        name='PacBio',
        quality_encoding='phred33',
        min_quality=10,
        typical_read_length=(500, 50_000),
        error_model='random',
        homopolymer_aware=True,
    ),
}


class PlatformHandler:
    """Detect and configure platform-specific parameters."""

    def __init__(self, platform: str | None = None):
        if platform and platform.lower() in PLATFORMS:
            self.config = PLATFORMS[platform.lower()]
        else:
            self.config = None  # will auto-detect

    def detect_platform(self, read_lengths: list[int],
                        mean_quality: float) -> PlatformConfig:
        """Heuristically detect the sequencing platform."""
        if self.config:
            return self.config

        median_len = sorted(read_lengths)[len(read_lengths) // 2]

        if median_len > 1000:
            if mean_quality < 15:
                self.config = PLATFORMS['nanopore']
            else:
                self.config = PLATFORMS['pacbio']
        else:
            self.config = PLATFORMS['illumina']
        return self.config

    def get_alignment_params(self) -> dict:
        """Return alignment parameters tuned for the detected platform."""
        cfg = self.config or PLATFORMS['illumina']

        if cfg.error_model == 'indel_heavy':
            return {
                'match_score': 2,
                'mismatch_penalty': -4,
                'gap_open': -4,
                'gap_extend': -1,
                'min_score_ratio': 0.15,  # very permissive for nanopore
            }
        elif cfg.error_model == 'random':
            return {
                'match_score': 2,
                'mismatch_penalty': -3,
                'gap_open': -5,
                'gap_extend': -1,
                'min_score_ratio': 0.20,
            }
        else:
            return {
                'match_score': 2,
                'mismatch_penalty': -6,
                'gap_open': -5,
                'gap_extend': -2,
                'min_score_ratio': 0.30,
            }
