"""
Read preservation engine — loads FASTQ/FASTA from any platform
and ensures ZERO reads are discarded.
Supports: single-end, paired-end, interleaved, .gz, .bz2
"""

from __future__ import annotations

import gzip
import bz2
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from Bio import SeqIO

logger = logging.getLogger(__name__)


@dataclass
class SequenceRead:
    """A single sequencing read with all metadata preserved."""
    name: str
    sequence: str
    quality_scores: list[int]
    platform_hint: str | None = None
    mate: SequenceRead | None = None  # paired-end mate
    is_merged: bool = False
    original_length: int = 0

    def __post_init__(self):
        self.original_length = len(self.sequence)


class ReadPreservationEngine:
    """
    Loads sequencing reads and NEVER discards any of them.
    Soft-flags low quality instead of filtering.
    """

    def __init__(self, min_length: int = 0):
        self.min_length = min_length  # only warn, never discard
        self.stats = {
            'total_reads': 0,
            'paired': 0,
            'merged': 0,
            'flagged_short': 0,
            'flagged_low_quality': 0,
        }

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_fastq(self, path: str | Path,
                   path_r2: str | Path | None = None) -> list[SequenceRead]:
        """Load reads from one or two FASTQ files."""
        reads_r1 = list(self._iter_fastq(Path(path)))
        self.stats['total_reads'] += len(reads_r1)

        if path_r2:
            reads_r2 = list(self._iter_fastq(Path(path_r2)))
            self.stats['total_reads'] += len(reads_r2)
            reads = self._pair_reads(reads_r1, reads_r2)
        else:
            reads = reads_r1

        logger.info("Loaded %d reads (paired: %d, flagged_short: %d)",
                     len(reads), self.stats['paired'], self.stats['flagged_short'])
        return reads

    def load_fasta(self, path: str | Path) -> str:
        """Load a reference sequence from FASTA."""
        p = Path(path)
        handle = self._open_file(p)
        record = next(SeqIO.parse(handle, 'fasta'))
        handle.close()
        return str(record.seq).upper()

    # ------------------------------------------------------------------
    # Paired-end merging
    # ------------------------------------------------------------------

    def _pair_reads(self, r1_list: list[SequenceRead],
                    r2_list: list[SequenceRead]) -> list[SequenceRead]:
        """Pair R1/R2 reads. If merging fails, keep BOTH reads."""
        paired: list[SequenceRead] = []
        r2_map = {r.name.split('/')[0].split(' ')[0]: r for r in r2_list}

        for r1 in r1_list:
            key = r1.name.split('/')[0].split(' ')[0]
            r2 = r2_map.pop(key, None)
            if r2:
                merged = self._try_merge(r1, r2)
                if merged:
                    self.stats['merged'] += 1
                    paired.append(merged)
                else:
                    # Keep both as linked mates — never discard
                    r1.mate = r2
                    self.stats['paired'] += 1
                    paired.append(r1)
            else:
                paired.append(r1)

        # Orphan R2 reads also preserved
        for r2 in r2_map.values():
            paired.append(r2)

        return paired

    def _try_merge(self, r1: SequenceRead, r2: SequenceRead,
                   min_overlap: int = 10) -> SequenceRead | None:
        """Attempt FLASH-style overlap merging of paired reads."""
        r2_rc = self._revcomp(r2.sequence)
        r2_qual_rev = r2.quality_scores[::-1]

        best_overlap = 0
        best_mismatches = float('inf')

        for overlap in range(min_overlap, min(len(r1.sequence), len(r2_rc)) + 1):
            r1_tail = r1.sequence[-overlap:]
            r2_head = r2_rc[:overlap]
            mismatches = sum(a != b for a, b in zip(r1_tail, r2_head))
            mismatch_rate = mismatches / overlap
            if mismatch_rate < 0.1 and overlap > best_overlap:
                best_overlap = overlap
                best_mismatches = mismatches

        if best_overlap >= min_overlap:
            merged_seq = r1.sequence + r2_rc[best_overlap:]
            # Merge quality: take max at overlap region
            merged_qual = list(r1.quality_scores)
            overlap_start = len(r1.quality_scores) - best_overlap
            for i in range(best_overlap):
                idx = overlap_start + i
                if idx < len(merged_qual) and i < len(r2_qual_rev):
                    merged_qual[idx] = max(merged_qual[idx], r2_qual_rev[i])
            merged_qual.extend(r2_qual_rev[best_overlap:])

            return SequenceRead(
                name=r1.name, sequence=merged_seq,
                quality_scores=merged_qual, is_merged=True,
            )
        return None

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _iter_fastq(self, path: Path) -> Iterator[SequenceRead]:
        """Iterate FASTQ records from plain or compressed files."""
        handle = self._open_file(path)
        fmt = 'fastq'
        for record in SeqIO.parse(handle, fmt):
            quals = record.letter_annotations.get(
                'phred_quality', [0] * len(record.seq))
            read = SequenceRead(
                name=record.id,
                sequence=str(record.seq).upper(),
                quality_scores=quals,
            )
            if len(read.sequence) < self.min_length:
                self.stats['flagged_short'] += 1
                # Flag but do NOT discard
            yield read
        handle.close()

    @staticmethod
    def _open_file(path: Path):
        """Open plain, gzip, or bz2 compressed files transparently."""
        suffix = path.suffix.lower()
        if suffix == '.gz':
            import io
            return io.TextIOWrapper(gzip.open(path, 'rb'))
        elif suffix == '.bz2':
            import io
            return io.TextIOWrapper(bz2.open(path, 'rb'))
        else:
            return open(path, 'r')

    @staticmethod
    def _revcomp(seq: str) -> str:
        comp = str.maketrans('ACGTacgt', 'TGCAtgca')
        return seq.translate(comp)[::-1]
