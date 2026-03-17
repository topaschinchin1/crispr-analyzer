"""
Comprehensive edit analysis engine.
Detects insertions, deletions, substitutions, complex indels,
base editing conversions, and prime editing outcomes.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .aligner import AlignmentResult
from .nucleases import NucleaseProfile

logger = logging.getLogger(__name__)


@dataclass
class EditEvent:
    """A single detected edit."""
    edit_type: str  # 'insertion', 'deletion', 'substitution', 'complex'
    ref_start: int
    ref_end: int
    ref_bases: str
    alt_bases: str
    size: int
    in_editing_window: bool = False
    distance_from_cut: int = 0


@dataclass
class ReadAnalysis:
    """Full analysis of one aligned read."""
    read_name: str
    classification: str  # 'unmodified', 'NHEJ', 'HDR', 'ambiguous', 'low_confidence'
    edits: list[EditEvent] = field(default_factory=list)
    total_insertions: int = 0
    total_deletions: int = 0
    total_substitutions: int = 0
    identity: float = 0.0
    alignment_method: str = ''


@dataclass
class SampleSummary:
    """Aggregate statistics for the entire sample."""
    total_reads: int = 0
    aligned_reads: int = 0
    unmodified_reads: int = 0
    modified_reads: int = 0
    nhej_reads: int = 0
    hdr_reads: int = 0
    ambiguous_reads: int = 0
    low_confidence_reads: int = 0
    editing_efficiency: float = 0.0
    insertion_sizes: list[int] = field(default_factory=list)
    deletion_sizes: list[int] = field(default_factory=list)
    substitution_positions: list[int] = field(default_factory=list)
    edit_position_counts: dict[int, int] = field(default_factory=dict)
    indel_spectrum: dict[int, int] = field(default_factory=dict)
    # Base editor specific
    base_conversion_counts: dict[str, int] = field(default_factory=dict)
    base_conversion_positions: dict[int, dict[str, int]] = field(default_factory=dict)


class AnalysisEngine:
    """
    Analyzes aligned reads to detect and classify all types of CRISPR edits.
    """

    def __init__(self, nuclease: NucleaseProfile, cut_site: int | None = None,
                 window_size: int = 50, hdr_template: str | None = None):
        self.nuclease = nuclease
        self.cut_site = cut_site
        self.window_size = window_size
        self.hdr_template = hdr_template

    def analyze_all(self, alignments: list[AlignmentResult],
                    reference: str) -> tuple[list[ReadAnalysis], SampleSummary]:
        """Analyze all aligned reads and return per-read results + summary."""
        results: list[ReadAnalysis] = []
        summary = SampleSummary()

        for aln in alignments:
            ra = self._analyze_single(aln, reference)
            results.append(ra)

        summary = self._build_summary(results)
        return results, summary

    # ------------------------------------------------------------------
    # Per-read analysis
    # ------------------------------------------------------------------

    def _analyze_single(self, aln: AlignmentResult,
                        reference: str) -> ReadAnalysis:
        edits = self._extract_edits(aln)

        # Classify the read
        if aln.classification == 'low_confidence':
            classification = 'low_confidence'
        elif not edits:
            classification = 'unmodified'
        elif self.hdr_template and self._check_hdr(aln, self.hdr_template):
            classification = 'HDR'
        elif any(e.edit_type in ('insertion', 'deletion', 'complex') for e in edits):
            classification = 'NHEJ'
        else:
            classification = 'ambiguous'

        n_ins = sum(e.size for e in edits if e.edit_type == 'insertion')
        n_del = sum(e.size for e in edits if e.edit_type == 'deletion')
        n_sub = sum(1 for e in edits if e.edit_type == 'substitution')

        # Base editor specific
        if self.nuclease.family == 'BaseEditors':
            self._annotate_base_editing(edits, reference)

        return ReadAnalysis(
            read_name=aln.query_name,
            classification=classification,
            edits=edits,
            total_insertions=n_ins,
            total_deletions=n_del,
            total_substitutions=n_sub,
            identity=aln.identity,
            alignment_method=aln.method,
        )

    def _extract_edits(self, aln: AlignmentResult) -> list[EditEvent]:
        """Walk the aligned sequences and extract all edit events."""
        edits: list[EditEvent] = []
        aligned_q = aln.aligned_query
        aligned_r = aln.aligned_ref

        if not aligned_q or not aligned_r:
            return edits

        ref_pos = aln.ref_start
        i = 0
        while i < len(aligned_q) and i < len(aligned_r):
            q = aligned_q[i]
            r = aligned_r[i]

            if q == r:
                ref_pos += 1
                i += 1
                continue

            if r == '-':
                # Insertion in query
                ins_start = i
                ins_bases = []
                while i < len(aligned_q) and i < len(aligned_r) and aligned_ref_at(aligned_r, i) == '-':
                    ins_bases.append(aligned_q[i])
                    i += 1
                edit = EditEvent(
                    edit_type='insertion',
                    ref_start=ref_pos, ref_end=ref_pos,
                    ref_bases='', alt_bases=''.join(ins_bases),
                    size=len(ins_bases),
                )
                self._annotate_position(edit)
                edits.append(edit)
            elif q == '-':
                # Deletion in query
                del_start = ref_pos
                del_bases = []
                while i < len(aligned_q) and i < len(aligned_r) and aligned_q[i] == '-':
                    del_bases.append(aligned_r[i])
                    ref_pos += 1
                    i += 1
                edit = EditEvent(
                    edit_type='deletion',
                    ref_start=del_start, ref_end=ref_pos,
                    ref_bases=''.join(del_bases), alt_bases='',
                    size=len(del_bases),
                )
                self._annotate_position(edit)
                edits.append(edit)
            else:
                # Substitution
                edit = EditEvent(
                    edit_type='substitution',
                    ref_start=ref_pos, ref_end=ref_pos + 1,
                    ref_bases=r, alt_bases=q,
                    size=1,
                )
                self._annotate_position(edit)
                edits.append(edit)
                ref_pos += 1
                i += 1

        # Merge adjacent events into complex indels
        edits = self._merge_complex_edits(edits)
        return edits

    def _annotate_position(self, edit: EditEvent) -> None:
        """Mark whether an edit is within the expected editing window."""
        if self.cut_site is not None:
            mid = (edit.ref_start + edit.ref_end) // 2
            edit.distance_from_cut = mid - self.cut_site
            edit.in_editing_window = abs(edit.distance_from_cut) <= self.window_size

        if self.nuclease.editing_window:
            w_start, w_end = self.nuclease.editing_window
            if self.cut_site is not None:
                abs_start = self.cut_site - w_end
                abs_end = self.cut_site + w_end
                edit.in_editing_window = (abs_start <= edit.ref_start <= abs_end)

    def _merge_complex_edits(self, edits: list[EditEvent],
                             max_gap: int = 3) -> list[EditEvent]:
        """Merge nearby insertion+deletion pairs into complex indels."""
        if len(edits) < 2:
            return edits

        merged: list[EditEvent] = [edits[0]]
        for edit in edits[1:]:
            prev = merged[-1]
            if (edit.ref_start - prev.ref_end <= max_gap and
                    prev.edit_type != 'substitution' and
                    edit.edit_type != 'substitution'):
                # Merge into complex
                merged[-1] = EditEvent(
                    edit_type='complex',
                    ref_start=prev.ref_start,
                    ref_end=max(prev.ref_end, edit.ref_end),
                    ref_bases=prev.ref_bases + edit.ref_bases,
                    alt_bases=prev.alt_bases + edit.alt_bases,
                    size=prev.size + edit.size,
                    in_editing_window=prev.in_editing_window or edit.in_editing_window,
                    distance_from_cut=prev.distance_from_cut,
                )
            else:
                merged.append(edit)
        return merged

    # ------------------------------------------------------------------
    # HDR detection
    # ------------------------------------------------------------------

    def _check_hdr(self, aln: AlignmentResult, template: str) -> bool:
        """Check if the read matches the HDR template."""
        template = template.upper()
        read_seq = aln.query_seq.upper()
        # Simple substring check; could use alignment for fuzzy matching
        return template in read_seq or read_seq in template

    # ------------------------------------------------------------------
    # Base editing analysis
    # ------------------------------------------------------------------

    def _annotate_base_editing(self, edits: list[EditEvent],
                               reference: str) -> None:
        """Annotate substitutions with base conversion types (C>T, A>G, etc.)."""
        for edit in edits:
            if edit.edit_type == 'substitution':
                conversion = f"{edit.ref_bases}>{edit.alt_bases}"
                edit.edit_type = f"base_conversion:{conversion}"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(self, results: list[ReadAnalysis]) -> SampleSummary:
        summary = SampleSummary()
        summary.total_reads = len(results)
        summary.aligned_reads = len(results)

        pos_counts: Counter = Counter()
        indel_spectrum: Counter = Counter()
        base_conv: Counter = Counter()
        base_conv_pos: defaultdict = defaultdict(Counter)

        for ra in results:
            if ra.classification == 'unmodified':
                summary.unmodified_reads += 1
            elif ra.classification == 'NHEJ':
                summary.nhej_reads += 1
                summary.modified_reads += 1
            elif ra.classification == 'HDR':
                summary.hdr_reads += 1
                summary.modified_reads += 1
            elif ra.classification == 'low_confidence':
                summary.low_confidence_reads += 1
            else:
                summary.ambiguous_reads += 1
                summary.modified_reads += 1

            for edit in ra.edits:
                pos_counts[edit.ref_start] += 1
                if edit.edit_type == 'insertion':
                    summary.insertion_sizes.append(edit.size)
                    indel_spectrum[edit.size] += 1
                elif edit.edit_type == 'deletion':
                    summary.deletion_sizes.append(edit.size)
                    indel_spectrum[-edit.size] += 1
                elif edit.edit_type.startswith('base_conversion:'):
                    conv = edit.edit_type.split(':')[1]
                    base_conv[conv] += 1
                    base_conv_pos[edit.ref_start][conv] += 1

        if summary.total_reads > 0:
            summary.editing_efficiency = (
                summary.modified_reads / summary.total_reads * 100
            )

        summary.edit_position_counts = dict(pos_counts)
        summary.indel_spectrum = dict(indel_spectrum)
        summary.base_conversion_counts = dict(base_conv)
        summary.base_conversion_positions = dict(base_conv_pos)

        return summary


def aligned_ref_at(aligned_ref: str, i: int) -> str:
    """Safe accessor for aligned reference string."""
    if i < len(aligned_ref):
        return aligned_ref[i]
    return ''
