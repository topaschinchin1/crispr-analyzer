"""
Advanced multi-strategy alignment engine.
Supports global, local, and semi-global alignment with NO read discarding.
Uses parasail for fast C-level Smith-Waterman / Needleman-Wunsch,
falls back to Biopython if parasail is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import parasail
    HAS_PARASAIL = True
except ImportError:
    HAS_PARASAIL = False

from Bio import Align

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Container for a single alignment result."""
    query_name: str
    query_seq: str
    ref_seq: str
    aligned_query: str
    aligned_ref: str
    score: int
    method: str  # 'global', 'local', 'semi_global', 'fuzzy'
    cigar: str
    query_start: int
    query_end: int
    ref_start: int
    ref_end: int
    identity: float
    classification: str = 'unclassified'  # set later by analysis
    raw_quality: Optional[list[int]] = None


@dataclass
class AlignmentParams:
    match_score: int = 2
    mismatch_penalty: int = -6
    gap_open: int = -5
    gap_extend: int = -2
    min_score_ratio: float = 0.30  # never used to discard, only to flag


class AdvancedAligner:
    """
    Multi-strategy aligner that NEVER discards reads.
    Tries global -> local -> semi-global -> fuzzy, picks the best.
    """

    def __init__(self, params: AlignmentParams | None = None):
        self.params = params or AlignmentParams()
        if HAS_PARASAIL:
            self._matrix = parasail.matrix_create("ACGT",
                                                  self.params.match_score,
                                                  self.params.mismatch_penalty)
        else:
            logger.info("parasail not found; using Biopython PairwiseAligner (slower)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(self, query_name: str, query_seq: str, ref_seq: str,
              quality_scores: list[int] | None = None) -> AlignmentResult:
        """
        Align a single read to the reference using multiple strategies.
        Returns the best alignment — NEVER returns None.
        """
        query_seq = query_seq.upper()
        ref_seq = ref_seq.upper()

        candidates: list[AlignmentResult] = []

        for method in ('global', 'local', 'semi_global'):
            try:
                result = self._run_alignment(query_name, query_seq, ref_seq, method)
                if result:
                    candidates.append(result)
            except Exception as exc:
                logger.debug("Alignment method %s failed for %s: %s",
                             method, query_name, exc)

        if not candidates:
            # Absolute fallback: fuzzy / trivial alignment
            candidates.append(self._fuzzy_align(query_name, query_seq, ref_seq))

        best = max(candidates, key=lambda r: r.score)

        # Attach quality scores
        best.raw_quality = quality_scores

        # Flag (but never discard) low-quality alignments
        # Use both score ratio AND identity — for fragment reads (shorter than
        # reference), score ratio alone is misleadingly low even when the
        # aligned region matches well.
        max_possible = len(query_seq) * self.params.match_score
        score_ratio = best.score / max_possible if max_possible > 0 else 0
        if score_ratio < self.params.min_score_ratio and best.identity < 0.5:
            best.classification = 'low_confidence'

        return best

    def align_long_read(self, query_name: str, query_seq: str, ref_seq: str,
                        window_size: int = 5000, overlap: int = 500,
                        quality_scores: list[int] | None = None) -> AlignmentResult:
        """
        Windowed alignment for long reads (nanopore/PacBio) to avoid
        memory blow-up with large sequences.
        """
        if len(query_seq) <= window_size and len(ref_seq) <= window_size:
            return self.align(query_name, query_seq, ref_seq, quality_scores)

        # Tile the query in overlapping windows, align each to reference
        best_result: AlignmentResult | None = None
        step = window_size - overlap

        for start in range(0, len(query_seq), step):
            chunk = query_seq[start:start + window_size]
            result = self.align(query_name, chunk, ref_seq)
            # Shift coordinates back
            result.query_start += start
            result.query_end += start
            if best_result is None or result.score > best_result.score:
                best_result = result

        if best_result is None:
            best_result = self._fuzzy_align(query_name, query_seq, ref_seq)

        best_result.raw_quality = quality_scores
        return best_result

    # ------------------------------------------------------------------
    # Internal alignment methods
    # ------------------------------------------------------------------

    def _run_alignment(self, name: str, query: str, ref: str,
                       method: str) -> AlignmentResult | None:
        if HAS_PARASAIL:
            return self._parasail_align(name, query, ref, method)
        return self._biopython_align(name, query, ref, method)

    # ---- parasail backend ------------------------------------------------

    def _parasail_align(self, name: str, query: str, ref: str,
                        method: str) -> AlignmentResult:
        go = abs(self.params.gap_open)
        ge = abs(self.params.gap_extend)

        if method == 'global':
            res = parasail.nw_trace_striped_32(query, ref, go, ge, self._matrix)
        elif method == 'local':
            res = parasail.sw_trace_striped_32(query, ref, go, ge, self._matrix)
        else:  # semi_global
            res = parasail.sg_trace_striped_32(query, ref, go, ge, self._matrix)

        traceback = res.traceback
        aligned_q = traceback.query
        aligned_r = traceback.ref
        cigar_str = self._traceback_to_cigar(aligned_q, aligned_r)

        matches = sum(1 for a, b in zip(aligned_q, aligned_r)
                      if a == b and a != '-')
        aligned_len = sum(1 for a, b in zip(aligned_q, aligned_r)
                          if not (a == '-' and b == '-'))
        identity = matches / max(aligned_len, 1)

        return AlignmentResult(
            query_name=name, query_seq=query, ref_seq=ref,
            aligned_query=aligned_q, aligned_ref=aligned_r,
            score=res.score, method=method, cigar=cigar_str,
            query_start=0, query_end=len(query),
            ref_start=0, ref_end=len(ref),
            identity=identity,
        )

    # ---- biopython backend -----------------------------------------------

    def _biopython_align(self, name: str, query: str, ref: str,
                         method: str) -> AlignmentResult:
        aligner = Align.PairwiseAligner()
        aligner.match_score = self.params.match_score
        aligner.mismatch_score = self.params.mismatch_penalty
        aligner.open_gap_score = self.params.gap_open
        aligner.extend_gap_score = self.params.gap_extend

        if method == 'global':
            aligner.mode = 'global'
        elif method == 'local':
            aligner.mode = 'local'
        else:
            aligner.mode = 'global'
            aligner.end_insertion_score = 0
            aligner.end_deletion_score = 0

        # Biopython align: target=ref, query=query
        alignments = aligner.align(ref, query)
        if not alignments:
            return self._fuzzy_align(name, query, ref)

        best = alignments[0]

        # Extract aligned sequences from the Alignment object
        aligned_r, aligned_q = self._extract_aligned_seqs(best, ref, query)

        matches = sum(1 for a, b in zip(aligned_q, aligned_r)
                      if a == b and a != '-')
        aligned_len = sum(1 for a, b in zip(aligned_q, aligned_r)
                          if not (a == '-' and b == '-'))
        identity = matches / max(aligned_len, 1)

        cigar_str = self._traceback_to_cigar(aligned_q, aligned_r)

        return AlignmentResult(
            query_name=name, query_seq=query, ref_seq=ref,
            aligned_query=aligned_q, aligned_ref=aligned_r,
            score=int(best.score), method=method, cigar=cigar_str,
            query_start=0, query_end=len(query),
            ref_start=0, ref_end=len(ref),
            identity=identity,
        )

    @staticmethod
    def _extract_aligned_seqs(alignment, target: str, query: str) -> tuple[str, str]:
        """Extract gapped aligned strings from a Biopython Alignment object."""
        # Use the alignment's coordinates to build gapped sequences
        try:
            coords = alignment.coordinates  # shape (2, n_segments+1)
            aligned_t = []
            aligned_q = []
            for i in range(coords.shape[1] - 1):
                t_start, t_end = coords[0, i], coords[0, i + 1]
                q_start, q_end = coords[1, i], coords[1, i + 1]
                t_len = abs(t_end - t_start)
                q_len = abs(q_end - q_start)

                if t_len > 0 and q_len > 0:
                    # Matched/mismatched block
                    aligned_t.append(target[t_start:t_end])
                    aligned_q.append(query[q_start:q_end])
                elif t_len > 0 and q_len == 0:
                    # Deletion in query (gap in query)
                    aligned_t.append(target[t_start:t_end])
                    aligned_q.append('-' * t_len)
                elif q_len > 0 and t_len == 0:
                    # Insertion in query (gap in target)
                    aligned_t.append('-' * q_len)
                    aligned_q.append(query[q_start:q_end])

            return ''.join(aligned_t), ''.join(aligned_q)
        except Exception:
            # Fallback: return raw sequences
            return target, query

    # ---- fuzzy fallback --------------------------------------------------

    def _fuzzy_align(self, name: str, query: str, ref: str) -> AlignmentResult:
        """
        Last-resort alignment: simple k-mer anchored approach.
        Guarantees a result so no read is ever discarded.
        """
        k = min(11, len(query) // 2, len(ref) // 2)
        if k < 4:
            # Sequence too short for k-mer approach; just pad/align trivially
            return AlignmentResult(
                query_name=name, query_seq=query, ref_seq=ref,
                aligned_query=query, aligned_ref=ref,
                score=0, method='fuzzy', cigar=f'{len(query)}M',
                query_start=0, query_end=len(query),
                ref_start=0, ref_end=len(ref),
                identity=0.0,
                classification='low_confidence',
            )

        # Build k-mer index for the reference
        ref_kmers: dict[str, list[int]] = {}
        for i in range(len(ref) - k + 1):
            kmer = ref[i:i + k]
            ref_kmers.setdefault(kmer, []).append(i)

        # Find best anchor
        best_offset = 0
        best_matches = 0
        for i in range(len(query) - k + 1):
            kmer = query[i:i + k]
            for pos in ref_kmers.get(kmer, []):
                offset = pos - i
                matches = sum(1 for j in range(len(query))
                              if 0 <= j + offset < len(ref)
                              and query[j] == ref[j + offset])
                if matches > best_matches:
                    best_matches = matches
                    best_offset = offset

        identity = best_matches / max(len(query), 1)

        return AlignmentResult(
            query_name=name, query_seq=query, ref_seq=ref,
            aligned_query=query, aligned_ref=ref,
            score=best_matches * self.params.match_score,
            method='fuzzy', cigar=f'{len(query)}M',
            query_start=0, query_end=len(query),
            ref_start=max(0, best_offset),
            ref_end=min(len(ref), best_offset + len(query)),
            identity=identity,
            classification='low_confidence',
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _traceback_to_cigar(aligned_query: str, aligned_ref: str) -> str:
        """Convert a pair of aligned strings to a CIGAR string."""
        if not aligned_query or not aligned_ref:
            return ''
        ops: list[str] = []
        for q, r in zip(aligned_query, aligned_ref):
            if q == '-':
                ops.append('D')
            elif r == '-':
                ops.append('I')
            elif q == r:
                ops.append('M')
            else:
                ops.append('X')

        # Run-length encode
        cigar_parts: list[str] = []
        if not ops:
            return ''
        current = ops[0]
        count = 1
        for op in ops[1:]:
            if op == current:
                count += 1
            else:
                cigar_parts.append(f'{count}{current}')
                current = op
                count = 1
        cigar_parts.append(f'{count}{current}')
        return ''.join(cigar_parts)
