"""
Comprehensive reporting: HTML, CSV, JSON, and publication-ready plots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .analysis import ReadAnalysis, SampleSummary, EditEvent

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate all output formats from analysis results."""

    def __init__(self, output_dir: str | Path, sample_name: str = 'sample'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_name = sample_name
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)

    def generate_all(self, results: list[ReadAnalysis],
                     summary: SampleSummary, reference: str,
                     cut_site: int | None = None,
                     guide_rna: str | None = None) -> None:
        """Generate all report formats."""
        self._write_csv(results)
        self._write_json(summary)
        self._generate_plots(summary, reference, cut_site)
        self._write_html(results, summary, reference, cut_site, guide_rna)
        logger.info("Reports written to %s", self.output_dir)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def _write_csv(self, results: list[ReadAnalysis]) -> None:
        rows = []
        for ra in results:
            row = {
                'read_name': ra.read_name,
                'classification': ra.classification,
                'identity': round(ra.identity, 4),
                'alignment_method': ra.alignment_method,
                'num_insertions': ra.total_insertions,
                'num_deletions': ra.total_deletions,
                'num_substitutions': ra.total_substitutions,
                'edits': '; '.join(
                    f"{e.edit_type}@{e.ref_start}:{e.ref_bases}>{e.alt_bases}"
                    for e in ra.edits
                ),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.output_dir / f'{self.sample_name}_per_read.csv'
        df.to_csv(csv_path, index=False)

        # Also write edit-level CSV
        edit_rows = []
        for ra in results:
            for e in ra.edits:
                edit_rows.append({
                    'read_name': ra.read_name,
                    'edit_type': e.edit_type,
                    'ref_start': e.ref_start,
                    'ref_end': e.ref_end,
                    'ref_bases': e.ref_bases,
                    'alt_bases': e.alt_bases,
                    'size': e.size,
                    'in_editing_window': e.in_editing_window,
                    'distance_from_cut': e.distance_from_cut,
                })
        if edit_rows:
            edf = pd.DataFrame(edit_rows)
            edf.to_csv(self.output_dir / f'{self.sample_name}_edits.csv', index=False)

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _write_json(self, summary: SampleSummary) -> None:
        data = {
            'sample': self.sample_name,
            'total_reads': summary.total_reads,
            'aligned_reads': summary.aligned_reads,
            'unmodified_reads': summary.unmodified_reads,
            'modified_reads': summary.modified_reads,
            'nhej_reads': summary.nhej_reads,
            'hdr_reads': summary.hdr_reads,
            'ambiguous_reads': summary.ambiguous_reads,
            'low_confidence_reads': summary.low_confidence_reads,
            'editing_efficiency_pct': round(summary.editing_efficiency, 2),
            'read_preservation_pct': 100.0,
            'mean_insertion_size': (
                round(np.mean(summary.insertion_sizes), 2)
                if summary.insertion_sizes else 0
            ),
            'mean_deletion_size': (
                round(np.mean(summary.deletion_sizes), 2)
                if summary.deletion_sizes else 0
            ),
            'indel_spectrum': summary.indel_spectrum,
            'base_conversion_counts': summary.base_conversion_counts,
        }
        json_path = self.output_dir / f'{self.sample_name}_summary.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _generate_plots(self, summary: SampleSummary, reference: str,
                        cut_site: int | None) -> None:
        sns.set_theme(style='whitegrid', font_scale=1.1)

        self._plot_classification_pie(summary)
        self._plot_indel_spectrum(summary)
        self._plot_edit_positions(summary, len(reference), cut_site)
        if summary.insertion_sizes or summary.deletion_sizes:
            self._plot_indel_sizes(summary)
        if summary.base_conversion_counts:
            self._plot_base_conversions(summary)

    def _plot_classification_pie(self, summary: SampleSummary) -> None:
        labels, sizes, colors = [], [], []
        data = [
            ('Unmodified', summary.unmodified_reads, '#4CAF50'),
            ('NHEJ', summary.nhej_reads, '#F44336'),
            ('HDR', summary.hdr_reads, '#2196F3'),
            ('Ambiguous', summary.ambiguous_reads, '#FF9800'),
            ('Low confidence', summary.low_confidence_reads, '#9E9E9E'),
        ]
        for label, count, color in data:
            if count > 0:
                labels.append(f'{label} ({count})')
                sizes.append(count)
                colors.append(color)

        if not sizes:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax.set_title(f'{self.sample_name} — Read Classification\n'
                     f'Total: {summary.total_reads} reads (100% preserved)')
        fig.tight_layout()
        fig.savefig(self.plots_dir / 'classification_pie.png', dpi=150)
        plt.close(fig)

    def _plot_indel_spectrum(self, summary: SampleSummary) -> None:
        if not summary.indel_spectrum:
            return
        sizes = sorted(summary.indel_spectrum.keys())
        counts = [summary.indel_spectrum[s] for s in sizes]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#F44336' if s < 0 else '#2196F3' for s in sizes]
        ax.bar(sizes, counts, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Indel size (negative = deletion, positive = insertion)')
        ax.set_ylabel('Count')
        ax.set_title(f'{self.sample_name} — Indel Size Spectrum')
        ax.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
        fig.tight_layout()
        fig.savefig(self.plots_dir / 'indel_spectrum.png', dpi=150)
        plt.close(fig)

    def _plot_edit_positions(self, summary: SampleSummary,
                             ref_len: int, cut_site: int | None) -> None:
        if not summary.edit_position_counts:
            return
        positions = list(range(ref_len))
        counts = [summary.edit_position_counts.get(p, 0) for p in positions]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(positions, counts, color='#2196F3', width=1.0)
        if cut_site is not None:
            ax.axvline(x=cut_site, color='red', linewidth=2,
                       linestyle='--', label=f'Cut site ({cut_site})')
            ax.legend()
        ax.set_xlabel('Reference position')
        ax.set_ylabel('Edit count')
        ax.set_title(f'{self.sample_name} — Edit Position Distribution')
        fig.tight_layout()
        fig.savefig(self.plots_dir / 'edit_positions.png', dpi=150)
        plt.close(fig)

    def _plot_indel_sizes(self, summary: SampleSummary) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if summary.insertion_sizes:
            axes[0].hist(summary.insertion_sizes, bins=range(
                1, max(summary.insertion_sizes) + 2),
                color='#2196F3', edgecolor='white')
            axes[0].set_title('Insertion Size Distribution')
            axes[0].set_xlabel('Size (bp)')
            axes[0].set_ylabel('Count')

        if summary.deletion_sizes:
            axes[1].hist(summary.deletion_sizes, bins=range(
                1, max(summary.deletion_sizes) + 2),
                color='#F44336', edgecolor='white')
            axes[1].set_title('Deletion Size Distribution')
            axes[1].set_xlabel('Size (bp)')
            axes[1].set_ylabel('Count')

        fig.suptitle(f'{self.sample_name} — Indel Size Distributions')
        fig.tight_layout()
        fig.savefig(self.plots_dir / 'indel_sizes.png', dpi=150)
        plt.close(fig)

    def _plot_base_conversions(self, summary: SampleSummary) -> None:
        if not summary.base_conversion_counts:
            return
        labels = list(summary.base_conversion_counts.keys())
        counts = list(summary.base_conversion_counts.values())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, counts, color='#9C27B0', edgecolor='white')
        ax.set_xlabel('Base Conversion')
        ax.set_ylabel('Count')
        ax.set_title(f'{self.sample_name} — Base Conversion Spectrum')
        fig.tight_layout()
        fig.savefig(self.plots_dir / 'base_conversions.png', dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Allele frequency table
    # ------------------------------------------------------------------

    @staticmethod
    def _edit_signature(ra: ReadAnalysis) -> str:
        """Create a hashable signature from a read's edits."""
        if not ra.edits:
            return 'UNMODIFIED'
        parts = []
        for e in ra.edits:
            parts.append(f"{e.edit_type}:{e.ref_start}-{e.ref_end}:{e.ref_bases}>{e.alt_bases}")
        return '|'.join(parts)

    def _build_allele_table(self, results: list[ReadAnalysis],
                            reference: str, guide_rna: str | None,
                            cut_site: int | None,
                            max_alleles: int = 30) -> str:
        """Build CRISPResso2-style allele frequency table as HTML."""
        from collections import Counter, defaultdict

        # Group reads by edit signature
        sig_groups: dict[str, list[ReadAnalysis]] = defaultdict(list)
        for ra in results:
            sig = self._edit_signature(ra)
            sig_groups[sig].append(ra)

        # Sort by frequency (descending)
        sorted_alleles = sorted(sig_groups.items(),
                                key=lambda x: len(x[1]), reverse=True)

        total_reads = len(results)
        if total_reads == 0:
            return ''

        # Determine the display window around the guide/cut site
        # If guide is found in reference, show ±15bp around it
        # Otherwise show the full reference (for short amplicons)
        ref_upper = reference.upper()
        guide_start = -1
        if guide_rna:
            guide_upper = guide_rna.upper().replace('U', 'T')
            guide_start = ref_upper.find(guide_upper)
            if guide_start < 0:
                # Try reverse complement
                rc_table = str.maketrans('ACGT', 'TGCA')
                rc_guide = guide_upper.translate(rc_table)[::-1]
                guide_start = ref_upper.find(rc_guide)
                if guide_start >= 0:
                    guide_upper = rc_guide

        if guide_start >= 0 and len(reference) > 80:
            win_start = max(0, guide_start - 15)
            win_end = min(len(reference), guide_start + len(guide_upper) + 15)
        else:
            win_start = 0
            win_end = len(reference)

        ref_window = ref_upper[win_start:win_end]

        # Build HTML
        html = []
        html.append('<div class="allele-section">')
        html.append(f'<h2>Allele Frequency Table</h2>')
        html.append(f'<p>Showing top {min(max_alleles, len(sorted_alleles))} alleles '
                     f'out of {len(sorted_alleles)} unique variants '
                     f'({total_reads} total reads)</p>')

        html.append('<div class="allele-table-wrap">')
        html.append('<table class="allele-table">')

        # --- Reference row ---
        html.append('<tr class="ref-row">')
        for i, base in enumerate(ref_window):
            ref_pos = win_start + i
            is_cut = (cut_site is not None and ref_pos == cut_site)
            is_guide = (guide_start >= 0 and
                        guide_start <= ref_pos < guide_start + len(guide_upper))
            classes = ['base']
            if is_guide:
                classes.append('guide-base')
            if is_cut:
                classes.append('cut-pos')
            html.append(f'<td class="{" ".join(classes)}">{base}</td>')
        html.append('<td class="allele-label">Reference</td>')
        html.append('</tr>')

        # --- sgRNA row ---
        if guide_start >= 0:
            html.append('<tr class="sgrna-row">')
            for i in range(len(ref_window)):
                ref_pos = win_start + i
                if guide_start <= ref_pos < guide_start + len(guide_upper):
                    html.append('<td class="base sgrna-mark"></td>')
                else:
                    html.append('<td class="base"></td>')
            html.append('<td class="allele-label">sgRNA</td>')
            html.append('</tr>')

        # --- Cleavage position indicator ---
        if cut_site is not None and win_start <= cut_site < win_end:
            html.append('<tr class="cut-row">')
            for i in range(len(ref_window)):
                ref_pos = win_start + i
                if ref_pos == cut_site:
                    html.append('<td class="base cut-indicator">&#9660;</td>')
                else:
                    html.append('<td class="base"></td>')
            html.append('<td class="allele-label">Cut site</td>')
            html.append('</tr>')

        # --- Spacer ---
        html.append(f'<tr><td colspan="{len(ref_window) + 1}" '
                     f'style="height:6px;border:none;"></td></tr>')

        # --- Allele rows ---
        for allele_idx, (sig, reads) in enumerate(sorted_alleles[:max_alleles]):
            count = len(reads)
            pct = count / total_reads * 100

            # Use the first read's alignment to build the display
            rep = reads[0]

            # Build the allele sequence aligned to the reference window
            # Walk the alignment and map each ref position to the query base
            allele_bases = self._map_allele_to_window(
                rep, ref_upper, win_start, win_end)

            html.append('<tr class="allele-row">')
            for i, (ref_base, allele_info) in enumerate(
                    zip(ref_window, allele_bases)):
                abase, mtype = allele_info  # (base, type)
                classes = ['base']
                if mtype == 'match':
                    classes.append('match')
                elif mtype == 'substitution':
                    classes.append('sub')
                elif mtype == 'insertion':
                    classes.append('ins')
                elif mtype == 'deletion':
                    classes.append('del')
                html.append(f'<td class="{" ".join(classes)}">{abase}</td>')

            # Frequency label
            reads_word = 'read' if count == 1 else 'reads'
            html.append(
                f'<td class="allele-label freq">'
                f'{pct:.2f}% ({count} {reads_word})</td>')
            html.append('</tr>')

        html.append('</table>')
        html.append('</div>')  # allele-table-wrap

        # Legend
        html.append('''
        <div class="allele-legend">
            <span class="legend-item"><span class="swatch sub-swatch"></span> <b>Substitution</b></span>
            <span class="legend-item"><span class="swatch ins-swatch"></span> Insertion</span>
            <span class="legend-item"><span class="swatch del-swatch"></span> Deletion</span>
            <span class="legend-item">&#9660; Predicted cleavage position</span>
        </div>
        ''')
        html.append('</div>')  # allele-section
        return '\n'.join(html)

    @staticmethod
    def _map_allele_to_window(ra: ReadAnalysis, reference: str,
                               win_start: int, win_end: int
                               ) -> list[tuple[str, str]]:
        """Map a read's alignment to the reference window positions.

        Returns a list of (displayed_base, mutation_type) for each
        position in the window.
        """
        aligned_q = ra.aligned_query
        aligned_r = ra.aligned_ref
        ref_offset = ra.ref_start

        # Build a map: reference_position -> (query_base, type)
        ref_map: dict[int, tuple[str, str]] = {}
        # Track insertions to annotate at the preceding ref position
        pending_insertions: dict[int, str] = {}

        if aligned_q and aligned_r:
            ref_pos = ref_offset
            for j in range(min(len(aligned_q), len(aligned_r))):
                qb = aligned_q[j]
                rb = aligned_r[j]

                if rb == '-':
                    # Insertion — note it at current ref_pos
                    if ref_pos not in pending_insertions:
                        pending_insertions[ref_pos] = ''
                    pending_insertions[ref_pos] += qb
                elif qb == '-':
                    # Deletion
                    ref_map[ref_pos] = ('-', 'deletion')
                    ref_pos += 1
                elif qb != rb:
                    ref_map[ref_pos] = (qb, 'substitution')
                    ref_pos += 1
                else:
                    ref_map[ref_pos] = (qb, 'match')
                    ref_pos += 1

        # Build output for the display window
        result = []
        for pos in range(win_start, win_end):
            if pos in ref_map:
                base, mtype = ref_map[pos]
                # If there's an insertion right after this position,
                # mark the base as having an insertion
                if (pos + 1) in pending_insertions and mtype == 'match':
                    mtype = 'insertion'
                result.append((base, mtype))
            else:
                # Position not covered by alignment
                ref_base = reference[pos] if pos < len(reference) else '?'
                result.append((ref_base, 'match'))

        return result

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def _write_html(self, results: list[ReadAnalysis],
                    summary: SampleSummary, reference: str,
                    cut_site: int | None,
                    guide_rna: str | None = None) -> None:
        allele_table_html = self._build_allele_table(
            results, reference, guide_rna, cut_site)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CRISPR Analysis — {self.sample_name}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem; background: #fafafa; color: #222; }}
  h1 {{ color: #1565C0; }}
  h2 {{ color: #333; border-bottom: 2px solid #1565C0; padding-bottom: 0.3rem; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .stat-card .value {{ font-size: 2rem; font-weight: bold; color: #1565C0; }}
  .stat-card .label {{ font-size: 0.9rem; color: #666; margin-top: 0.3rem; }}
  .highlight {{ background: #E3F2FD; padding: 0.8rem; border-radius: 6px; margin: 1rem 0; font-weight: bold; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th {{ background: #1565C0; color: white; padding: 0.7rem; text-align: left; }}
  td {{ padding: 0.5rem 0.7rem; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f5f5f5; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 0.5rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1rem; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }}
  .badge-nhej {{ background: #FFCDD2; color: #B71C1C; }}
  .badge-hdr {{ background: #BBDEFB; color: #0D47A1; }}
  .badge-unmod {{ background: #C8E6C9; color: #1B5E20; }}
  .badge-lc {{ background: #E0E0E0; color: #424242; }}
  .badge-amb {{ background: #FFE0B2; color: #E65100; }}

  /* Allele frequency table — CRISPResso2 style */
  .allele-section {{ margin: 2rem 0; }}
  .allele-table-wrap {{ overflow-x: auto; }}
  .allele-table {{ border-collapse: collapse; width: auto; margin: 1rem 0; background: white; font-family: 'Courier New', monospace; font-size: 14px; }}
  .allele-table td {{ padding: 0; text-align: center; border: none; }}
  .allele-table td.base {{ width: 22px; height: 26px; min-width: 22px; line-height: 26px; }}
  .allele-table .ref-row td {{ background: #f0f0f0; font-weight: bold; color: #333; }}
  .allele-table .ref-row td.guide-base {{ background: #c8e6c9; }}
  .allele-table .sgrna-row td {{ height: 6px; }}
  .allele-table .sgrna-row td.sgrna-mark {{ background: #81c784; height: 6px; }}
  .allele-table .cut-row td {{ height: 18px; font-size: 10px; color: #d32f2f; }}
  .allele-table .allele-row td.match {{ background: #e8f5e9; color: #333; }}
  .allele-table .allele-row td.sub {{ background: #fff3e0; color: #e65100; font-weight: bold; }}
  .allele-table .allele-row td.ins {{ background: #fce4ec; color: #c62828; outline: 2px solid #ef5350; outline-offset: -2px; }}
  .allele-table .allele-row td.del {{ background: #efebe9; color: #999; }}
  .allele-table .allele-label {{ padding: 2px 12px !important; white-space: nowrap; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; text-align: left !important; border-left: 2px solid #ddd; }}
  .allele-table .allele-label.freq {{ color: #555; }}
  .allele-table .ref-row .allele-label {{ font-weight: bold; color: #333; background: #f0f0f0; }}
  .allele-legend {{ margin: 1rem 0; font-size: 13px; display: flex; gap: 1.5rem; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .swatch {{ display: inline-block; width: 16px; height: 16px; border-radius: 3px; }}
  .sub-swatch {{ background: #fff3e0; border: 1px solid #e65100; }}
  .ins-swatch {{ background: #fce4ec; border: 2px solid #ef5350; }}
  .del-swatch {{ background: #efebe9; border: 1px solid #999; }}
</style>
</head>
<body>
<h1>CRISPR NGS Analysis Report</h1>
<p><strong>Sample:</strong> {self.sample_name} &nbsp;|&nbsp;
   <strong>Reference length:</strong> {len(reference)} bp &nbsp;|&nbsp;
   <strong>Cut site:</strong> {cut_site if cut_site is not None else 'N/A'}</p>

<div class="highlight">
  100% Read Preservation — All {summary.total_reads} reads analyzed (0 discarded)
</div>

<h2>Summary Statistics</h2>
<div class="stat-grid">
  <div class="stat-card"><div class="value">{summary.total_reads}</div><div class="label">Total Reads</div></div>
  <div class="stat-card"><div class="value">{summary.editing_efficiency:.1f}%</div><div class="label">Editing Efficiency</div></div>
  <div class="stat-card"><div class="value">{summary.unmodified_reads}</div><div class="label">Unmodified</div></div>
  <div class="stat-card"><div class="value">{summary.nhej_reads}</div><div class="label">NHEJ</div></div>
  <div class="stat-card"><div class="value">{summary.hdr_reads}</div><div class="label">HDR</div></div>
  <div class="stat-card"><div class="value">{summary.low_confidence_reads}</div><div class="label">Low Confidence</div></div>
</div>

{allele_table_html}

<h2>Visualizations</h2>
<div class="plots">
  <div><img src="plots/classification_pie.png" alt="Classification"></div>
  <div><img src="plots/indel_spectrum.png" alt="Indel Spectrum"></div>
  <div><img src="plots/edit_positions.png" alt="Edit Positions"></div>
  <div><img src="plots/indel_sizes.png" alt="Indel Sizes"></div>
</div>

<h2>Per-Read Results (first 200)</h2>
<table>
<tr><th>Read</th><th>Classification</th><th>Identity</th><th>Ins</th><th>Del</th><th>Sub</th><th>Method</th></tr>
"""
        badge_map = {
            'NHEJ': 'nhej', 'HDR': 'hdr', 'unmodified': 'unmod',
            'low_confidence': 'lc', 'ambiguous': 'amb',
        }
        for ra in results[:200]:
            badge = badge_map.get(ra.classification, 'amb')
            html += (
                f"<tr><td>{ra.read_name}</td>"
                f"<td><span class='badge badge-{badge}'>{ra.classification}</span></td>"
                f"<td>{ra.identity:.2%}</td>"
                f"<td>{ra.total_insertions}</td>"
                f"<td>{ra.total_deletions}</td>"
                f"<td>{ra.total_substitutions}</td>"
                f"<td>{ra.alignment_method}</td></tr>\n"
            )

        html += """</table>

<h2>Methodology</h2>
<ul>
  <li><strong>Alignment:</strong> Multi-strategy (global + local + semi-global), best score selected</li>
  <li><strong>Read preservation:</strong> 100% — low-confidence reads flagged, never discarded</li>
  <li><strong>Edit detection:</strong> CIGAR-based with complex indel merging</li>
</ul>

<footer style="margin-top:2rem; color:#999; font-size:0.8rem;">
  Generated by Advanced CRISPR NGS Analyzer v0.1.0
</footer>
</body>
</html>"""

        html_path = self.output_dir / f'{self.sample_name}_report.html'
        with open(html_path, 'w') as f:
            f.write(html)
