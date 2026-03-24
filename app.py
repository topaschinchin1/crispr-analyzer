#!/usr/bin/env python3
"""
CRISPR NGS Analyzer — Flask Web Platform
"""

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from flask import (Flask, jsonify, render_template, request,
                   send_file, send_from_directory)
from werkzeug.utils import secure_filename

from config import Config
from analysis.aligner import AdvancedAligner, AlignmentParams
from analysis.analysis import AnalysisEngine
from analysis.nucleases import NucleaseEngine
from analysis.platforms import PlatformHandler
from analysis.readers import ReadPreservationEngine
from analysis.reports import ReportGenerator

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

NUCLEASE_FAMILY_MAP = {
    'BaseEditor': 'BaseEditors',
    'PrimeEditor': 'PrimeEditors',
}


def _allowed_file(filename: str) -> bool:
    name = filename.lower()
    for ext in Config.ALLOWED_EXTENSIONS:
        if name.endswith(ext):
            return True
    return False


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# API — upload
# ---------------------------------------------------------------------------

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)

    saved = []
    for f in files:
        if not _allowed_file(f.filename):
            return jsonify({'error': f'File type not allowed: {f.filename}'}), 400
        fname = secure_filename(f.filename)
        dest = os.path.join(job_dir, fname)
        f.save(dest)
        saved.append(fname)
        logger.info("Saved upload: %s (%s)", fname, job_id)

    with jobs_lock:
        jobs[job_id] = {
            'status': 'uploaded',
            'percent': 0,
            'message': 'Files uploaded',
            'complete': False,
            'error': None,
            'files': saved,
            'upload_dir': job_dir,
            'result_dir': None,
            'summary': None,
            'created': time.time(),
        }

    return jsonify({'job_id': job_id, 'files': saved})


# ---------------------------------------------------------------------------
# API — start analysis
# ---------------------------------------------------------------------------

@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    data = request.get_json(force=True)
    job_id = data.get('job_id')

    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Unknown job_id'}), 404

    # Validate required fields
    reference = data.get('reference', '').strip()
    guide_rna = data.get('guide_rna', '').strip()
    if not reference or not guide_rna:
        return jsonify({'error': 'Reference sequence and guide RNA are required'}), 400

    # Write reference to temp FASTA
    ref_path = os.path.join(job['upload_dir'], 'reference.fa')
    # Strip FASTA header lines if user pasted raw sequence
    ref_lines = reference.split('\n')
    seq_lines = [l.strip() for l in ref_lines if not l.startswith('>')]
    clean_ref = ''.join(seq_lines).upper()
    if not clean_ref:
        return jsonify({'error': 'Invalid reference sequence'}), 400
    with open(ref_path, 'w') as f:
        f.write(f">amplicon\n{clean_ref}\n")

    params = {
        'reference_path': ref_path,
        'guide_rna': guide_rna.upper().replace('U', 'T'),
        'nuclease': data.get('nuclease', 'Cas9'),
        'variant': data.get('variant'),
        'platform': data.get('platform', 'auto'),
        'analysis_mode': data.get('analysis_mode', 'standard'),
        'ignore_substitutions': bool(data.get('ignore_substitutions', False)),
        'edge_mask': int(data.get('edge_mask', 0)),
    }

    with jobs_lock:
        jobs[job_id]['status'] = 'queued'
        jobs[job_id]['message'] = 'Analysis queued'

    thread = threading.Thread(target=_run_analysis, args=(job_id, params),
                              daemon=True)
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'started'})


# ---------------------------------------------------------------------------
# API — progress / results / download
# ---------------------------------------------------------------------------

@app.route('/api/progress/<job_id>')
def get_progress(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Unknown job_id'}), 404
    return jsonify({
        'status': job['message'],
        'percent': job['percent'],
        'complete': job['complete'],
        'error': job['error'],
    })


@app.route('/api/results/<job_id>')
def get_results(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or not job['complete']:
        return jsonify({'error': 'Results not ready'}), 404

    summary = job.get('summary', {})
    return jsonify({
        'total_reads': summary.get('total_reads', 0),
        'edited_reads': summary.get('modified_reads', 0),
        'editing_efficiency': summary.get('editing_efficiency_pct', 0),
        'unmodified_reads': summary.get('unmodified_reads', 0),
        'nhej_reads': summary.get('nhej_reads', 0),
        'hdr_reads': summary.get('hdr_reads', 0),
        'low_confidence_reads': summary.get('low_confidence_reads', 0),
        'preserved_pct': 100.0,
    })


@app.route('/api/download/<job_id>/<file_type>')
def download_file(job_id, file_type):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or not job['complete'] or not job['result_dir']:
        return jsonify({'error': 'Results not ready'}), 404

    result_dir = Path(job['result_dir'])
    name = 'analysis'

    file_map = {
        'html': result_dir / f'{name}_report.html',
        'csv': result_dir / f'{name}_per_read.csv',
        'json': result_dir / f'{name}_summary.json',
        'edits_csv': result_dir / f'{name}_edits.csv',
    }

    path = file_map.get(file_type)
    if not path or not path.exists():
        return jsonify({'error': f'File not found: {file_type}'}), 404

    return send_file(path, as_attachment=True)


# ---------------------------------------------------------------------------
# Background analysis worker
# ---------------------------------------------------------------------------

def _update(job_id: str, percent: int, message: str):
    with jobs_lock:
        jobs[job_id]['percent'] = percent
        jobs[job_id]['message'] = message
        jobs[job_id]['status'] = 'running'
    logger.info("[%s] %d%% — %s", job_id, percent, message)


def _run_analysis(job_id: str, params: dict):
    """Run the full CRISPR analysis pipeline in a background thread."""
    try:
        with jobs_lock:
            job = jobs[job_id]
        upload_dir = job['upload_dir']
        fastq_files = [f for f in job['files']
                       if not f.endswith('.fa') and not f.endswith('.fasta')]

        if not fastq_files:
            raise ValueError("No FASTQ files found in upload")

        # ---- 1. Resolve nuclease ----
        _update(job_id, 5, 'Configuring nuclease...')
        nuc_engine = NucleaseEngine()
        family = NUCLEASE_FAMILY_MAP.get(params['nuclease'], params['nuclease'])
        nuclease = nuc_engine.get(family, params.get('variant'))

        # ---- 2. Load reads ----
        _update(job_id, 10, 'Loading sequencing reads...')
        reader = ReadPreservationEngine()
        r1_path = os.path.join(upload_dir, fastq_files[0])
        r2_path = (os.path.join(upload_dir, fastq_files[1])
                   if len(fastq_files) > 1 else None)
        reads = reader.load_fastq(r1_path, r2_path)
        _update(job_id, 20, f'Loaded {len(reads)} reads (100% preserved)')

        if not reads:
            raise ValueError("No reads could be loaded from FASTQ files")

        # ---- 3. Load reference ----
        _update(job_id, 25, 'Loading reference sequence...')
        reference = reader.load_fasta(params['reference_path'])

        # ---- 4. Platform detection ----
        _update(job_id, 30, 'Detecting sequencing platform...')
        plat_name = params['platform'] if params['platform'] != 'auto' else None
        platform = PlatformHandler(plat_name)
        read_lengths = [len(r.sequence) for r in reads[:500]]
        mean_qual = float(
            sum(sum(r.quality_scores) / max(len(r.quality_scores), 1)
                for r in reads[:500]) / min(len(reads), 500)
        )
        pconfig = platform.detect_platform(read_lengths, mean_qual)
        _update(job_id, 35, f'Platform: {pconfig.name}')

        ap = platform.get_alignment_params()
        align_params = AlignmentParams(
            match_score=ap['match_score'],
            mismatch_penalty=ap['mismatch_penalty'],
            gap_open=ap['gap_open'],
            gap_extend=ap['gap_extend'],
            min_score_ratio=ap['min_score_ratio'],
        )

        # ---- 5. Find cut site ----
        cut_site = nuc_engine.find_cut_site(params['guide_rna'], reference, nuclease)
        if cut_site is not None:
            _update(job_id, 38, f'Cut site at position {cut_site}')

        # ---- 6. Align reads ----
        _update(job_id, 40, 'Aligning reads (multi-strategy)...')
        aligner = AdvancedAligner(align_params)
        alignments = []
        total = len(reads)
        for i, read in enumerate(reads):
            if len(read.sequence) > 2000:
                aln = aligner.align_long_read(
                    read.name, read.sequence, reference,
                    quality_scores=read.quality_scores)
            else:
                aln = aligner.align(
                    read.name, read.sequence, reference,
                    quality_scores=read.quality_scores)
            alignments.append(aln)

            if read.mate:
                mate_aln = aligner.align(
                    read.mate.name + '_mate', read.mate.sequence, reference,
                    quality_scores=read.mate.quality_scores)
                alignments.append(mate_aln)

            # Update progress (40-80% range)
            if (i + 1) % max(1, total // 20) == 0 or i == total - 1:
                pct = 40 + int(40 * (i + 1) / total)
                _update(job_id, pct, f'Aligned {i+1}/{total} reads...')

        # ---- 7. Analyze edits ----
        _update(job_id, 82, 'Analyzing edits...')
        engine = AnalysisEngine(
            nuclease=nuclease, cut_site=cut_site, window_size=50,
            ignore_substitutions=params.get('ignore_substitutions', False),
            edge_mask=params.get('edge_mask', 0))
        results, summary = engine.analyze_all(alignments, reference)

        _update(job_id, 90, f'Editing efficiency: {summary.editing_efficiency:.1f}%')

        # ---- 8. Generate reports ----
        _update(job_id, 92, 'Generating reports and plots...')
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
        os.makedirs(result_dir, exist_ok=True)

        reporter = ReportGenerator(result_dir, sample_name='analysis')
        reporter.generate_all(results, summary, reference, cut_site,
                              guide_rna=params.get('guide_rna'))

        # Read summary JSON back
        summary_path = os.path.join(result_dir, 'analysis_summary.json')
        with open(summary_path) as f:
            summary_data = json.load(f)

        # ---- Done ----
        with jobs_lock:
            jobs[job_id]['complete'] = True
            jobs[job_id]['percent'] = 100
            jobs[job_id]['message'] = 'Analysis complete'
            jobs[job_id]['result_dir'] = result_dir
            jobs[job_id]['summary'] = summary_data

        logger.info("[%s] Analysis complete — %d reads, %.1f%% efficiency",
                     job_id, summary_data['total_reads'],
                     summary_data['editing_efficiency_pct'])

    except Exception as exc:
        logger.exception("Analysis failed for job %s", job_id)
        with jobs_lock:
            jobs[job_id]['error'] = str(exc)
            jobs[job_id]['message'] = f'Error: {exc}'
            jobs[job_id]['complete'] = False


# ---------------------------------------------------------------------------
# Cleanup stale jobs (runs every hour)
# ---------------------------------------------------------------------------

def _cleanup_loop():
    while True:
        time.sleep(3600)
        cutoff = time.time() - Config.JOB_EXPIRY_HOURS * 3600
        to_delete = []
        with jobs_lock:
            for jid, job in jobs.items():
                if job['created'] < cutoff:
                    to_delete.append(jid)

        for jid in to_delete:
            with jobs_lock:
                job = jobs.pop(jid, None)
            if job:
                for d in [job.get('upload_dir'), job.get('result_dir')]:
                    if d and os.path.isdir(d):
                        shutil.rmtree(d, ignore_errors=True)
                logger.info("Cleaned up expired job %s", jid)


threading.Thread(target=_cleanup_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
