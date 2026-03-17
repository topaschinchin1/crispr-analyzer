"""Configuration for the CRISPR Web Platform."""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(32).hex())
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
    ALLOWED_EXTENSIONS = {'.fastq', '.fq', '.fa', '.fasta', '.gz', '.bz2'}
    JOB_EXPIRY_HOURS = 24
    MAX_CONCURRENT_JOBS = 4
