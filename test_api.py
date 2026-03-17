#!/usr/bin/env python3
"""End-to-end API test for the CRISPR web platform."""

import io
import json
import random
import time
import requests

BASE = 'http://localhost:5000'
random.seed(42)

# -- Synthetic data --
REFERENCE = (
    "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    "AATTCCGGAATTCCGGAATTCCGGAATTCCGGAATTCCGGAATTCCGG"
    "TTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCC"
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA"
    "CCAATTGGCCAATTGGCCAATTGGCCAATTGGCCAATTGG"
    "GGTTAACCGGTTAACCGGTTAACCGGTTAACCGGTTAACC"
)
GUIDE = REFERENCE[100:120]


def make_fastq(n=50):
    lines = []
    for i in range(n):
        seq = list(REFERENCE)
        # 60% mutated at cut site
        if random.random() < 0.6:
            pos = 117
            if random.random() < 0.5:
                del seq[pos:pos+3]
            else:
                seq.insert(pos, ''.join(random.choices('ACGT', k=4)))
        seq_str = ''.join(seq)
        qual = 'I' * len(seq_str)
        lines.append(f"@read_{i}\n{seq_str}\n+\n{qual}\n")
    return ''.join(lines)


print("1. Uploading FASTQ...")
fastq_data = make_fastq(50)
files = {'files': ('test.fastq', io.BytesIO(fastq_data.encode()), 'application/octet-stream')}
r = requests.post(f'{BASE}/api/upload', files=files)
assert r.status_code == 200, f"Upload failed: {r.text}"
job_id = r.json()['job_id']
print(f"   Job ID: {job_id}")

print("2. Starting analysis...")
r = requests.post(f'{BASE}/api/analyze', json={
    'job_id': job_id,
    'reference': REFERENCE,
    'guide_rna': GUIDE,
    'nuclease': 'Cas9',
    'variant': 'SpCas9',
    'platform': 'illumina',
    'analysis_mode': 'standard',
})
assert r.status_code == 200, f"Analyze failed: {r.text}"
print(f"   Status: {r.json()['status']}")

print("3. Polling progress...")
for _ in range(60):
    time.sleep(1)
    r = requests.get(f'{BASE}/api/progress/{job_id}')
    p = r.json()
    print(f"   {p['percent']}% — {p['status']}")
    if p['complete']:
        break
    if p['error']:
        print(f"   ERROR: {p['error']}")
        break

print("4. Fetching results...")
r = requests.get(f'{BASE}/api/results/{job_id}')
results = r.json()
print(json.dumps(results, indent=2))

print("\n5. Testing downloads...")
for ft in ['html', 'csv', 'json']:
    r = requests.get(f'{BASE}/api/download/{job_id}/{ft}')
    print(f"   {ft}: {r.status_code} ({len(r.content)} bytes)")

print("\nAll tests passed!")
