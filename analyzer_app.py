from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

APP_TITLE = 'ConeDystrophy Genetic Analyzer 1.4 — Modern UI'

STANDARD_COLUMNS = [
    'patient_id',
    'sample_id',
    'family_id',
    'gene',
    'chromosome',
    'position',
    'ref',
    'alt',
    'genotype',
    'zygosity',
    'variant_id',
    'hgvs_c',
    'hgvs_p',
    'transcript',
    'consequence',
    'quality',
    'depth',
    'allele_depth',
    'filter',
    'population_frequency',
    'clinical_significance',
    'inheritance',
]

ALIASES = {
    'patient_id': [
        'patient_id',
        'patient',
        'patientid',
        'subject_id',
        'subject',
        'pacjent',
        'id_pacjenta',
    ],
    'sample_id': [
        'sample_id',
        'sample',
        'sampleid',
        'probka',
        'id_probki',
    ],
    'family_id': ['family_id', 'family', 'pedigree', 'rodzina'],
    'gene': [
        'gene',
        'gene_symbol',
        'symbol',
        'hgnc',
        'gene.refgene',
        'generef',
        'genename',
    ],
    'chromosome': ['chromosome', 'chrom', 'chr', '#chrom'],
    'position': [
        'position',
        'pos',
        'start',
        'coordinate',
        'genomic_position',
    ],
    'ref': ['ref', 'reference', 'reference_allele'],
    'alt': ['alt', 'alternative', 'alternate', 'alternative_allele'],
    'genotype': ['genotype', 'gt', 'geno'],
    'zygosity': ['zygosity', 'zyg', 'zygosity_status'],
    'variant_id': ['variant_id', 'variant', 'rsid', 'rs_id'],
    'hgvs_c': ['hgvs_c', 'hgvsc', 'coding_hgvs', 'cdna_change'],
    'hgvs_p': ['hgvs_p', 'hgvsp', 'protein_hgvs', 'protein_change'],
    'transcript': ['transcript', 'transcript_id', 'feature'],
    'consequence': ['consequence', 'effect', 'annotation', 'variant_effect'],
    'quality': ['quality', 'qual'],
    'depth': ['depth', 'dp'],
    'allele_depth': ['allele_depth', 'ad'],
    'filter': ['filter', 'vcf_filter'],
    'population_frequency': ['population_frequency', 'af', 'gnomad_af'],
    'clinical_significance': ['clinical_significance', 'clinvar', 'significance'],
    'inheritance': ['inheritance', 'mode_of_inheritance'],
}

VALID_CHROMS = {str(i) for i in range(1, 23)} | {'X', 'Y', 'MT'}
GT_RE = re.compile('^(?:\\d+|\\.)[\\/|](?:\\d+|\\.)$|^(?:\\d+|\\.)$')

def norm_col(s):
    return str(s).strip().lower().replace(' ', '_').replace('-', '_')

def suggest_mapping(columns):
    reverse = {norm_col(v): k for k, vals in ALIASES.items() for v in vals}
    return {c: reverse.get(norm_col(c), '') for c in columns}

def normalize_chrom(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip()
    if s.lower().startswith('chr'):
        s = s[3:]
    s = s.upper()
    return 'MT' if s == 'M' else s

def normalize_gt(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip()
    map_ = {'HET': '0/1', 'HETEROZYGOUS': '0/1', 'HETEROZYGOTA': '0/1', 'HOM': '1/1', 'HOMOZYGOUS': '1/1', 'HOMOZYGOTA': '1/1', 'WT': '0/0', 'WILDTYPE': '0/0'}
    return map_.get(s.upper(), s)

def zygosity(gt):
    if pd.isna(gt):
        return 'missing'
    s = str(gt).strip()
    if s in {'', '.', './.', '.|.'}:
        return 'missing'
    sep = '|' if '|' in s else '/' if '/' in s else None
    if not sep:
        return 'unknown'
    p = s.split(sep)
    if len(p) != 2 or '.' in p:
        return 'missing'
    if p[0] == p[1] == '0':
        return 'homozygous_reference'
    if p[0] == p[1] and p[0] != '0':
        return 'homozygous_alternative'
    if p[0] != p[1]:
        return 'heterozygous'
    return 'unknown'

def variant_type(ref, alt):
    if pd.isna(ref) or pd.isna(alt):
        return 'unknown'
    r, a = (str(ref).strip(), str(alt).strip())
    if ',' in a:
        return 'multiallelic'
    if len(r) == 1 and len(a) == 1:
        return 'SNV'
    if len(r) < len(a):
        return 'insertion'
    if len(r) > len(a):
        return 'deletion'
    if len(r) == len(a) and len(r) > 1:
        return 'MNV'
    return 'complex'

def standardize(df):
    out = df.copy()
    if 'chromosome' in out:
        out['chromosome'] = out['chromosome'].map(normalize_chrom)
    if 'gene' in out:
        out['gene'] = out['gene'].astype('string').str.strip().str.upper()
    if 'genotype' in out:
        out['genotype'] = out['genotype'].map(normalize_gt)
        out['zygosity'] = out['genotype'].map(zygosity)
    if 'position' in out:
        out['position'] = pd.to_numeric(out['position'], errors='coerce').astype('Int64')
    for c in ['ref', 'alt']:
        if c in out:
            out[c] = out[c].astype('string').str.strip().str.upper()
    for c in ['patient_id', 'sample_id', 'consequence', 'hgvs_c', 'hgvs_p', 'variant_id']:
        if c in out:
            out[c] = out[c].astype('string').str.strip()
    if {'ref', 'alt'} <= set(out.columns):
        out['variant_type'] = [variant_type(r, a) for r, a in zip(out['ref'], out['alt'])]
    return out

def read_vcf(path):
    opener = gzip.open if str(path).lower().endswith('.gz') else open
    rows, samples = ([], [])
    with opener(path, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                p = line.rstrip().split('\t')
                samples = p[9:]
                continue
            if line.startswith('#') or not line.strip():
                continue
            p = line.rstrip().split('\t')
            if len(p) < 8:
                continue
            chrom, pos, vid, ref, alt, qual, filt, info = p[:8]
            info_map = {}
            for tok in info.split(';'):
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    info_map[k] = v
            gene = info_map.get('GENE') or info_map.get('SYMBOL') or info_map.get('Gene') or ''
            cons = info_map.get('Consequence') or info_map.get('ANN') or info_map.get('CSQ') or ''
            fmt = p[8].split(':') if len(p) > 8 else []
            vals = p[9:] if len(p) > 9 else []
            if vals:
                for i, sv in enumerate(vals):
                    fm = dict(zip(fmt, sv.split(':')))
                    sid = samples[i] if i < len(samples) else f'sample_{i + 1}'
                    rows.append({'patient_id': sid, 'sample_id': sid, 'gene': gene, 'chromosome': chrom, 'position': pos, 'ref': ref, 'alt': alt, 'genotype': fm.get('GT', ''), 'variant_id': '' if vid == '.' else vid, 'quality': '' if qual == '.' else qual, 'depth': fm.get('DP', ''), 'allele_depth': fm.get('AD', ''), 'filter': filt, 'consequence': cons})
            else:
                rows.append({'patient_id': '', 'sample_id': '', 'gene': gene, 'chromosome': chrom, 'position': pos, 'ref': ref, 'alt': alt, 'genotype': '', 'variant_id': '' if vid == '.' else vid, 'quality': '' if qual == '.' else qual, 'filter': filt, 'consequence': cons})
    return pd.DataFrame(rows)

def read_data(path, sheet_name=None):
    p = Path(path)
    n = p.name.lower()
    if n.endswith('.vcf') or n.endswith('.vcf.gz'):
        return read_vcf(p)
    if p.suffix.lower() == '.csv':
        return pd.read_csv(p, sep=None, engine='python')
    if p.suffix.lower() in {'.tsv', '.txt'}:
        return pd.read_csv(p, sep=None, engine='python')
    if p.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(p, sheet_name=sheet_name if sheet_name is not None else 0)
    raise ValueError('Nieobsługiwany format pliku.')

def run_app():
    print("ConeDystrophy Genetic Analyzer - development stage 11/34")

