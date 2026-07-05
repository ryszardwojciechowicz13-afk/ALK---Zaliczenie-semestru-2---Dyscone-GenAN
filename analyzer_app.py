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

def run_app():
    print("ConeDystrophy Genetic Analyzer - development stage 03/34")

