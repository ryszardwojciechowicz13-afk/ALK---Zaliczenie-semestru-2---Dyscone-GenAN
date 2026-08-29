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

def excel_sheet_score(df):
    mapped = suggest_mapping(df.columns)
    recognized = {v for v in mapped.values() if v}
    core = {'patient_id', 'gene', 'chromosome', 'position', 'ref', 'alt', 'genotype'}
    return 10 * len(recognized & core) + len(recognized)

def suggest_excel_sheet(path):
    xls = pd.ExcelFile(path)
    preferred = ['Genetic_Data_Clean', 'Genetic_Data_QC_Test', 'Clean_Data', 'Filtered_Data', 'Variants']
    for name in preferred:
        if name in xls.sheet_names:
            return (name, xls.sheet_names)
    best = None
    best_score = -1
    for name in xls.sheet_names:
        try:
            probe = pd.read_excel(path, sheet_name=name, nrows=10)
            score = excel_sheet_score(probe)
            if score > best_score:
                best_score = score
                best = name
        except Exception:
            pass
    return (best or xls.sheet_names[0], xls.sheet_names)

def run_qc(df):
    issues = []

    def add(mask, code, severity, msg):
        if mask is None:
            return
        mask = mask.fillna(False) if hasattr(mask, 'fillna') else mask
        for idx in df.index[mask][:5000]:
            issues.append({'row': idx, 'severity': severity, 'code': code, 'message': msg})
    required = ['patient_id', 'chromosome', 'position', 'ref', 'alt']
    missing_cols = [c for c in required if c not in df]
    for c in required:
        if c in df:
            add(df[c].isna() | (df[c].astype('string').str.strip() == ''), f'MISSING_{c.upper()}', 'ERROR', f'Brak wartości: {c}')
    if 'chromosome' in df:
        c = df['chromosome'].map(normalize_chrom)
        add(~c.isin(VALID_CHROMS) & c.notna(), 'INVALID_CHROMOSOME', 'ERROR', 'Nieprawidłowy chromosom')
    if 'position' in df:
        p = pd.to_numeric(df['position'], errors='coerce')
        add(p.isna() | (p <= 0), 'INVALID_POSITION', 'ERROR', 'Nieprawidłowa pozycja genomowa')
    if 'genotype' in df:
        g = df['genotype'].map(normalize_gt).astype('string')
        add(g.notna() & (g != '') & ~g.str.match(GT_RE, na=False), 'INVALID_GENOTYPE', 'WARNING', 'Nierozpoznany genotyp')
    if {'ref', 'alt'} <= set(df.columns):
        r = df['ref'].astype('string').str.upper().str.strip()
        a = df['alt'].astype('string').str.upper().str.strip()
        rx = '^[ACGTN*.-]+(?:,[ACGTN*.-]+)*$'
        add(r.notna() & ~r.str.match(rx, na=False), 'INVALID_REF', 'ERROR', 'Nieprawidłowy REF')
        add(a.notna() & ~a.str.match(rx, na=False), 'INVALID_ALT', 'ERROR', 'Nieprawidłowy ALT')
        add((r == a) & r.notna(), 'REF_EQUALS_ALT', 'WARNING', 'REF i ALT są identyczne')
    dupcols = [c for c in ['patient_id', 'chromosome', 'position', 'ref', 'alt', 'genotype'] if c in df]
    if dupcols:
        add(df.duplicated(dupcols, keep=False), 'DUPLICATE', 'WARNING', 'Potencjalny duplikat')
    issues = pd.DataFrame(issues, columns=['row', 'severity', 'code', 'message'])
    errors = int((issues['severity'] == 'ERROR').sum()) if len(issues) else 0
    warnings = int((issues['severity'] == 'WARNING').sum()) if len(issues) else 0
    miss = []
    for c in df.columns:
        m = int(df[c].isna().sum())
        miss.append({'column': c, 'missing_count': m, 'missing_percent': 100 * m / max(len(df), 1)})
    summary = {'rows': len(df), 'columns': len(df.columns), 'missing_required_columns': missing_cols, 'errors': errors, 'warnings': warnings, 'duplicate_rows': int(df.duplicated(dupcols, keep=False).sum()) if dupcols else 0, 'quality_score': max(0, 100 - 100 * (errors + 0.25 * warnings) / max(len(df), 1))}
    return (summary, issues, pd.DataFrame(miss))

def clean_data(df, opts):
    out = df.copy()
    log = []
    if opts.get('trim'):
        for c in out.select_dtypes(include=['object', 'string']).columns:
            out[c] = out[c].astype('string').str.strip()
        log.append('Usunięto zbędne spacje.')
    if opts.get('na'):
        out = out.replace('^\\s*$', pd.NA, regex=True).replace(['NA', 'N/A', 'NULL', 'None', 'none', '-', 'blank'], pd.NA)
        log.append('Ujednolicono braki danych.')
    if opts.get('std'):
        out = standardize(out)
        log.append('Wykonano standaryzację danych genetycznych.')
    if opts.get('empty'):
        n = len(out)
        out = out.dropna(how='all')
        log.append(f'Usunięto {n - len(out)} pustych rekordów.')
    if opts.get('dups'):
        subset = [c for c in ['patient_id', 'chromosome', 'position', 'ref', 'alt', 'genotype'] if c in out]
        n = len(out)
        out = out.drop_duplicates(subset=subset if subset else None)
        log.append(f'Usunięto {n - len(out)} duplikatów.')
    return (out.reset_index(drop=True), log)

def summary(df):
    patients = df['patient_id'].nunique() if 'patient_id' in df else 0
    genes = df['gene'].nunique() if 'gene' in df else 0
    pp = df.groupby('patient_id').size() if 'patient_id' in df and len(df) else pd.Series(dtype=float)
    return {'patients': int(patients), 'variants': len(df), 'genes': int(genes), 'mean': float(pp.mean()) if len(pp) else 0, 'median': float(pp.median()) if len(pp) else 0, 'min': int(pp.min()) if len(pp) else 0, 'max': int(pp.max()) if len(pp) else 0}

def top_genes(df, n=30):
    if 'gene' not in df:
        return pd.DataFrame(columns=['gene', 'patients', 'variants'])
    g = df.dropna(subset=['gene']).groupby('gene')
    out = g.size().rename('variants').to_frame()
    out['patients'] = g['patient_id'].nunique() if 'patient_id' in df else 0
    return out.reset_index().sort_values(['patients', 'variants'], ascending=False).head(n)

def top_variants(df, n=50):
    cols = [c for c in ['chromosome', 'position', 'ref', 'alt'] if c in df]
    if len(cols) < 4:
        return pd.DataFrame()
    g = df.groupby(cols, dropna=False)
    out = g.size().rename('records').to_frame()
    out['patients'] = g['patient_id'].nunique() if 'patient_id' in df else 0
    return out.reset_index().sort_values(['patients', 'records'], ascending=False).head(n)

class PandasModel(QAbstractTableModel):

    def __init__(self, df=None):
        super().__init__()
        self.df = df if df is not None else pd.DataFrame()

    def set_df(self, df):
        self.beginResetModel()
        self.df = df if df is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return len(self.df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            v = self.df.iat[index.row(), index.column()]
            return '' if pd.isna(v) else str(v)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return str(self.df.columns[section]) if orientation == Qt.Horizontal else str(section + 1)

class DataTable(QWidget):

    def __init__(self, title=''):
        super().__init__()
        self.setObjectName('tableCard')
        l = QVBoxLayout(self)
        l.setContentsMargins(16, 14, 16, 16)
        l.setSpacing(10)
        header = QHBoxLayout()
        header.setSpacing(8)
        if title:
            q = QLabel(title)
            q.setObjectName('sectionTitle')
            header.addWidget(q)
        header.addStretch()
        self.count_lbl = QLabel('0 rekordów')
        self.count_lbl.setObjectName('mutedLabel')
        header.addWidget(self.count_lbl)
        l.addLayout(header)
        self.view = QTableView()
        self.model = PandasModel()
        self.view.setModel(self.model)
        self.view.setAlternatingRowColors(True)
        self.view.setShowGrid(False)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.view.verticalHeader().setVisible(False)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setStretchLastSection(True)
        l.addWidget(self.view)

    def set_df(self, df, limit=1000):
        base = df if df is not None else pd.DataFrame()
        self.model.set_df(base.head(limit).copy())
        shown = min(len(base), limit)
        self.count_lbl.setText(f'{shown:,} / {len(base):,} rekordów'.replace(',', ' '))

class PlotCanvas(FigureCanvas):

    def __init__(self):
        self.fig = Figure(figsize=(8, 5), facecolor='#FFFFFF')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setStyleSheet('background: transparent; border: none;')

    def reset_axes(self):
        self.fig.clear()
        self.fig.set_facecolor('#FFFFFF')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#FFFFFF')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        return self.ax

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1540, 920)
        self.setMinimumSize(1180, 720)
        self.raw_df = self.clean_df = self.filtered_df = None
        self.mapping = {}
        self.source_path = ''
        self.qc_summary = {}
        self.qc_issues = pd.DataFrame()
        self.missing_df = pd.DataFrame()
        self.map_boxes = {}
        self.nav_buttons = []
        self.build_ui()
        self.apply_modern_theme()
        self.set_nav_index(0)
        self.statusBar().showMessage('Gotowy — wczytaj dane, aby rozpocząć analizę.')

    def apply_modern_theme(self):
        self.setStyleSheet('\n            * { font-family: "Segoe UI", Arial, sans-serif; }\n            QMainWindow, QWidget#appRoot { background: #F5F7FB; color: #172033; }\n            QWidget#sidebar { background: #0F172A; }\n            QLabel#brandTitle { color: #F8FAFC; font-size: 20px; font-weight: 700; }\n            QLabel#brandSubtitle { color: #94A3B8; font-size: 11px; }\n            QLabel#sidebarSection { color: #64748B; font-size: 10px; font-weight: 700; letter-spacing: 1px; }\n            QPushButton#nav {\n                color: #CBD5E1; background: transparent; border: none; border-radius: 9px;\n                text-align: left; padding: 10px 12px; font-size: 13px; font-weight: 600;\n            }\n            QPushButton#nav:hover { background: #172033; color: #FFFFFF; }\n            QPushButton#nav:checked { background: #2563EB; color: #FFFFFF; }\n            QPushButton#sidebarAction {\n                background: #172033; color: #E2E8F0; border: 1px solid #334155; border-radius: 9px;\n                padding: 9px 12px; text-align: left; font-weight: 600;\n            }\n            QPushButton#sidebarAction:hover { background: #1E293B; border-color: #475569; }\n            QLabel#pageTitle { color: #111827; font-size: 25px; font-weight: 700; }\n            QLabel#pageSubtitle { color: #64748B; font-size: 12px; }\n            QLabel#sectionTitle { color: #1E293B; font-size: 14px; font-weight: 700; }\n            QLabel#mutedLabel { color: #64748B; font-size: 11px; }\n            QLabel#statusPill { background: #EFF6FF; color: #1D4ED8; border-radius: 8px; padding: 7px 10px; font-weight: 600; }\n            QFrame#metricCard, QGroupBox, QWidget#tableCard, QFrame#card {\n                background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;\n            }\n            QFrame#metricCard { min-height: 112px; }\n            QLabel#metricCaption { color: #64748B; font-size: 11px; font-weight: 600; }\n            QLabel#metricValue { color: #0F172A; font-size: 28px; font-weight: 700; }\n            QLabel#metricHint { color: #94A3B8; font-size: 10px; }\n            QGroupBox { font-weight: 700; padding-top: 14px; margin-top: 8px; }\n            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 5px; color: #334155; }\n            QPushButton {\n                background: #FFFFFF; color: #334155; border: 1px solid #CBD5E1; border-radius: 8px;\n                padding: 8px 13px; font-weight: 600;\n            }\n            QPushButton:hover { background: #F8FAFC; border-color: #94A3B8; }\n            QPushButton#primaryButton { background: #2563EB; color: #FFFFFF; border: 1px solid #2563EB; }\n            QPushButton#primaryButton:hover { background: #1D4ED8; border-color: #1D4ED8; }\n            QPushButton#successButton { background: #059669; color: #FFFFFF; border: 1px solid #059669; }\n            QPushButton#successButton:hover { background: #047857; }\n            QLineEdit, QComboBox, QSpinBox {\n                background: #FFFFFF; color: #172033; border: 1px solid #CBD5E1; border-radius: 8px;\n                padding: 7px 9px; min-height: 18px;\n            }\n            QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border: 1px solid #3B82F6; }\n            QTextEdit { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 9px; padding: 7px; }\n            QProgressBar { background: #E2E8F0; border: none; border-radius: 5px; height: 9px; text-align: center; color: transparent; }\n            QProgressBar::chunk { background: #2563EB; border-radius: 5px; }\n            QTableView {\n                background: #FFFFFF; alternate-background-color: #F8FAFC; border: none; border-radius: 8px;\n                selection-background-color: #DBEAFE; selection-color: #172033;\n            }\n            QHeaderView::section {\n                background: #F1F5F9; color: #475569; padding: 8px; border: none; border-bottom: 1px solid #E2E8F0;\n                font-weight: 700; font-size: 11px;\n            }\n            QSplitter::handle { background: #E2E8F0; width: 1px; height: 1px; }\n            QScrollArea { background: transparent; border: none; }\n            QScrollBar:vertical { background: #F1F5F9; width: 10px; margin: 4px 2px 4px 2px; border-radius: 5px; }\n            QScrollBar::handle:vertical { background: #CBD5E1; min-height: 34px; border-radius: 5px; }\n            QScrollBar::handle:vertical:hover { background: #94A3B8; }\n            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }\n            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }\n            QCheckBox { spacing: 8px; }\n            QCheckBox::indicator { width: 17px; height: 17px; }\n            QStatusBar { background: #FFFFFF; color: #64748B; border-top: 1px solid #E2E8F0; }\n        ')

    def page_header(self, title, subtitle=''):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(2)
        t = QLabel(title)
        t.setObjectName('pageTitle')
        lay.addWidget(t)
        if subtitle:
            s = QLabel(subtitle)
            s.setObjectName('pageSubtitle')
            s.setWordWrap(True)
            lay.addWidget(s)
        return box

    def card(self):
        f = QFrame()
        f.setObjectName('card')
        return f

    def primary(self, text, slot=None):
        b = QPushButton(text)
        b.setObjectName('primaryButton')
        if slot:
            b.clicked.connect(slot)
        return b

    def set_nav_index(self, index):
        self.stack.setCurrentIndex(index)
        for i, b in enumerate(self.nav_buttons):
            b.setChecked(i == index)

    def build_ui(self):
        root = QWidget()
        root.setObjectName('appRoot')
        self.setCentralWidget(root)
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        nav = QWidget()
        nav.setObjectName('sidebar')
        nav.setFixedWidth(252)
        nl = QVBoxLayout(nav)
        nl.setContentsMargins(18, 22, 18, 18)
        nl.setSpacing(7)
        brand = QLabel('ConeDystrophy')
        brand.setObjectName('brandTitle')
        nl.addWidget(brand)
        sub = QLabel('GENETIC ANALYZER  •  v1.3')
        sub.setObjectName('brandSubtitle')
        nl.addWidget(sub)
        nl.addSpacing(20)
        sec = QLabel('ANALIZA')
        sec.setObjectName('sidebarSection')
        nl.addWidget(sec)
        self.stack = QStackedWidget()
        self.stack.setContentsMargins(0, 0, 0, 0)
        pages = [('Dashboard', self.pg_dash), ('Import i mapowanie', self.pg_import), ('Kontrola jakości', self.pg_qc), ('Czyszczenie', self.pg_clean), ('Filtrowanie', self.pg_filter), ('Pacjenci', self.pg_patient), ('Geny', self.pg_gene), ('Warianty', self.pg_variant), ('Statystyki', self.pg_stats), ('Wykresy', self.pg_plot), ('Raport i eksport', self.pg_export), ('Ustawienia', self.pg_settings)]
        for i, (name, fn) in enumerate(pages):
            b = QPushButton(name)
            b.setObjectName('nav')
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.clicked.connect(lambda checked=False, x=i: self.set_nav_index(x))
            nl.addWidget(b)
            self.nav_buttons.append(b)
            self.stack.addWidget(fn())
        nl.addStretch()
        sec2 = QLabel('PROJEKT')
        sec2.setObjectName('sidebarSection')
        nl.addWidget(sec2)
        bs = QPushButton('Zapisz projekt')
        bs.setObjectName('sidebarAction')
        bs.clicked.connect(self.save_project)
        bl = QPushButton('Otwórz projekt')
        bl.setObjectName('sidebarAction')
        bl.clicked.connect(self.load_project)
        nl.addWidget(bs)
        nl.addWidget(bl)
        footer = QLabel('Lokalne przetwarzanie danych')
        footer.setObjectName('brandSubtitle')
        nl.addSpacing(5)
        nl.addWidget(footer)
        h.addWidget(nav)
        h.addWidget(self.stack, 1)

    def make_metric_box(self, name, hint=''):
        g = QFrame()
        g.setObjectName('metricCard')
        l = QVBoxLayout(g)
        l.setContentsMargins(18, 15, 18, 14)
        l.setSpacing(5)
        cap = QLabel(name)
        cap.setObjectName('metricCaption')
        l.addWidget(cap)
        v = QLabel('0')
        v.setObjectName('metricValue')
        l.addWidget(v)
        l.addStretch()
        hh = QLabel(hint or '')
        hh.setObjectName('metricHint')
        l.addWidget(hh)
        return (g, v)

    def _page_layout(self, w):
        l = QVBoxLayout(w)
        l.setContentsMargins(26, 22, 26, 24)
        l.setSpacing(16)
        return l

    def pg_dash(self):
        w = QWidget()
        l = self._page_layout(w)
        l.addWidget(self.page_header('Dashboard', 'Szybki przegląd jakości i struktury aktualnie analizowanego zbioru danych.'))
        r = QHBoxLayout()
        r.setSpacing(12)
        self.b1, self.m1 = self.make_metric_box('PACJENCI', 'unikalne identyfikatory')
        self.b2, self.m2 = self.make_metric_box('REKORDY / WARIANTY', 'aktualny zbiór roboczy')
        self.b3, self.m3 = self.make_metric_box('GENY', 'unikalne symbole genów')
        self.b4, self.m4 = self.make_metric_box('JAKOŚĆ QC', 'wynik kontroli jakości')
        for x in [self.b1, self.b2, self.b3, self.b4]:
            r.addWidget(x)
        l.addLayout(r)
        chart = self.card()
        cl = QVBoxLayout(chart)
        cl.setContentsMargins(16, 14, 16, 12)
        ch = QHBoxLayout()
        tt = QLabel('Najczęstsze geny w kohorcie')
        tt.setObjectName('sectionTitle')
        ch.addWidget(tt)
        ch.addStretch()
        cl.addLayout(ch)
        self.dplot = PlotCanvas()
        cl.addWidget(self.dplot, 1)
        l.addWidget(chart, 1)
        self.dstatus = QLabel('Wczytaj dane, aby rozpocząć.')
        self.dstatus.setObjectName('statusPill')
        self.dstatus.setWordWrap(True)
        l.addWidget(self.dstatus)
        return w

    def pg_import(self):
        w = QWidget()
        l = self._page_layout(w)
        l.addWidget(self.page_header('Import i mapowanie', 'Wczytaj dane, sprawdź rozpoznane kolumny i sprowadź je do wspólnego standardu analitycznego.'))
        top = self.card()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(16, 12, 16, 12)
        b = self.primary('Wczytaj plik danych', self.import_ui)
        self.file_lbl = QLabel('Nie wybrano pliku')
        self.file_lbl.setObjectName('mutedLabel')
        tl.addWidget(b)
        tl.addWidget(self.file_lbl, 1)
        l.addWidget(top)
        sp = QSplitter(Qt.Horizontal)
        self.raw_table = DataTable('Podgląd danych wejściowych')
        sp.addWidget(self.raw_table)
        self.map_scroll = QScrollArea()
        self.map_scroll.setWidgetResizable(True)
        self.map_scroll.setFrameShape(QFrame.NoFrame)
        self.map_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.map_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.map_scroll.setMinimumWidth(360)
        self.map_widget = self.card()
        self.map_widget.setMinimumWidth(330)
        self.map_form = QFormLayout(self.map_widget)
        self.map_form.setContentsMargins(16, 14, 16, 18)
        self.map_form.setHorizontalSpacing(12)
        self.map_form.setVerticalSpacing(10)
        self.map_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.map_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.map_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        mh = QLabel('Mapowanie kolumn')
        mh.setObjectName('sectionTitle')
        self.map_form.addRow(mh)
        hint = QLabel('Przewiń w dół, aby zobaczyć wszystkie kolumny.')
        hint.setObjectName('mutedLabel')
        hint.setWordWrap(True)
        self.map_form.addRow(hint)
        ab = self.primary('Zastosuj mapowanie i standaryzację', self.apply_mapping)
        self.map_form.addRow(ab)
        self.map_scroll.setWidget(self.map_widget)
        sp.addWidget(self.map_scroll)
        sp.setStretchFactor(0, 1)
        sp.setStretchFactor(1, 0)
        sp.setSizes([980, 440])
        l.addWidget(sp, 1)
        return w

    def pg_qc(self):
        w = QWidget()
        l = self._page_layout(w)
        l.addWidget(self.page_header('Kontrola jakości', 'Automatyczne wykrywanie braków, błędnych wartości, nieprawidłowych genotypów i duplikatów.'))
        top = self.card()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(16, 12, 16, 12)
        b = self.primary('Uruchom kontrolę jakości', self.qc_ui)
        self.qc_lbl = QLabel('QC nieuruchomione')
        self.qc_lbl.setObjectName('statusPill')
        tl.addWidget(b)
        tl.addWidget(self.qc_lbl, 1)
        l.addWidget(top)
        self.qprog = QProgressBar()
        self.qprog.setRange(0, 100)
        l.addWidget(self.qprog)
        sp = QSplitter(Qt.Vertical)
        self.qtable = DataTable('Błędy i ostrzeżenia')
        self.mtable = DataTable('Brakujące wartości')
        sp.addWidget(self.qtable)
        sp.addWidget(self.mtable)
        l.addWidget(sp, 1)
        return w

    def pg_clean(self):
        w = QWidget()
        l = self._page_layout(w)
        l.addWidget(self.page_header('Czyszczenie danych', 'Zastosuj bezpieczne, powtarzalne operacje czyszczenia bez modyfikowania oryginalnego pliku RAW.'))
        g = QGroupBox('Opcje czyszczenia')
        gl = QGridLayout(g)
        gl.setContentsMargins(16, 18, 16, 14)
        gl.setHorizontalSpacing(22)
        gl.setVerticalSpacing(10)
        self.c_trim = QCheckBox('Usuń zbędne spacje')
        self.c_na = QCheckBox('Ujednolić braki danych')
        self.c_std = QCheckBox('Standaryzuj dane')
        self.c_empty = QCheckBox('Usuń puste rekordy')
        self.c_dup = QCheckBox('Usuń duplikaty')
        for c in [self.c_trim, self.c_na, self.c_std, self.c_empty, self.c_dup]:
            c.setChecked(True)
        for i, c in enumerate([self.c_trim, self.c_na, self.c_std, self.c_empty, self.c_dup]):
            gl.addWidget(c, i // 3, i % 3)
        l.addWidget(g)
        bar = QHBoxLayout()
        b = self.primary('Uruchom czyszczenie', self.clean_ui)
        bar.addWidget(b)
        bar.addStretch()
        l.addLayout(bar)
        self.clog = QTextEdit()
        self.clog.setReadOnly(True)
        self.clog.setMaximumHeight(120)
        self.clog.setPlaceholderText('Dziennik operacji czyszczenia pojawi się tutaj.')
        l.addWidget(self.clog)
        self.ctable = DataTable('Wyczyszczone dane')
        l.addWidget(self.ctable, 1)
        return w

    def pg_filter(self):
        w = QWidget()
        l = self._page_layout(w)
        l.addWidget(self.page_header('Filtrowanie', 'Zawężaj dane po pacjencie, genie, chromosomie, zygotyczności, typie wariantu i dowolnej frazie.'))
        g = QGroupBox('Filtry analityczne')
        q = QGridLayout(g)
        q.setContentsMargins(16, 18, 16, 14)
        q.setHorizontalSpacing(12)
        q.setVerticalSpacing(10)
        self.fp = QComboBox()
        self.fg = QComboBox()
        self.fc = QComboBox()
        self.fz = QComboBox()
        self.ft = QComboBox()
        self.fcons = QLineEdit()
        self.fs = QLineEdit()
        self.fcons.setPlaceholderText('np. missense')
        self.fs.setPlaceholderText('np. ABCA4, rs123, P001')
        fields = [('Pacjent', self.fp), ('Gen', self.fg), ('Chromosom', self.fc), ('Zygotyczność', self.fz), ('Typ wariantu', self.ft), ('Konsekwencja zawiera', self.fcons)]
        for i, (lab, wd) in enumerate(fields):
            q.addWidget(QLabel(lab), i // 3 * 2, i % 3)
            q.addWidget(wd, i // 3 * 2 + 1, i % 3)
        q.addWidget(QLabel('Wyszukiwanie globalne'), 4, 0)
        q.addWidget(self.fs, 5, 0, 1, 2)
        b = self.primary('Zastosuj filtry', self.filter_ui)
        q.addWidget(b, 5, 2)
        l.addWidget(g)
        self.flbl = QLabel('Brak aktywnych filtrów.')
        self.flbl.setObjectName('statusPill')
        l.addWidget(self.flbl)
        self.ftable = DataTable('Wyniki filtrowania')
        l.addWidget(self.ftable, 1)
        return w

def run_app():
    print("ConeDystrophy Genetic Analyzer - development stage 28/34")

