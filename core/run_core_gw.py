#!/usr/bin/env python3
"""Reproducible QuAcK core-IP benchmark: G0W0@HF, SRG-evGW@HF, SRG-qsGW.

No QuAcK source modifications. Hydrogen uses aug-cc-pVTZ while other
atoms use aug-cc-pCVTZ, combined into one NWChem basis file.

Usage:
    python3 run_core_gw.py --test --prepare
    python3 run_core_gw.py --test --methods g0w0
    python3 run_core_gw.py --test --methods evgw
    python3 run_core_gw.py --test
    python3 run_core_gw.py --test --collect
    python3 run_core_gw.py --clean-integrals  # remove old int/ folders only
    python3 run_core_gw.py --all --collect --refresh-references --out /path/to/benchmark
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

BASIS = "aug-cc-pCVTZ"
BASIS_FILE = "aug-cc-pcvtz"
BASE_URL = "https://raw.githubusercontent.com/pfloos/QUESTDB/main/charged/core_IPs"
METHODS = ("g0w0", "evgw", "qsgw")
# The public method identifier is always "evgw". Its GW input uses SRG
# regularization (s=500); no separate unregularized evGW or SRG alias exists.
RESULT_LABELS = {"g0w0": "G0W0_HF", "evgw": "evgw", "qsgw": "qsGW_SRG"}
CC_METHODS = ("CC2", "CCSD", "CC3", "CCSDT", "CC4", "CCSDTQ")
VALIDATION = {"ch4", "nh3", "h2o", "hf", "co_c", "co_o", "hcl"}

# CVS-FCI/ACVTZ, Tables 2 and 3 of Marie, Burth, Loos, JCTC (2026),
# https://doi.org/10.1021/acs.jctc.6c00713 . Site suffixes are by
# occurrence of the named element in the QUESTDB XYZ file, NOT QP energy.
# Fields: json_stem, geometry_stem, element, site_index, CVS-FCI(eV).
RAW_REFERENCES = """
hf HF F 1 694.104
h2o H2O O 1 539.870
nh3 NH3 N 1 405.630
ch4 CH4 C 1 290.851
bf BF F 1 695.430
co_o CO O 1 542.470
co_c CO C 1 296.257
n2 N2 N 1 409.880
f2 F2 F 1 696.410
hcn_n HCN N 1 406.770
hcn_c HCN C 1 293.400
hno_o HNO O 1 541.600
hno_n HNO N 1 409.840
hof_f HOF F 1 694.170
hof_o HOF O 1 542.550
nh2f_f NH2F F 1 692.970
nh2f_n NH2F N 1 408.440
ch3f_f CH3F F 1 692.570
ch3f_c CH3F C 1 293.560
ch2o_o CH2O O 1 539.370
ch2o_c CH2O C 1 294.460
ch2nh_n CH2NH N 1 405.590
ch2nh_c CH2NH C 1 292.580
c2h2_c C2H2 C 1 291.280
sio_o SiO O 1 537.200
pn_n PN N 1 405.740
cs_c CS C 1 293.390
hcl HCl Cl 1 2821.420
h2s H2S S 1 2471.690
ph3 PH3 P 1 2145.730
sih4 SiH4 Si 1 1843.300
cs_s CS S 1 2472.810
pn_p PN P 1 2147.470
sio_si SiO Si 1 1845.300
c2h4_c C2H4 C 1 290.820
ch3oh_o CH3OH O 1 539.040
ch3oh_c CH3OH C 1 292.500
co2_o CO2 O 1 541.240
co2_c CO2 C 1 297.710
nno_o NNO O 1 541.440
nno_n2 NNO N 2 412.600
nno_n1 NNO N 1 408.700
fcho_f FCHO F 1 693.940
fcho_o FCHO O 1 540.100
fcho_c FCHO C 1 297.030
hnco_o HNCO O 1 540.140
hnco_n HNCO N 1 406.400
hnco_c HNCO C 1 295.940
hcooh_o1 HCOOH O 1 540.680
hcooh_o2 HCOOH O 2 538.870
hcooh_c HCOOH C 1 295.680
h2cnn_n1 H2CNN N 1 409.430
h2cnn_n2 H2CNN N 2 407.120
h2cnn_c H2CNN C 1 291.660
h2cco_o H2CCO O 1 540.170
h2cco_c2 H2CCO C 2 294.740
h2cco_c1 H2CCO C 1 291.150
ch2s_s CH2S S 1 2471.110
ch2s_c CH2S C 1 292.280
ocs_s OCS S 1 2472.090
ocs_o OCS O 1 540.250
ocs_c OCS C 1 295.370
ch3cn_n CH3CN N 1 405.650
ch3cn_c2 CH3CN C 2 292.790
ch3cn_c1 CH3CN C 1 292.770
ch3nc_n CH3NC N 1 406.710
ch3nc_c1 CH3NC C 1 293.360
ch3nc_c2 CH3NC C 2 292.330
h2ncho_o H2NCHO O 1 537.560
h2ncho_n H2NCHO N 1 406.530
h2ncho_c H2NCHO C 1 294.480
ch3no_o CH3NO O 1 540.420
ch3no_n CH3NO N 1 409.120
ch3no_c CH3NO C 1 292.020
c2h3f_f CH2CHF F 1 693.090
c2h3f_c2 CH2CHF C 2 293.440
c2h3f_c1 CH2CHF C 1 291.030
ch3cho_o CH3CHO O 1 538.480
ch3cho_c2 CH3CHO C 2 293.970
ch3cho_c1 CH3CHO C 1 291.360
n2ch2_n N2CH2 N 1 407.990
n2ch2_c N2CH2 C 1 292.920
p2 P2 P 1 2145.610
cl2 Cl2 Cl 1 2822.270
""".strip()


def references():
    entries = []
    for line in RAW_REFERENCES.splitlines():
        key, geometry, element, site, value = line.split()
        entries.append(dict(key=key, geometry=geometry, element=element,
                            site=int(site), fci_paper_eV=float(value)))
    if len(entries) != 84 or len({x['key'] for x in entries}) != 84:
        raise ValueError("Internal benchmark table must contain 84 distinct IPs")
    return entries


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=35) as response:
        raw = response.read()
    destination.write_bytes(raw)
    return raw


def find_source(path, local_root, cache, remote_rel, required=True, refresh=False):
    """Read local QUESTDB data, or download a reference with explicit refresh.

    Refreshing never falls back to a stale cached JSON. A missing optional
    JSON (HTTP 404) is distinct from a network failure, which is fatal.
    """
    if local_root:
        local_root = Path(local_root)
        variants = [local_root / 'charged' / 'core_IPs' / path,
                    local_root / path, local_root / 'core_IPs' / path]
        for candidate in variants:
            if candidate.is_file():
                return candidate
        if required:
            raise FileNotFoundError(f"Not found in supplied QUESTDB: {path}")
        return None
    dest = cache / path
    if not refresh and dest.is_file() and dest.stat().st_size:
        return dest
    try:
        download(BASE_URL + '/' + remote_rel, dest)
    except HTTPError as exc:
        if exc.code == 404 and not required:
            # A stale JSON from a previous checkout must not survive a 404.
            dest.unlink(missing_ok=True)
            return None
        raise RuntimeError(f"Cannot fetch {remote_rel} from QUESTDB: HTTP {exc.code}") from exc
    except (OSError, URLError) as exc:
        # In a strict refresh, never silently use old data (or a paper value).
        if refresh or required:
            raise RuntimeError(f"Cannot refresh QUESTDB reference {remote_rel}: {exc}") from exc
        return None
    return dest


def parse_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    atoms = []
    for line in lines[2:2+n]:
        tokens = line.split()
        if len(tokens) < 4:
            raise ValueError(f"Malformed XYZ: {path}")
        atoms.append((tokens[0].capitalize(), tuple(map(float, tokens[1:4]))))
    if len(atoms) != n:
        raise ValueError(f"Incomplete XYZ: {path}")
    return atoms


def load_reference(row, local_root, cache, refresh=False):
    """Build one reference record, keeping a full provenance audit.

    Missing/null JSON values remain missing, except the CVS-FCI target for
    which a *clearly labeled* paper fallback is available. The published GW
    number in QUESTDB is G0W0@PBEh, not our current G0W0@HF calculation.
    """
    out = dict(row)
    out.update(cvs_fci_eV=row['fci_paper_eV'],
               reference_source='paper_Table2_or_3',
               questdb_json_present=False,
               questdb_json_sha256=None,
               questdb_cvs_fci_eV=None,
               questdb_minus_paper_eV=None,
               g0w0_pbeh_eV=None)
    for key in CC_METHODS:
        out[key] = None
    src = find_source(row['key']+'.json', local_root, cache,
                      row['key']+'.json', required=False, refresh=refresh)
    if src is None:
        out['reference_note'] = 'QUESTDB JSON missing; paper CVS-FCI fallback only'
        return out
    if not src.stat().st_size:
        raise ValueError(f"Empty QUESTDB JSON: {src}")
    raw = src.read_bytes()
    try:
        j = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid QUESTDB JSON: {src}: {exc}") from exc
    if not isinstance(j, dict):
        raise ValueError(f"QUESTDB JSON is not an object: {src}")
    out['questdb_json_present'] = True
    out['questdb_json_sha256'] = hashlib.sha256(raw).hexdigest()
    out['reference_note'] = ''

    def read_number(container, field):
        section = j.get(container)
        if not isinstance(section, dict):
            return None
        value = section.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite {container}/{field} in {src}")
            return value
        return None

    sci = j.get('sCI')
    if isinstance(sci, dict):
        sci_energy = sci.get('Energy')
        if isinstance(sci_energy, dict):
            val = sci_energy.get(BASIS)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                val = float(val)
                if not math.isfinite(val):
                    raise ValueError(f"Nonfinite sCI/Energy/{BASIS} in {src}")
                out['cvs_fci_eV'] = val
                out['questdb_cvs_fci_eV'] = val
                out['questdb_minus_paper_eV'] = val-row['fci_paper_eV']
                out['reference_source'] = 'QUESTDB_sCI'
    if out['questdb_cvs_fci_eV'] is None:
        out['reference_note'] = 'sCI/Energy/aug-cc-pCVTZ missing; paper CVS-FCI fallback'
    for key in CC_METHODS:
        out[key] = read_number(key, BASIS)
    out['g0w0_pbeh_eV'] = read_number('GW', BASIS)
    return out


def replace_block(filetext, header, modifications, header_colon=False):
    """Update values by names from commented QuAcK input headers."""
    lines = filetext.splitlines(keepends=True)
    updated = False
    for i, line in enumerate(lines[:-1]):
        if not line.lstrip().startswith('#'):
            continue
        if header_colon:
            match = re.match(r'^\s*#\s*'+re.escape(header)+r'\s*:\s*(.*)$',line)
        else:
            match = re.match(r'^\s*#\s*'+re.escape(header)+r'\s+(.*)$',line)
        if not match:
            continue
        names = match.group(1).split() if header_colon else [header] + match.group(1).split()
        original = lines[i+1].split()
        if len(original) < len(names):
            raise ValueError(f"QuAcK input block {header}: {len(original)} values but {len(names)} keys")
        for name,value in modifications.items():
            if name not in names:
                raise ValueError(f"QuAcK {header} input has no option {name}")
            original[names.index(name)] = str(value)
        lines[i+1] = '  '+'    '.join(original)+'\n'
        updated = True
        break
    if not updated:
        raise ValueError(f"Could not find QuAcK input block {header}")
    return ''.join(lines)


def make_inputs(quack, job_dir, method):
    template_dir = quack/'input'
    methods_txt = (template_dir/'methods.default').read_text()
    options_txt = (template_dir/'options.default').read_text()
    methods_txt = replace_block(methods_txt, 'RHF', {'RHF':'T'})
    which = {'g0w0':'G0W0','evgw':'evGW','qsgw':'qsGW'}[method]
    methods_txt = replace_block(methods_txt, 'G0W0', {which:'T'})
    settings = {'lin':'F', 'eta':'0.0', 'TDA_W':'F',
                'reg':'T' if method in ('evgw','qsgw') else 'F',
                'linDM':'F'}
    options_txt = replace_block(options_txt, 'GW', settings, header_colon=True)
    options_txt = replace_block(options_txt, 'RPA/MP2',
        {'CVS-Alpha':'0','CVS-Beta':'0','FC-Alpha':'0','FC-Beta':'0'},header_colon=True)
    options_txt = replace_block(options_txt, 'spin', {'TDA':'F'},header_colon=True)
    inp=job_dir/'input'
    inp.mkdir(parents=True,exist_ok=True)
    (inp/'methods').write_text(methods_txt)
    (inp/'options').write_text(options_txt)
    for name in ['dft','geminal','hpc_flags','mom_occupations.default']:
        src=template_dir/name
        if src.is_file():
            shutil.copy2(src, inp/name.replace('.default',''))
    # Verify the explicit protocol before running.
    assert 'TDA_W reg' in options_txt and '0.0' in options_txt


def get_basis(quack, out):
    """Use aug-cc-pCVTZ (heavy atoms) + aug-cc-pVTZ (hydrogen).

    Repair an existing cached basis without duplicating H shells.
    """
    dest = out/'basis'/BASIS_FILE
    local = quack/'basis'/BASIS_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or not dest.stat().st_size:
        if local.is_file() and local.stat().st_size:
            shutil.copy2(local, dest)
        else:
            try:
                import basis_set_exchange as bse
            except ImportError as exc:
                raise RuntimeError('Install basis-set-exchange: python3 -m pip install basis-set-exchange') from exc
            dest.write_text(bse.get_basis(BASIS, fmt='nwchem',
                             uncontract_general=True, uncontract_spdf=True))
    content = dest.read_text()
    h_shell = re.compile(r'(?mi)^\s*H\s+(?:S|P|D|F|G|H|I)\b')
    if h_shell.search(content):
        return dest
    try:
        import basis_set_exchange as bse
    except ImportError as exc:
        raise RuntimeError('H basis missing: install basis-set-exchange') from exc
    hydrogen = bse.get_basis('aug-cc-pVTZ', elements=['H'], fmt='nwchem',
                             header=False, uncontract_general=True, uncontract_spdf=True)
    lines = [line for line in hydrogen.splitlines()
             if not re.match(r'^\s*(?:BASIS\b|END\s*$)',line,re.I)]
    h_block = '\n'.join(lines).strip()
    if not h_shell.search(h_block):
        raise RuntimeError('BSE returned no hydrogen shells')
    ends = list(re.finditer(r'(?mi)^[ \t]*END[ \t]*$', content))
    if not ends:
        raise ValueError('No NWChem END statement found in basis file')
    at = ends[-1].start()
    dest.write_text(content[:at].rstrip() + '\n\n' + h_block + '\n' + content[at:])
    print('Added H aug-cc-pVTZ shells to the aug-cc-pCVTZ basis')
    return dest


def make_job(quack, out, row, method, xyz_source, basis_source):
    job = out/'jobs'/row['geometry'].lower()/method
    (job/'mol').mkdir(parents=True,exist_ok=True)
    (job/'basis').mkdir(parents=True,exist_ok=True)
    shutil.copy2(xyz_source,job/'mol'/(row['geometry']+'.xyz'))
    shutil.copy2(basis_source,job/'basis'/BASIS_FILE)
    make_inputs(quack,job,method)
    meta = dict(geometry=row['geometry'],method=method,basis=BASIS,
                eta=0.0,TDA_W=False,linearization=False,
                SRG=(method in ('evgw','qsgw')),SRG_flow=500.0 if method in ('evgw','qsgw') else None,
                xyz_sha256=hashlib.sha256(xyz_source.read_bytes()).hexdigest(),
                basis_sha256=hashlib.sha256(basis_source.read_bytes()).hexdigest())
    (job/'protocol.json').write_text(json.dumps(meta,indent=2)+'\n')
    return job


FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?'
ORBROW = re.compile(r'^\s*\|\s*(\d+)\s*\|\s*('+FLOAT+r')\s*\|\s*('+FLOAT+r')\s*\|\s*('+FLOAT+r')\s*\|\s*('+FLOAT+r')\s*\|\s*$',re.M)
HEADING = {
    'g0w0': re.compile(r'G0W0@RHF calculation',re.I),
    'evgw': re.compile(r'Self-consistent evG\s*\d+W\s*\d+@RHF calculation',re.I),
    'qsgw': re.compile(r'Self-consistent qsG\s*\d+W\s*\d+@RHF calculation',re.I),
}
CONV = re.compile(r'Convergence\s*=\s*('+FLOAT+r')',re.I)
QP_FAIL = re.compile(r'^\s*(\d+)\s+\d+\s+'+FLOAT+r'\s+'+FLOAT+r'\s+'+FLOAT+r'\s+Cvg Failed!',re.M|re.I)


QP_TABLE_HEADER = re.compile(r'(?m)^\s*Orb\.\s+It\.\s+e_QPlin\s*\(eV\)', re.I)


def parse_output(log, method, threshold=1e-5):
    """Read final GW orbital table and only FINAL-iteration QP failures.

    Failure of a non-target QP root is kept as a diagnostic, NOT as a
    failure of the entire evGW calculation.
    """
    heads = list(HEADING[method].finditer(log))
    if not heads:
        return dict(ok=False, reason='No GW orbital table in output', orbitals={})
    last_heading = heads[-1]
    tail = log[last_heading.end():]
    end = re.search(r'(?:G0W0|evGW|qsGW)@RHF HOMO\s+energy\s*=', tail, re.I)
    block = tail[:end.start()] if end else tail[:30000]
    orbitals = {}
    for m in ORBROW.finditer(block):
        i, ehf, sig, z, egw = m.groups()
        orbitals[int(i)] = dict(orbital=int(i), hf_eV=float(ehf.replace('D','E')),
                                sigma_eV=float(sig.replace('D','E')),
                                z=float(z.replace('D','E')),
                                qp_energy_eV=float(egw.replace('D','E')))
    # Only the QP listing immediately preceding the final printed table
    # corresponds to the final evGW iteration. Ignore early failures.
    all_failed = sorted({int(i) for i in QP_FAIL.findall(log)})
    last_qp_headers = [m for m in QP_TABLE_HEADER.finditer(log)
                       if m.start() < last_heading.start()]
    final_failed = []
    if last_qp_headers and method != 'qsgw':
        final_output = log[last_qp_headers[-1].end():last_heading.start()]
        final_failed = sorted({int(i) for i in QP_FAIL.findall(final_output)})
    diagnostics = dict(qp_failed_orbitals=final_failed,
                       qp_failed_orbitals_any_iteration=all_failed)
    if re.search(r'(?i)convergence failed', log):
        return dict(ok=False, reason='QuAcK reports convergence failure',
                    orbitals=orbitals, **diagnostics)
    if not orbitals:
        return dict(ok=False, reason='No orbital rows in final GW table',
                    orbitals={}, **diagnostics)
    if method in ('evgw','qsgw'):
        cv = CONV.findall(block)
        if not cv:
            return dict(ok=False, reason='No final self-consistency residual',
                        orbitals=orbitals, **diagnostics)
        residual = float(cv[-1].replace('D','E'))
        if not math.isfinite(residual) or residual > threshold * 1.1:
            return dict(ok=False, reason=f'GW residual {residual:g} exceeds threshold',
                        orbitals=orbitals, convergence=residual, **diagnostics)
        return dict(ok=True, reason='converged', orbitals=orbitals,
                    convergence=residual, **diagnostics)
    return dict(ok=True, reason='table found', orbitals=orbitals, **diagnostics)


def check_regularized_job(job, method):
    """Reject legacy unregularized evGW logs when collecting SRG results."""
    if method not in ('evgw', 'qsgw'):
        return None
    protocol_path = job/'protocol.json'
    options_path = job/'input'/'options'
    try:
        metadata = json.loads(protocol_path.read_text())
        if (metadata.get('method') != method or metadata.get('SRG') is not True
                or metadata.get('SRG_flow') != 500.0):
            return 'Legacy or incompatible job: SRG metadata does not match'
        options_text = options_path.read_text()
        for i, line in enumerate(options_text.splitlines()[:-1]):
            m = re.match(r'^\s*#\s*GW\s*:\s*(.*)$', line)
            if m:
                names = m.group(1).split()
                values = options_text.splitlines()[i+1].split()
                if len(names) != len(values) or values[names.index('reg')] != 'T':
                    return 'Legacy or incompatible job: GW reg is not T'
                return None
        return 'Missing GW options block'
    except (ValueError, IndexError, KeyError, OSError) as exc:
        return f'Cannot verify SRG input settings: {exc}'


def probe_log(job, method):
    log_file=job/'quack.log'
    if not log_file.is_file():
        return dict(ok=False,reason='No log',orbitals={})
    problem = check_regularized_job(job, method)
    if problem:
        return dict(ok=False,reason=problem,orbitals={})
    try:
        return parse_output(log_file.read_text(errors='replace'), method)
    except Exception as exc:
        return dict(ok=False,reason='parse error: '+str(exc),orbitals={})


def job_fingerprint(job):
    h=hashlib.sha256()
    for name in ('input/methods','input/options','protocol.json'):
        h.update((job/name).read_bytes())
    return h.hexdigest()


def delete_job_integrals(job):
    """Remove only this job's disposable QuAcK ``int/`` directory.

    ``quack.log``, ``run.json``, inputs, geometry and orbital assignments
    are untouched. Never follow an ``int`` or job-directory symlink.
    Returns the number of bytes removed, or None if there was no directory.
    """
    job = Path(job)
    integrals = job/'int'
    if job.is_symlink() or integrals.is_symlink():
        raise OSError(f'Refusing to clean a symlink: {integrals}')
    if not integrals.exists():
        return None
    if not integrals.is_dir():
        raise OSError(f'Expected an integral directory, found: {integrals}')
    total = 0
    for path in integrals.rglob('*'):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    shutil.rmtree(integrals)
    return total


def cleanup_job_integrals(job):
    """Best-effort cleanup; a disk-cleanup problem must not hide results."""
    try:
        removed = delete_job_integrals(job)
    except OSError as exc:
        print(f'  WARNING: cannot remove integrals for {job}: {exc}',file=sys.stderr)
        return
    if removed is not None:
        print(f'  CLEAN  {job.parent.name}/{job.name}/int '
              f'({removed / (1024**2):.1f} MiB freed)',flush=True)


def clean_existing_integrals(out):
    """Clean old benchmark jobs without running QuAcK or reading references."""
    jobs = out/'jobs'
    if jobs.is_symlink():
        raise OSError(f'Refusing to traverse a symlink: {jobs}')
    if not jobs.exists():
        return 0, 0
    count = total = 0
    for molecule in sorted(jobs.iterdir()):
        if molecule.is_symlink() or not molecule.is_dir():
            continue
        for job in sorted(molecule.iterdir()):
            if job.is_symlink() or not job.is_dir():
                continue
            try:
                removed = delete_job_integrals(job)
            except OSError as exc:
                print(f'  WARNING: cannot remove integrals for {job}: {exc}',file=sys.stderr)
                continue
            if removed is not None:
                count += 1
                total += removed
                print(f'  CLEAN  {molecule.name}/{job.name}/int '
                      f'({removed / (1024**2):.1f} MiB)',flush=True)
    return count, total


def run_job(quack, job, row, method, force=False, dry_run=False, keep_integrals=False):
    previous=probe_log(job,method)
    old=job/'run.json'
    fingerprint=job_fingerprint(job)
    old_record={}
    if old.is_file():
        try:old_record=json.loads(old.read_text())
        except ValueError:pass
    if previous['ok'] and old_record.get('input_fingerprint')==fingerprint and not force:
        print(f'  CACHED {row["geometry"]} {method}')
        if not keep_integrals and not dry_run:
            cleanup_job_integrals(job)
        return previous
    command=[sys.executable,str(quack/'PyDuck.py'),'--working_dir',str(job),
             '-b',BASIS_FILE,'-x',row['geometry'],'-nc']
    if dry_run:
        print('  DRY RUN:', ' '.join(command))
        return dict(ok=False,reason='dry run',orbitals={})
    print(f'  RUN    {row["geometry"]} {method}',flush=True)
    env=os.environ.copy()
    env['QUACK_ROOT']=str(quack)
    start=time.monotonic()
    try:
        with (job/'quack.log').open('w') as logfile:
            proc=subprocess.run(command,cwd=job,env=env,stdout=logfile,
                                stderr=subprocess.STDOUT,check=False)
        parsed=probe_log(job,method)
        parsed['returncode']=proc.returncode
        if proc.returncode != 0:
            parsed.update(ok=False,reason=f'PyDuck exit code {proc.returncode}: '+parsed['reason'])
        # Failures of other QP roots are not fatal for the selected core orbital.
        (job/'run.json').write_text(json.dumps({k:v for k,v in parsed.items() if k!='orbitals'} |
                                           {'seconds':round(time.monotonic()-start,3),'input_fingerprint':fingerprint},indent=2)+'\n')
        print(f'  {"OK" if parsed["ok"] else "FAILED"}   {row["geometry"]} {method}: {parsed["reason"]}',flush=True)
        return parsed
    finally:
        # Once PyDuck/QuAcK has exited, its integrals can safely be removed.
        # Do this even for failed jobs, so a long benchmark cannot fill disk.
        if not keep_integrals:
            cleanup_job_integrals(job)


def assign_orbitals(geometry, atoms, basis_path, out):
    """Match canonical RHF 1s-like MOs to atomic sites by Mulliken AO populations.

    The same neutral geometry and (uncontracted NWChem) basis as QuAcK.
    Orbital index is 1-based. Symmetry-degenerate sites require manual review.
    """
    cache=out/'orbital_maps'/(geometry.lower()+'.json')
    if cache.is_file():
        return json.loads(cache.read_text())
    try:
        import numpy as np
        from pyscf import gto, scf
    except ImportError as exc:
        raise RuntimeError('PySCF and numpy are needed for core-orbital/site matching') from exc
    elements=sorted({a[0] for a in atoms})
    basis={a:gto.basis.parse_nwchem.load(str(basis_path),a) for a in elements}
    mol=gto.M(atom=[(a,coord) for a,coord in atoms],basis=basis,charge=0,spin=0,
              unit='Angstrom',verbose=0,cart=False)
    mf=scf.RHF(mol)
    mf.conv_tol=1e-10
    mf.max_cycle=256
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f'PySCF reference HF failed to converge for {geometry}')
    heavy=[(j,a[0]) for j,a in enumerate(atoms) if a[0]!='H']
    size=len(heavy)
    s=mf.get_ovlp()
    c=mf.mo_coeff[:,:size]
    sc=s@c
    slices=mol.aoslice_by_atom()
    weights=[]
    for index,symbol in heavy:
        lo,hi=map(int,slices[index,2:4])
        pop=(c[lo:hi,:]*sc[lo:hi,:]).sum(axis=0)
        weights.append([float(v) for v in pop])
    best=None
    second=-float('inf')
    for perm in itertools.permutations(range(size)):
        score=sum(weights[atom_index][orb] for atom_index,orb in enumerate(perm))
        if best is None or score>best[0]:
            if best is not None:second=max(second,best[0])
            best=(score,perm)
        else:
            second=max(second,score)
    if best is None:
        raise RuntimeError(f'No core orbitals identified in {geometry}')
    records=[]
    per_element={}
    for pos,(atom_index,symbol) in enumerate(heavy):
        site=per_element.get(symbol,0)+1
        per_element[symbol]=site
        orbital=best[1][pos]+1
        records.append(dict(atom_index=atom_index+1,element=symbol,site=site,
                            orbital=orbital,hf_energy_eV=float(mf.mo_energy[orbital-1]*27.211386245988),
                            local_population=weights[pos][orbital-1]))
    result=dict(geometry=geometry,assignment=records,score_gap=best[0]-second,
                ambiguous=best[0]-second<0.10,
                comment='Site numbering follows occurrence in QUESTDB XYZ; inspect ambiguous assignments.')
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_text(json.dumps(result,indent=2)+'\n')
    return result


def extract_results(rows, out, orbital_maps, reference_by_key):
    results, all_orbitals = [], []
    for row in rows:
        ref = reference_by_key[row['key']]
        mapped = orbital_maps.get(row['geometry'])
        selected = None
        if mapped:
            matches = [a for a in mapped['assignment']
                       if a['element'].lower()==row['element'].lower() and a['site']==row['site']]
            if len(matches)==1:
                selected = matches[0]['orbital']
        if selected is None and row['key'] in VALIDATION:
            selected = 2 if row['key']=='co_c' else 1
        record = dict(key=row['key'], geometry=row['geometry'], element=row['element'],
                      site=row['site'], orbital=selected,
                      assignment_ambiguous=bool(mapped and mapped['ambiguous']),
                      CVS_FCI_eV=ref['cvs_fci_eV'], reference_source=ref['reference_source'],
                      questdb_json_present=ref['questdb_json_present'],
                      reference_note=ref['reference_note'],
                      G0W0_PBEh_eV=ref['g0w0_pbeh_eV'])
        for name in CC_METHODS:
            record[name+'_eV'] = ref[name]
        if mapped and selected is not None:
            detail = next((a for a in mapped['assignment'] if a['orbital']==selected), None)
            if detail:
                record['atom_index'] = detail['atom_index']
                record['HF_reference_eV'] = detail['hf_energy_eV']
                record['site_population'] = detail['local_population']
        for method, label in RESULT_LABELS.items():
            job = out/'jobs'/row['geometry'].lower()/method
            parsed = probe_log(job,method)
            orb = parsed['orbitals'].get(selected) if selected is not None else None
            failures = parsed.get('qp_failed_orbitals',[])
            target_failed = selected in failures if selected is not None else False
            status = parsed['reason']
            if target_failed:
                status += '; target core QP Newton root failed in FINAL iteration'
            elif failures:
                status += f'; other final QP roots failed: {failures}'
            if orb and mapped and record.get('HF_reference_eV') is not None and method != 'qsgw':
                if abs(orb['hf_eV'] - record['HF_reference_eV']) > 0.5:
                    target_failed = True
                    status += '; HF orbital mismatch vs PySCF'
            valid = bool(parsed['ok'] and orb and not target_failed and
                         math.isfinite(orb['qp_energy_eV']))
            record[label+'_IP_eV'] = -orb['qp_energy_eV'] if valid else None
            if method != 'qsgw':
                record[label+'_Z'] = orb['z'] if valid and math.isfinite(orb['z']) else None
                z = record[label+'_Z']
                record[label+'_low_Z'] = (z < 0.5) if z is not None else None
            else:
                record[label+'_Z_diagnostic'] = (orb['z'] if valid and math.isfinite(orb['z']) else None)
            record[label+'_status'] = status
            record[label+'_QP_failed_final'] = bool(target_failed)
            v = record[label+'_IP_eV']
            record[label+'_error_eV'] = v - record['CVS_FCI_eV'] if v is not None else None
            for idx, data in sorted(parsed['orbitals'].items()):
                all_orbitals.append(dict(geometry=row['geometry'],
                                         method=label, job_directory=method,
                                         run_converged=parsed['ok'], run_status=parsed['reason'],
                                         qp_failed_final=(idx in failures), **data))
        results.append(record)
    unique = {(r['geometry'],r['method'],r['orbital']):r for r in all_orbitals}
    return results, list(unique.values())


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for row in rows:w.writerow(row)


def summarize(rows):
    methods={'G0W0_HF':'G0W0_HF_IP_eV','evgw':'evgw_IP_eV',
             'qsGW_SRG':'qsGW_SRG_IP_eV','G0W0_PBEh':'G0W0_PBEh_eV'}
    methods.update({x:x+'_eV' for x in CC_METHODS})
    summaries=[]
    for group,group_rows in [('all',rows)]+[(x,[r for r in rows if r['element']==x])
                                           for x in sorted({r['element'] for r in rows})]:
        for method,column in methods.items():
            errors=[r[column]-r['CVS_FCI_eV'] for r in group_rows
                    if r.get(column) is not None]
            if errors:
                summaries.append(dict(group=group,method=method,n=len(errors),
                                      MSE_eV=sum(errors)/len(errors),
                                      MAE_eV=sum(map(abs,errors))/len(errors),
                                      RMSE_eV=math.sqrt(sum(x*x for x in errors)/len(errors)),
                                      MaxAE_eV=max(map(abs,errors))))
    return summaries


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    which=p.add_mutually_exclusive_group()
    which.add_argument('--test',action='store_true',help='Six small molecules; seven core IPs (default)')
    which.add_argument('--all',action='store_true',help='All 84 benchmark core IPs; missing QUESTDB data are reported explicitly')
    p.add_argument('--quack',type=Path,default=Path(os.environ.get('QUACK_ROOT','')) if os.environ.get('QUACK_ROOT') else None,
                   help='Path to compiled QuAcK repository (or QUACK_ROOT env variable)')
    p.add_argument('--questdb',type=Path,help='Local QUESTDB checkout or charged/core_IPs folder (otherwise download & cache)')
    p.add_argument('--out',type=Path,default=Path('core_gw_results'),help='Output directory')
    p.add_argument('--prepare',action='store_true',help='Generate input files, do not run')
    p.add_argument('--collect',action='store_true',help='Rebuild tables from existing logs without running jobs')
    p.add_argument('--refresh-references',action='store_true',
                   help='Refetch all QUESTDB target JSON files (or read --questdb checkout); never reuse stale cached JSON')
    p.add_argument('--dry-run',action='store_true',help='Prepare and print QuAcK commands, do not run')
    p.add_argument('--force',action='store_true',help='Re-run even successful jobs')
    p.add_argument('--keep-integrals',action='store_true',help='Keep each job int/ directory (default: delete after QuAcK exits)')
    p.add_argument('--clean-integrals',action='store_true',help='Remove old jobs/*/*/int directories and exit, preserving logs/results')
    p.add_argument('--reset',action='store_true',help='DELETE the entire benchmark --out directory and exit (logs, cache, tables)')
    p.add_argument('--methods',default='g0w0,evgw,qsgw',
                   help='Comma-separated subset: g0w0,evgw,qsgw (evgw always uses SRG)')
    p.add_argument('--skip-qsgw',action='store_true',help='Legacy shortcut for --methods g0w0,evgw')
    p.add_argument('--max-molecules',type=int,help='Stop after this many distinct geometries')
    args=p.parse_args(argv)
    selected_methods=[v.strip().lower() for v in args.methods.split(',') if v.strip()]
    if not selected_methods or any(v not in METHODS for v in selected_methods):
        p.error('--methods must be a comma-separated subset of g0w0,evgw,qsgw')
    selected_methods=list(dict.fromkeys(selected_methods))
    if args.skip_qsgw:
        selected_methods=[v for v in selected_methods if v!='qsgw']
    # Reset is a standalone operation. Never wipe then quietly run calculations.
    out_raw=args.out.expanduser()
    out=out_raw.resolve()
    quack=args.quack.expanduser().resolve() if args.quack is not None else None
    if args.reset:
        if args.prepare or args.collect or args.force or args.dry_run or args.clean_integrals or args.keep_integrals:
            p.error('--reset cannot be combined with --prepare, --collect, --force or --dry-run')
        if out_raw.is_symlink():
            p.error('--reset refuses to delete a symlink')
        home=Path.home().resolve()
        cwd=Path.cwd().resolve()
        forbidden={Path('/'),home,cwd}
        # Do not delete any parent of QuAcK or QUESTDB or a location within them.
        for source in (quack, args.questdb.expanduser().resolve() if args.questdb else None):
            if source is not None and (out == source or out in source.parents or source in out.parents):
                p.error(f'--reset refuses to delete a path related to source repository: {source}')
        if out in forbidden or out == Path('/mnt/data'):
            p.error(f'Unsafe --reset destination: {out}')
        marker=out/'.core_gw_benchmark'
        if out.name != 'core_gw_results' and not marker.is_file():
            p.error('--reset only accepts a directory named core_gw_results or a directory created by this runner')
        if out.exists():
            if not out.is_dir():
                p.error(f'Reset target is not a directory: {out}')
            shutil.rmtree(out)
            print(f'Reset complete. Deleted benchmark directory: {out}')
            print('All previous QuAcK logs, basis and QUESTDB caches, results and statistics were removed.')
        else:
            print(f'Nothing to reset; directory does not exist: {out}')
        return 0
    if args.clean_integrals:
        if args.prepare or args.collect or args.force or args.dry_run or args.keep_integrals:
            p.error('--clean-integrals is a standalone maintenance operation')
        if out_raw.is_symlink():
            p.error('--clean-integrals refuses an output-directory symlink')
        if out.name != 'core_gw_results' and not (out/'.core_gw_benchmark').is_file():
            p.error('--clean-integrals requires core_gw_results or a runner-created directory')
        if not out.is_dir():
            print(f'Nothing to clean: {out} does not exist')
            return 0
        try:
            count, freed = clean_existing_integrals(out)
        except OSError as exc:
            p.error(str(exc))
        print(f'Removed {count} integral directories; freed '
              f'{freed/(1024**3):.2f} GiB. Logs and CSVs retained.')
        return 0
    if args.collect:
        # Avoid silently writing empty tables into the wrong --out folder.
        logs = list(out.glob('jobs/*/*/quack.log'))
        if not logs:
            p.error(f'--collect found no jobs/*/*/quack.log in {out}; check --out')
        print(f'Collecting from {len(logs)} existing QuAcK logs in {out}', flush=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/'.core_gw_benchmark').write_text('QuAcK core-IP benchmark output directory\n')
    cache=out/'questdb_cache' 
    allrows=references()
    rows=allrows if args.all else [x for x in allrows if x['key'] in VALIDATION]
    geometries=list(dict.fromkeys(r['geometry'] for r in rows))
    if args.max_molecules is not None:
        geometries=geometries[:args.max_molecules]
        rows=[r for r in rows if r['geometry'] in geometries]
    if not args.collect:
        if not quack or not (quack/'PyDuck.py').is_file():
            p.error('Please provide --quack /path/to/QuAcK or set QUACK_ROOT')
        if not (quack/'bin'/'QuAcK').is_file() and not (args.prepare or args.dry_run):
            p.error('QuAcK is not compiled: expected $QUACK_ROOT/bin/QuAcK')
        for required in ('methods.default','options.default'):
            if not (quack/'input'/required).is_file():
                p.error('Missing QuAcK input template: '+required)
    if args.collect:
        # Core mappings were already saved during preparation; no PySCF required.
        orbital_maps={}
        for mol in geometries:
            mp=out/'orbital_maps'/(mol.lower()+'.json')
            if mp.is_file():orbital_maps[mol]=json.loads(mp.read_text())
    else:
        basis=get_basis(quack,out)
        orbital_maps={}
        for mol in geometries:
            xyz=find_source('geometries/'+mol+'.xyz',args.questdb,cache,
                            'geometries/'+mol+'.xyz')
            atom=parse_xyz(xyz)
            # All-mode site assignments require a neutral RHF calculation.
            # Validation targets have fixed and explicitly checked orbital ranks.
            if args.all:
                orbital_maps[mol]=assign_orbitals(mol,atom,basis,out)
            for method in selected_methods:
                row=next(r for r in rows if r['geometry']==mol)
                job=make_job(quack,out,row,method,xyz,basis)
                if not args.prepare:
                    run_job(quack,job,row,method,force=args.force,dry_run=args.dry_run,
                            keep_integrals=args.keep_integrals)
    print(f'Reading {len(rows)} QUESTDB reference records'
          + (' (forced refresh)' if args.refresh_references else ''), flush=True)
    manifest=[load_reference(r,args.questdb,cache,
                             refresh=args.refresh_references) for r in rows]
    reference_by_key={r['key']:r for r in manifest}
    missing_json=[r['key'] for r in manifest if not r['questdb_json_present']]
    fallback_fci=[r['key'] for r in manifest if r['reference_source'] != 'QUESTDB_sCI']
    print(f'QUESTDB JSON present: {len(rows)-len(missing_json)}/{len(rows)}')
    if missing_json:
        print('WARNING: Missing QUESTDB JSON (paper CVS-FCI fallback): '
              + ', '.join(missing_json), file=sys.stderr)
    if fallback_fci:
        print('WARNING: CVS-FCI values not provided by JSON: '
              + ', '.join(fallback_fci), file=sys.stderr)
    discrepancies=[r for r in manifest if r['questdb_minus_paper_eV'] is not None
                   and abs(r['questdb_minus_paper_eV']) > 0.5]
    if discrepancies:
        print('WARNING: QUESTDB and paper CVS-FCI differ by >0.5 eV: '
              + ', '.join(f"{r['key']} ({r['questdb_minus_paper_eV']:+.3f})"
                          for r in discrepancies), file=sys.stderr)
    results,orbital_rows=extract_results(rows,out,orbital_maps,reference_by_key)
    write_csv(out/'results.csv',results)
    write_csv(out/'orbitals.csv',orbital_rows)
    write_csv(out/'statistics.csv',summarize(results))
    write_csv(out/'reference_manifest.csv',manifest)
    write_csv(out/'reference_audit.csv', [
        dict(key=r['key'], reference_source=r['reference_source'],
             questdb_json_present=r['questdb_json_present'],
             reference_note=r['reference_note'],
             questdb_json_sha256=r['questdb_json_sha256'],
             cvs_fci_eV=r['cvs_fci_eV'],
             questdb_minus_paper_eV=r['questdb_minus_paper_eV'],
             n_available_methods=sum(r[m] is not None for m in CC_METHODS)
                                 + int(r['g0w0_pbeh_eV'] is not None))
        for r in manifest])
    print(f'\nSaved {len(results)} core-IP targets in {out}/results.csv')
    print(f'Saved GW orbital tables in {out}/orbitals.csv')
    print(f'Saved method errors in {out}/statistics.csv')
    print(f'Saved reference provenance in {out}/reference_audit.csv')
    print('Note: missing QUESTDB GW/CC data stay blank; these are not zero-valued reference results.')
    if args.all:
        print('IMPORTANT: review orbital_maps/*.json for nearly degenerate or symmetry-equivalent cores.')
    if args.prepare or args.dry_run:
        print('No quantum-chemistry calculations were executed.' if args.prepare else 'Dry-run complete.')

if __name__=='__main__':
    main()
