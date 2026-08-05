#!/usr/bin/env python3
"""Regenerate sample_data.csv — the de-identified demo dataset served by the
"Load sample data" button.

Source is Va_data/WHOVA_anonymized.csv, which is only PARTIALLY anonymised: the
interviewer name and phone were replaced, but respondent phones, GPS, submitter
names, facility names, village names and free-text narratives were left intact.

Rather than blacklist known-bad columns (which missed the deceased-name field on
a first pass, because it is named by question ID rather than by "name"), a column
survives only if every value it holds LOOKS like coded data: a number, an ISO
date/time, or a lowercase_underscore answer token. Anything free-text is dropped
by construction. Id10010 is a vetted exception — see ALLOW below.
"""
import csv, re, os, sys

SRC = 'Va_data/WHOVA_anonymized.csv'
OUT = 'sample_data.csv'
N   = 1000

NUM   = re.compile(r'^-?\d+(\.\d+)?$')
DATE  = re.compile(r'^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$')
TIME  = re.compile(r'^\d{1,2}:\d{2}(:\d{2})?$')
LOWER = re.compile(r'^[a-z0-9_]+$')
VOCAB = {'yes','no','dk','ref','dnk','na','n_a','other','none','true','false','male',
         'female','undetermined','not_applicable','refused_to_answer','doesnt_know',
         'dont_know','no_answer',
         # numeric sentinels — without these a single 'NaN' drops a whole clinical
         # column, which is how Id10120 (illness duration) went missing once.
         'nan','inf','-inf','null','nil'}

# Dropped regardless of shape: device/submission identifiers that link a row back
# to the original ODK submission or handset.
FORCE_DROP = {'meta-instanceID','KEY','deviceid','DeviceID','SubmitterID','SubmitterName',
              'meta-instanceName','phonenumber','vaid',
              # phone columns: the respondent's number was never anonymised, and the
              # interviewer's has no analytical value either way
              'Id10010_group-Id10010Phone','Id10007_group-Id10007Phone',
              # 561 distinct serial numbers — a re-identification key, not an answer
              'intro-D2SN'}

# Location columns cannot be caught by value shape: a coordinate is just a number.
# Household GPS is the single most re-identifying field in the file, so it is matched
# by name and dropped before any shape test runs.
LOCATION = re.compile(r'gps|latitude|longitude|altitude|coord|geopoint', re.I)

# Kept despite holding capitalised text: verified to contain ONLY the synthetic
# replacement names from WHOVA_name_mapping.csv (0 of 326 values are real).
# Retained because the interviewer breakdown and ICI-by-interviewer need it.
ALLOW = {'Id10010_group-Id10010'}


def safe(v):
    s = v.strip()
    if not s:
        return True
    # A long run of digits is a phone number or a serial, never a VA answer —
    # real answers are ages, counts and durations. FormVersion (8 digits) is the
    # only legitimate long number, hence the 9-digit floor.
    if s.isdigit() and len(s) >= 9:
        return False
    if NUM.match(s) or DATE.match(s) or TIME.match(s):
        return True
    return all(LOWER.match(t) or t.lower() in VOCAB for t in s.split())


def main():
    if not os.path.exists(SRC):
        sys.exit(f'missing source: {SRC}')
    csv.field_size_limit(10**7)
    with open(SRC, encoding='utf-8', errors='replace') as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        rows = [r for r in rd if any(c.strip() for c in r)]

    # Choose the rows first, then judge each column against exactly those rows.
    # Probing a fixed prefix of the source instead would miss a stray value in a
    # later row that the stride sampler goes on to include — which is how the
    # village name 'Mnyaki' survived in Id10057_death_outside.
    step = max(1, len(rows) // N)
    picked = [rows[i] for i in range(0, len(rows), step)][:N]

    keep = []
    for i, h in enumerate(hdr):
        if h in FORCE_DROP or LOCATION.search(h):
            continue
        if h in ALLOW:
            keep.append(i)
            continue
        vals = [r[i] for r in picked if i < len(r) and r[i].strip()]
        # Every value must pass, not merely most: a near-empty column holding one
        # stray free-text answer (a village name typed into an "other" box) would
        # survive any percentage threshold.
        if all(safe(v) for v in vals):
            keep.append(i)

    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['instanceID'] + [hdr[i] for i in keep])
        for n, row in enumerate(picked, 1):
            row = row + [''] * (len(hdr) - len(row))
            w.writerow([f'sample-{n:05d}'] + [row[i] for i in keep])

    print(f'{OUT}: {len(picked)} records, {len(keep)+1} columns, '
          f'{os.path.getsize(OUT)/1048576:.2f} MB '
          f'(dropped {len(hdr)-len(keep)} of {len(hdr)} source columns)')


if __name__ == '__main__':
    main()
