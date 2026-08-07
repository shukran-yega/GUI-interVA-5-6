#!/usr/bin/env python3
"""Truncate sample_data.csv in place to the first N records.

Keeps every column. Uses the csv module rather than `head -n` because the
narrative fields (Id10476, comment, Id10434/6, Id10444) contain embedded
newlines — a line-based cut would slice a record in half and corrupt the file.

Writes to a temp file and renames, so an interrupted run cannot leave a
half-written CSV in place of the original.
"""
import csv
import os
import sys

TARGET = 'sample_data.csv'
N = 3000


def main():
    if not os.path.exists(TARGET):
        sys.exit(f'missing: {TARGET}')

    csv.field_size_limit(10 ** 7)
    tmp = TARGET + '.tmp'
    before = os.path.getsize(TARGET)

    with open(TARGET, encoding='utf-8', errors='replace', newline='') as fi, \
         open(tmp, 'w', encoding='utf-8', newline='') as fo:
        rd, wr = csv.reader(fi), csv.writer(fo)
        try:
            hdr = next(rd)
        except StopIteration:
            os.remove(tmp)
            sys.exit('file is empty')
        wr.writerow(hdr)
        kept = 0
        for row in rd:
            if not any(c.strip() for c in row):
                continue
            wr.writerow(row)
            kept += 1
            if kept >= N:
                break

    os.replace(tmp, TARGET)
    after = os.path.getsize(TARGET)
    print(f'{TARGET}: {kept} records, {len(hdr)} columns, '
          f'{after/1048576:.2f} MB (was {before/1048576:.2f} MB)')


if __name__ == '__main__':
    main()
