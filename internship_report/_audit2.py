import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
src = (BASE / 'report.tex').read_text(encoding='utf-8')
labels = re.findall(r'\\label\{([^}]+)\}', src)
dup = [l for l, c in Counter(labels).items() if c > 1]
print('labels:', len(labels), 'dups:', dup)
refs = set(re.findall(r'\\(?:eq)?ref\{([^}]+)\}', src))
missing = [r for r in refs if r not in labels]
print('refs:', len(refs), 'missing targets:', missing)
cites = set(
    k.strip()
    for grp in re.findall(r'\\cite\{([^}]+)\}', src)
    for k in grp.split(',')
)
bib_src = (BASE / 'references.bib').read_text(encoding='utf-8')
bib = set(re.findall(r'@\w+\{([^,]+),', bib_src))
print('cites:', len(cites), 'missing bib:', [c for c in cites if c not in bib])
print('uncited bib:', [b for b in bib if b not in cites])
# table column integrity for the two new tables: scan only the tabular
# block that follows each label, so nearby floats cannot leak in.
for name, spec in [('tab:claims', 4), ('tab:simplex-violation', 5)]:
    i = src.find('\\label{%s}' % name)
    j = src.find('\\begin{tabular}', i)
    k = src.find('\\end{tabular}', j)
    body = src[j:k]
    for row in body.split('\\\\'):
        row = row.strip()
        if not row or '&' not in row:
            continue
        amp = row.count('&')
        multi = sum(int(m) for m in re.findall(r'\\multicolumn\{(\d+)\}', row))
        n_multi = len(re.findall(r'\\multicolumn\{', row))
        eff = amp + 1 - n_multi + multi
        status = 'OK' if eff == spec else 'BAD'
        print(name, status, 'cols=', eff, '|', row[:60])
