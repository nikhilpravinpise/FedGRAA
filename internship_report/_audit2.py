import re
from collections import Counter
src = open('report.tex', encoding='utf-8').read()
labels = re.findall(r'\\label\{([^}]+)\}', src)
dup = [l for l, c in Counter(labels).items() if c > 1]
print('labels:', len(labels), 'dups:', dup)
refs = set(re.findall(r'\\(?:eq)?ref\{([^}]+)\}', src))
missing = [r for r in refs if r not in labels]
print('refs:', len(refs), 'missing targets:', missing)
cites = set(k.strip() for grp in re.findall(r'\\cite\{([^}]+)\}', src) for k in grp.split(','))
bib = set(re.findall(r'@\w+\{([^,]+),', open('references.bib', encoding='utf-8').read()))
print('cites:', len(cites), 'missing bib:', [c for c in cites if c not in bib])
print('uncited bib:', [b for b in bib if b not in cites])
# table column integrity for the two new tables
for name, spec in [('tab:claims', 4), ('tab:simplex-violation', 5)]:
    i = src.find('\\label{%s}' % name)
    j = src.rfind('\\begin{tabular}', 0, i)
    k = src.find('\\end{tabular}', i)
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
