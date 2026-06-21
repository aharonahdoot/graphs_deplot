import re, sys
def load(fn):
    d = {}
    for ln in open(fn):
        m = re.search(r'(\S.*?)\s+n=\s*(\d+)\s+dmax', ln)
        if m:
            d[m.group(1).strip()] = int(m.group(2))
    return d
a = load(sys.argv[1]); b = load(sys.argv[2])
la, lb = sys.argv[1].split('/')[-1], sys.argv[2].split('/')[-1]
print(f"{'image':24s} {la:>12} {lb:>12}")
for k in b:
    o = a.get(k, '?'); n = b[k]
    flag = '  <== changed' if o != n else ''
    print(f"{k:24s} {str(o):>12} {n:>12}{flag}")
