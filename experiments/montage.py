import cv2, glob, os, numpy as np, sys
ovs = sorted(glob.glob("experiments/out/markers/*.png"))
grp = sys.argv[1] if len(sys.argv) > 1 else "all"
cell_w = 470
per_row = 3
def load(f):
    im = cv2.imread(f)
    h, w = im.shape[:2]
    s = cell_w / w
    im = cv2.resize(im, (cell_w, int(h*s)))
    name = os.path.basename(f)[:-4]
    cv2.rectangle(im, (0,0),(cell_w,18),(0,0,0),-1)
    cv2.putText(im, name, (3,13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,255,255),1)
    return im
ims = [load(f) for f in ovs]
# pad to equal heights per row
rows = []
for i in range(0, len(ims), per_row):
    chunk = ims[i:i+per_row]
    h = max(c.shape[0] for c in chunk)
    chunk = [cv2.copyMakeBorder(c,0,h-c.shape[0],0,0,cv2.BORDER_CONSTANT,value=(255,255,255)) for c in chunk]
    while len(chunk) < per_row:
        chunk.append(np.full((h,cell_w,3),255,np.uint8))
    rows.append(np.hstack(chunk))
w = max(r.shape[1] for r in rows)
rows = [cv2.copyMakeBorder(r,0,0,0,w-r.shape[1],cv2.BORDER_CONSTANT,value=(255,255,255)) for r in rows]
# split into pages of 5 rows
for p in range(0, len(rows), 5):
    page = np.vstack(rows[p:p+5])
    out = f"experiments/out/montage_{p//5}.png"
    cv2.imwrite(out, page)
    print("wrote", out, page.shape)
