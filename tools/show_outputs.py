import sys, nbformat
nb = nbformat.read(sys.argv[1], 4)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
for i, c in enumerate(nb.cells):
    if c.cell_type != "code": continue
    for o in c.get("outputs", []):
        t = o.get("text") or "".join(o.get("data", {}).get("text/plain", "")) if o.get("output_type") != "error" else "ERROR " + o.get("ename","") + ": " + o.get("evalue","")
        if t: print(f"--- cell {i} ---\n" + str(t)[-n:])
