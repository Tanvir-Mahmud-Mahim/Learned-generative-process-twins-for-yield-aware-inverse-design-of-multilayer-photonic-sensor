"""Inject generated table rows into paper/main.tex between markers."""
import os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "paper")
t = open(os.path.join(P, "main.tex")).read()
for name in ["twin", "yield", "transfer"]:
    rows = open(os.path.join(P, "table_%s.tex" % name)).read().rstrip("%\n")
    rows = "\n".join("            " + r for r in rows.split("\n"))
    t = re.sub(r"(%BEGIN-TABLE-" + name + r"\n).*?(\s*%END-TABLE-" + name + ")",
               r"\1" + rows.replace("\\", "\\\\") + r"\2", t, flags=re.S)
open(os.path.join(P, "main.tex"), "w").write(t)
print("injected tables")
