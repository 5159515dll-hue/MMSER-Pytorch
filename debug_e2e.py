#!/usr/bin/env python3
"""端到端测试：CAJ -> 修正PDF -> DOCX"""
import fitz
import re
import subprocess
from pathlib import Path

caj = Path("/Users/dailulu/Desktop/医疗保险支付方式改革对我国公立医院效率的影响研究_胡璇.caj")
pdf_path = Path("/tmp/test_e2e.pdf")
docx_path = Path("/tmp/test_e2e.docx")

# 1) CAJ -> PDF
print("1) CAJ -> PDF")
r = subprocess.run(["caj2pdf", "convert", str(caj), "-o", str(pdf_path)], capture_output=True, text=True)
print(f"   exit={r.returncode}")

# 2) Fix PDF
print("2) Fix PDF")
doc = fitz.open(str(pdf_path))
modified = False
for page in doc:
    if page.rotation != 0:
        page.set_rotation(0)
        modified = True
    for cx in page.get_contents():
        stream = doc.xref_stream(cx)
        text = stream.decode("latin-1")
        new_text, n = re.subn(
            r'([-\d.]+)\s+0\s+0\s+-([\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+cm',
            r'\1 0 0 \2 \3 0 cm',
            text,
        )
        if n > 0:
            doc.update_stream(cx, new_text.encode("latin-1"))
            modified = True

print(f"   modified={modified}")
if modified:
    fixed_path = Path("/tmp/test_e2e_fixed.pdf")
    doc.save(str(fixed_path), deflate=True)
    doc.close()
    fixed_path.replace(pdf_path)
    print(f"   saved fixed pdf")
else:
    doc.close()

# 3) Verify the fixed PDF is actually correct
print("3) Verify fixed PDF page 0")
doc2 = fitz.open(str(pdf_path))
p = doc2[0]
for cx in p.get_contents():
    stream = doc2.xref_stream(cx).decode("latin-1")
    print(f"   stream: {repr(stream[:80])}")
doc2.close()

# 4) PDF -> DOCX
print("4) PDF -> DOCX")
from pdf2docx import Converter
cv = Converter(str(pdf_path))
cv.convert(str(docx_path))
cv.close()
print(f"   done: {docx_path}")
print(f"   docx size: {docx_path.stat().st_size} bytes")
