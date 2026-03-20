#!/usr/bin/env python3
"""诊断 caj2pdf 输出的 PDF 并测试修正。"""
import fitz
import re
import subprocess

caj = "/Users/dailulu/Desktop/医疗保险支付方式改革对我国公立医院效率的影响研究_胡璇.caj"
pdf_orig = "/tmp/test_orig.pdf"
pdf_fixed = "/tmp/test_fixed.pdf"

# 1) CAJ -> PDF
subprocess.run(["caj2pdf", "convert", caj, "-o", pdf_orig], capture_output=True)

# 2) 检查原始 PDF
doc = fitz.open(pdf_orig)
page = doc[0]
print("=== Page 0 原始内容流 ===")
for cx in page.get_contents():
    stream = doc.xref_stream(cx)
    text = stream.decode("latin-1")
    print(f"  stream len={len(text)}")
    print(f"  first 300 chars: {repr(text[:300])}")

# 3) 尝试修正
modified = False
for pg in doc:
    for cx in pg.get_contents():
        stream = doc.xref_stream(cx)
        text = stream.decode("latin-1")
        new_text, n = re.subn(
            r"([-\d.]+)\s+0\s+0\s+-([\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+cm",
            r"\1 0 0 \2 \3 0 cm",
            text,
        )
        if n > 0:
            doc.update_stream(cx, new_text.encode("latin-1"))
            modified = True
            print(f"  Fixed page {pg.number}, {n} subs")

print(f"modified = {modified}")

if modified:
    doc.save(pdf_fixed, deflate=True)
    doc.close()

    # 验证
    doc2 = fitz.open(pdf_fixed)
    page2 = doc2[0]
    print("\n=== Page 0 修正后内容流 ===")
    for cx in page2.get_contents():
        stream = doc2.xref_stream(cx).decode("latin-1")
        print(f"  first 300 chars: {repr(stream[:300])}")
    doc2.close()
else:
    doc.close()
    print("没有需要修正的内容")
