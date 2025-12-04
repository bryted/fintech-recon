from pathlib import Path
lines = Path('numbered.txt').read_text().splitlines()
out = []
for entry in lines:
    part = entry.split(':',1)
    out.append(part[1] if len(part) else entry)
Path('reconciliation_engine.py').write_text('\n'.join(out))
