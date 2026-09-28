#!/usr/bin/env python3
"""Materialize every shipped TOML case plus the documented SPE11C r5 variant."""
from pathlib import Path
import json, re, shutil, subprocess, zipfile, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'pyopmspe11'
EXE = ROOT / '.venv/bin/pyopmspe11'

def main():
    jobs = []
    for category, source in [('benchmark', 'benchmark'), ('examples', 'examples'), ('tests', 'tests/configs')]:
        for cfg in sorted((REPO/source).rglob('*.toml')):
            relative = cfg.relative_to(REPO/source)
            jobs.append((category, cfg, ROOT/category/relative.with_suffix('')))
    derived = ROOT/'configs/derived/spe11c/r5_Cart_50m-50m-10m_no_dispersion.toml'
    derived.parent.mkdir(parents=True, exist_ok=True)
    original = (REPO/'benchmark/spe11c/r1_Cart_50m-50m-10m.toml').read_text()
    derived.write_text('# Derived locally following upstream docs/text/benchmark/spe11c.rst; not a shipped submission config.\n'+re.sub(r'^dispersion = .*$', 'dispersion = [0, 0, 0, 0, 0, 0, 0]', original, flags=re.M))
    jobs.append(('derived', derived, ROOT/'derived/spe11c'/derived.stem))
    # Process the very large C case last.
    jobs.sort(key=lambda j: j[1].stem == 'r4_cp_8m-8mish-8mish')
    if sys.argv[1:]: jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    manifest = ROOT/('extra_manifest.json' if sys.argv[1:] else 'manifest.json')
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    # Recover successful outputs after an interrupted orchestration run.
    known = {r['folder'] for r in records if r['returncode']==0}
    for category,cfg,dst in jobs:
        log=dst/'generation.log'
        folder=str(dst.relative_to(ROOT))
        decks=list(dst.glob('*.DATA'))
        if folder not in known and decks and log.exists() and 'pyopmspe11: success' in log.read_text():
            source=REPO/('tests/configs' if category=='tests' else category)
            config=ROOT/'configs'/category/cfg.relative_to(source) if category!='derived' else cfg
            records.append(dict(category=category,config=str(config.relative_to(ROOT)),folder=folder,decks=[str(d.relative_to(ROOT)) for d in decks],origin='generated',returncode=0))
    completed = {r['folder'] for r in records if r['returncode']==0 and r['decks'] and all((ROOT/p).is_file() for p in r['decks'])}
    records = [r for r in records if r['folder'] in completed]
    jobs = [j for j in jobs if str(j[2].relative_to(ROOT)) not in completed]
    def generate(job):
        category, cfg, dst = job
        dst.mkdir(parents=True, exist_ok=True)
        config_copy = ROOT/'configs'/category/cfg.relative_to(REPO/('tests/configs' if category=='tests' else category)) if category!='derived' else cfg
        config_copy.parent.mkdir(parents=True, exist_ok=True)
        if cfg != config_copy: shutil.copy2(cfg, config_copy)
        ref = REPO/'tests/decks'/cfg.parent.name/cfg.stem if category=='benchmark' else None
        print('START', category, cfg.stem, flush=True)
        if ref is not None and ref.is_dir():
            for file in ref.iterdir():
                if file.suffix=='.zip':
                    with zipfile.ZipFile(file) as z: z.extractall(dst)
                else: shutil.copy2(file, dst/file.name)
            code=0
            origin='upstream reference'
        else:
            with (dst/'generation.log').open('w') as log:
                code=subprocess.run([str(EXE),'-i',str(cfg),'-o',str(dst),'-m','deck','-f','0'],stdout=log,stderr=subprocess.STDOUT).returncode
            origin='generated'
        decks=list(dst.glob('*.DATA'))
        print('DONE', category, cfg.stem, 'status', code, flush=True)
        return dict(category=category,config=str(config_copy.relative_to(ROOT)),folder=str(dst.relative_to(ROOT)),decks=[str(d.relative_to(ROOT)) for d in decks],origin=origin,returncode=code)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(generate, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            manifest.write_text(json.dumps(records,indent=2)+'\n')
    if any(r['returncode'] or not r['decks'] for r in records): raise SystemExit('Some decks failed; inspect manifest and logs')
if __name__=='__main__': main()
