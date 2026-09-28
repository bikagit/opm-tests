#!/usr/bin/env python3
"""Generate the two documented convergence series, without running Flow."""
from pathlib import Path
import json, subprocess
from mako.template import Template
ROOT=Path(__file__).resolve().parent
rows=[]
template=Template(filename=str(ROOT/'pyopmspe11/convergence/spe11b.mako'))
for domain,sizes in [('full',['40','20','10','5']),('lower',['320','160','80','40','20','10','5'])]:
 for i,size in enumerate(sizes):
  name=f'{domain}_cp{i}-z{size}mish-x{size}m'
  cfg=ROOT/'configs/convergence'/f'{name}.toml'
  cfg.parent.mkdir(parents=True,exist_ok=True)
  cfg.write_text(template.render(i=i,domain=domain))
  dst=ROOT/'convergence'/name
  dst.mkdir(parents=True,exist_ok=True)
  cmd=[str(ROOT/'.venv/bin/pyopmspe11'),'-i',str(cfg),'-o',str(dst),'-m','deck','-f','0']
  if domain=='lower': cmd+=['-n','lower']
  print('START',name,flush=True)
  with (dst/'generation.log').open('w') as log:
   code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
  rows.append(dict(category='convergence',config=str(cfg.relative_to(ROOT)),folder=str(dst.relative_to(ROOT)),decks=[str(p.relative_to(ROOT)) for p in dst.glob('*.DATA')],origin='generated from upstream template',returncode=code))
  (ROOT/'convergence_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
  print('DONE',name,code,flush=True)
