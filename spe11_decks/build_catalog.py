#!/usr/bin/env python3
"""Verify generated include references and write a linked deck catalogue."""
from pathlib import Path
import json, re, tomllib
ROOT=Path(__file__).resolve().parent
records=json.loads((ROOT/'manifest.json').read_text())
convergence=ROOT/'convergence_manifest.json'
if convergence.exists(): records += json.loads(convergence.read_text())
extra=ROOT/'extra_manifest.json'
if extra.exists(): records += json.loads(extra.read_text())
records=list({r['folder']:r for r in records}.values())
records.sort(key=lambda r:(r['category'],r['folder']))
lines=['# SPE11 deck catalogue','', 'Source: [OPM/pyopmspe11](https://github.com/OPM/pyopmspe11), commit `47f2b44fd6b8bebe7ac8392a1e92ae3b4f885bf7`.','', '## Which are the official benchmark configurations?','', 'The 13 configurations under `benchmark/` are the current upstream OPM benchmark reproduction configurations: SPE11A r1–r5, SPE11B r1–r4, and SPE11C r1–r4. There is no single universal official OPM deck per case. The upstream benchmark gallery identifies this family as reproducing the OPM team results. Examples, regression tests, and locally derived variants are separately labelled below; they are not labelled as benchmark submissions.','', 'Sources: [benchmark overview](https://opm.github.io/pyopmspe11/benchmark.html), [SPE11A](https://opm.github.io/pyopmspe11/benchmark/spe11a.html), [SPE11B](https://opm.github.io/pyopmspe11/benchmark/spe11b.html), [SPE11C](https://opm.github.io/pyopmspe11/benchmark/spe11c.html).','', '### SPE11C historical correction','', 'Upstream documents that Well 1 in the submitted SPE11C results was approximately 100 m above the intended 300 m benchmark depth. Current files are current upstream reproduction inputs, not a guarantee of byte-for-byte historical submission inputs. The additional r5 comparison is locally derived from current r1 by setting dispersion to zero, exactly as described in upstream documentation; it is not a separately shipped configuration or an original submission.','', '## Available decks','', '| Category | Case / variant | Model | Grid dimensions | Cells | Deck | Configuration | Origin |','| --- | --- | --- | --- | ---: | --- | --- | --- |']
errors=[]
for rec in records:
 cfg=tomllib.loads((ROOT/rec['config']).read_text())
 if rec['returncode'] or not rec['decks']:
  errors.append(rec['folder'])
  lines.append(f"| {rec['category']} | {cfg['spe11']} / {Path(rec['folder']).name} | {cfg['model']} | — | — | Generation failed; [log]({rec['folder']}/generation.log) | [TOML]({rec['config']}) | No runnable deck |")
  continue
 for name in rec['decks']:
  path=ROOT/name
  content=path.read_text()
  dims=tuple(map(int,re.search(r'\bDIMENS\s+(\d+)\s+(\d+)\s+(\d+)',content).groups()))
  visited=set()
  def check(p):
   if p in visited: return
   visited.add(p)
   # Stream large grid includes; their numeric lines can be several GB long.
   pattern=re.compile(rb"(?mi)^[ \t]*INCLUDE\s+['\"]?([^'\"\s]+)['\"]?\s+/")
   tail=b''
   with p.open('rb') as stream:
    while block:=stream.read(1024*1024):
     data=tail+block
     for match in pattern.finditer(data):
      inc=p.parent/match.group(1).decode()
      if not inc.is_file(): raise RuntimeError(f'Missing include {inc}')
      check(inc)
     tail=data[-4096:]
  check(path)
  rec['dimensions']=dims
  rec['include_files_checked']=len(visited)-1
  n=dims[0]*dims[1]*dims[2]
  lines.append(f"| {rec['category']} | {cfg['spe11']} / {path.parent.name} | {cfg['model']} | {' × '.join(map(str,dims))} | {n:,} | [{path.name}]({name}) | [TOML]({rec['config']}) | {rec['origin']} |")
lines += ['', '## Scope and usage','', 'This catalogue materializes every shipped TOML case under upstream `benchmark/`, `examples/`, and `tests/configs/`, plus the documented derived SPE11C r5. Arbitrary resolutions and physics settings admit unlimited additional decks. The 11 documented convergence-study configurations (four full-domain and seven lower-domain) are also included from the upstream Mako template and labelled separately. Further optional refinements are parameterized in `pyopmspe11/convergence/convergence.py` and are not enumerated here.','', 'All decks retain their companion include files. The six supplied reference variants are copied directly (the zipped C grid is extracted); other variants are generated with the installed upstream generator in deck-only mode. No simulations were run. All listed include references were checked.','', 'Run a deck from its containing directory:','', '```bash','flow CASE.DATA --output-dir=output','```','', 'For benchmark reproduction, also use the `flow` command and solver options in the corresponding TOML. In particular, SPE11A r5 differs from r4 in solver tolerances outside the DATA file. The benchmark documentation linked above supplies reporting and postprocessing commands. Plain `flow CASE.DATA` does not reproduce all those solver/reporting settings.','', 'The original convenient `SPE11A/`, `SPE11B/`, and `SPE11C/` folders remain: they correspond to A r2, B r1, and C r1.','', 'Regenerate the fixed collection with `.venv/bin/python generate_all.py` and the documented convergence series with `.venv/bin/python generate_convergence.py`; rebuild and verify this catalogue with `.venv/bin/python build_catalog.py`. The isolated environment contains the generator and dependencies. The upstream README specifies Flow 2026.04 or current master; master-specific configuration features still need a compatible executable.','']
if errors: lines += ['## Unavailable upstream configurations', '', 'The upstream lower-domain 320 m convergence template produces z_n[15] = z_n[16] = 0; the current generator rejects these values. Its unmodified TOML and error log are retained. No runnable deck is claimed for that configuration.', '', 'The special_issue_convective example also fails in the current upstream generator with a corner-point refinement array-shape mismatch (50 values into 49 slots). Its unchanged configuration and traceback are retained.', '', 'Incomplete generations: '+', '.join(errors)]
(ROOT/'README.md').write_text('\n'.join(lines))
(ROOT/'verified_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print(f'Checked {len(records)-len(errors)} cases; failures: {errors}')
if errors: raise SystemExit(1)
