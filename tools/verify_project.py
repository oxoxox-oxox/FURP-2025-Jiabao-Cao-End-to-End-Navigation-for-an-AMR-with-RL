"""Verify documented evidence, original STPS logic and actual simulator/model interfaces."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'


def run(*args):
    result = subprocess.run([sys.executable,'-X','utf8',*args],cwd=ROOT,
                            capture_output=True,text=True,encoding='utf-8')
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode:
        raise RuntimeError(f'Command failed with {result.returncode}: {args}')


def main():
    (ROOT/'artifacts').mkdir(exist_ok=True)
    for name in ('README.md','README.zh-CN.md'):
        text=(ROOT/name).read_text(encoding='utf-8')
        links=re.findall(r'\]\(([^)]+)\)',text)+re.findall(r'(?:src|href)="([^"]+)"',text)
        for link in links:
            if not link.startswith(('https:','http:','#')):
                assert (ROOT/link.split('#')[0]).exists(),(name,link)
    for line in (ROOT/'docs/assets/SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split('  ',1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    run('tools/summarize_results.py')
    run('-m','unittest','discover','-s','tests','-v')
    run('tools/smoke_stps.py')
    run('tools/smoke_model.py')
    print('VERIFICATION PASSED: archived statistics, switching tests, real IR-SIM, CPU model, links and poster hashes.')
    print('Full trained-policy evaluation remains dependent on the original checkpoint files.')


if __name__ == '__main__':
    main()
