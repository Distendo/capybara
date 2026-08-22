#!/usr/bin/env python3
import argparse, json, os, re, shutil, signal, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

HOME = Path(os.environ.get('CAPYBARA_HOME', str(Path.home()/'.capybara')))
BIN = HOME/'bin'; MODELS = Path(os.environ.get('CAPYBARA_MODELS', str(HOME/'models'))); RUN = HOME/'run'
SERVER = BIN/'llama-server'; PORT=int(os.environ.get('CAPYBARA_PORT','11434')); HOST=os.environ.get('CAPYBARA_HOST','127.0.0.1')
BASE=f'http://{HOST}:{PORT}'; OPENAI=f'{BASE}/v1'


def die(msg): print(f'capybara: {msg}', file=sys.stderr); raise SystemExit(1)
def models(): MODELS.mkdir(parents=True, exist_ok=True); return sorted(MODELS.glob('*.gguf'), key=lambda p:p.stat().st_mtime, reverse=True)
def model_path(name):
    p=Path(name)
    if p.exists(): return p
    exact=MODELS/name
    if exact.exists(): return exact
    stem=name[:-5] if name.lower().endswith('.gguf') else name
    for p in models():
        if p.name==name or p.stem==stem or p.stem.lower()==stem.lower(): return p
    return exact

def http(path, data=None, method='GET', timeout=5):
    req=urllib.request.Request(BASE+path, data=json.dumps(data).encode() if data is not None else None, headers={'Content-Type':'application/json'}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw=r.read(); return json.loads(raw) if raw else {}
    except (urllib.error.URLError, urllib.error.HTTPError): return None

def running(): return http('/health', timeout=2) is not None

def server_start(model=None, extra=None):
    if running(): return
    if not SERVER.exists(): die('llama-server is not installed')
    if model is None:
        model=os.environ.get('CAPYBARA_MODEL')
    p=model_path(model) if model else (models()[0] if models() else None)
    if p is None or not p.exists(): die('no GGUF model installed')
    RUN.mkdir(parents=True, exist_ok=True); logfile=RUN/'server.log'; pidfile=RUN/'server.pid'
    threads=os.environ.get('CAPYBARA_THREADS', str(os.cpu_count() or 4)); ctx=os.environ.get('CAPYBARA_CONTEXT','10240')
    batch=os.environ.get('CAPYBARA_BATCH','2048'); ubatch=os.environ.get('CAPYBARA_UBATCH','512'); layers=os.environ.get('CAPYBARA_GPU_LAYERS','999'); parallel=os.environ.get('CAPYBARA_PARALLEL','1')
    cmd=[str(SERVER),'--model',str(p),'--host',HOST,'--port',str(PORT),'--threads',threads,'--threads-batch',threads,'--ctx-size',ctx,'--batch-size',batch,'--ubatch-size',ubatch,'--n-gpu-layers',layers,'--parallel',parallel,'--cont-batching','--flash-attn','on']
    if extra: cmd.extend(extra)
    with open(logfile,'ab') as f: proc=subprocess.Popen(cmd,stdout=f,stderr=f)
    pidfile.write_text(str(proc.pid))
    for _ in range(40):
        if running(): return
        if proc.poll() is not None: break
        time.sleep(.25)
    die(f'server failed to start; see {logfile}')

def stop_server():
    pid=RUN/'server.pid'
    if not pid.exists(): print('Capybara is not running'); return
    try: os.kill(int(pid.read_text()), signal.SIGTERM)
    except ProcessLookupError: pass
    pid.unlink(missing_ok=True); print('Capybara stopped')

def hf_parse(spec):
    # owner/repo[:quant] -> llama.cpp -hf style query
    if spec.startswith(('http://','https://')): return None
    if '/' not in spec: return None
    repo, sep, quant=spec.rpartition(':')
    if not sep or '/' not in repo: repo, quant=spec, None
    return repo, quant

def hf_download(spec):
    parsed=hf_parse(spec)
    if not parsed: return False
    repo, quant=parsed
    api=f'https://huggingface.co/api/models/{repo}'
    try:
        with urllib.request.urlopen(api, timeout=20) as r: info=json.load(r)
    except Exception as e: die(f'Hugging Face lookup failed: {e}')
    gg=[s['rfilename'] for s in info.get('siblings',[]) if s.get('rfilename','').lower().endswith('.gguf')]
    if not gg: die(f'no GGUF files found in {repo}')
    if quant:
        q=quant.lower()
        cand=[f for f in gg if q in f.lower()]
        if not cand: die(f'no GGUF matching quantization {quant} in {repo}')
    else:
        cand=[f for f in gg if 'q4_k_m' in f.lower()] or gg
    # Prefer the single direct file if possible
    filename=cand[0]
    dest=MODELS/Path(filename).name; MODELS.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/{repo}/resolve/main/{filename}?download=true'
    curl=shutil.which('curl')
    if curl:
        cmd=[curl,'-L','--fail','--retry','4','--retry-delay','2','--progress-bar','-o',str(dest),url]
        r=subprocess.run(cmd); 
        if r.returncode: dest.unlink(missing_ok=True); die('download failed')
    else:
        try: urllib.request.urlretrieve(url,dest)
        except Exception as e: dest.unlink(missing_ok=True); die(f'download failed: {e}')
    print(f'installed {dest}'); return True

def pull(source):
    MODELS.mkdir(parents=True,exist_ok=True)
    if source.startswith(('http://','https://')):
        filename=Path(source.split('?',1)[0].rstrip('/')).name
        if not filename.lower().endswith('.gguf'): filename += '.gguf'
        dest=MODELS/filename
        if shutil.which('curl'):
            r=subprocess.run([shutil.which('curl'),'-L','--fail','--retry','4','--retry-delay','2','--progress-bar','-o',str(dest),source])
            if r.returncode: dest.unlink(missing_ok=True); die('download failed')
        else:
            try: urllib.request.urlretrieve(source,dest)
            except Exception as e: dest.unlink(missing_ok=True); die(f'download failed: {e}')
        print(f'installed {dest}'); return
    p=Path(source)
    if p.exists():
        dest=MODELS/p.name; shutil.copy2(p,dest); print(f'installed {dest}'); return
    if hf_download(source): return
    die('model must be a local GGUF path, direct URL, or Hugging Face repo[:quant]')

def chat_request(model, messages, stream=True):
    data={'model':Path(model).name,'messages':messages,'stream':stream}
    req=urllib.request.Request(OPENAI+'/chat/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'},method='POST')
    try:
        return urllib.request.urlopen(req,timeout=None)
    except Exception as e: die(str(e))

def chat(model):
    history=[]; print(f'Capybara running {Path(model).name}. /bye exits.')
    while True:
        try: prompt=input('>>> ')
        except (EOFError,KeyboardInterrupt): print(); return
        if prompt.strip() in {'/bye','/exit','/quit'}: return
        if not prompt.strip(): continue
        history.append({'role':'user','content':prompt})
        try:
            with chat_request(model,history,True) as r:
                while True:
                    line=r.readline()
                    if not line: break
                    s=line.decode(errors='replace').strip()
                    if not s.startswith('data:'): continue
                    payload=s[5:].strip()
                    if payload=='[DONE]': break
                    try:
                        obj=json.loads(payload); delta=obj.get('choices',[{}])[0].get('delta',{}).get('content','')
                        if delta: print(delta,end='',flush=True)
                    except Exception: pass
            print()
        except SystemExit: raise
        except Exception as e: print(f'capybara: {e}')

def do_list():
    print('NAME\tSIZE\tMODIFIED')
    for p in models():
        size=p.stat().st_size; units=['B','KB','MB','GB']; n=float(size); i=0
        while n>=1024 and i<3: n/=1024; i+=1
        print(f'{p.stem}\t{n:.1f} {units[i]}\t{time.strftime("%Y-%m-%d %H:%M",time.localtime(p.stat().st_mtime))}')

def do_show(name):
    p=model_path(name)
    if not p.exists(): die('model not found')
    print(f'Name: {p.stem}\nPath: {p}\nSize: {p.stat().st_size} bytes')
    cli=BIN/'llama-cli'
    if cli.exists():
        try:
            r=subprocess.run([str(cli),'--help'],capture_output=True,text=True); print('Engine: llama.cpp')
        except Exception: pass

def do_rm(name):
    p=model_path(name)
    if not p.exists(): die('model not found')
    p.unlink(); print(f'deleted {name}')

def do_cp(src,dst):
    p=model_path(src)
    if not p.exists(): die('source model not found')
    out=MODELS/(dst if dst.lower().endswith('.gguf') else dst+'.gguf'); shutil.copy2(p,out); print(f'copied {src} -> {out.name}')

def generate(model,prompt):
    if not running(): server_start(model)
    req=urllib.request.Request(BASE+'/api/generate',data=json.dumps({'model':Path(model).name,'prompt':prompt,'stream':True}).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=None) as r:
        for line in r:
            try:
                obj=json.loads(line); print(obj.get('response',''),end='',flush=True)
            except Exception: pass
    print()

def create_model(name,file):
    text=Path(file).read_text()
    m=re.search(r'^FROM\s+(.+)$',text,re.I|re.M)
    if not m: die('Modelfile needs FROM')
    base=model_path(m.group(1).strip())
    if not base.exists(): die(f'base model not installed: {m.group(1).strip()}')
    out=MODELS/(name if name.endswith('.gguf') else name+'.gguf'); shutil.copy2(base,out)
    (out.with_suffix('.capybara.json')).write_text(json.dumps({'name':name,'base':m.group(1).strip(),'modelfile':str(Path(file).resolve())},indent=2))
    print(f'created {name}')

def main():
    ap=argparse.ArgumentParser(prog='capybara'); ap.add_argument('-v','--version',action='store_true'); sub=ap.add_subparsers(dest='cmd')
    sub.add_parser('serve'); p=sub.add_parser('run'); p.add_argument('model'); p.add_argument('prompt',nargs='?'); p=sub.add_parser('pull'); p.add_argument('model'); p=sub.add_parser('push'); p.add_argument('model'); p=sub.add_parser('show'); p.add_argument('model'); sub.add_parser('list'); sub.add_parser('ls'); p=sub.add_parser('ps'); p=sub.add_parser('stop'); p.add_argument('model',nargs='?'); p=sub.add_parser('rm'); p.add_argument('model'); p=sub.add_parser('cp'); p.add_argument('source'); p.add_argument('destination'); p=sub.add_parser('generate'); p.add_argument('model'); p.add_argument('prompt'); p=sub.add_parser('create'); p.add_argument('-f','--file',default='Modelfile'); p.add_argument('name',nargs='?'); sub.add_parser('signin'); sub.add_parser('signout'); sub.add_parser('version'); sub.add_parser('logs'); sub.add_parser('help'); p=sub.add_parser('launch'); p.add_argument('integration'); p.add_argument('--model')
    a=ap.parse_args()
    if a.version or a.cmd=='version': print('Capybara 0.2.0 (llama.cpp backend)'); return
    if a.cmd in {None,'help'}: ap.print_help(); return
    if a.cmd=='serve': server_start(); print(f'http://{HOST}:{PORT}'); return
    if a.cmd=='run':
        p=model_path(a.model)
        if not p.exists():
            if not hf_download(a.model): die('model not found')
            p=model_path(a.model)
        if a.prompt: server_start(p); generate(p,a.prompt)
        else: server_start(p); chat(p)
        return
    if a.cmd=='pull': pull(a.model); return
    if a.cmd in {'list','ls'}: do_list(); return
    if a.cmd=='show': do_show(a.model); return
    if a.cmd=='rm': do_rm(a.model); return
    if a.cmd=='cp': do_cp(a.source,a.destination); return
    if a.cmd=='ps': print('RUNNING' if running() else 'STOPPED'); return
    if a.cmd=='stop': stop_server(); return
    if a.cmd=='generate': p=model_path(a.model); server_start(p); generate(p,a.prompt); return
    if a.cmd=='create': create_model(a.name or 'custom',a.file); return
    if a.cmd=='logs':
        log=RUN/'server.log'; print(log.read_text(errors='replace') if log.exists() else 'No logs.'); return
    if a.cmd=='signin': print('Registry auth is not required for public Hugging Face GGUF pulls.'); return
    if a.cmd=='signout': print('Signed out.'); return
    if a.cmd=='push': die('push requires a Capybara registry endpoint; not implemented in local mode')
    if a.cmd=='launch':
        exe=shutil.which(a.integration)
        if not exe: die(f'{a.integration} not installed')
        env=os.environ.copy(); env['OPENAI_BASE_URL']=OPENAI; env['OPENAI_API_BASE']=OPENAI
        if a.model: env['CAPYBARA_MODEL']=a.model
        subprocess.run([exe],env=env,check=False); return
    die(f'unknown command: {a.cmd}')

if __name__=='__main__': main()
