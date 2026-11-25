import os, uuid, json, threading, subprocess, re, shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
ROOT=Path(os.getenv('DATA_DIR','data')).resolve(); ROOT.mkdir(parents=True,exist_ok=True)
app=FastAPI(); app.add_middleware(CORSMiddleware,allow_origins=os.getenv('ORIGINS','http://localhost:5173').split(','),allow_methods=['GET','POST'],allow_headers=['*'])
from fastapi.responses import FileResponse
@app.get('/models/{id}/export/{filename}')
def model_file(id:str,filename:str):
    if not re.fullmatch('[a-f0-9]{32}',id) or not re.fullmatch(r'[A-Za-z0-9_.-]+\.ply',filename): raise HTTPException(404)
    p=ROOT/id/'export'/filename
    if not p.is_file(): raise HTTPException(404)
    return FileResponse(p)
lock=threading.Lock()
class Job(BaseModel):
    url:str
    engine:str='gsplat'
    frames:int=Field(200,ge=20,le=800)
    iterations:int=Field(7000,ge=100,le=50000)
def video_id(url):
    p=urlparse(url); v=p.path.strip('/') if p.hostname=='youtu.be' else parse_qs(p.query).get('v',[''])[0] if p.hostname in ('youtube.com','www.youtube.com','m.youtube.com') and p.path=='/watch' else ''
    if p.scheme!='https' or not re.fullmatch(r'[A-Za-z0-9_-]{11}',v): raise ValueError('Invalid YouTube URL')
    return v
def save(d,state):
    tmp=d/'state.tmp'; tmp.write_text(json.dumps(state)); tmp.replace(d/'state.json')
def run(d,state,stage,cmd):
    state.update(stage=stage); save(d,state)
    with (d/'training.log').open('a') as log:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            log.write(line); log.flush(); state['last_log']=line.strip()[-500:]; save(d,state)
        if p.wait(): raise RuntimeError(stage+' failed; see log')
def worker(d,j,v):
    state={'id':d.name,'status':'running','engine':j.engine}
    with lock:
        try:
            if v is not None:
                run(d,state,'download',['yt-dlp','--no-playlist','--max-filesize','500M','-f','bv*[height<=1080]+ba/b[height<=1080]','--merge-output-format','mp4','-o',str(d/'video.%(ext)s'),'https://www.youtube.com/watch?v='+v])
            video=next(d.glob('video.*'))
            run(d,state,'reconstruction',['ns-process-data','video','--data',str(video),'--output-dir',str(d/'scene'),'--num-frames-target',str(j.frames)])
            if j.engine=='gsplat':
                run(d,state,'training',['ns-train','splatfacto','--data',str(d/'scene'),'--output-dir',str(d/'runs'),'--max-num-iterations',str(j.iterations),'--vis','tensorboard'])
                config=next((d/'runs').rglob('config.yml'))
                run(d,state,'export',['ns-export','gaussian-splat','--load-config',str(config),'--output-dir',str(d/'export')])
            else:
                run(d,state,'training',['python','-m','backend.keras_engine',str(d/'scene'),str(d/'export'),str(j.iterations)])
            model=next((d/'export').glob('*.ply')); state.update(status='completed',stage='done',model=f'/models/{d.name}/export/{model.name}')
        except Exception as e: state.update(status='failed',error=str(e))
        save(d,state)
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/jobs',status_code=202)
def create(j:Job):
    try: v=video_id(j.url)
    except ValueError as e: raise HTTPException(400,str(e))
    if j.engine not in ('gsplat','keras'): raise HTTPException(400,'Unknown engine')
    d=ROOT/uuid.uuid4().hex; d.mkdir(); save(d,{'id':d.name,'status':'queued'}); threading.Thread(target=worker,args=(d,j,v),daemon=True).start(); return {'id':d.name}
MAX_UPLOAD=500*1024*1024
@app.post('/jobs/upload',status_code=202)
def upload(video:UploadFile=File(...),engine:str=Form('gsplat'),frames:int=Form(200,ge=20,le=800),iterations:int=Form(7000,ge=100,le=50000)):
    if engine not in ('gsplat','keras'): raise HTTPException(400,'Unknown engine')
    ext=Path(video.filename or '').suffix.lower()
    if ext not in ('.mp4','.mov','.avi','.webm','.mkv','.m4v'): raise HTTPException(415,'Unsupported video format')
    d=ROOT/uuid.uuid4().hex; d.mkdir()
    try:
        total=0
        with (d/('video'+ext)).open('wb') as target:
            while chunk:=video.file.read(1024*1024):
                total+=len(chunk)
                if total>MAX_UPLOAD: raise HTTPException(413,'Video exceeds 500 MiB')
                target.write(chunk)
        if not total: raise HTTPException(400,'Empty video')
        j=Job(url='',engine=engine,frames=frames,iterations=iterations)
        save(d,{'id':d.name,'status':'queued','source':'upload'})
        threading.Thread(target=worker,args=(d,j,None),daemon=True).start()
        return {'id':d.name}
    except Exception:
        shutil.rmtree(d,ignore_errors=True)
        raise
    finally: video.file.close()
@app.get('/jobs/{id}')
def status(id:str):
    if not re.fullmatch('[a-f0-9]{32}',id): raise HTTPException(404)
    p=ROOT/id/'state.json'
    if not p.exists(): raise HTTPException(404)
    return json.loads(p.read_text())
