from fastapi.testclient import TestClient
from backend import main

def test_upload(tmp_path,monkeypatch):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    calls=[]
    monkeypatch.setattr(main.threading,'Thread',lambda **kw:type('Stub',(),{'start':lambda self:calls.append(kw['args'])})())
    r=TestClient(main.app).post('/jobs/upload',files={'video':('clip.mp4',b'example','video/mp4')},data={'engine':'keras'})
    assert r.status_code==202
    assert (tmp_path/r.json()['id']/'video.mp4').read_bytes()==b'example'
    assert calls[0][2] is None

def test_invalid_upload(tmp_path,monkeypatch):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    c=TestClient(main.app)
    assert c.post('/jobs/upload',files={'video':('bad.txt',b'x')}).status_code==415
    assert c.post('/jobs/upload',files={'video':('empty.mp4',b'')}).status_code==400
    monkeypatch.setattr(main,'MAX_UPLOAD',2)
    assert c.post('/jobs/upload',files={'video':('big.mp4',b'abc')}).status_code==413
    assert not list(tmp_path.iterdir())
