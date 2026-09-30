"""Keyboard teleoperation in the browser: record human demonstrations for the insertion task.

  python scripts/teleop_server.py            # then open http://localhost:8020 (VS Code forwards the port)

Keys (hold): W/S +x/-x, A/D +y/-y, R/F up/down, Q/E rotate, Shift = fine (x0.25). Space toggles the gripper.
Enter saves the episode (only if it succeeded), Backspace discards it and starts a new scene.
Each saved episode goes to data/teleop/episode_<seed>.npz with obs, act and the operator label "human".
Seeds start at 200000 so teleop scenes never overlap the scripted demos (0-499) or the evaluation (100000+).
"""

import io
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assembly_lab.env import CONTROL_HZ, MAX_STEPS, AssemblyEnv, EpisodeConfig

OUT = ROOT / "data" / "teleop"
OUT.mkdir(parents=True, exist_ok=True)
SEED0 = 200_000


class Session:
    def __init__(self):
        self.env = AssemblyEnv()
        self.renderer = mujoco.Renderer(self.env.model, 360, 480)
        self.lock = threading.Lock()
        self.keys = set()
        self.grip = -1.0
        self.frame = b""
        self.saved = len(list(OUT.glob("episode_*.npz")))
        self.seed = SEED0 + self.saved
        self.new_scene()

    def new_scene(self):
        while (OUT / f"episode_{self.seed}.npz").exists():
            self.seed += 1
        rng = np.random.default_rng(self.seed + 55_555)
        off = float(rng.uniform(0, 0.0025)) if rng.random() < 0.5 else 0.0
        self.cfg = EpisodeConfig(seed=self.seed, perception_offset=off)
        self.obs = self.env.reset(self.cfg)
        self.O, self.A = [], []
        self.success = False
        self.grip = -1.0

    def action(self):
        k = self.keys
        s = 0.25 if "shift" in k else 1.0
        a = np.zeros(5)
        a[0] = s * (("w" in k) - ("s" in k))
        a[1] = s * (("a" in k) - ("d" in k))
        a[2] = s * (("r" in k) - ("f" in k))
        a[3] = s * (("q" in k) - ("e" in k))
        a[4] = self.grip
        return a

    def tick(self):
        with self.lock:
            if not self.success and self.env.t < MAX_STEPS:
                a = self.action()
                self.O.append(self.obs)
                self.A.append(a.astype(np.float32))
                self.obs, self.success, _ = self.env.step(a)
            img = np.concatenate([self.env.render(self.renderer, "front"), self.env.render(self.renderer, "close")], axis=1)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=80)
            self.frame = buf.getvalue()

    def save(self):
        with self.lock:
            if not self.success:
                return False
            info = dict(
                seed=self.seed,
                operator="human",
                perception_offset_mm=self.cfg.perception_offset * 1000,
                steps=self.env.t,
                peak_force=float(self.env.peak_force),
                err_xy_mm=float(np.linalg.norm(self.env.placement_error()[0]) * 1000),
                saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            np.savez_compressed(
                OUT / f"episode_{self.seed}.npz", obs=np.array(self.O), act=np.array(self.A), info=json.dumps(info)
            )
            self.saved += 1
            self.seed += 1
            self.new_scene()
            return True

    def discard(self):
        with self.lock:
            self.seed += 1
            self.new_scene()

    def status(self):
        with self.lock:
            f = float(np.linalg.norm(self.env.wrench()[:3]))
            return dict(
                seed=self.seed,
                step=self.env.t,
                max_steps=MAX_STEPS,
                success=self.success,
                gripper="closed" if self.grip > 0 else "open",
                wrist_force_N=round(f, 1),
                perception_offset_mm=round(self.cfg.perception_offset * 1000, 2),
                saved=self.saved,
            )


S = Session()
app = FastAPI()


def loop():
    period = 1.0 / CONTROL_HZ
    while True:
        t0 = time.time()
        S.tick()
        time.sleep(max(0.0, period - (time.time() - t0)))


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/stream")
def stream():
    def gen():
        while True:
            yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + S.frame + b"\r\n"
            time.sleep(1.0 / CONTROL_HZ)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.post("/keys")
def keys(body: dict):
    S.keys = {k.lower() for k in body.get("down", [])}
    return {}


@app.post("/grip")
def grip():
    S.grip = -S.grip
    return {}


@app.post("/save")
def save():
    return {"saved": S.save()}


@app.post("/discard")
def discard():
    S.discard()
    return {}


@app.get("/status")
def status():
    return JSONResponse(S.status())


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>assembly-lab teleop</title><style>
body{margin:0;background:#12151b;color:#e6e9ef;font:15px/1.5 system-ui,sans-serif}main{max-width:980px;margin:auto;padding:16px}
img{width:100%;border-radius:8px;background:#000}#st{display:flex;flex-wrap:wrap;gap:8px 18px;margin:10px 0}
b{color:#7aa7ff}.ok{color:#5fd38d}.keys{color:#9aa3b2;font-size:14px}</style></head><body><main>
<h2>assembly-lab teleoperation</h2>
<img src="/stream" alt="robot camera views">
<div id=st></div>
<p class=keys>Hold W/S (x), A/D (y), R/F (up/down), Q/E (rotate), Shift = fine. Space = gripper.
Enter = save (only after success), Backspace = discard and new scene. Click the page once so it receives keys.</p>
</main><script>
const down=new Set();const map={' ':'space'};
function send(){fetch('/keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({down:[...down]})})}
addEventListener('keydown',e=>{const k=e.key.toLowerCase();
 if(k===' '){e.preventDefault();if(!e.repeat)fetch('/grip',{method:'POST'});return}
 if(k==='enter'){fetch('/save',{method:'POST'}).then(r=>r.json()).then(j=>{if(!j.saved)alert('Not saved: the part is not seated and released yet.')});return}
 if(k==='backspace'){e.preventDefault();fetch('/discard',{method:'POST'});return}
 if(e.shiftKey)down.add('shift');down.add(k);send()});
addEventListener('keyup',e=>{down.delete(e.key.toLowerCase());if(!e.shiftKey)down.delete('shift');send()});
addEventListener('blur',()=>{down.clear();send()});
setInterval(()=>fetch('/status').then(r=>r.json()).then(s=>{document.getElementById('st').innerHTML=
 `<span>scene <b>${s.seed}</b></span><span>step <b>${s.step}/${s.max_steps}</b></span><span>gripper <b>${s.gripper}</b></span>`+
 `<span>wrist force <b>${s.wrist_force_N} N</b></span><span>perception error <b>${s.perception_offset_mm} mm</b></span>`+
 `<span class=${s.success?'ok':''}>${s.success?'SUCCESS - press Enter':'not done'}</span><span>saved <b>${s.saved}</b></span>`}),300);
</script></body></html>"""

if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    port = int(os.environ.get("TELEOP_PORT", "8020"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
