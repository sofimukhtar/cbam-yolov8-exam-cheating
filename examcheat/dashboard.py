from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Exam Monitor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#f7f8fa;color:#1d2330}
h1{font-size:20px;margin:0 0 4px} .meta{color:#5b6475;font-size:13px;margin-bottom:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.card{background:#fff;border-radius:8px;padding:10px;border-left:6px solid #2e7d32;box-shadow:0 1px 2px #0002}
.card.Yellow{border-color:#f9a825}.card.Red{border-color:#c62828}
.id{font-weight:600}.s{font-size:22px;font-variant-numeric:tabular-nums}.small{font-size:12px;color:#5b6475}
</style></head><body>
<h1>Examination monitoring</h1><div class="meta" id="meta">waiting for data...</div>
<div class="grid" id="grid"></div>
<script>
async function tick(){
  try{
    const r = await fetch('state'); const s = await r.json();
    if(!s.students){return;}
    document.getElementById('meta').textContent =
      `t = ${s.time_s.toFixed(0)} s | thresholds ${s.thresholds.join(' / ')} | step ${s.latency_ms.toFixed(0)} ms | ambiguous ${s.ambiguous}`;
    const ids = Object.keys(s.students).sort((a,b)=>s.students[b].S - s.students[a].S);
    document.getElementById('grid').innerHTML = ids.map(id=>{
      const x = s.students[id];
      return `<div class="card ${x.alert}"><div class="id">${id}</div><div class="s">${x.S.toFixed(3)}</div>
      <div class="small">${x.alert} | observed ${(100*x.observed_fraction).toFixed(0)} %</div></div>`;}).join('');
  }catch(e){}
}
setInterval(tick, 1000); tick();
</script></body></html>"""


def create_app(state_path: Path) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

    @app.get("/state")
    def state():
        try:
            return jsonify(json.loads(Path(state_path).read_text()))
        except (FileNotFoundError, json.JSONDecodeError):
            return jsonify({})

    return app
