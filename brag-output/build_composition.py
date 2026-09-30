"""Render composition/index.html from index.template.html using measured WAV durations."""
import json
from pathlib import Path

HERE = Path(__file__).parent
COMP = HERE / "composition"
MUSIC_LEN = 117.384

durs = json.loads((HERE / "voiceover-durations.json").read_text())
V = [durs[f"scene{i:02d}"] for i in range(1, 10)]

# scene durations: lead + narration + tail
LEAD = [1.2] + [0.35] * 8
TAIL = [0.5] + [0.4] * 7 + [2.9]
D = [LEAD[i] + V[i] + TAIL[i] for i in range(9)]

S = [0.0]
for i in range(8):
    S.append(round(S[i] + D[i], 3))
TOT = round(sum(D), 3)

VO_START = [round(S[i] + LEAD[i], 3) for i in range(9)]

# music loops
music_lines = []
t = 0.0
i = 1
while t < TOT - 0.001:
    d = round(min(MUSIC_LEN, TOT - t), 3)
    music_lines.append(
        f'  <audio id="music-{i}" data-start="{round(t,3)}" data-duration="{d}" '
        f'data-track-index="10" data-volume="0.13" '
        f'src="assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3"></audio>'
    )
    t += MUSIC_LEN
    i += 1

# sfx moments (aligned to timeline beats)
SFX = [
    round(VO_START[0] + 0.4, 3),   # title drop (scene 1)
    round(S[2] + 18.4, 3),         # F-RET-001 hero card lands (scene 3)
    round(S[4] + 34.4, 3),         # verdict card lands (scene 5)
    round(S[8] + 30.6, 3),         # end card (scene 9)
]

subs = {
    "TOTAL": f"{TOT:.3f}",
    **{f"SS{i+1}": f"{S[i]:.3f}" for i in range(9)},
    **{f"SD{i+1}": f"{D[i]:.3f}" for i in range(9)},
    **{f"VOS{i+1}": f"{VO_START[i]:.3f}" for i in range(9)},
    **{f"VOD{i+1}": f"{V[i]:.3f}" for i in range(9)},
    **{f"SFX{i+1}": f"{SFX[i]:.3f}" for i in range(4)},
    "MUSIC_LINES": "\n".join(music_lines),
    "JS_S1": f"{S[0]:.3f}", "JS_S2": f"{S[1]:.3f}", "JS_S3": f"{S[2]:.3f}",
    "JS_S4": f"{S[3]:.3f}", "JS_S5": f"{S[4]:.3f}", "JS_S6": f"{S[5]:.3f}",
    "JS_S7": f"{S[6]:.3f}", "JS_S8": f"{S[7]:.3f}", "JS_S9": f"{S[8]:.3f}",
    "JS_TOT": f"{TOT:.3f}",
}

html = (HERE / "index.template.html").read_text(encoding="utf-8")
for k, v in subs.items():
    html = html.replace("{{" + k + "}}", v)
leftover = [t for t in html.split("{{") if "}}" in t.split("}}")[0]]
if leftover:
    raise SystemExit(f"unsubstituted tokens: {[t.split('}}')[0] for t in leftover]}")
(COMP / "index.html").write_text(html, encoding="utf-8")

print(f"scenes: {[round(d,2) for d in D]}")
print(f"total: {TOT:.3f}s = {TOT/60:.2f} min")
print(f"music loops: {len(music_lines)}")
print(f"sfx at: {SFX}")
