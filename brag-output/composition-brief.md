# Hyperframes Composition Brief: ForensiQ — whiteboard explainer lecture

## Objective

Create a long-form whiteboard explainer lecture (NOT a 15–25s launch video —
TUTOR_BRIEF override) teaching ForensiQ to a smart junior. Narration ON
(`--voice`), Kokoro `af_heart`, target ~6 minutes. Calm, precise, friendly;
every keyword defined on screen at first use.

## Output

- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: sum of scene audio lengths + pads (compute from the generated
  voiceover WAVs; do NOT hardcode scene lengths before audio exists)

## Source Material

- Project root: `D:\aria\Projects\ForensiQ`
- Primary files read: README.md, docs/TAXONOMY.md, docs/PLAYBOOKS.md,
  src/forensiq/{ingest/schema.py, taxonomy/classifier.py,
  attribution/{blame.py, health.py, replay.py}, patterns/{cluster.py,
  drift.py}, report/rca.py, cli.py}
- Product name: ForensiQ
- Tagline: "the flight recorder + detective for AI pipelines"
- Key visual moments to recreate: the trace→spans unroll sketch, the
  taxonomy family wall (real ids + signatures), the two-pass rules→LLM
  flow, the z-score blame sketch, the drift line chart with day-15 kink,
  the RCA page mock with blame graph
- Copy that must appear verbatim:
  - Failure ids: F-RET-001 · F-RET-002 · F-PROMPT-001 · F-GEN-002 ·
    F-GEN-003 · F-TOOL-001 · F-INFRA-001 · F-INFRA-002 · F-INFRA-003
  - Signatures: `hit_count == 0` · `finish_reason == "length"` ·
    "any 3/5/8-gram repeated ≥ 5" · "cost > p95 × 3"
  - Verdict: `F-RET-002 → {"retrieve": 1.0}`
  - Drift: baseline 14d · trailing 3d · α = 0.01 · 2-of-3 windows · CUSUM
  - CLI: `forensiq demo` · `forensiq report --file traces.jsonl --out rca.md`
    · `forensiq watch` · `forensiq replay --taxonomy F-PROMPT-001`
  - Numbers: precision 1.0 · recall 1.0 · blame top-1 100% · 8/8 clusters ·
    128 tests ~4s offline · 560 traces · day-15 drift
  - "drift detection measures change, not badness"

## Creative Direction

- Tone preset: polished
- Creative direction: whiteboard lecture — patient senior engineer teaching
  one system; chalk-on-board aesthetic; zero hype; definitions pinned as
  cards
- Angle: "bad answers since Tuesday and nobody knows why" carries the
  whole lecture; ForensiQ = black box + detective
- Hook: the Mon→Tue calendar flip and the hanging question
- Outro / punchline: "Don't guess. Read the recording."
- Avoid:
  - Generic SaaS language, launch energy, hype adjectives
  - Abstract filler visuals (every sketch is a concept from the repo)
  - Beat-grid text reveals that outrun the narration

## Visual Identity

- Background: `#13211d` chalkboard green-slate, subtle vignette
- Text: chalk white `#f2f0e9` (body ≥ 28px at 1080p)
- Accent: amber `#f5b942` (failures/alarms), teal `#63d3c3` (evidence/
  verdicts), chalk blue `#9ec5e8` (diagram strokes)
- Display font: 'Ink Free', system-ui fallback (chalk handwriting feel);
  body font: system-ui, sans-serif
- Visual references from the project: taxonomy table structure
  (id | title | stage | signature), pipeline stations embed → retrieve →
  rerank → generate, mermaid blame-graph arrow style, RCA section headings

## Storyboard

Use `brag-plan.md` as the creative contract. Nine scenes:

1. The Tuesday problem — ~40s — Mon→Tue calendar, broken answer, log wall,
   Ouija line, title card
2. The black box — ~50s — flight recorder analogy, trace→spans unroll with
   quoted attrs, JSONL/Langfuse ingest, normalisation note
3. Naming the failure — ~60s — five family cards, real ids + signatures,
   playbook mapping
4. Two-pass classification — ~50s — 9 rules desk → leftovers → LLM desk
   (optional, EchoMock fallback)
5. Blame ranking — ~55s — stage gauges, z-score sketch, agreement rule,
   verdict card, 100% top-1
6. The slow leak — ~60s — 28-day decay chart, baseline/trailing windows,
   p-value vs α, 2-of-3 rule, CUSUM, honest caveat
7. The report — ~50s — RCA page mock: exec summary, blame graph, evidence,
   playbook; postmortem/tickets; byte-identical note
8. The numbers — ~45s — six stat tiles count up with plain-word captions
9. Recap + end card — ~50s+ — six recap lines, closing card holds

Keyword definition cards: pinned top-right of the board as each term first
appears; term + one plain sentence; hold ≥ 0.3s/word.

## Audio

- Audio role: warm quiet bed under continuous narration
- Audio arc: bed starts low, stays low (narration is ~100% of runtime),
  ends with the final frame
- Music: `assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`
  (copy from skill assets); loop back-to-back (two+ elements at data-start 0
  and data-start <source-length>) to cover full runtime; volume 0.13
- Music cue guidance: preset JSON exists in the skill's music/cues/
  directory; natural timing chosen over beat locks — narration-paced
  lecture, cues would fight reading floors
- Audio-reactive treatment: none (narration-led lecture — documented decision)
- Audio-coupled moments:
  - keyword definition cards — soft drop pop-in
  - failure-id card landing — single soft impact
  - stats count-up on the numbers board — subtle ticks allowed, no dense SFX
- SFX selection guidance: prefer low high-frequency-risk files
  (interface/drop_001, impact/impactSoft_medium_*); ≤ 5 SFX total; volume
  0.4–0.6, quieter than the bed
- Exact SFX choice: Hyperframes chooses filenames/timestamps after animation
  exists; copy chosen files into `composition/assets/sfx/`
- Voiceover: per-scene WAVs generated via `npx hyperframes tts --voice
  af_heart` into `composition/assets/voiceover/voice_01.wav … voice_09.wav`
  (via `gen_voiceover.py`); each wired on its own track (`data-track-index`
  3..11, data-volume 1); scene `data-start`/`data-duration` derive from
  measured WAV durations (0.35s lead-in, ~0.35s tail per scene; +1.2s title
  settle on scene 1; +2.5s end hold on scene 9). Root `data-duration` =
  total.

## Hyperframes Instructions

Load `hyperframes-core`, `hyperframes-animation`, `hyperframes-creative`,
`hyperframes-keyframes`, `hyperframes-cli`. /brag owns the angle; Hyperframes
owns composition structure, exact timing, lint, render.

Requirements:
- Single standalone `index.html`, one paused GSAP timeline registered at
  `window.__timelines["forensiq-lecture"]` matching the root
  `data-composition-id`.
- Never tween `.clip` elements (animate inner wrappers); no CSS-transform +
  GSAP conflicts; every `<audio>` has an id; no `crossorigin`; no network at
  render time (GSAP via CDN is standard — prefer a local copy if check flags
  failed requests).
- Scenes are contiguous `.clip` sections; entrance fades on inner wrappers;
  chalk-stroke diagram accents may draw in (opacity/scale, no heavy motion).
- All text ≥ 28px body; chalk-on-dark contrast ≥ WCAG AA (light chalk on
  #13211d passes; amber text ≥ 32px bold or use for borders/underlines;
  kcard defs 21.5px on rgba(255,255,255,0.05) panels — proven passing in the
  sibling episode's check run).
- Deterministic: no Math.random, no Date.now, finite repeats only.
- Run `npx hyperframes check` before render — the single gate; fix every
  error including WCAG contrast findings.
- Render with `--quality delivery` (long-form lecture, final delivery).
