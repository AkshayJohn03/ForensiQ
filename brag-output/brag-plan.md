# Brag Plan: ForensiQ — whiteboard explainer lecture

## What is this app?

ForensiQ is failure forensics for AI pipelines: it ingests traces, names the
failure from a taxonomy, ranks the guilty stage with quoted evidence, detects
statistical drift, and writes the RCA report a manager reads the morning
after. The flight recorder + detective for RAG systems.

**TUTOR_BRIEF override:** this is NOT a 15–25s launch video. It is a
whiteboard explainer lecture per the `--tone` paragraph — patient senior
engineer at a whiteboard teaching one system to a smart junior, narration on,
5+ minutes, every keyword defined on screen at first use.

## The 9-question rubric

1. **What is the app?** Failure forensics for AI (RAG) pipelines — records
   traces, names failures, blames stages, detects drift, writes the report.
2. **Most impressive claim?** Precision/recall 1.0 on planted failures; blame
   top-1 100%; drift honest at α=0.01 — measured against planted ground
   truth, not vibes.
3. **Visual hook?** The evidence board: a failure id card (F-RET-001) with
   its quoted span signature — `hit_count: 0`.
4. **What to show from the actual repo?** The taxonomy table (real ids),
   the two-pass flow (rules → LLM leftovers), the blame equation (z-score +
   evidence weight), the chi-square drift alarm, the mermaid-style RCA
   structure, the real CLI (`forensiq demo / report / watch / replay`).
5. **Shortest satisfying video?** Not applicable — long-form lecture, target
   ~6 minutes (task brief: 4–6+ min).
6. **Tone?** Preset `polished`; direction: TUTOR_BRIEF --tone paragraph
   verbatim (whiteboard lecture, calm, precise, friendly, no hype).
7. **Audio?** Warm quiet music bed under continuous narration (narration is
   ~100% of runtime); sparse soft SFX on card landings only.
8. **Share caption?** "Since Tuesday the assistant gives bad answers and
   nobody knows why. ForensiQ is the flight recorder + detective for AI
   pipelines: it names the failure, points at the guilty stage with quoted
   evidence, and catches the slow leak before you do."
9. **User flow worth showing?** CLI flow: `traces.jsonl` → `forensiq
   ingest` → classification → blame → `rca.md`. Recreated as the recurring
   board motif (traces in, report out).

## The angle

"Since Tuesday, the assistant gives bad answers and nobody knows why."
Debugging without a recording is Ouija-board work. ForensiQ is the black box
plus the detective: record first, then name, then blame with evidence, then
watch for the slow leak, then write the report. The lecture follows the
pipeline's own five layers — ingest, taxonomy, attribution, patterns, report
— one whiteboard sketch per concept.

## Hook (first 30 seconds)

The scenario on the board: a calendar flipping from fine-Monday to
broken-Tuesday, an assistant answer going wrong, and the question hanging:
"nobody knows why". Then the title card: ForensiQ — the flight recorder +
detective for AI pipelines.

## Key moments (the middle)

- The trace anatomy sketch: a request unrolled into spans (embed → retrieve
  → rerank → generate) with quoted attributes — `hit_count: 0`.
- The taxonomy wall: five family cards (F-RET / F-PROMPT / F-GEN / F-TOOL /
  F-INFRA) with real ids and one-line signatures.
- The two-pass flow: 9 rules stamp verdicts in milliseconds; only ambiguous
  leftovers walk to the LLM desk (which may be empty — it's optional).
- The blame equation: z-score vs healthy baseline + evidence weight must
  agree → `F-RET-002 → {"retrieve": 1.0}`.
- The drift alarm: baseline 14 days vs trailing 3 days, chi-square p-value
  crossing α=0.01, the 2-of-3 rule, CUSUM for slow ramps.
- The RCA page: executive summary → mermaid blame graph → evidence →
  playbook fix.

## Outro / punchline

"That's ForensiQ — the flight recorder and the detective. When the assistant
goes weird on a Tuesday, you don't guess. You read the recording."

## User flow worth showing

traces (JSONL / Langfuse) → ForensiQ (classify → blame → drift) → rca.md.
Shown as the board's recurring left-to-right flow and concretely in scenes
2, 5, and 7.

## Tone

- Preset: polished
- Creative direction: TUTOR_BRIEF --tone paragraph (whiteboard explainer
  lecture; patient senior engineer; no hype; definitions pinned on screen)
- Interpretation: long-form pacing, calm reveals, generous holds sized to
  narration, chalk aesthetic, zero launch energy.

## Format: landscape — 1920x1080
## Duration: sum of scene narration WAVs + pads (target ≈ 6 min; task floor 4 min, ceiling 7 min)

## Visual identity

- Background: `#13211d` chalkboard green-slate (series-consistent with
  sibling episodes), subtle vignette
- Text: chalk white `#f2f0e9`; dim `#d5d2c6` (body ≥ 28px at 1080p; kcard
  defs 21.5px — proven passing in the sibling episode)
- Accent: amber `#f5b942` (failures/alarms), teal `#63d3c3` (evidence/
  verdicts), chalk blue `#9ec5e8` (diagram strokes)
- Display font: 'Ink Free' (local TTF), body system-ui
- Strongest visual element: the failure-id evidence cards and the trace→
  spans unroll sketch

## Share copy (draft)

"Since Tuesday the assistant gives bad answers and nobody knows why.
ForensiQ is the flight recorder + detective for AI pipelines — it names the
failure from a taxonomy, points at the guilty stage with quoted evidence,
and catches the slow leak before you do. 6-minute whiteboard lecture inside."

## Audio direction

- Role: warm quiet bed under continuous narration
- Music: `assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`
  (skill asset, looped back-to-back; volume 0.13)
- Music cue guidance: preset exists but narration-paced lecture chooses
  natural timing — beat locks would fight reading floors (documented
  decision, same as sibling episodes)
- Audio-reactive treatment: none (narration-led lecture — documented)
- Audio-coupled moments: keyword definition cards soft pop-in; failure-id
  card landing (soft impact); stats count-up on the numbers board (subtle)
- SFX posture: sparse; ≤ 5 total; volume 0.4–0.6, quieter than the bed
- Restraint rule: nothing may outrun or mask the narration

## Storyboard

Nine scenes; durations measured from generated narration WAVs (0.35s lead-in
+ ~0.35s tail per scene; +1.2s title settle on scene 1; +2.5s end hold on
scene 9).

### Scene 1 — The Tuesday problem — narration-led (~40s)
Calendar Mon(fine)→Tue(broken); assistant answer going wrong; log wall;
Ouija-board line; title card "ForensiQ — the flight recorder + detective".
Keyword cards: "trace" (preview), "Ouija-board debugging".
Audio: bed starts low; title settle. Transition: soft crossfade.

### Scene 2 — The black box (~50s)
Flight-recorder analogy; a request unrolled into spans (embed → retrieve →
rerank → generate) with attributes; JSONL + Langfuse ingest notes;
normalisation ("hybrid/search → retrieve").
Keyword cards: trace, span, span_id/parent_id, JSONL, Langfuse.

### Scene 3 — Naming the failure: the taxonomy (~60s)
Five family cards with real ids + signatures; failure-id card lands with
soft impact; each id maps to a playbook.
Keyword cards: failure taxonomy, F-RET-001, signature, playbook.

### Scene 4 — Two-pass classification (~50s)
Rules-first flow sketch: 9 deterministic rules (evidence-quoting,
milliseconds, unit-tested) → ambiguous leftovers → LLM desk (optional,
EchoMockClient fallback). Trade-off stated honestly.
Keyword cards: two-pass classification, LLMClient Protocol.

### Scene 5 — Blame ranking with evidence (~55s)
Assembly-line stages with health gauges; z-score sketch (baseline mean, the
abnormal span 3σ out); agreement rule (label + numbers); verdict card
`F-RET-002 → retrieve 1.0`; blame top-1 100% teaser.
Keyword cards: blame ranking, z-score, healthy baseline, evidence chain.

### Scene 6 — The slow leak: chi-square drift (~60s)
Quality-decay line chart across 28 days with day-15 kink; baseline 14d vs
trailing 3d windows; p-value dial crossing α=0.01; 2-of-3 window rule;
CUSUM ramp line; honest caveat ("measures change, not badness").
Keyword cards: statistical drift, chi-square test, p-value, alpha, CUSUM.

### Scene 7 — The report a manager reads (~50s)
RCA page mock: executive summary, failure mix, mermaid-style blame graph +
timeline, quoted evidence, mapped playbook row; postmortem + ticket exports;
byte-identical determinism note.
Keyword cards: RCA, mermaid diagram, determinism.

### Scene 8 — The numbers (~45s)
Six stat tiles count up: precision 1.0, recall 1.0, blame top-1 100%,
clusters 8/8, drift α=0.01 (stable weeks silent), 128 tests (~4s offline);
replay honesty note ("top_k 5→10 does not fix an empty index").
Keyword cards: planted ground truth, precision, recall.

### Scene 9 — Recap + end card (~50s+2.5s hold)
Six recap lines (record → name → blame → watch → replay → report); end card:
"ForensiQ — the flight recorder + the detective. Don't guess. Read the
recording."

**Music mood for this video:** warm, quiet, professional (happy-beats vol 12
at 0.13 under narration)
**Audio summary:** low bed under 100% narration, three soft SFX accents,
fade with the final frame.

## Voiceover script

See `voiceover-script.txt` (scene-delimited; generated to per-scene WAVs via
`npx hyperframes tts --voice af_heart`; scene timings derive from measured
WAV durations — never hardcoded).
