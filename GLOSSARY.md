# ForensiQ Glossary — every keyword, in plain English

The study companion to the repo and the video. Every term is defined in one to
three plain sentences, then tied to why it matters in *this* project. Grouped
the way the system itself is layered: first you record, then you name, then
you blame, then you spot patterns, then you write it up.

---

## 1. Tracing — the black-box recorder

**trace**
One complete record of a single request moving through the pipeline, from the
moment it arrived to the moment an answer left. A trace is the whole story;
everything else is detail inside it.
*Why it matters here:* ForensiQ's unit of analysis is the trace — every
question ("what failed, where, and is it getting worse?") is asked over a
cohort of traces.

**span**
One step inside a trace, with a name, a duration, a status, and attributes —
"the retrieve step took 44 ms and found 5 hits". A trace is a tree of spans.
*Why it matters here:* Failure rules read span attributes (`hit_count == 0`,
`finish_reason == "length"`) — no spans, no evidence.

**span_id**
The unique identifier of one span, so steps can be referenced unambiguously.
*Why it matters here:* Evidence chains quote spans by id; the report's blame
graph is built from these connections.

**parent_id**
The span_id of the step that caused this one, turning a flat list of spans
into a tree. `parent_id: s2` on the generate span means "this answer was
produced from what s2 retrieved".
*Why it matters here:* Parent links let ForensiQ walk upstream — a refusal in
generate traces back to an empty retrieve before blaming the model.

**stage**
The role a span plays in the assembly line: `embed → retrieve → rerank →
graph → generate`, plus `guard / tool / agent`. Stage is the vocabulary blame
ranking speaks.
*Why it matters here:* "Which stage is guilty?" is only answerable if every
span carries a stage label.

**stage vocabulary / normalisation**
Different exporters name the same step differently — RAG_showcase says
`hybrid`, others say `retrieval` or `search`. Normalisation maps them all to
one canonical name on ingest (`ingest/schema.py::STAGE_ALIASES`).
*Why it matters here:* Without it, every new data source would need new
rules; with it, foreign traces ingest with zero glue code.

**Langfuse**
The industry-standard flight recorder for LLM apps: a service that collects
traces and spans from your pipeline. ForensiQ can pull traces from Langfuse's
REST API directly (`forensiq ingest --langfuse`).
*Why it matters here:* ForensiQ deliberately does not reinvent recording —
Langfuse records, ForensiQ detects. It reads Langfuse exports unchanged.

**OpenTelemetry**
The vendor-neutral standard for traces and spans (the `trace_id`/`span_id`/
`parent_id` shape itself). Langfuse and most recorders speak its vocabulary.
*Why it matters here:* ForensiQ's Trace+Span model follows the same shape, so
anything OpenTelemetry-shaped is close to ingestable already.

**JSONL**
JSON Lines: one JSON object per line of a file. It is the lingua franca for
dumping traces because you can append a line per request and stream it.
*Why it matters here:* `JSONLTraceLoader` is the simplest way in — one trace
per line, no server, no keys, fully offline.

**flight recorder analogy**
Aeroplanes carry a black box that records everything, so that after an
incident you reconstruct what happened rather than guess. Tracing is the same
idea for software requests.
*Why it matters here:* It is the design stance of the whole repo — record
first, autopsy second. Debugging without traces is Ouija-board work.

**top_k**
How many candidate documents the retriever is told to return (top 5, top 10).
*Why it matters here:* It appears in span attrs, and counterfactual replay's
first "what-if" is `top_k 5 → 10` — which the honest effect model refuses to
sell as a fix (see dilution).

**hit score**
The similarity score the retriever attached to each returned document, e.g.
`hit_scores: [0.81, 0.77, 0.61]`. High means "this chunk looks relevant".
*Why it matters here:* F-RET-002 is literally "the mean hit score fell below
0.5", and retrieval health cards are built from these numbers.

**hit_count**
How many documents the retriever actually returned for a query.
*Why it matters here:* `hit_count == 0` is the signature of F-RET-001, the
emptiest, loudest failure in the catalogue.

**tokens_in / tokens_out / cost**
How many tokens a generate span consumed going in and coming out, and what it
cost. Per-span accounting that rolls up into per-trace cost.
*Why it matters here:* The cost-spike rule (F-INFRA-002) compares a trace's
cost against the cohort's p95 × 3 — spend anomalies are failures too.

**finish_reason**
Why the model stopped generating: `"stop"` (it chose to), `"length"` (it ran
out of context window), or an error. Emitted by every OpenAI-compatible API.
*Why it matters here:* `finish_reason == "length"` plus a near-ceiling prompt
is the exact signature of prompt truncation (F-PROMPT-001).

**RAG (retrieval-augmented generation)**
The pattern where a model answers strictly from documents a retriever fetched
first, instead of from memory. The pipeline being autopsied here:
embed → retrieve → rerank → graph → generate.
*Why it matters here:* ForensiQ is forensics for exactly this shape of
system; the stage vocabulary *is* the RAG assembly line.

---

## 2. Failure taxonomy — naming the failure

**failure taxonomy**
A named catalogue of failure types with signatures — the difference between
"the answer was bad" and "this is F-RET-001, empty retrieval set, and here is
the span that proves it".
*Why it matters here:* The taxonomy (`docs/TAXONOMY.md`) is the backbone:
every failure ForensiQ emits carries one of its ids.

**failure family (F-RET / F-PROMPT / F-GEN / F-TOOL / F-INFRA)**
The five prefixes grouping failures by where they live: retrieval (F-RET),
the prompt (F-PROMPT), generation (F-GEN), tool calls (F-TOOL), and
infrastructure (F-INFRA). Family maps to stage, stage maps to owner.
*Why it matters here:* The family prefix tells you whose desk the incident
lands on before you read another word.

**F-RET-001 — empty retrieval set**
The retriever returned nothing: `hit_count == 0`. The model then answers from
memory, which is how confident nonsense happens.
*Why it matters here:* Highest-precedence retrieval failure — an empty
context explains a lot of downstream weirdness, so it is checked before
generation symptoms.

**F-RET-002 — low-recall context**
Retrieval returned documents, but weak ones: mean hit score under the
threshold (default 0.5). The model was fed wet cardboard.
*Why it matters here:* The most common silent killer of RAG quality — the
pipeline "works", the answers are just slowly wrong.

**F-RET-003 — retrieval score collapse (drift)**
Hit scores decay across days rather than in one trace — a cohort-level
pattern, caught by the drift detector, not by per-trace rules.
*Why it matters here:* The "slow leak" scenario: nothing crashes, quality
just bleeds out over weeks.

**F-PROMPT-001 — prompt truncation**
The prompt hit the context ceiling and was silently cut: `finish_reason ==
"length"` with `prompt_chars ≥ 0.9 × context_limit`.
*Why it matters here:* Truncation is invisible in the answer — the model
never says "by the way, half my instructions were cut off".

**F-GEN-001 — structured output format break**
The output was supposed to be JSON but doesn't parse. Everything downstream
that expected structure now chokes.
*Why it matters here:* A common agent-pipeline failure that is cheap to
detect (`json.loads` fails) and cheap to fix (schema enforcement).

**F-GEN-002 — repetition loop**
The model got stuck repeating itself — the same 3/5/8-gram appears five or
more times. Tokens burn, nothing is said.
*Why it matters here:* The canonical decoder failure, and a cost problem as
much as a quality one.

**F-GEN-003 — ungrounded refusal**
The model said "I can't answer" even though the retrieved context was usable.
A false refusal.
*Why it matters here:* The refusal list deliberately mirrors RAG_showcase's
abention phrasings so that *correct* abstentions are not misclassified as
failures.

**F-GEN-004 — ungrounded claim**
The answer asserts things the context doesn't support — flagged by the
groundedness heuristic rather than a pass-1 rule.
*Why it matters here:* Fabrication is the failure users actually notice; the
health card catches it when the classifier's rules can't.

**F-TOOL-001 — tool execution error**
A tool call (API, database, calculator) failed: the tool span's status is
`error`.
*Why it matters here:* Agent pipelines fail in their tools more often than
in their models; the rule is one line because the signature is clean.

**F-INFRA-001 — stage timeout**
A step exceeded its time budget: span status `timeout`. Checked first, before
anything else.
*Why it matters here:* Hard infrastructure signals outrank clever inference —
a timeout explains everything downstream of it.

**F-INFRA-002 — cost spike**
One trace cost more than the cohort's 95th-percentile cost times three.
Cohort-relative on purpose: absolute thresholds rot when you change models.
*Why it matters here:* Cost anomalies are usually symptoms of retry loops or
runaway prompts — both fixable once named.

**F-INFRA-003 — guard abort**
A guard/guardrail blocked the request, or an error span matches no other
rule — the catch-all. The only id where the LLM pass (pass 2) is the primary
detector.
*Why it matters here:* Every taxonomy needs an "anything else" row, or novel
failures get silently dropped instead of surfaced.

**precedence rules**
When several rules could fire on one trace, they are evaluated in a fixed
order and the first match wins: upstream causes before downstream symptoms
(an empty-context refusal is a *retrieval* problem, so F-RET-001 outranks the
refusal signature F-GEN-003).
*Why it matters here:* One primary failure per trace, chosen deterministically
— so two runs agree and reports stay diffable.

**two-pass classification**
The classifier's design: pass 1 is deterministic rules that quote evidence;
pass 2 sends only the ambiguous leftovers (error spans, guard blocks) to an
LLM behind a protocol.
*Why it matters here:* Rules are fast, testable, and always available — the
LLM is optional, so classification keeps working during the very outage that
caused the failures.

**LLMClient Protocol / EchoMockClient**
`LLMClient` is the interface pass-2 classifiers talk through;
`EchoMockClient` is the offline stand-in that keeps the pipeline testable
with no key and no network.
*Why it matters here:* It is why the whole 128-test suite runs offline in
seconds, and why an LLM outage degrades coverage instead of breaking
forensics.

**planted ground truth**
A test corpus where the failures were inserted deliberately, each labelled
with the failure it is supposed to be. Scoring against planted truth measures
the classifier, not the corpus.
*Why it matters here:* ForensiQ's headline numbers — precision/recall 1.0,
blame top-1 100% — mean "against failures we planted on purpose", which is
the honest way to claim detection works.

**offline synthetic corpus**
A generated dataset (`ingest/synthetic.py`) that simulates 28 days of
pipeline traffic with 8 planted failure modes and a day-15 drift event — no
network, no keys, no real user data.
*Why it matters here:* It makes the demo (`forensiq demo`) and the tests
reproducible by anyone, anywhere, forever.

**seeded RNG**
A random number generator started from a fixed seed, producing the same
"random" sequence every run. The synthetic corpus is seeded.
*Why it matters here:* Same seed → same 560 traces → byte-identical reports
and stable test scores. Randomness you can replay is a fixture, not noise.

**precision**
Of the failures the classifier flagged, how many were real: `true positives /
all flagged`. High precision means it doesn't cry wolf.
*Why it matters here:* 1.0 precision on the planted corpus means every alarm
was a real planted failure — no false alarms to train you to ignore.

**recall**
Of the real failures present, how many the classifier caught: `true
positives / all actual`. High recall means nothing slipped past.
*Why it matters here:* 1.0 recall on planted failures means the rulebook
found every single one — the forensic net has no known holes at these modes.

**F1**
The harmonic mean of precision and recall — one number that only looks good
when *both* are good. A classifier can game precision or recall alone; F1
makes gaming hard.
*Why it matters here:* It is the fair scoreboard for pass-1 rules: any
threshold tweak that trades one for the other shows up immediately.

---

## 3. Attribution — pointing at the guilty stage

**blame ranking**
For each failure, rank the pipeline stages by how likely each is the root
cause — with evidence quoted from the trace, not vibes. Output: e.g.
`F-RET-002 → {"retrieve": 1.0}`.
*Why it matters here:* Naming the failure says *what* broke; blame ranking
says *where* — the question a manager actually asks first.

**z-score**
How many standard deviations a measurement sits from its baseline mean:
`(x − μ) / σ`. A z-score of −3 means "three sigma worse than normal".
*Why it matters here:* Blame ranking computes per-stage directional z-scores
of span attributes against the healthy baseline — the numeric anomaly has to
agree with the failure label before a stage is accused.

**healthy baseline**
The average behaviour of *healthy* traces only, per stage — failed traces are
excluded so the yardstick stays clean.
*Why it matters here:* If sick traces contaminated the baseline, "abnormal"
would quietly mean "average", and blame would point nowhere.

**anomaly**
A measurement far enough from baseline that luck is an unlikely explanation —
operationally, a large directional z-score on a stage attribute.
*Why it matters here:* Anomaly is the raw material of blame; the z-score is
what turns "the retrieve span looked weird" into a number.

**stage health cards**
Per-stage scorecards with stage-specific metrics: retrieval gets hit rate,
k-coverage and mean score; generation gets groundedness; rerank gets its
delta. Not one overall "quality" number.
*Why it matters here:* Retrieval sickness and generation sickness have
different owners and different fixes — a single blended number optimizes for
dashboards and against diagnosis.

**groundedness**
A heuristic score of how much of the answer is actually supported by the
retrieved context. ForensiQ's version double-penalizes fabricated numbers and
capitalised entities ("anchored tokens") and has a negation guard.
*Why it matters here:* It powers F-GEN-004 and the generation health card —
the lexical, offline way to smell fabrication.

**counterfactual replay**
Re-run the failure through a simulated pipeline with one knob changed —
"would top_k=10 have fixed it?" — and report the predicted effect honestly.
*Why it matters here:* It validates a candidate fix before you page anyone;
`forensiq replay --file traces.jsonl --taxonomy F-PROMPT-001`.

**what-if analysis**
The general act of asking "what would have happened if X were different?" —
counterfactual replay is its disciplined, recorded form.
*Why it matters here:* It turns the postmortem's last line from "we should
raise top_k" into "we tested raising top_k; it dilutes the mean score and
does not fix this".

**dilution (replay effect model)**
The honest finding that raising top_k adds marginal hits *below* the current
mean, pulling the average down — more context is not automatically better
context. And an empty index stays empty no matter the top_k.
*Why it matters here:* It is the difference between a simulation that sells
fixes and one that respects the physics of ranking; the pessimism is
documented, not hidden.

**PipelineAdapter protocol**
The interface replay uses to execute the counterfactual pipeline. The shipped
`SimulatedAdapter` is an estimate; wire a real adapter and ForensiQ records
predicted-vs-actual for calibration.
*Why it matters here:* The protocol is what keeps the honesty structural —
the tool admits what it is simulating instead of laundering a guess into a
measurement.

---

## 4. Patterns & drift — the slow leak

**statistical drift**
A gradual shift in the pipeline's failure mix over days or weeks — not a
crash, a leak. Measured by comparing recent windows against a baseline.
*Why it matters here:* The day-15 retrieval degradation in the demo corpus is
the canonical case: nothing errors, quality just bleeds.

**chi-square test**
A statistical test for count data: "do today's failure counts differ from the
baseline mix more than random sampling noise allows?" Chi-square goodness-of-
fit gives a calibrated p-value for exactly that question.
*Why it matters here:* It is the drift detector's engine — chosen over KL
divergence because sparse counts (≤8 classes, tens of failures) need a real
null distribution, which chi-square provides with a small-count pooling
guard.

**p-value**
The probability of seeing a difference at least this big if nothing had
actually changed. Small p-value → "probably a real change, not luck".
*Why it matters here:* Alerts fire when p < α, so the false-alarm rate is a
chosen number (1%), not a guessed threshold.

**significance level (alpha)**
The p-value threshold you accept before calling a change real. ForensiQ uses
α = 0.01 — a 1% tolerance for false alarms.
*Why it matters here:* `FORENSIQ_ALPHA=0.01` is the difference between an
alarm you trust and one you mute; the stable window staying quiet at α=0.01
is a tested property, not a hope.

**baseline window / trailing window**
Drift compares a recent trailing window (default 3 days) against a longer
baseline (default 14 days). Baseline is "what normal looked like"; trailing
is "what this week looks like".
*Why it matters here:* Both are config (`FORENSIQ_BASELINE_DAYS`,
`FORENSIQ_WINDOW_DAYS`) — the comparison, not the calendar, decides.

**small-count pooling**
Chi-square expects non-tiny expected counts; failure classes with too few
expected occurrences are pooled into an "other" bucket before testing.
*Why it matters here:* It keeps the p-value honest on sparse data — without
it, rare classes would manufacture significance.

**2-of-3 trailing rule**
An alert requires 2 of the last 3 overlapping windows to be significant.
Under the null, such a run has probability ~3×10⁻⁴ per position.
*Why it matters here:* One noisy window cannot reset a sustained signal, and
one noisy window cannot fire a false alarm either — both directions are
protected.

**CUSUM**
Cumulative sum: a sequential statistic that accumulates small deviations and
alarms when their running total crosses a boundary. It notices slow ramps
that windowed tests are sluggish on.
*Why it matters here:* Applied per failure class, CUSUM catches the "score
decays 2% a week" pattern the chi-square window is too blunt to feel early.

**TF-IDF**
Term frequency × inverse document frequency: weight a word by how often it
appears in one document and how rare it is across all documents. A robust,
old-school way to turn texts into vectors.
*Why it matters here:* Hand-rolled in `patterns/cluster.py` and applied to
failure evidence strings — no sklearn needed to group similar failures.

**clustering**
Grouping items so similar ones land together, without labels. Here: group
failure traces by their evidence text so repeated incidents form families.
*Why it matters here:* The demo's clustering recovers all 8 planted failure
families perfectly — proof the grouping tracks real structure.

**agglomerative clustering**
Bottom-up clustering: start with every item alone, repeatedly merge the two
closest clusters, stop at a threshold. Average linkage = cluster distance is
the average pairwise distance.
*Why it matters here:* It is the specific algorithm used, chosen because it
is simple, deterministic, and needs no cluster count up front.

**k-means**
Top-up clustering: pick k centres, assign points to the nearest, recompute
centres, repeat. Used as the fallback shape check.
*Why it matters here:* `patterns/cluster.py` uses it to validate the
agglomerative grouping when the family count is known — two cheap views of
the same structure.

**cluster purity**
How homogeneous each cluster is: if nearly all members share one true label,
purity is near 1. The score that says "these groups mean something".
*Why it matters here:* The test asserts intra-cluster distance < inter-
cluster distance on planted families — purity is the metric behind "recovers
all 8 families perfectly".

---

## 5. Reporting — the artifact a manager reads

**RCA (root cause analysis)**
A structured incident document: what failed, which stage caused it, the
evidence, the fix. ForensiQ generates one automatically from traces.
*Why it matters here:* `forensiq report --out rca.md` is the product — the
morning-after artifact that turns 560 traces into a readable story.

**postmortem**
The written account of an incident produced after the fact: timeline, cause,
impact, actions. A blameless postmortem is the cultural norm — fix systems,
not people.
*Why it matters here:* ForensiQ exports a postmortem document alongside the
RCA — the taxonomy + blame output is shaped to slot into one.

**playbook**
The pre-agreed first response for a failure type: F-RET-001 → "check index
freshness and embed-model consistency…". Stored per id, overridable via
`docs/playbooks.yaml`.
*Why it matters here:* Every report row ends in a mapped fix, so the report
doesn't just diagnose — it hands you the next action.

**mermaid diagram**
Markdown-native diagrams-as-text (flowcharts, gantt-style timelines) that
render on GitHub and in most markdown viewers.
*Why it matters here:* The RCA embeds a mermaid blame graph and incident
timeline — diagrams that live in the repo and diff like code.

**evidence chain**
The quoted span attributes that justify a classification or blame decision —
`hit_count: 0`, `finish_reason: "length"`, the repeated n-gram.
*Why it matters here:* Every claim in the report is quotable; the evidence is
the difference between an accusation and a verdict.

**blame graph**
The mermaid chart in the RCA showing which stages absorbed the blame and how
strongly, per failure id.
*Why it matters here:* A manager reads the graph in ten seconds and knows
where the meeting starts.

**executive summary**
The report's first section: the failure mix, the top root cause, the drift
status — written for someone who will read nothing else.
*Why it matters here:* It is the senior-engineer-to-manager translation
layer, generated deterministically, not hand-written at 2 a.m.

**ticket export**
Machine-readable exports of failures (with verbatim evidence) for filing
issues — `failures.json`, ticket-ready rows.
*Why it matters here:* It closes the loop from detection to action: the
failure becomes a work item without copy-paste archaeology.

**cohort**
The set of traces analysed together — one report covers one cohort.
Cohort-level views include the failure mix, cost p95 and blame distribution.
*Why it matters here:* Some rules (F-INFRA-002) and all drift statistics only
make sense over a cohort; single traces have no "normal" to compare against.

**determinism / byte-identical reports**
Reports contain no wall-clock data — windows come from trace timestamps,
tables are sorted, evidence picks are first-in-trace-order. Two runs over the
same corpus produce byte-identical output.
*Why it matters here:* Byte-identical reports make report diffing a legitimate
review tool: if the file changed, the pipeline changed.

**trace sampling (production note)**
Capture 1–10% of healthy traces but 100% of errored or negatively-flagged
ones. ForensiQ's statistics need hundreds of failures, not millions of
traces.
*Why it matters here:* It is how the tool stays affordable in production —
drift and blame run fine on the sampled failure-complete diet.
