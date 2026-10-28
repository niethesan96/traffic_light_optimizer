# Spec: Interview Prep Package — Multi-Agent Traffic RL Project (interview-prep-spec.md)

## 1. What this request is

The user has a **interview tomorrow (project walkthrough, 45–60 minutes, Software Engineer role)** about the
**Multi-Agent Traffic Network Optimization** project (this repo). They need to:

1. Explain the project end-to-end verbally (no screen share, no live demo).
2. Answer follow-up questions confidently, including tough ones.
3. Possibly be asked to **design/describe the pipeline or architectural workflow on the spot**.

The user gets **~1 hour of prep time**, so the deliverable must be **scannable: bullets over prose**, prioritized
so the most important material comes first.

**No code changes are to be made.** This spec only describes prep documents the user will read.

---

## 2. Critical context discovered during the interview (interview answers)

| Question | Answer |
|---|---|
| Interview type | **Project walkthrough** — they explain the project, interviewer asks follow-ups |
| Interview length | **45–60 minutes** |
| User's RL comfort level | **"Lean on scaffolding"** — mostly used the pipeline built by others/AI; needs to learn to explain it |
| Role interviewed for | **Software Engineer** (project is a side project, not the core job) |
| Live demo possible? | **No** — purely verbal explanation |
| Resume accuracy | **35% claim is aspirational/planned**, not backed by a real experiment yet |
| Real data | **None** — the "validated against real Bangalore intersections" line is also aspirational |
| If pressed on MAPPO / 35% claim | **Commit-to-roadmap**: say "Phase 5 target, here's my plan to run it" |
| Resume | **Keep as-is** for tomorrow; include **reworded defensible bullets** for later use |
| Framing of results | **"Pipeline-works" framing** — full working pipeline + 223 tests, learning proof pending |
| Topics user most wants covered | **"Why RL at all"**, **real-world scale-up**, **business value for a MNC company** |
| Post-interview plan | **Not wanted** — tomorrow only |
| Prep format priority | **Timed narrative** (60 s / 3 min / 10 min talk tracks) + **Concept cheat sheet** first; other deliverables still wanted |

---

## 3. Ground truth vs. resume claims (must be reflected honestly in every deliverable)

### 3.1 REAL — what is actually built and verified in this repo

| Fact | Source |
|---|---|
| SUMO 3×3 signalized grid: 16 nodes, 48 directed edges, 9 TLS intersections (`N_1_1`–`N_3_3`) | PROJECT_STATUS.md Phase 1 |
| Fixed-time baseline: avg waiting **18.94 s**, avg travel time **47.35 s**, throughput **848/856 vehicles** (99.1 %), avg queue 4.51, max queue 9 | PROJECT_STATUS.md Phase 2 |
| Signal plan: 90 s cycle = 41 + 3 + 1 + 41 + 3 + 1 (NS/EW green, yellow, all-red) | PROJECT_STATUS.md |
| PettingZoo `ParallelEnv` with **9 independent agents** (`agent_1_1`–`agent_3_3`) | Phase 3 |
| Observation: **18-dim** local vector (phase, time-in-phase, queue ×4, waiting ×4, count ×4, speed ×4). Strictly local, zero-padded | Phase 3 |
| Action: **Discrete(2)** — KEEP / SWITCH, min green 10 s, yellow/all-red always complete; wraps | Phase 3 |
| Reward: `r_i = α·(W_i^{t−1} − W_i^t) + β·(Q_i^{t−1} − Q_i^t)`, α=β=1.0 default | Phase 3 |
| Episode: truncation at max_steps, step length 10 s, default 3600 s episode | Phase 3 |
| **Independent PPO** (Phase 4): 9 separate ActorCritic MLPs, 18→64→64→actor+critic, ~5,571 params/agent, **no parameter sharing, no centralized critic, no cross-agent info** | Phase 4 |
| PPO details: clipped surrogate ε=0.2, GAE γ=0.99 λ=0.95, entropy coeff 0.01, value loss coeff 0.5, grad clip 0.5, 4 epochs, minibatch 64, rollout 128, target KL 0.015, lr 3e-4 | configs/default.yaml |
| **223 tests passing** (28 grid builder, 14 signal plans, 21 demand, 50 fixed-time, 54 multi-agent env, 33 PPO, 23 bugfix regressions) | PROJECT_STATUS.md |
| Smoke test evidence: non-zero observations, varying rewards [−11, +11], finite losses, non-zero advantages, entropy moved from ln(2)=0.6931 → 0.6928 | CHANGELOG correction pass |
| 6 major bugs found & fixed during correction pass (TLS phase mismatch, netconvert TLS file, signal_state key, network regen, tripinfo parse timing, demand coverage) | CHANGELOG |
| `demand_type: boundary` (default) vs `network_wide` (OD-based, full 9-intersection coverage) | configs |

### 3.2 ASPIRATIONAL — on the resume but NOT in the repo (must not be presented as done)

| Resume claim | Repo reality |
|---|---|
| **MAPPO / CTDE (centralized training, decentralized execution)** | Phase 5, **not started** — `agents/mappo/` is an empty stub; gate says do NOT start until real training run proves learning |
| **RAY RLlib** | Only a **commented-out** line in `requirements.txt` (`# ray[rllib]>=2.5.0`); never used |
| **Message passing** between agents | Nothing implemented — only a `get_adjacency_map()` stub comment "for message passing later" in `grid_builder.py`; PHASE3_SPEC.md says CTDE was the *design intent* |
| **"Green wave" emergent coordination** | Only a linear **offset** computation exists in `signal_plans.py` (fixed-time baseline), not an emergent RL behavior |
| **35 % waiting-time reduction vs independent agents** | No completed training run; no comparative evaluation. Non-blocking smoke tests only. Explicitly flagged in NEXT_SESSION.md as "NOT a claim that PPO learns" |
| **Validated against real Bangalore traffic data** | No real dataset anywhere; demand is synthetic (`traffic_demand.py` generators) |
| 223 tests | ✅ real (the one resume line fully backed) |

### 3.3 Honesty strategy chosen by user

- Tomorrow: **keep the resume as-is** and speak within the "pipeline-works" framing.
- If the interviewer presses on MAPPO / 35 % / real data → **commit-to-roadmap script**:
  - "MAPPO with a centralized critic is the Phase-5 design; the repo today has the independent-PPO baseline implemented and fully tested, and I'm about to run the 10k-timestep experiment that gates the MAPPO upgrade."
  - Never state a % improvement as a measured result; state it as the *target* of the planned comparison.
- The user also wants **reworded defensible resume bullets** (for their own use later — kept resume as-is for tomorrow).

---

## 4. Deliverables (a prep package, in priority order for a ~1-hour read)

All in **markdown**, with ASCII diagrams. Suggested home: single file
`interview-prep.md` at repo root **or** split into `prep/01-talk-track.md`, `prep/02-cheatsheet.md`, `prep/03-architecture.md`, `prep/04-qa-bank.md` — user's choice; single scannable file is preferred given 1 hour.

### 4.1 (Priority 1) Timed talk track

Three versions of the same narrative:

- **60-second elevator** — problem (traffic signal timing is fixed and suboptimal), approach (model each of 9 intersections in a SUMO 3×3 grid as an RL agent), current state (full pipeline + fixed-time baseline + independent PPO trained/validated, 223 tests), forward (MAPPO/CTDE as next step).
- **3-minute walkthrough** — add: PettingZoo env abstractions, observation/action/reward definitions, why independent PPO first (baseline before MAPPO), evaluation story (fixed-time 18.94 s/47.35 s/848 throughput as reference), and the resume headline as **roadmap**, not result.
- **10-minute deep-dive** — add: netconvert pipeline (nodes→edges→connections→TLS programs→route demand generation), the 6-bug correction pass as a **debugging war story** (great for SWE signal), the PPO math at a high level, and honest status on training runs.

Narrative must be **memorizable**: numbered beats, one-line-per-beat, no jargon the user hasn't had prepped in §4.2/§4.4.

### 4.2 (Priority 2) Concept cheat sheet — must be ~1 page, flashcard style

Must cover, with one-sentence definitions and one "why it matters here" line each:

- RL fundamentals: agent/environment, state/observation/action/reward, policy, value function, discount factor γ.
- **PPO**: policy-gradient method, clipped surrogate objective (show formula), why clipping stabilizes training.
- **GAE (λ, γ)**: bias-variance trade-off, why 0.95/0.99 are common defaults.
- Entropy bonus & target KL: exploration / trust-region safety valves.
- **IPPO vs MAPPO vs CTDE**: define all three; explain *why the repo uses IPPO as the Phase-4 baseline* (control condition) and *how MAPPO would add a centralized value head using the global state*.
- CTDE discipline: why it is a *design intent* in this repo (per PHASE3_SPEC.md) separate from what's merged today.
- Reward shaping for traffic: the Δwaiting + Δqueue formulation, why per-agent local reward, why it can plateau; credit assignment is hard.
- SUMO/TraCI basics: netconvert, .net.xml/.rou.xml, tripinfo.xml, step_length, why 10 s step (speeds up RL rollouts).
- Key numbers to memorize: 18.94 s / 47.35 s / 848 throughput / 99.1 % completion / 223 tests / 18-dim obs / Discrete(2) / 90 s cycle / 9 agents.

### 4.3 Architecture / pipeline walkthrough (whiteboard-ready)

Must be **ASCII diagrams the user can redraw**, aligned with repo layout:

```
configs/*.yaml ──▶ scripts/build_network.py ──▶ sumo_network/grid_builder.py (nodes, edges, connections, TLS)
                                        └──▶ sumo_network/traffic_demand.py (.rou.xml flows)
data/synthetic/{network,demand}/
        │
        ▼
envs/multi_intersection_env.py  (PettingZoo ParallelEnv)
   ├── envs/single_intersection.py  (per-agent observation/action/reward gluing TraCI)
   ├── envs/observation.py (18-dim), envs/action.py (KEEP/SWITCH), envs/reward.py
        │
        ▼
agents/independent_ppo/{network,buffer,ppo,trainer}.py   (9 × ActorCritic, PPO updates)
        │
        ▼
scripts/train_independent_ppo.py / evaluate_independent_ppo.py ──▶ results/ + evaluation/metrics.py
                                                   └──▶ fixed-time baseline comparison (Phase 2)
```

Also include:

1. **A step-by-step "one environment step" trace** the user can narrate: SUMO step → per-agent observation extraction → policy forward pass → action → TLS state change → reward computation → buffer append → periodic GAE/PPO update.
2. **A target MAPPO/CTDE slide-in sketch**: shared centralized critic consuming concatenated local obs + phase offsets, decentralized actors unchanged, what changes in `trainer.py` buffer plumbing (per PHASE3_SPEC.md, neighbor-phase observation is the natural message-passing v2).
3. A **"how to redesign this for a MNC/production deployment"** section: real detector data ingestion (loop detectors / SCADA), offline calibration, safe staged rollout (shadow mode → percentage control), monitoring dashboards, retraining cadence, and evaluation gates.

### 4.4 (Priority 3) Q&A bank — with pencil, repo-sourced model answers

Must include, at minimum:

- **■ High-risk "caught" questions** (need rehearsed honest pivots):
  - "Where is MAPPO? You said CTDE." → pivot answer given explicitly.
  - "Where's the 35 % measured?" → "that's the headline target of the upcoming MAPPO-vs-IPPO comparison; the independent-PPO stage is complete and benchmarked against fixed-time baseline."
  - "Show me the Bangalore data." → "synthetic OD-based demand is integrated; real-data ingestion is the next milestone."
- **■ What the user explicitly requested:**
  - *Why RL at all?* (vs. simpler rule-based/adaptive traffic control: combinatorial multi-agent coordination, non-stationary partial observability, learned rather than hand-tuned policies, but honest about the cost — sample inefficiency)
  - *Real-world scale-up* (from 3×3 synthetic grid to a real city corridor: data acquisition, sim-to-real gap map, safety constraints, SPSA/actuated baseline)
  - *Why is this good for a MNC?* (smart-city signal timing, energy/idle reduction, commuter KPIs, portable methodology across cities, rides as internal infra for traffic-to-RL transfer) — frame as: repurpose to adjacent problems: demand forecasting, routing, EV charging, logistics.
- Standard project walkthrough Qs: architecture of the env, why 9 agents not 1, why Discrete(2), why 18-dim obs, why min-green, reward formula rationale, step_length choice, handling corner intersection N_3_3 (single-phase always green), evaluation procedure, test suite story (what 223 tests cover), the 6-bug correction pass told as debugging narrative, what would you do differently.
- Forward-looking Qs: what happens if independent agents conflict (gridlock), what a centralized critic buys you, credit assignment in multi-agent RL, non-stationarity.

Every answer must be **short (3–6 sentences), grounded in the concrete numbers in §3.1**, and never introduce a number that isn't in this spec.

### 4.5 Reworded resume bullets (for later use, not tomorrow)

Produce 3–4 defensible reformulations of the current bullets, e.g.:

- "Built a SUMO 3×3 grid simulation with a PettingZoo multi-agent environment (9 RL agents), configured fixed-time and boundary/network-wide demand baselines."
- "Implemented independent-PPO training (IPPO, clipped objective + GAE) with a 223-test suite; smoke-tested the end-to-end pipeline end-to-end."
- "Designed the CTDE upgrade path (MAPPO with centralized critic, neighbor-phase message passing) currently slated for Phase 5."
- Explicit note: these preserve intellectual honesty while staying strong; the user decides whether to swap in later (they kept the resume as-is for tomorrow).

---

## 5. Non-goals (explicitly, per user's answers)

- No fraud/overstatement: never coach the user to present aspirational items as measured results.
- No code changes, no new experiments, no MAPPO implementation tonight.
- No post-interview "make the resume true" roadmap (user said tomorrow only).
- No delivery/deployment tooling changes.

## 6. Acceptance criteria (definition of "prep package is ready")

1. **Can be read end-to-end in ≤ 1 hour** (word counts: talk tracks ≤ 1,500 words total; cheat sheet ≤ 1 page; architecture ≤ 1.5 pages; Q&A ≤ 15 questions).
2. Every quantitative claim inside the package matches PROJECT_STATUS.md / configs/default.yaml exactly — no invented numbers.
3. Every aspirational item (MAPPO/Ray/message-passing/35 %/Bangalore data) is either (a) clearly framed as roadmap/Phase-5, or (b) absent — never stated as delivered.
4. User can, without opening the repo aloud:
   - State 5 real numbers (18.94 s, 47.35 s, 848 throughput, 18-dim, 223 tests).
   - Draw the ASCII architecture diagram from memory.
   - Deliver the 60-second talk track.
   - Answer "why RL at all?" in under a minute without any overclaim.
   - Pivot honestly if MAPPO or 35 % is questioned.
5. Nothing added past the 3 deliverables + resume bullets (no log-in doc, no marketing).
6. User does not need to run code to use the package (no demo available).

## 7. Open questions / deferred

- Single file vs. 4-file split (§4 intro) → default to single `interview-prep.md` unless user says otherwise.
- Exact company/industry of the interview → not asked; spec assumes generic SWE; MNC-value framing kept generic so it applies broadly.
- Whether the user wants a spoken-language version (interview likely in English) → assumed English; translate only if user requests.

*Spec generated after a 4-round interview with the user; repo facts sourced from PROJECT_STATUS.md, CHANGELOG.md, NEXT_SESSION.md, PHASE3_SPEC.md, report1.md, configs/*.yaml, and code searches confirming MAPPO/RLlib/data absence.*
