# Interview Prep — Multi-Agent Traffic RL Project
*Read top-to-bottom in ~1 hour. Everything below is grounded in the repo; nothing here overclaims.*

## The one rule for tomorrow
Speak in **"pipeline-works" framing**: the pipeline is real and fully tested; MAPPO / 35% / real-data validation are the **Phase-5 roadmap**, never "results". If pushed → commit-to-roadmap (Section 5).

**5 numbers to memorize:** `18.94 s` avg wait · `47.35 s` avg travel · `848` vehicles throughput · `18-dim` observation · `223 tests`.

---

# 0. Word-for-word script: Problem → Design → Workflow
*Say this as flowing speech. Natural pace ≈ 4 minutes. Bracketed tags are your mental beats — don't say them.*

## [THE PROBLEM] — "What problem does this solve?"

"The core problem I'm tackling is **traffic signal coordination across a network** — not just one intersection.

Today, most real intersections run **fixed-time plans**: one cycle length, one set of green times, chosen in advance, applied all day. That's optimal for exactly one traffic pattern — and wrong the moment traffic shifts. A plan tuned for morning rush is wasteful at 2 PM and gridlocking at 8 PM.

Worse, each intersection is controlled **in isolation**. But traffic isn't isolated — cars that leave one intersection arrive at the next. If my intersection dumps a queue straight into your red light, locally 'greedy' signals create spillover, and the network jams even though every intersection is 'doing its best'. In other words, **the locally optimal decision is not the network optimum** — that gap is exactly what coordination is supposed to close.

The catch is that coordination design is **combinatorial**: 9 intersections × multiple phases × continuous timing choices — you can't hand-tune that. And you can't experiment freely on real streets, because mistakes mean real congestion and real safety risk.

So my solution: **learn coordination in a simulator**, where we can run thousands of cheap, safe experiments — and treat the learned policy as a decision layer that could later be calibrated against real detector data."

## [THE DESIGN] — "How is it designed?"

"I designed it as **two clean layers**.

**Layer 1 — the simulation layer.** I build a **3×3 grid city in SUMO, the standard open-source traffic simulator**: 9 signalized intersections, 48 directed edges, generated entirely from a YAML config through a script chain — netconvert compiles nodes, edges, connections, and traffic-light programs into the network file, and a demand generator produces vehicle flows. I generate **two demand patterns**: boundary-entry flows, and OD-based network-wide flows so all 9 intersections actually carry traffic.

**Layer 2 — the learning layer.** Around that simulation I built a **PettingZoo multi-agent environment** — the standard API for multi-agent RL — with **one agent per intersection, 9 agents acting simultaneously**.

Three deliberate design decisions I'd highlight:

**One — decentralized observations.** Each agent sees only **its own intersection**: an 18-dimensional vector — current phase, time-in-phase, and per incoming lane: queue length, waiting vehicles, vehicle count, mean speed. Why? Because a deployable controller shouldn't need to see the whole world — and a local-only observation creates exactly the coordination gap that makes this problem interesting.

**Two — safety lives in the environment, not in the policy.** The agent only chooses **KEEP or SWITCH**. The environment enforces minimum green time — 10 seconds — and always completes yellow and all-red phases. So the agent can never produce an unsafe transition, no matter what it learns.

**Three — the reward is improvement, per agent, local:** the change in total waiting time plus the change in queue length since the last step. Dense every 10 seconds, directly aligned with the KPI we care about — average waiting time.

On the algorithm side, I started with **Independent PPO** — nine separate actor-critic networks, no parameter sharing, no centralized critic. That's not a shortcut; it's my **control condition**. Phase 5 of my design is **MAPPO — centralized training, decentralized execution** — a shared critic that sees the full network during training, while execution stays fully local. Having the independent baseline means MAPPO's coordination benefit gets **measured, not assumed**."

## [THE WORKFLOW] — "Walk me through how it runs."

"The pipeline runs end-to-end like this:

**Step 1 — Build.** A config file drives network construction: fields specify the 3×3 grid, signal timing — a 90-second cycle of 41-second greens plus yellow and all-red — and a demand level. Build scripts emit the SUMO network and the route-demand files.

**Step 2 — Simulate.** The PettingZoo environment launches SUMO and connects through its API, TraCI. One environment step = 10 simulated seconds.

**Step 3 — Act.** Every step, each of the 9 agents reads its 18-dim local observation and its policy — an actor-critic network — picks KEEP or SWITCH. The environment applies it safely: min-green checked, yellow/all-red completed, phases wrap correctly.

**Step 4 — Reward & learn.** Each agent gets its improvement reward — Δ waiting plus Δ queue. Transitions go into a per-agent rollout buffer. Every **128 steps**, the PPO update runs: advantages computed with **GAE** — gamma 0.99, lambda 0.95 — then four epochs of clipped-surrogate updates with an entropy bonus and a KL safeguard that stops the update if the policy moves too fast.

**Step 5 — Evaluate.** The agent checkpoints are evaluated deterministically per-vehicle metrics from SUMO's tripinfo log, then compared against my **fixed-time baseline**: **18.94 seconds average waiting, 47.35 seconds average travel time, 848 of 856 vehicles completed — 99.1 percent**. That's the number everything must beat.

**Step 6 — Guard.** The whole system sits under a **223-test suite** — network topology, demand generation, the environment, PPO math, and regression tests for every bug I've fixed.

**Where it stands honestly:** the entire pipeline — network builder, multi-agent env, baseline, independent-PPO trainer and evaluator — is implemented and verified end-to-end, including smoke-run validation. Full-scale training runs and the MAPPO upgrade are the next phase — the design is ready and gated on those runs."

# 1. Talk track

## 60-second version (6 beats)
1. **Problem:** Fixed-time traffic signals waste capacity — one timing for all conditions. I wanted to learn if RL can coordinate a network of intersections better than fixed timing.
2. **What I built:** A simulated 3×3 grid city in **SUMO** — 9 signalized intersections, 48 directed edges — controlled by **9 independent RL agents**, one per intersection.
3. **Environment:** A **PettingZoo ParallelEnv** — every step, each agent gets a local 18-dim observation (queue lengths, waiting times, vehicle counts, speeds, current phase) and picks KEEP or SWITCH for its signal.
4. **Algorithm:** **Independent PPO** — 9 separate actor-critic networks, no parameter sharing, trained with the clipped PPO objective and GAE.
5. **Proof it's solid:** **223 passing tests**, a fixed-time baseline benchmarked at **18.94 s** average waiting / **848** vehicles per hour completed, and an end-to-end training pipeline that smoke-tests clean.
6. **Next:** The MAPPO upgrade — centralized training, decentralized execution — so agents can coordinate like a real "green wave". That's Phase 5.

## 3-minute version (adds "why", per-beat)
1–4. *Same as 60-second, plus:*
- **Why simulation first:** real intersections are dangerous/slow to experiment on. SUMO is the standard open-source traffic simulator; TraCI gives per-step control.
- **Why 9 separate agents, not one big brain:** independent PPO is the **control condition**. Before building MAPPO (centralized critic), I needed a baseline: *does even fully-decentralized control work, and how much is lost without coordination?* That measures MAPPO's value.
- **Why discrete KEEP/SWITCH:** raw phase selection with safety constraints is complex; keep/switch maps cleanly onto min-green/yellow/all-red physics that SUMO enforces anyway.
5. **Benchmark story:** I built a **fixed-time baseline** first (90 s cycle, 41 s green NS/EW). Measured: **18.94 s** avg wait, **47.35 s** avg travel, **848/856** vehicles completed (99.1%). That's the number RL must beat.
6. **Status honesty:** Training runs are **in progress** — the pipeline is verified end-to-end (smoke test showed non-zero rewards, finite losses, entropy starting to move). The MAPPO/CTDE phase — and the coordination gains I expect from it — is the *designed next step*, gated on the first real 10k-timestep run.
7. **Engineering depth (offer proactively):** Along the way I fixed **6 real bugs** — phase-count mismatches, a netconvert misconfiguration, a tripinfo parse-order bug — each caught by test and locked in with **23 regression tests**. Mention least one: *"my favorite: the evaluation script read tripinfo.xml after closing the environment — which deleted the file. Classic ordering bug, now regression-tested."*

## 10-minute version (adds war story + math + roadmap)
Everything in 3-minute, plus:
- **Network construction pipeline:** YAML config → netconvert builds nodes/edges/connections + auto-generates TLS programs → Python post-processing sets phase durations → route/demand generator (.rou.xml) → run.
- **The bug-fix war story (pick 2 of 6):**
  1. *TLS phase mismatch:* env assumed 6 phases; SUMO networks actually had 4 or 1. Fix = `detect_tls_phases()` reading real phases from SUMO at runtime — hardcoding assumptions vs. reading ground truth.
  2. *Demand coverage:* boundary-only demand left 5/9 intersections with zero traffic — agents there had nothing to learn. Added OD-based `network_wide` demand.
- **PPO in one breath:** collect 128 steps per agent → compute advantages with GAE (γ=0.99, λ=0.95) → 4 epochs of clipped-surrogate updates (ε=0.2) with entropy bonus 0.01 and target-KL early stop at 0.015.
- **Roadmap (this is where the resume headline lives):** *"The headline result I'm building toward is MAPPO with a centralized critic — the expected win is coordinated 'green wave' behavior and a large waiting-time reduction over independent agents. The independent-PPO baseline is done and tested; the first full training run gates that comparison."*

---

# 2. Concept cheat sheet (flashcard style)

| Concept | One-liner | Why it matters here |
|---|---|---|
| **RL loop** | Agent sees state → acts → gets reward → environment transitions. | One SUMO step = one env step; 9 agents act simultaneously (ParallelEnv). |
| **State vs observation** | State = full environment truth; observation = what the agent can see. | Each agent sees only **its** intersection: 18 floats, strictly local. |
| **Policy π(a\|s)** | The agent's decision function; we optimize its expected return. | Actor network outputs P(SWITCH) vs P(KEEP). |
| **Value function V(s)** | Predicted future discounted reward from this state. | Critic head; used to compute advantages. |
| **Discount γ** | How much future rewards count vs now. | 0.99 — episode is ~360 steps, so distant rewards still matter. |
| **PPO (clipped)** | Policy-gradient method; clips the new/old probability ratio to [1−ε, 1+ε] so one update can't destroy the policy. | ε=0.2 — stable training on a fragile simulator. |
| **GAE(λ)** | Advantage estimate blending TD error and Monte-Carlo returns; λ trades bias vs variance. | λ=0.95 standard. |
| **Entropy bonus** | Reward for keeping the policy random → exploration. | 0.01; smoke test showed it moving off ln(2)=0.693 — the first learning signal. |
| **Target KL** | Stop a PPO update early if the policy moved too far. | 0.015 — trust-region safety valve. |
| **IPPO** | Independent PPO: N agents, N separate networks, no info sharing. | This repo's Phase-4 baseline / control condition. |
| **MAPPO / CTDE** | Centralized Training, Decentralized Execution: training uses a critic that sees global state; execution stays local. | Phase-5 target: fixes credit assignment across agents. |
| **Non-stationarity** | From one agent's view, the world changes because *other agents* learn too. | Why independent agents can plateau; MAPPO's main motivation. |
| **Reward shaping** | r = α·Δwaiting + β·Δqueue — improvement-based, local only. | α=β=1.0; credit assignment across agents is the hard part. |
| **SUMO/TraCI** | Traffic simulator + Python control API. netconvert builds networks; tripinfo.xml logs per-vehicle stats. | step_length = 10 s to make RL rollouts tractable; tripinfo is the metrics source of truth. |
| **PettingZoo** | Standard API for multi-agent RL envs (like Gymnasium, but multi-agent). | Lets me swap agents/algos cleanly. |

**Numbers card:** 18.94 s · 47.35 s · 848/856 (99.1%) · avg queue 4.51 · 90 s cycle = 41+3+1+41+3+1 · 18-dim obs · Discrete(2) · 9 agents · 5571 params/agent · 223 tests · min-green 10 s · step 10 s.

---

# 3. Architecture (whiteboard-ready)

## 3.1 Data & training pipeline
```
configs/*.yaml
     │
     ▼
scripts/build_network.py
     ├──▶ sumo_network/grid_builder.py      (nodes, edges, connections, TLS via netconvert)
     └──▶ sumo_network/traffic_demand.py    (.rou.xml: boundary flows | OD-based network-wide)
              │
     data/synthetic/{network,demand}/   (net.xml, rou.xml)
              │
     ▼
envs/multi_intersection_env.py   ← PettingZoo ParallelEnv (9 agents)
     ├── envs/single_intersection.py   (TraCI glue: obs extraction, action, reward)
     └── envs/{observation,action,reward}.py   (18-dim obs, KEEP/SWITCH, Δwait+Δqueue)
              │
     ▼
agents/independent_ppo/{network,buffer,ppo,trainer}.py   (9 × ActorCritic 18→64→64)
              │
     ▼
scripts/train_independent_ppo.py ──▶ results/ (checkpoints, training_logs.json, CSV)
scripts/evaluate_independent_ppo.py ──▶ evaluation/metrics.py (tripinfo.xml)
                                        └──▶ vs fixed-time baseline (scripts/run_fixed_time.py)
```

## 3.2 One environment step (narrate this)
1. `env.step(actions)` — SUMO advances 10 s.
2. Per agent: TraCI extracts 4 incoming lanes → queue, waiting, count, speed + phase & time-in-phase → **18-dim obs**.
3. Each ActorCritic policy samples KEEP/SWITCH (randomly during training, argmax at eval).
4. If SWITCH & min-green (10 s) satisfied → advance to next phase; yellow/all-red always complete.
5. Reward `r_i = (W_prev − W_now) + (Q_prev − Q_now)` computed **per agent, local only**.
6. Transition goes into each agent's rollout buffer; every **128 steps** → GAE → PPO update (4 epochs, minibatch 64, KL guard).

## 3.3 MAPPO/CTDE upgrade slide-in (Phase 5)
- **Unchanged:** actors, env, obs, actions, execution path.
- **New:** one **centralized critic** (or shared critic) consuming global state — concatenated local obs + neighbor phases.
- **Changes in code:** `trainer.py` buffer plumbing to route global state to the critic; PHASE3_SPEC.md's neighbor-phase observation is the natural *"message passing v2"*.
- **Framing line:** *"Independent PPO exists precisely to measure how much MAPPO's centralized critic buys."*

## 3.4 Real-world / MNC scale-up (when asked "how does this get used?")
```
real detector loops / SCADA feed ─▶ calibration (match real flows/timings to sim)
      └─▶ offline RL training in calibrated sim
             └─▶ shadow mode (advisory only, live monitoring)
                    └─▶ % rollout (control 10% → 50% → 100% of cycle plans)
                           └─▶ dashboards: wait/travel/queue KPIs, drift alarms, retraining cadence
```
- **Why a MNC cares:** traffic wait time = fuel, emissions, commuter hours. A proven sim-to-decision pipeline is reusable across cities — same methodology, new detector data + network topology; adjacent wins in routing, demand forecasting, logistics.
- **Honest limits to state:** sim-to-real gap, safety constraints (never unsafe transitions), and that today's repo is the **simulation & algorithm** layer of that diagram.

---

# 4. Q&A bank (model answers — say them in your own words)

**Q: Why RL at all — why not just better fixed timing or adaptive heuristics?**
> Fixed-time plans are optimal for one traffic pattern and degrade for others. Adaptive heuristics (loop-detector actuation) improve a single intersection but don't coordinate a network — the Nash equilibrium of locally-greedy decisions isn't network-optimal. A 3×3 grid's joint phase policy space is combinatorial; RL learns a policy mapping traffic states to coordination without hand-tuning every scenario. That said, I'm honest about the cost: sample inefficiency — which is why I train in simulation.

**Q: Why 9 agents instead of one controller?**
> One centralized controller doesn't scale — a city grid has hundreds of intersections. The point of the project is the *coordination* question, so I model the realistic case: local control with local observations. 9 is small enough to debug, big enough to exhibit inter-intersection conflicts.

**Q: Why independent PPO first (if asked "why no MAPPO yet")? — [PIVOT ANSWER]**
> Deliberate sequencing. MAPPO is Phase 5 of my plan; Phase 4 built the IPPO control condition so the centralized critic's contribution is *measured*, not assumed. The environment and whole pipeline are done and regression-tested; the first full training run gates the MAPPO upgrade.

**Q: Where's the 35% number from? — [PIVOT ANSWER]**
> That's the headline **target** for the MAPPO-vs-IPPO comparison I'm running. What's measured today: the fixed-time baseline at 18.94 s wait / 47.35 s travel / 848 throughput, and a verified training pipeline. I don't quote experiment numbers before the experiment runs.

**Q: What about real-world / Bangalore-scale validation? — [PIVOT ANSWER]**
> The current demand is synthetic (boundary and OD-based network-wide generators). Calibrating against real detector data is the next milestone in the scale-up path — it's the standard order: algorithm works in sim, then calibrate, then shadow-mode.

**Q: Walk me through one training step.** *(Use §3.2 trace verbatim.)*

**Q: What's in the observation? Why local only?**
> 18 floats: current phase, time-in-phase, then ×4 lanes each: queue, waiting vehicles, count, mean speed. Local-only is the point — it makes agents cheap to deploy (no inter-intersection comms needed) and creates the coordination gap that MAPPO will close.

**Q: Why KEEP/SWITCH and not direct phase selection?**
> Safety: min-green (10 s) and forced yellow/all-red completion live in the env logic, so the agent can never produce an unsafe transition. Switch wraps modulo the program's phase count — and yes, intersection N_3_3 has a 1-phase always-green program, so its agent correctly can't act.

**Q: Why this reward?**
> Improvement-based: r = α·Δwaiting + β·Δqueue, α=β=1.0. Dense (every 10 s step gives signal), bounded, aligned with the KPI (avg wait). Alternative — raw waiting sums — is sparse and dominated by incoming traffic, not the agent's decisions.

**Q: Your hardest bug?**
> The tripinfo parse-order one — I closed the SUMO environment, which deletes tripinfo.xml, *before* reading the metrics from it. Fix: parse first, close after. Locked with regression tests. Second favorite: the env hardcoded 6 phases while the network actually had 4 or 1 — now I read phases from SUMO at runtime instead of trusting my model of the world.

**Q: How do you know the system isn't broken when training runs?**
> The smoke test checklist: non-zero observations, rewards that vary ([-11, +11] observed), non-zero advantages, finite policy/value losses, entropy departing ln(2). Plus 223 tests across topology, demand, env, PPO, and regressions.

**Q: What would independent agents do badly at? (expect this — it's your MAPPO motivation)**
> Non-stationarity and credit assignment. Each agent's world includes 8 learning agents, so from its view the MDP is shifting. Coordination failures like gridlock or "green wave" mismatch are *joint* problems the local reward can't assign blame for. That's exactly what a centralized critic fixes.

**Q: How would you scale to a real city?** *(Use §3.4 diagram. Calibrate → shadow → partial rollout → full; monitoring + retraining; state sim layer is what's built.)*

**Q: What's the business value?**
> Reduced waiting = fuel, emissions, and commuter time saved. More importantly, the *pipeline* is city-agnostic: same architecture, swap in real detector data and topology. That's a repeatable methodology, not a one-off demo — and the techniques transfer to routing, demand forecasting, and logistics optimization.

**Q: What would you do differently?**
> Run real-network-wide demand from day one (boundary demand left 5/9 intersections starving, which hid problems); detect TLS phases from SUMO from the start rather than hardcoding; and timebox the fixed-time baseline's polish — it was important, but I over-invested there before having a learning signal.

---

# 5. Emergency card — if cornered

Say exactly this, calmly, then redirect to your strongest ground:
> "To be precise about status: the complete pipeline — SUMO network build, PettingZoo multi-agent env, fixed-time baseline, independent-PPO training and evaluation, 223 tests — is implemented and verified. MAPPO with the centralized critic, RLlib integration, and real-data calibration are the designed Phase-5 roadmap. The 35% reduction is the target of that upcoming MAPPO-vs-IPPO comparison, not a measured result yet. What I can speak to deeply is what's built and what I learned building it."

Then pivot to: the 3-minute walkthrough beat #7 (bug-fix war story) or §3.4 scale-up — whoever asked, answer on terrain you own.

# 6. Reworded resume bullets (for later use — resume stays as-is for tomorrow)
- Built a SUMO 3×3-grid simulation with a PettingZoo multi-agent environment (9 RL agents), 18-dim local observations, and safe KEEP/SWITCH signal actions with min-green constraints.
- Implemented independent-PPO (clipped objective + GAE) across 9 actor-critic agents and a fixed-time benchmark (18.94 s avg wait, 848 veh/h); full pipeline covered by a 223-test suite.
- Designed the CTDE upgrade path — MAPPO with centralized critic and neighbor-phase message passing — gated on the first 10k-timestep training run.
