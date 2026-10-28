"""Paired fixed-time, independent PPO and MAPPO evaluation on held-out seeds."""
from pathlib import Path
import argparse
import json
import hashlib
import csv
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from agents.mappo.trainer import MAPPOTrainer
from agents.independent_ppo.trainer import IndependentPPOTrainer
from envs.multi_intersection_env import MultiIntersectionEnv


def evaluate(config, controller, seed, duration, disable_messages=False, trace_path=None):
    env = MultiIntersectionEnv(config, seed=seed, demand_seed=seed, episode_duration=duration)
    queues = []
    trace = []
    try:
        obs, _ = env.reset(seed=seed)
        if trace_path:
            direction_links={}
            for agent,tls in env.get_agent_intersection_map().items():
                ew,ns=[],[]
                for k,connections in enumerate(env._traci.trafficlight.getControlledLinks(tls)):
                    if not connections:
                        continue
                    origin,destination=connections[0][0].split('_to_')
                    (ew if int(origin.split('_')[1])==int(destination.split('_')[0]) else ns).append(k)
                direction_links[agent]=(tls,ew,ns)
            def observe_signals(traci):
                now=traci.simulation.getTime()
                for agent,(tls,ew,ns) in direction_links.items():
                    signal=traci.trafficlight.getRedYellowGreenState(tls)
                    trace.append(dict(time=now,agent=agent,phase=traci.trafficlight.getPhase(tls),
                        ew_green=int(any(signal[k] in 'Gg' for k in ew)),
                        ns_green=int(any(signal[k] in 'Gg' for k in ns))))
            env.simulation_observer=observe_signals
        if controller is not None:
            if controller.agent_names != env.possible_agents or controller.obs_dim != env.get_observation_dim():
                raise ValueError('Checkpoint observation dimensions/agent order do not match scenario')
        while True:
            if controller is None:
                actions = {}  # leave SUMO's fixed program in control
            elif isinstance(controller, MAPPOTrainer):
                if disable_messages:
                    with torch.no_grad():
                        actions = {name: controller.model.actor_logits(
                            torch.as_tensor(obs[name]),
                            torch.zeros(4, controller.model.obs_dim), i).argmax(-1).item()
                            for i, name in enumerate(controller.agent_names)}
                else:
                    actions = controller.actions(obs, deterministic=True)
            elif hasattr(controller, 'actions'):
                actions = controller.actions(obs, deterministic=True, disable_messages=disable_messages)
            else:
                with torch.no_grad():
                    actions = {n: controller.networks[n].get_action_probs(
                        torch.as_tensor(obs[n]).unsqueeze(0)).argmax(-1).item()
                        for n in controller.agent_names}
            obs, _, terminated, truncated, infos = env.step(actions)
            queues.append(sum(info['local_queue_length'] for info in infos.values()))
            if any(terminated.values()) or any(truncated.values()):
                break
        remaining = env._traci.vehicle.getIDCount()
        simulated_seconds = env._traci.simulation.getTime()
        pending = env._traci.simulation.getPendingVehicles()
        # SUMO flushes XML on connection close. Preserve it until parsing ends.
        env._traci.close()
        env._traci = None
        trips = ET.parse(env.get_tripinfo_path()).getroot().findall('tripinfo')
        completed = [t for t in trips if float(t.get('arrival', '-1')) >= 0]
        waits = [float(t.get('waitingTime', '0')) for t in completed]
        travels = [float(t.get('duration', '0')) for t in completed]
        stops = [float(t.get('waitingCount', '0')) for t in completed]
        if trace_path:
            path = Path(trace_path)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('w',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=list(trace[0]))
                writer.writeheader()
                writer.writerows(trace)
        return dict(seed=seed, simulated_seconds=simulated_seconds,
                    completed=len(completed), remaining=remaining,
                    pending=len(pending),
                    avg_waiting_time=float(np.mean(waits)) if waits else None,
                    total_waiting_time=float(sum(waits)),
                    avg_travel_time=float(np.mean(travels)) if travels else None,
                    avg_stops=float(np.mean(stops)) if stops else None,
                    nonstop_fraction=float(np.mean(np.asarray(stops)==0)) if stops else None,
                    avg_sampled_queue=float(np.mean(queues)))
    finally:
        env.close()


def summarize(episodes):
    result = {}
    for key in ('avg_waiting_time', 'avg_travel_time', 'completed', 'remaining',
                'pending', 'avg_sampled_queue', 'avg_stops', 'nonstop_fraction'):
        values = [e[key] for e in episodes if e.get(key) is not None]
        result[key] = dict(mean=float(np.mean(values)), std=float(np.std(values))) if values else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/coordination.yaml')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mappo')
    group.add_argument('--rllib', help='Exported RLModule checkpoint directory')
    parser.add_argument('--independent')
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument('--episode-duration', type=int, default=3600)
    parser.add_argument('--output', default='results/comparison.json')
    parser.add_argument('--reuse-fixed-time',
                        help='Reuse fixed-time episodes with matching scenario, seeds and duration')
    parser.add_argument('--trace-dir',help='Optional per-controller signal traces for progression analysis')
    args = parser.parse_args()
    if args.episode_duration <= 10:
        parser.error('episode duration must exceed one decision interval')
    torch.set_num_threads(1)
    if args.rllib:
        from agents.mappo.rllib_integration import RLlibController
        mappo = RLlibController(args.rllib)
    else:
        mappo = MAPPOTrainer.load(args.mappo)
    controllers = {'fixed_time': None, 'mappo': mappo, 'mappo_no_messages': mappo}
    if args.independent:
        controllers['independent_ppo'] = args.independent
    result = dict(config=args.config, seeds=args.seeds, duration=args.episode_duration,
                  checkpoint_paths=dict(mappo=args.mappo, rllib=args.rllib, independent=args.independent),
                  episodes={}, summary={})
    result['scenario'] = yaml.safe_load(Path(args.config).read_text())
    result['mappo_joint_training_decisions'] = mappo.global_step
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, controller in controllers.items():
        if name == 'independent_ppo':
            controller = IndependentPPOTrainer.load_checkpoint(controller)
        if name == 'fixed_time' and args.reuse_fixed_time:
            saved = json.loads(Path(args.reuse_fixed_time).read_text())
            if (saved.get('scenario') != result['scenario'] or saved.get('seeds') != args.seeds
                    or saved.get('duration') != args.episode_duration):
                raise ValueError('Cached fixed-time evaluation does not match this experiment')
            episodes = saved['episodes']['fixed_time']
            result['reused_fixed_time'] = args.reuse_fixed_time
        else:
            episodes = [evaluate(args.config, controller, seed, args.episode_duration,
                disable_messages=name == 'mappo_no_messages',
                trace_path=(Path(args.trace_dir)/f'{name}_seed{seed}.csv') if args.trace_dir else None)
                for seed in args.seeds]
        result['episodes'][name] = episodes
        result['summary'][name] = summarize(episodes)
        output.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(name, json.dumps(result['summary'][name]), flush=True)
    files = [Path(args.config)]
    if args.mappo:
        files.append(Path(args.mappo))
    else:
        files.extend(p for p in Path(args.rllib).rglob('*') if p.is_file())
    if args.independent:
        files.extend(sorted(Path(args.independent).glob('*.pt')))
        files.append(Path(args.independent) / 'metadata.json')
    result['file_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result['paired_waiting_reduction_percent'] = {}
    for baseline in ('fixed_time', 'independent_ppo'):
        if baseline not in result['episodes']:
            continue
        changes = [(b['avg_waiting_time']-m['avg_waiting_time']) / b['avg_waiting_time'] * 100
                   for b, m in zip(result['episodes'][baseline], result['episodes']['mappo'])
                   if b['avg_waiting_time'] is not None and b['avg_waiting_time'] > 0
                   and m['avg_waiting_time'] is not None]
        change_summary = dict(mean=float(np.mean(changes)), per_seed=changes) if changes else None
        if changes and len(changes) > 1:
            samples = np.random.default_rng(0).choice(
                changes, size=(10000, len(changes)), replace=True).mean(axis=1)
            change_summary['ci95_across_evaluation_seeds'] = np.quantile(
                samples, [0.025, 0.975]).tolist()
            change_summary['interval_method'] = 'percentile bootstrap over evaluation seeds'
        result['paired_waiting_reduction_percent'][baseline] = change_summary
    output.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
