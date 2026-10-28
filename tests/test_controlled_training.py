from pathlib import Path
import numpy as np
import pytest
import torch
from agents.mappo.model import ResidualMAPPOModel
from agents.independent_ppo.network import ActorCritic
from agents.mappo.trainer import MAPPOTrainer
from envs.reward import RewardConfig, compute_reward

NAMES = [f'agent_{r}_{c}' for r in range(1,4) for c in range(1,4)]


def test_queue_reward_penalizes_persistent_waiting():
    config = RewardConfig(mode='queue_cost',queue_weight=1,reward_scale=.1)
    assert compute_reward(100,100,4,4,config) == pytest.approx(-.4)
    assert compute_reward(100,100,0,0,config) == 0


def test_residual_warm_start_preserves_all_independent_actions_and_values(tmp_path):
    torch.set_num_threads(1)
    trainer=MAPPOTrainer(NAMES,50,model_version=2)
    observations=torch.rand(4,9,50)
    with torch.no_grad():
        expected_logits=torch.stack([n(observations[:,i])[0]
            for i,n in enumerate(trainer.model.local_experts)],1)
        expected_values=torch.stack([n.get_value(observations[:,i])
            for i,n in enumerate(trainer.model.local_experts)],1)
        torch.testing.assert_close(trainer.model.logits(observations),expected_logits)
        torch.testing.assert_close(trainer.model.values(observations),expected_values)
    trainer.save(tmp_path/'model.pt')
    restored=MAPPOTrainer.load(tmp_path/'model.pt')
    torch.testing.assert_close(restored.model.logits(observations),expected_logits)


def test_declared_multilane_space_matches_observation():
    from envs.multi_intersection_env import MultiIntersectionEnv
    env=MultiIntersectionEnv('configs/coordination.yaml')
    assert all(env.observation_space(n).shape==(50,) for n in NAMES)
    env.close()


def test_rllib_actor_does_not_read_joint_state(tmp_path):
    pytest.importorskip('ray.rllib')
    from gymnasium import spaces
    from ray.rllib.core.columns import Columns
    from agents.mappo.rllib_integration import TrafficPPOModule, pack_observations, RLlibController
    obs={n:np.random.default_rng(i).random(50,dtype=np.float32) for i,n in enumerate(NAMES)}
    packed=pack_observations(obs,NAMES)
    tensor=torch.as_tensor(np.stack([packed[n] for n in NAMES]))
    module=TrafficPPOModule(observation_space=spaces.Box(-np.inf,np.inf,(709,),np.float32),
        action_space=spaces.Discrete(2),model_config={'agent_names':NAMES,'local_dim':50})
    original=module.forward_inference({Columns.OBS:tensor})[Columns.ACTION_DIST_INPUTS]
    tensor[:,259:]+=10
    following=module.forward_inference({Columns.OBS:tensor})[Columns.ACTION_DIST_INPUTS]
    torch.testing.assert_close(original,following)
    assert module.compute_values({Columns.OBS:tensor}).shape==(9,)
    module.save_to_path(str(tmp_path/'module'))
    controller=RLlibController(tmp_path/'module')
    messages=np.stack([np.zeros(50),obs[NAMES[1]],obs[NAMES[3]],np.zeros(50)])
    assert controller.local_action(obs[NAMES[0]],messages,0)==original[0].argmax().item()


def test_bangalore_projection_preserves_measured_daily_total(tmp_path):
    import csv
    import yaml
    import xml.etree.ElementTree as ET
    from sumo_network.grid_builder import GridBuilder, GridConfig
    from sumo_network.traffic_demand import DemandConfig, TrafficDemandGenerator
    rows=list(csv.DictReader(open('data/bangalore/bilekahalli_2013_counts.csv')))
    assert sum(int(r['vehicles_24h']) for r in rows)==95805
    config=DemandConfig.from_yaml('configs/bangalore_counts.yaml')
    path=TrafficDemandGenerator(config).generate_network_wide_with_grid_builder(
        GridBuilder(GridConfig(boundary_ring=True)),tmp_path/'demand.xml')
    rates=[float(f.get('vehsPerHour')) for f in ET.parse(path).getroot().findall('flow')]
    assert len(rates)==12
    assert sum(rates)==pytest.approx(95805/24,abs=1e-5)


def test_platoons_preserve_demand_volume_and_sort_departures(tmp_path):
    import xml.etree.ElementTree as ET
    from sumo_network.grid_builder import GridBuilder, GridConfig
    from sumo_network.traffic_demand import DemandConfig, TrafficDemandGenerator
    config=DemandConfig(duration=3600,flow_rate=3600,demand_level='custom',
        corridor_weights=[1.]*12,platoon_period=120,platoon_on_time=40)
    path=TrafficDemandGenerator(config).generate_network_wide_with_grid_builder(
        GridBuilder(GridConfig(boundary_ring=True)),tmp_path/'platoons.xml')
    flows=ET.parse(path).getroot().findall('flow')
    assert len(flows)>360
    begins=[float(f.get('begin')) for f in flows]
    assert begins==sorted(begins)
    total=sum((float(f.get('end'))-float(f.get('begin')))*float(f.get('vehsPerHour'))/3600 for f in flows)
    assert total==pytest.approx(3600,abs=.05)


def test_signal_trace_records_physics_steps(tmp_path):
    from scripts.compare_controllers import evaluate
    import csv
    pytest.importorskip('libsumo')
    path=tmp_path/'trace.csv'
    metrics=evaluate('configs/cooperative.yaml',None,7001,30,trace_path=path)
    rows=list(csv.DictReader(path.open()))
    assert metrics['simulated_seconds']==30
    assert len(rows)==20*9
    assert sorted({float(r['time']) for r in rows})==list(range(11,31))


def test_measured_profile_has_no_random_count_perturbation(tmp_path):
    from dataclasses import replace
    from sumo_network.grid_builder import GridBuilder, GridConfig
    from sumo_network.traffic_demand import DemandConfig, TrafficDemandGenerator
    config=DemandConfig.from_yaml('configs/bangalore_short_links.yaml')
    grid=GridBuilder(GridConfig(boundary_ring=True))
    a=TrafficDemandGenerator(config).generate_network_wide_with_grid_builder(grid,tmp_path/'a.xml')
    b=TrafficDemandGenerator(replace(config,seed=config.seed+1)).generate_network_wide_with_grid_builder(grid,tmp_path/'b.xml')
    assert a.read_bytes()==b.read_bytes()
