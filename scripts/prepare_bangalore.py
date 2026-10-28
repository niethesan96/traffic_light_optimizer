"""Prepare an auditable count-driven benchmark, without inventing city geometry."""
from pathlib import Path
import csv
import hashlib
import json
import yaml


def main():
    source = Path('data/bangalore/bilekahalli_2013_counts.csv')
    rows = list(csv.DictReader(source.open()))
    counts = [int(r['vehicles_24h']) for r in rows]
    if len(counts) != 6 or any(v < 0 for v in counts):
        raise ValueError('Expected six nonnegative daily movement counts')
    config = yaml.safe_load(Path('configs/cooperative.yaml').read_text())
    config['paths']['demand_dir'] = 'data/bangalore/demand'
    config['demand'].update(flow_rate=sum(counts)/24, demand_level='custom',
                          corridor_weights=[v/2 for v in counts for _ in range(2)])
    config['data_provenance'] = dict(source_csv=str(source), table=4,
        doi='10.17577/IJERTV2IS70850', sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        observation_date='2013-03-06', transformation='daily counts / 24; projected movement proportions',
        validation_scope='count-driven synthetic stress test; no measured delay validation')
    Path('configs/bangalore_counts.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    Path('data/bangalore/manifest.json').write_text(json.dumps(dict(
        config['data_provenance'], daily_total=sum(counts),
        average_hourly_network_target=sum(counts)/24),indent=2))
    print(json.dumps(dict(daily_total=sum(counts), average_hourly_network_target=sum(counts)/24)))


if __name__ == '__main__':
    main()
