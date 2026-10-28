"""Render a saved controller comparison as a chart and an evidence report."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison')
    args = parser.parse_args()
    path = Path(args.comparison)
    data = json.loads(path.read_text())
    names = list(data['summary'])
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for axis, key, title in zip(axes,
            ('avg_waiting_time', 'completed', 'avg_sampled_queue'),
            ('Completed-trip waiting time (s)', 'Completed trips', 'Sampled network queue (vehicles)')):
        summaries = [data['summary'][n][key] for n in names]
        axis.bar(range(len(names)), [s['mean'] if s else 0 for s in summaries],
                 yerr=[s['std'] if s else 0 for s in summaries], capsize=4,
                 color=['#6b7280', '#2563eb', '#60a5fa', '#d97706'][:len(names)])
        axis.set_xticks(range(len(names)), [n.replace('_', '\n') for n in names], fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Paired SUMO evaluation — mean ± SD across evaluation seeds')
    figure.savefig(path.with_suffix('.png'), dpi=160)
    lines = ['# Controller comparison', '',
             f"Scenario: `{data['config']}`. Seeds: {data['seeds']}. "
             f"Episode duration: {data['duration']} seconds.", '',
             '| Controller | Wait (s) | Travel (s) | Completed | Remaining | Pending | Queue |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        summaries = data['summary'][name]
        formatted = [f"{summaries[k]['mean']:.2f}" if summaries[k] else 'N/A'
                     for k in ('avg_waiting_time', 'avg_travel_time', 'completed',
                               'remaining', 'pending', 'avg_sampled_queue')]
        lines.append('| ' + name + ' | ' + ' | '.join(formatted) + ' |')
    lines.extend(['', '## Measured paired waiting-time changes', ''])
    for baseline, changes in data['paired_waiting_reduction_percent'].items():
        if changes:
            lines.append(f"- MAPPO vs {baseline}: **{changes['mean']:.2f}%** mean reduction. "
                         'Negative values indicate worse waiting time.')
    lines.extend(['', 'These are synthetic scenario results from one trained policy seed. '
                  'Read waiting time together with throughput and backlog. '
                  'A no-message inference ablation is not a separately retrained baseline. '
                  'This comparison does not establish green waves or Bangalore validity.', '',
                  f'![Comparison]({path.with_suffix(".png").name})'])
    path.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
