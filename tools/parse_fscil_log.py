import argparse
import json
import math
import re


ACC_RE = re.compile(
    r'\[(\d+)\]Evaluation results : acc : ([\d.]+|nan) ; acc_base : ([\d.]+|nan) ; acc_inc : ([\d.]+|nan)')
OLD_NEW_RE = re.compile(
    r'\[(\d+)\]Evaluation results : acc_incremental_old : ([\d.]+|nan) ; acc_incremental_new : ([\d.]+|nan)')


def _to_float(val):
    if val == 'nan':
        return float('nan')
    return float(val)


def _summarize(stats):
    overall = [s['acc'] for s in stats]
    old = [s['acc_old'] for s in stats]
    novel = [s['acc_new'] for s in stats]
    forgetting = []
    for idx, value in enumerate(old):
        if idx == 0:
            forgetting.append(0.0)
        else:
            forgetting.append(max(old[:idx]) - value)
    def _avg(values, start=0):
        vals = values[start:]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))
    return dict(
        overall_avg=_avg(overall, 0),
        old_avg=_avg(old, 1),
        novel_avg=_avg(novel, 1),
        forgetting_avg=_avg(forgetting, 1),
        forgetting=forgetting,
    )


def parse_log(path):
    session_stats = {}
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            acc_match = ACC_RE.search(line)
            if acc_match:
                sess = int(acc_match.group(1))
                session_stats.setdefault(sess, {})
                session_stats[sess].update(dict(
                    session=sess,
                    acc=_to_float(acc_match.group(2)),
                    acc_base=_to_float(acc_match.group(3)),
                    acc_inc=_to_float(acc_match.group(4)),
                ))
            old_match = OLD_NEW_RE.search(line)
            if old_match:
                sess = int(old_match.group(1))
                session_stats.setdefault(sess, {})
                session_stats[sess].update(dict(
                    acc_old=_to_float(old_match.group(2)),
                    acc_new=_to_float(old_match.group(3)),
                ))
    stats = []
    for sess in sorted(session_stats.keys()):
        item = session_stats[sess]
        acc = item.get('acc', 0.0)
        acc_base = item.get('acc_base', 0.0)
        acc_inc = item.get('acc_inc', 0.0)
        acc_old = item.get('acc_old', float('nan'))
        acc_new = item.get('acc_new', float('nan'))
        if math.isnan(acc_old):
            acc_old = acc_base if sess > 1 else acc
        if math.isnan(acc_new):
            acc_new = acc_inc if sess > 1 else acc
        def _sanitize(val):
            if isinstance(val, float) and math.isnan(val):
                return None
            return val
        stats.append(dict(
            session=sess,
            acc=_sanitize(acc),
            acc_base=_sanitize(acc_base),
            acc_inc=_sanitize(acc_inc),
            acc_old=_sanitize(acc_old),
            acc_new=_sanitize(acc_new),
        ))
    summary = _summarize(stats)
    return dict(sessions=stats, summary=summary)


def main():
    parser = argparse.ArgumentParser(description='Parse FSCIL log and summarize metrics')
    parser.add_argument('log', help='path to log file')
    parser.add_argument('--out', default=None, help='path to json output')
    args = parser.parse_args()
    metrics = parse_log(args.log)
    output = json.dumps(metrics, indent=2, ensure_ascii=True)
    print(output)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            handle.write(output + '\n')


if __name__ == '__main__':
    main()
