"""Paired-bootstrap ROUGE-1 F gap of every backend against the cc.hi.300 baseline on the same documents
(1000 resamples, seed 42), computed from results/perdoc. Writes results/extrinsic_gaps.jsonl."""
import json
import numpy as np

B = "cc.hi.300 (Facebook)"
safe = lambda s: s.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")
pd = lambda m, d, s: f"results/perdoc/{safe(m)}__{safe(d)}__{s}.jsonl"
r1 = lambda p: np.array([json.loads(l)["rouge-1"]["f"] for l in open(p)])
rows = [json.loads(l) for l in open("results/extrinsic.jsonl")]
with open("results/extrinsic_gaps.jsonl", "w") as out:
    for r in rows:
        if r["model"] == B:
            continue
        d, s = r["dataset"], r["split"]
        x, base = r1(pd(r["model"], d, s)), r1(pd(B, d, s))
        idx = np.random.default_rng(42).integers(0, len(x), (1000, len(x)))
        diff = x - base
        ci = np.percentile(diff[idx].mean(1), [2.5, 97.5])
        g = {"model": r["model"], "dataset": d, "split": s, "r1_gap": round(float(diff.mean()), 4),
             "r1_gap_ci95": [round(float(c), 4) for c in ci], "significant": bool(ci[0] > 0 or ci[1] < 0)}
        out.write(json.dumps(g) + "\n")
        print(f"{d:9} {s:4} {r['model']:22} R1 {r['rouge1_f']:.3f} R2 {r['rouge2_f']:.3f} RL {r['rougeL_f']:.3f}  "
              f"dR1 {g['r1_gap']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]{' *' if g['significant'] else ''}")
