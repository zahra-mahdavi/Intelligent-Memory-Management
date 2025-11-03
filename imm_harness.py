
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, random, time, json, sys
from collections import deque, defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd

class TierCache:
    def __init__(self, name, capacity):
        self.name, self.capacity = name, capacity
        self.deque, self.set = deque(), set()
        self.evictions = 0
    def __contains__(self, page): return page in self.set
    def add(self, page):
        if page in self.set:
            self.deque.remove(page); self.deque.append(page); return None
        evicted=None
        if len(self.deque)>=self.capacity and self.capacity>0:
            evicted=self.deque.popleft(); self.set.remove(evicted); self.evictions += 1
        if self.capacity>0:
            self.deque.append(page); self.set.add(page)
        return evicted
    def touch(self,page):
        if page in self.set:
            self.deque.remove(page); self.deque.append(page)
    def remove(self,page):
        if page in self.set:
            self.deque.remove(page); self.set.remove(page)
    def occupancy(self): return len(self.deque)

class BasePolicy:
    def __init__(self, tiers_cfg: List[Dict[str,Any]]):
        self.tiers=[TierCache(t["name"],t["capacity"]) for t in tiers_cfg]
    def place_on_miss(self,page): self.tiers[-1].add(page)
    def on_hit(self,page,idx):
        if idx>0: self.tiers[idx].remove(page); self.tiers[idx-1].add(page)
    def prefetch(self,_): return []

class PredictivePolicy(BasePolicy):
    def __init__(self, tiers_cfg: List[Dict[str,Any]], n=2, prefetch_budget=2):
        super().__init__(tiers_cfg)
        self.n=n; self.prefetch_budget=prefetch_budget
        self.context=deque(maxlen=n); self.counts=defaultdict(Counter); self.last_retrain_ms=0.0
    def observe(self,page):
        start=time.time()
        if len(self.context)==self.n:
            key=tuple(self.context); self.counts[key][page]+=1
        self.context.append(page); self.last_retrain_ms=(time.time()-start)*1000.0
    def predict_next(self):
        if len(self.context)<self.n: return []
        key=tuple(self.context); ctr=self.counts.get(key)
        if not ctr: return []
        return [p for p,_ in ctr.most_common(self.prefetch_budget)]
    def prefetch(self,_page):
        preds=self.predict_next(); placed=[]
        for p in preds:
            if any(p in tc for tc in self.tiers): continue
            for idx in range(len(self.tiers)-2,-1,-1):
                self.tiers[idx].add(p); placed.append(p); break
        return placed

def generate_dynamic_trace(num_pages,trace_length,hotset_size,stickiness,drift_pages,protected_ratio,seed=42):
    rng=random.Random(seed)
    protected=set(rng.sample(range(num_pages),max(0,int(protected_ratio*num_pages))))
    pool=[p for p in range(num_pages) if p not in protected]
    hotset=deque(rng.sample(pool, min(hotset_size, len(pool))), maxlen=hotset_size)
    trace=[]
    for t in range(trace_length):
        if t>0 and stickiness>0 and (t % stickiness == 0):
            candidates=[p for p in pool if p not in hotset]
            replace=list(hotset)[:min(drift_pages, len(hotset))]
            for r in replace:
                try: hotset.remove(r)
                except ValueError: pass
            if candidates:
                k=min(len(replace), len(candidates))
                hotset.extend(rng.sample(candidates, k))
        page = rng.choice(list(hotset)) if (rng.random()<0.85 and hotset) else rng.randrange(num_pages)
        trace.append((page, page in protected))
    return trace, protected

def run_sim(policy, tiers_cfg, trace, security_check_ns=15, check_invariants=True):
    s={"steps":0,"fast_hits":0,"med_hits":0,"slow_hits":0,"misses":0,
       "latency_ns_sum":0.0,"energy_sum":0.0,
       "prefetch_requests":0,"prefetch_correct":0,"retrain_time_ms_sum":0.0,"security_checks":0}
    last_pref=set()
    for page,prot in trace:
        s["steps"]+=1
        if prot:
            s["security_checks"]+=1
            s["latency_ns_sum"]+=security_check_ns; s["energy_sum"]+=security_check_ns*0.01
        hit=False; idx_hit=None
        for idx,(tc,cfg) in enumerate(zip(policy.tiers,tiers_cfg)):
            if page in tc: hit=True; idx_hit=idx; tc.touch(page); policy.on_hit(page,idx); break
        if hit:
            if idx_hit==0:s["fast_hits"]+=1
            elif idx_hit==1:s["med_hits"]+=1
            else:s["slow_hits"]+=1
            latency=tiers_cfg[idx_hit]["latency_ns"]; energy=tiers_cfg[idx_hit]["energy"]
        else:
            policy.place_on_miss(page); s["misses"]+=1
            latency=tiers_cfg[-1]["latency_ns"]*2; energy=tiers_cfg[-1]["energy"]*1.5
        s["latency_ns_sum"]+=latency; s["energy_sum"]+=energy
        if isinstance(policy,PredictivePolicy):
            if last_pref: s["prefetch_requests"]+=1
            if page in last_pref: s["prefetch_correct"]+=1
            policy.observe(page); last_pref=set(policy.prefetch(page)); s["retrain_time_ms_sum"]+=policy.last_retrain_ms
    steps=max(1,s["steps"])
    out = {
        "steps":steps,
        "fast_hit_rate":s["fast_hits"]/steps,
        "med_hit_rate":s["med_hits"]/steps,
        "slow_hit_rate":s["slow_hits"]/steps,
        "miss_rate":s["misses"]/steps,
        "avg_latency_ns":s["latency_ns_sum"]/steps,
        "avg_energy":s["energy_sum"]/steps,
        "prefetch_accuracy": (s["prefetch_correct"]/s["prefetch_requests"]) if s["prefetch_requests"] else 0.0,
        "avg_retrain_time_ms": s["retrain_time_ms_sum"]/steps,
        "security_checks": s["security_checks"],
    }
    for i,tc in enumerate(policy.tiers):
        out[f"evictions_t{i}"] = tc.evictions
        out[f"occupancy_t{i}"] = tc.occupancy()
    if check_invariants:
        rate_sum = out["fast_hit_rate"] + out["med_hit_rate"] + out["slow_hit_rate"] + out["miss_rate"]
        if abs(1.0 - rate_sum) > 1e-9:
            raise AssertionError(f"Rate sum invariant violated: {rate_sum}")
        lats=[t["latency_ns"] for t in tiers_cfg]; ens=[t["energy"] for t in tiers_cfg]
        if not (lats[0] <= lats[1] <= lats[2] and ens[0] <= ens[1] <= ens[2]):
            raise AssertionError("Tier ordering invariant violated for latency/energy")
    return out

def load_config(path:str) -> Dict[str,Any]:
    import json
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def export_champsim_trace(trace, out_path:str):
    with open(out_path,"w",encoding="utf-8") as f:
        for page,_ in trace: f.write(f"{page}\n")

def parse_grid(s: Optional[str]) -> Dict[str, list]:
    if not s: return {}
    grid={}
    for part in s.split(";"):
        if not part.strip(): continue
        key,vals = part.split("=")
        grid[key.strip()] = [int(v.strip()) for v in vals.split(",") if v.strip()]
    return grid

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--num_pages", type=int, default=2000)
    p.add_argument("--trace_length", type=int, default=2000)
    p.add_argument("--hotset_size", type=int, default=80)
    p.add_argument("--stickiness", type=int, default=400)
    p.add_argument("--drift_pages", type=int, default=20)
    p.add_argument("--protected_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policy", type=str, default="LRU", choices=["LRU","PREDICTIVE"])
    p.add_argument("--ngram", type=int, default=2)
    p.add_argument("--prefetch_budget", type=int, default=2)
    p.add_argument("--out_csv", type=str, default="imm_results.csv")
    p.add_argument("--grid", type=str, default=None, help="e.g., 'ngram=2,3;prefetch_budget=1,2'")
    p.add_argument("--export_champsim_trace", type=str, default=None)
    args=p.parse_args()

    tiers=[{"name":"Fast","capacity":256,"latency_ns":80,"energy":1.0},
           {"name":"Medium","capacity":512,"latency_ns":250,"energy":2.0},
           {"name":"Slow","capacity":1024,"latency_ns":1200,"energy":4.5}]
    if args.config:
        cfg=load_config(args.config)
        if "tiers" in cfg: tiers=cfg["tiers"]

    run_params = dict(
        num_pages = args.num_pages,
        trace_length = args.trace_length,
        hotset_size = args.hotset_size,
        stickiness = args.stickiness,
        drift_pages = args.drift_pages,
        protected_ratio = args.protected_ratio,
        seed = args.seed,
    )
    random.seed(args.seed)
    trace,_=generate_dynamic_trace(**run_params)

    if args.export_champsim_trace:
        export_champsim_trace(trace, args.export_champsim_trace)

    base_row = {**run_params, "tiers": json.dumps(tiers)}

    rows=[]
    def run_with(policy_name, ngram, k):
        if policy_name=="LRU":
            pol=BasePolicy(tiers)
        else:
            pol=PredictivePolicy(tiers, n=ngram, prefetch_budget=k)
        res=run_sim(pol, tiers, trace, check_invariants=True)
        res.update(base_row)
        res.update({
            "policy": "Baseline-LRU" if policy_name=="LRU" else f"Predictive-n{ngram}-k{k}",
            "ngram": ngram if policy_name!="LRU" else 0,
            "prefetch_budget": k if policy_name!="LRU" else 0,
        })
        return res

    grid = parse_grid(args.grid)
    if grid:
        rows.append(run_with("LRU", 0, 0))
        ngs = grid.get("ngram",[args.ngram])
        ks  = grid.get("prefetch_budget",[args.prefetch_budget])
        for n in ngs:
            for k in ks:
                rows.append(run_with("PREDICTIVE", n, k))
    else:
        if args.policy=="LRU":
            rows.append(run_with("LRU", 0, 0))
        else:
            rows.append(run_with("PREDICTIVE", args.ngram, args.prefetch_budget))

    import pandas as pd
    df=pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))
    print(f"Saved -> {args.out_csv}")

if __name__=="__main__":
    main()
