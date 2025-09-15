
import argparse, random, time
from collections import deque, defaultdict, Counter
import pandas as pd

class TierCache:
    def __init__(self, name, capacity):
        self.name, self.capacity = name, capacity
        self.deque, self.set = deque(), set()
    def __contains__(self, page): return page in self.set
    def add(self, page):
        if page in self.set:
            self.deque.remove(page); self.deque.append(page); return None
        evicted=None
        if len(self.deque)>=self.capacity:
            evicted=self.deque.popleft(); self.set.remove(evicted)
        self.deque.append(page); self.set.add(page); return evicted
    def touch(self,page):
        if page in self.set:
            self.deque.remove(page); self.deque.append(page)
    def remove(self,page):
        if page in self.set:
            self.deque.remove(page); self.set.remove(page)

class BasePolicy:
    def __init__(self, tiers): self.tiers=[TierCache(t["name"],t["capacity"]) for t in tiers]
    def place_on_miss(self,page): self.tiers[-1].add(page)
    def on_hit(self,page,idx):
        if idx>0: self.tiers[idx].remove(page); self.tiers[idx-1].add(page)
    def prefetch(self,_): return []

class PredictivePolicy(BasePolicy):
    def __init__(self, tiers,n=2,prefetch_budget=2):
        super().__init__(tiers); self.n=n; self.prefetch_budget=prefetch_budget
        self.context=deque(maxlen=n); self.counts=defaultdict(Counter); self.last_retrain_ms=0.0
    def observe(self,page):
        start=time.time()
        if len(self.context)==self.n:
            key=tuple(self.context); self.counts[key][page]+=1
        self.context.append(page); self.last_retrain_ms=(time.time()-start)*1000.0
    def predict_next(self):
        if len(self.context)<self.n: return []
        key=tuple(self.context); 
        if key not in self.counts: return []
        return [p for p,_ in self.counts[key].most_common(self.prefetch_budget)]
    def prefetch(self,page):
        preds=self.predict_next(); placed=[]
        for p in preds:
            if any(p in tc for tc in self.tiers): continue
            for idx in range(len(self.tiers)-2,-1,-1):
                self.tiers[idx].add(p); placed.append(p); break
        return placed

def generate_dynamic_trace(num_pages,length,hotset_size,stickiness,drift_pages,protected_ratio,seed=42):
    random.seed(seed)
    protected=set(random.sample(range(num_pages),int(protected_ratio*num_pages)))
    pool=[p for p in range(num_pages) if p not in protected]
    hotset=deque(random.sample(pool,min(hotset_size,len(pool))),maxlen=hotset_size)
    trace=[]
    for t in range(length):
        if t%stickiness==0 and t>0:
            candidates=[p for p in pool if p not in hotset]
            replace=list(hotset)[:min(drift_pages,len(hotset))]
            for r in replace:
                if r in hotset: hotset.remove(r)
            if candidates: hotset.extend(random.sample(candidates,min(len(replace),len(candidates))))
        page=random.choice(list(hotset)) if random.random()<0.85 else random.randrange(num_pages)
        trace.append((page,page in protected))
    return trace,protected

def run_sim(policy,tiers_cfg,trace,security_check_ns=15):
    s={"steps":0,"fast_hits":0,"med_hits":0,"slow_hits":0,"misses":0,"latency_ns_sum":0,"energy_sum":0.0,
       "prefetch_requests":0,"prefetch_correct":0,"retrain_time_ms_sum":0.0,"security_checks":0}
    last_pref=set()
    for page,prot in trace:
        s["steps"]+=1
        if prot:
            s["security_checks"]+=1; s["latency_ns_sum"]+=security_check_ns; s["energy_sum"]+=security_check_ns*0.01
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
            if last_pref: s["prefetch_requests"]+=1; 
            if page in last_pref: s["prefetch_correct"]+=1
            policy.observe(page); last_pref=set(policy.prefetch(page)); s["retrain_time_ms_sum"]+=policy.last_retrain_ms
    steps=max(1,s["steps"])
    return {"steps":steps,"fast_hit_rate":s["fast_hits"]/steps,"med_hit_rate":s["med_hits"]/steps,
            "slow_hit_rate":s["slow_hits"]/steps,"miss_rate":s["misses"]/steps,
            "avg_latency_ns":s["latency_ns_sum"]/steps,"avg_energy":s["energy_sum"]/steps,
            "prefetch_accuracy":(s["prefetch_correct"]/s["prefetch_requests"]) if s["prefetch_requests"] else 0.0,
            "avg_retrain_time_ms":(s["retrain_time_ms_sum"]/steps),"security_checks":s["security_checks"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--num_pages",type=int,default=2000)
    p.add_argument("--trace_length",type=int,default=2000)
    p.add_argument("--hotset_size",type=int,default=80)
    p.add_argument("--stickiness",type=int,default=400)
    p.add_argument("--drift_pages",type=int,default=20)
    p.add_argument("--protected_ratio",type=float,default=0.05)
    p.add_argument("--predictive",action="store_true")
    p.add_argument("--ngram",type=int,default=2)
    p.add_argument("--prefetch_budget",type=int,default=2)
    p.add_argument("--out_csv",type=str,default="imm_results.csv")
    a=p.parse_args()
    tiers=[{"name":"Fast","capacity":256,"latency_ns":80,"energy":1.0},
           {"name":"Medium","capacity":512,"latency_ns":250,"energy":2.0},
           {"name":"Slow","capacity":1024,"latency_ns":1200,"energy":4.5}]
    trace,_=generate_dynamic_trace(a.num_pages,a.trace_length,a.hotset_size,a.stickiness,a.drift_pages,a.protected_ratio)
    results=[]
    base=BasePolicy(tiers); r1=run_sim(base,tiers,trace); r1["policy"]="Baseline-LRU"; results.append(r1)
    if a.predictive:
        pred=PredictivePolicy(tiers,n=a.ngram,prefetch_budget=a.prefetch_budget)
        r2=run_sim(pred,tiers,trace); r2["policy"]=f"Predictive-n{a.ngram}-k{a.prefetch_budget}"; results.append(r2)
    df=pd.DataFrame(results); df.to_csv(a.out_csv,index=False); print(df.to_string(index=False)); print(f"Saved -> {a.out_csv}")

if __name__=="__main__": main()
