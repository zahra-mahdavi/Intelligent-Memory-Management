
import time, random
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
        key=tuple(self.context)
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
    protected=set(random.sample(range(num_pages),max(0,int(protected_ratio*num_pages))))
    pool=[p for p in range(num_pages) if p not in protected]
    hotset=deque(random.sample(pool,min(hotset_size,len(pool))),maxlen=hotset_size)
    trace=[]
    for t in range(length):
        if stickiness>0 and t%stickiness==0 and t>0:
            candidates=[p for p in pool if p not in hotset]
            replace=list(hotset)[:min(drift_pages,len(hotset))]
            for r in replace:
                try:
                    hotset.remove(r)
                except ValueError:
                    pass
            if candidates: 
                k=min(len(replace),len(candidates))
                if k>0:
                    hotset.extend(random.sample(candidates,k))
        page=random.choice(list(hotset)) if random.random()<0.85 and len(hotset)>0 else random.randrange(num_pages)
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
            if page in tc: 
                hit=True; idx_hit=idx; tc.touch(page); policy.on_hit(page,idx); break
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
            if len(last_pref)>0: s["prefetch_requests"]+=1
            if page in last_pref: s["prefetch_correct"]+=1
            policy.observe(page); last_pref=set(policy.prefetch(page)); s["retrain_time_ms_sum"]+=policy.last_retrain_ms
    steps=max(1,s["steps"])
    return {"steps":steps,"fast_hit_rate":s["fast_hits"]/steps,"med_hit_rate":s["med_hits"]/steps,
            "slow_hit_rate":s["slow_hits"]/steps,"miss_rate":s["misses"]/steps,
            "avg_latency_ns":s["latency_ns_sum"]/steps,"avg_energy":s["energy_sum"]/steps,
            "prefetch_accuracy":(s["prefetch_correct"]/s["prefetch_requests"]) if s["prefetch_requests"] else 0.0,
            "avg_retrain_time_ms":(s["retrain_time_ms_sum"]/steps),"security_checks":s["security_checks"]}
