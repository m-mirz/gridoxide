#!/usr/bin/env python3
"""Where a GPU batch power flow's time goes, from an Nsight Systems trace.

    nsys profile -o run --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        target/release/examples/gpu_bench <case.m> 256 sweep
    nsys export --type sqlite run.nsys-rep
    scripts/bench/nsys_batch_breakdown.py run.sqlite

Inside the `batch.chunk` NVTX ranges of `GpuBatchSolver`, leaving out the
first quarter (gpu_bench's untimed run of four): the device's time in cuDSS's
kernels, in ours (`k_*`), in copies, and idle. A kernel counts to the range
that launched it: the host's ranges alone mislead, as cuDSS's phases only
queue work and the wait falls in `batch.mismatch`, the next synchronization.
"""
import sqlite3,sys,re,bisect,collections
def merge(iv):
    iv=sorted(iv);out=[]
    for a,b in iv:
        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out
def clip(iv,w):
    return [(max(a,w[0]),min(b,w[1])) for a,b in iv if b>w[0] and a<w[1]]
def main(path,label):
    c=sqlite3.connect(path); c.row_factory=sqlite3.Row
    names={r[0]:r[1] for r in c.execute('select id,value from StringIds')}
    tabs={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    ev=[(r['start'],r['end'],r['text'] or names.get(r['textId'])) for r in c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')]
    ev=[e for e in ev if e[2] and e[2].startswith('batch.')]
    chunks=sorted((a,b) for a,b,t in ev if t=='batch.chunk')
    k=len(chunks)//4 or 0   # the first of four solves is the untimed warm-up
    win=chunks[k:]
    lo,hi=win[0][0],win[-1][1]
    wall=sum(b-a for a,b in win)
    ranges=sorted((a,b,t) for a,b,t in ev if t!='batch.chunk' and a>=lo and b<=hi)
    starts=[r[0] for r in ranges]
    def phase(t):
        i=bisect.bisect_right(starts,t)-1
        return ranges[i][2] if i>=0 and t<=ranges[i][1] else 'other'
    rt={r['correlationId']:r['start'] for r in c.execute('select correlationId,start from CUPTI_ACTIVITY_KIND_RUNTIME')}
    kern=[]
    for r in c.execute('select start,end,demangledName,shortName,correlationId from CUPTI_ACTIVITY_KIND_KERNEL'):
        kern.append((r['start'],r['end'],names.get(r['shortName'],''),r['correlationId']))
    copies=[(r['start'],r['end'],r['correlationId']) for r in c.execute('select start,end,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY')]
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in tabs:
        copies+=[(r['start'],r['end'],r['correlationId']) for r in c.execute('select start,end,correlationId from CUPTI_ACTIVITY_KIND_MEMSET')]
    t=collections.Counter(); byphase=collections.Counter(); top=collections.Counter(); ncu=0
    busy=[];cud=[];own=[];cp=[]
    for a,b,n,cid in kern:
        if not (lo<=a and b<=hi+10**9): continue
        ours=n.startswith('k_')
        (own if ours else cud).append((a,b)); busy.append((a,b))
        top[n]+=b-a; ncu+=not ours
        byphase[phase(rt.get(cid,a))]+=b-a
    for a,b,cid in copies:
        if lo<=a<=hi: cp.append((a,b)); busy.append((a,b)); byphase[phase(rt.get(cid,a))]+=b-a
    size=lambda iv: sum(b-a for a,b in iv)
    tot=lambda iv: sum(size(clip(merge(iv),w)) for w in win)
    B=tot(busy); C=tot(cud); O=tot(own); M=tot(cp)
    print(f'## {label}: {len(win)} chunks, wall {wall/1e6:.1f} ms')
    print(f'   GPU busy {B/1e6:.1f} ms = {100*B/wall:.0f}% of wall; of busy: cuDSS kernels {100*C/B:.1f}%, ours {100*O/B:.1f}%, copies/memsets {100*M/B:.1f}%')
    print(f'   of wall: cuDSS {100*C/wall:.1f}%, ours {100*O/wall:.1f}%, copies {100*M/wall:.1f}%, GPU idle {100*(wall-B)/wall:.1f}%')
    w=collections.Counter()
    for a,b,tt in ranges: w[tt]+=b-a
    print('   host range wall % / device work % of busy: '+', '.join(f'{p[6:]} {100*w[p]/wall:.1f}/{100*byphase[p]/B:.1f}' for p,_ in w.most_common()))
    print('   top: '+', '.join(f'{n[:34]} {100*v/B:.1f}%' for n,v in top.most_common(5)), f'| {ncu} cuDSS kernels')
for a in sys.argv[1:]:
    main(a, a.split('/')[-1].replace('.sqlite',''))
