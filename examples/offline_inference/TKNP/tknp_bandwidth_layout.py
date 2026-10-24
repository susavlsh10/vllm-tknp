"""Opt-in root-load proxy for homogeneous decode graph ownership.

The physical KV-capacity deficit of root approximates weight/buffer memory.
Treat that deficit as bandwidth work, multiplied by root dense passes (two
for DBO). This is a measured-memory proxy, not a calibrated latency model.
"""


def bandwidth_aware_counts(capacities, total, *, root_dense_passes=1,
                           minimum_root=2, minimum_peer=2, root_maximum=None):
    degree=len(capacities)
    if (degree<2 or total<0 or total%degree or root_dense_passes<1
            or any(type(c) is not int or c<0 for c in capacities)):
        raise ValueError("Invalid homogeneous TKNP graph budget")
    limits=list(capacities)
    if root_maximum is not None:limits[0]=min(limits[0],root_maximum)
    counts=[minimum_root]+[minimum_peer]*(degree-1)
    if sum(counts)>total or any(q>c for q,c in zip(counts,limits)) or sum(limits)<total:
        raise ValueError("Root-aware quotas cannot fit this graph/cohort shape")
    overhead=max(0,max(capacities[1:])-capacities[0])*root_dense_passes
    for _ in range(total-sum(counts)):
        candidates=[i for i in range(degree) if counts[i]<limits[i]]
        owner=min(candidates,key=lambda i:(counts[i]+(overhead if i==0 else 0),i==0,i))
        counts[owner]+=1
    return counts
