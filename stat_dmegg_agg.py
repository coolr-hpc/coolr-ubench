#!/usr/bin/env python

import json, sys
import numpy as np

if len(sys.argv) < 2:
    print 'Usage: %s fn' % sys.argv[0]
    sys.exit(0)

fn =  sys.argv[1]

try:
    f = open( fn )
except:
    print 'Unable to open', fn


dgemm_agg = []

while True:
    l = f.readline();
    if not l:
        break

    if l[0] != '{':  # quick hack.
        continue

    j = json.loads(l)
    v = j['dgemm_agg']
    dgemm_agg.append(v)


print 'mean=%.2lf std=%.2lf min=%.2lf max=%.2lf' %  \
       (np.mean(dgemm_agg), np.std(dgemm_agg),  \
        np.min(dgemm_agg), np.max(dgemm_agg))

f.close()


