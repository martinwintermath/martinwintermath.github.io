#!/usr/bin/env python3
"""Exact-rational construction of the circle realization-space example.

Builds all clusters, checks the relations and nondegeneracies, and exports
one rational homogeneous Lawrence vertex matrix. By default, five rational
circle points are checked, including all construction counts. The all-input statement
is established symbolically in the paper, not by this computation.

Usage:
  python3 circle_construction.py
  python3 circle_construction.py --export ./
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
import json


def circuits(which, point=(F(1), F(0))):
    assert which == "circle"
    x, y = point
    assert x*x + y*y == 1, point
    inputs = {'u': x + 3, 'v': y + 3}
    gates = [
        ('two', '+', '1', '1'),
        ('three', '+', 'two', '1'),
        ('six', '+', 'three', 'three'),
        ('eighteen', '*', 'three', 'six'),
        ('a', '*', 'u', 'u'),
        ('b', '*', 'v', 'v'),
        ('c', '+', 'a', 'b'),
        ('L', '+', 'c', 'eighteen'),
        ('d', '+', 'u', 'v'),
        ('e', '*', 'six', 'd'),
        ('R', '+', 'e', '1'),
    ]
    return inputs, gates


def local_value(label, w):
    kind, name = label
    if kind == 'zero': return F(0)
    if kind == 'pos': return w[name]
    if kind == 'neg': return -w[name]
    if kind == 'inv': return 1 / w[name]
    raise ValueError(label)


def Q(a, b, c, d, f):
    if f is None:
        assert a != d and c != b
        return (a-d)/(c-b)
    assert a != f and c != b
    return ((a-d)*(c-f))/((a-f)*(c-b))


def build(which, point=(F(1), F(0))):
    inputs, gates = circuits(which, point)
    w = {'1': F(1), **inputs}
    for z, op, a, b in gates:
        assert z not in w
        w[z] = w[a]+w[b] if op == '+' else w[a]*w[b]
    assert w['L'] == w['R'], (which,w['L'],w['R'])
    assert all(x > 1 for k,x in w.items() if k != '1')

    def zero(): return ('zero','')
    def pos(s): return ('pos', s)
    def neg(s): return ('neg', s)
    def inv(s): return ('inv', s)
    clusters = []
    relations = []  # (cluster IDs for a,b,c,d,f; where f=None means infinity)
    def add_cluster(labels, rels):
        idx = len(clusters)
        labels = list(dict.fromkeys(labels))
        vals = [local_value(label,w) for label in labels]
        assert all(vals[i] < vals[i+1] for i in range(len(vals)-1)), (idx,labels,vals)
        assert zero() in labels
        clusters.append(labels)
        for a,b,c,d,f in rels:
            assert all(q in labels for q in (a,b,c,d))
            assert f is None or f in labels
            relations.append(((idx,a),(idx,b),(idx,c),(idx,d),None if f is None else (idx,f)))

    add_cluster([neg('1'),zero(),pos('1')], [(zero(),zero(),neg('1'),pos('1'),None)])
    variables = [*inputs.keys(), *(z for z,op,a,b in gates)]
    for t in variables:
        add_cluster([neg(t),zero(),inv(t),pos('1'),pos(t)], [
            (zero(),zero(),neg(t),pos(t),None),
            (pos('1'),pos('1'),inv(t),pos(t),zero()),
        ])
    for z,op,a,b in gates:
        if op == '+':
            add_cluster([neg(a),zero(),pos('1'),pos(b),pos(z)], [
                (zero(),pos(b),neg(a),pos(z),None),
            ])
        else:
            add_cluster([zero(),inv(a),pos('1'),pos(b),pos(z)], [
                (pos('1'),pos(b),inv(a),pos(z),zero()),
            ])
    add_cluster([neg('L'),zero(),pos('1'),pos('R')], [
        (zero(),zero(),neg('L'),pos('R'),None),
    ])
    num_local_relations=len(relations)

    occurrences=defaultdict(list)
    for i,cluster in enumerate(clusters):
        for key in cluster:
            if key[0] != 'zero': occurrences[key].append((i,key))
    for occ in occurrences.values():
        for later in occ[1:]:
            earlier=occ[0]
            relations.append(((earlier[0],zero()),later,(later[0],zero()),earlier,None))
    num_links = len(relations)-num_local_relations

    offsets=[F(0)]
    for i in range(len(clusters)-1):
        m_next=local_value(clusters[i+1][0],w)
        M_prev=local_value(clusters[i][-1],w)
        offsets.append(offsets[-1]+M_prev-m_next)
    coords={}
    for i,cluster in enumerate(clusters):
        for key in cluster:
            coords[(i,key)] = offsets[i] + local_value(key,w)
    all_finite=sorted(set(coords.values()))
    assert len(all_finite)==sum(len(c) for c in clusters)-(len(clusters)-1)
    assert len(all_finite) == len(coords)-(len(clusters)-1)
    assert coords[(0,zero())]==0 and coords[(0,pos('1'))]==1
    for j in range(len(clusters)-1):
        assert coords[(j,clusters[j][-1])] == coords[(j+1,clusters[j+1][0])]
    assert len(set(coords[(j,zero())] for j in range(len(clusters))))==len(clusters)

    for nr,(a,b,c,d,f) in enumerate(relations):
        aa,bb,cc,dd=(coords[v] for v in (a,b,c,d))
        ff=None if f is None else coords[f]
        assert aa!=dd and cc!=bb and aa!=cc and bb!=dd, (which,nr,a,b,c,d)
        if ff is not None: assert aa!=ff and cc!=ff
        assert Q(aa,bb,cc,dd,ff)==1, (which,nr,(aa,bb,cc,dd,ff))
    B=len(all_finite)+1
    K=len(relations)
    rank=K+2
    num_elems=B+4*K
    dimension=rank+num_elems-1
    vertices=2*num_elems
    return {
        'which':which,'values':w,'clusters':clusters,'relations':relations,
        'line_coords':coords, 'all_finite':all_finite,
        'number_value_clusters':len(w)-1,
        'number_additions':sum(op=='+' for z,op,a,b in gates),
        'number_multiplications':sum(op=='*' for z,op,a,b in gates),
        'number_clusters':len(clusters),
        'nonzero_occurrences':sum(len(o) for o in occurrences.values()),
        'distinct_nonzero_symbols':len(occurrences),
        'local_relations':num_local_relations,'links':num_links,
        'n_L':B,'K':K,'rank':rank,'matroid_elements':num_elems,
        'dimension':dimension,'vertices':vertices,
    }


def sparse_V(result):
    """Return rank x n homogeneous matrix as a list of sparse rational columns."""
    K=result['K']; n=result['matroid_elements']
    cols=[]
    for t in result['all_finite']:
        cols.append({0:t,1:F(1)})
    cols.append({0:F(1)}) # infinity
    def p(t):
        if t is None: return (F(1),F(0))
        return (t,F(1))
    for i, (a,b,c,d,f) in enumerate(result['relations']):
        coords=result['line_coords']
        aa,bb,cc,dd=(coords[v] for v in (a,b,c,d))
        h=1/(aa-dd); k=1/(cc-bb)
        j=2+i
        cols.extend([
            {j:F(1)},
            {0:F(1),j:F(1)},
            {0:aa*h,1:h,j:F(1)},
            {0:cc*k,1:k,j:F(1)},
        ])
    cols=[{idx:val for idx,val in col.items() if val} for col in cols]
    assert len(cols)==n
    return cols


def export_lawrence(result, dest):
    """Sparse integer/rational columns for the full Lawrence point configuration.
    Homogeneous rows = rank + n; each pair has its own e_i row.
    The affine hyperplane is sum of last n coordinates = 1.
    """
    V=sparse_V(result)
    r=result['rank']; n=len(V)
    out=[]
    for i,col in enumerate(V):
        first=col.copy(); first[r+i]=F(1)
        second={r+i:F(1)}
        for vertex in (first,second):
            out.append([[j,str(val)] for j,val in sorted(vertex.items())])
    obj={'description':'Columns of homogeneous Lawrence vertex matrix',
         'example':result['which'], 'rows':r+n,'cols':2*n,
         'rank':r+n,'polytope_dimension':r+n-1, 'columns':out}
    Path(dest).write_text(json.dumps(obj,separators=(',',':')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',type=Path,help='Directory for sparse JSON rational Lawrence vertex matrices')
    args=parser.parse_args()
    points = [(F(1), F(0)), (F(-1), F(0)), (F(0), F(1)),
              (F(3, 5), F(4, 5)), (F(-3, 5), F(-4, 5))]
    expected = {
        'number_value_clusters': 13, 'number_additions': 7,
        'number_multiplications': 4, 'number_clusters': 26,
        'nonzero_occurrences': 98, 'distinct_nonzero_symbols': 41,
        'local_relations': 39, 'links': 57, 'n_L': 100, 'K': 96,
        'rank': 98, 'matroid_elements': 484, 'dimension': 581, 'vertices': 968,
    }
    representative = None
    for point in points:
        data = build('circle', point)
        for key, value in expected.items():
            assert data[key] == value, (point, key, data[key], value)
        if representative is None:
            representative = data
        print(f'Circle point ({point[0]}, {point[1]}): exact checks passed')
    for key, value in expected.items():
        print(f'circle {key:29s}: {value}')
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        export_lawrence(representative, args.export/'circle_lawrence_sparse.json')
    print('All finite-precision-free rational relation and nondegeneracy checks passed.')


if __name__=='__main__': main()
