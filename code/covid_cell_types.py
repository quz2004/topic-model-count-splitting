"""Collapse Stephenson et al. fine cell types to lineages (pre-specified in reports/step11_replication_plan.md)."""
def coarse(s):
    t = s.lower()
    if 'monocyte' in t and ('cd16-positive' in t or 'cd14-low' in t): return 'non-classical monocyte'
    if 'cd14' in t and 'monocyte' in t: return 'classical monocyte'
    if 'monocyte' in t: return 'classical monocyte'
    if 'dendritic' in t: return 'conventional dendritic cell'
    if 'platelet' in t or 'megakaryocyte' in t: return 'platelet'
    if 'natural killer' in t or 'nk' in t.split() : return 'natural killer cell'
    if 'nkt' in t.replace(' ', '') : return 'natural killer cell'
    if 'cd4' in t or 't-helper' in t or 'regulatory t' in t: return 'CD4-positive, alpha-beta T cell'
    if 'cd8' in t or 'gamma-delta' in t or 'mait' in t or 'mucosal invariant' in t: return 'CD8-positive, alpha-beta T cell'
    if 'plasma' in t or 'b cell' in t or 'plasmablast' in t: return 'B cell'
    return 'other'
