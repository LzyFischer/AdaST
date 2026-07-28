# Preliminary experiment at scale — N=20, T=12000, H=1, readout=last

3 seeds. MAE is per-node z-scored, so 0.798 is predict-the-mean for Gaussian data.

## attention

### raw MAE

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.594 | 0.695 | 0.594 |
| SD | 0.728 | 0.600 | 0.599 |
| SC | 0.709 | 0.678 | 0.671 |

### ROW-normalised (per dataset: which architecture wins)

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.000 | 1.000 | 0.004 |
| SD | 1.000 | 0.008 | 0.000 |
| SC | 1.000 | 0.186 | 0.000 |

### COLUMN-normalised (per architecture: which dataset it likes)

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.000 | 1.000 | 0.000 |
| SD | 1.000 | 0.000 | 0.062 |
| SC | 0.860 | 0.820 | 1.000 |

- diagonal by ROW reading: **2/3**  (this is the claim "the matching architecture wins on each dataset")
- diagonal by COLUMN reading: **2/3**  (this is what `normalize_columns_minmax` plots)

## conv

### raw MAE

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.604 | 0.695 | 0.666 |
| SD | 0.724 | 0.598 | 0.600 |
| SC | 0.706 | 0.677 | 0.678 |

### ROW-normalised (per dataset: which architecture wins)

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.000 | 1.000 | 0.676 |
| SD | 1.000 | 0.000 | 0.012 |
| SC | 1.000 | 0.000 | 0.042 |

### COLUMN-normalised (per architecture: which dataset it likes)

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.000 | 1.000 | 0.839 |
| SD | 1.000 | 0.000 | 0.000 |
| SC | 0.853 | 0.815 | 1.000 |

- diagonal by ROW reading: **2/3**  (this is the claim "the matching architecture wins on each dataset")
- diagonal by COLUMN reading: **2/3**  (this is what `normalize_columns_minmax` plots)

## mlp

### raw MAE

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.578 | 0.699 | 0.581 |
| SD | 0.712 | 0.599 | 0.599 |
| SC | 0.687 | 0.678 | 0.659 |

### ROW-normalised (per dataset: which architecture wins)

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.000 | 1.000 | 0.024 |
| SD | 1.000 | 0.000 | 0.001 |
| SC | 1.000 | 0.682 | 0.000 |

### COLUMN-normalised (per architecture: which dataset it likes)

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.000 | 1.000 | 0.000 |
| SD | 1.000 | 0.000 | 0.234 |
| SC | 0.815 | 0.790 | 1.000 |

- diagonal by ROW reading: **3/3**  (this is the claim "the matching architecture wins on each dataset")
- diagonal by COLUMN reading: **2/3**  (this is what `normalize_columns_minmax` plots)

## gcn

### raw MAE

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.578 | 0.695 | 0.579 |
| SD | 0.712 | 0.598 | 0.600 |
| SC | 0.687 | 0.677 | 0.659 |

### ROW-normalised (per dataset: which architecture wins)

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.000 | 1.000 | 0.010 |
| SD | 1.000 | 0.000 | 0.013 |
| SC | 1.000 | 0.626 | 0.000 |

### COLUMN-normalised (per architecture: which dataset it likes)

| data \ arch | T | S | ST |
|---|---|---|---|
| TD | 0.000 | 1.000 | 0.000 |
| SD | 1.000 | 0.000 | 0.258 |
| SC | 0.815 | 0.815 | 1.000 |

- diagonal by ROW reading: **3/3**  (this is the claim "the matching architecture wins on each dataset")
- diagonal by COLUMN reading: **2/3**  (this is what `normalize_columns_minmax` plots)
