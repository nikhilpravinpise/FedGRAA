# FedGRAA-v2 communication-overhead table

Per-cell median `bytes_sent` (SecAgg-instrumented, T-12's byte-exactness gate) for each FedGRAA-v2 frontier arm, expressed as a ratio against the two T3 near-zero-cost approximation baselines (celm, spectralfed), paired with the rare-class-recall delta so the "is exact worth N times the bytes?" question has both axes on one row.

## CIC-IDS-2017

| v2 arm | vs. | v2 GB | approx GB | byte ratio | v2 rare-acc | approx rare-acc | rare-acc delta |
|---|---|---|---|---|---|---|---|
| v2_m1 | celm | 5.6072 | 0.9344 | 6.00x | 0.8806 | 0.0000 | +0.8806 |
| v2_m1 | spectralfed | 5.6072 | 0.9344 | 6.00x | 0.8806 | 0.0000 | +0.8806 |
| v2_m2 | celm | 3.7382 | 0.9344 | 4.00x | 0.9181 | 0.0000 | +0.9181 |
| v2_m2 | spectralfed | 3.7382 | 0.9344 | 4.00x | 0.9181 | 0.0000 | +0.9181 |
| v2_m3 | celm | 2.8037 | 0.9344 | 3.00x | 0.9347 | 0.0000 | +0.9347 |
| v2_m3 | spectralfed | 2.8037 | 0.9344 | 3.00x | 0.9347 | 0.0000 | +0.9347 |
| v2_m5 | celm | 1.8692 | 0.9344 | 2.00x | 0.8889 | 0.0000 | +0.8889 |
| v2_m5 | spectralfed | 1.8692 | 0.9344 | 2.00x | 0.8889 | 0.0000 | +0.8889 |

## UNSW-NB15

| v2 arm | vs. | v2 GB | approx GB | byte ratio | v2 rare-acc | approx rare-acc | rare-acc delta |
|---|---|---|---|---|---|---|---|
| v2_m1 | celm | 2.9086 | 0.4847 | 6.00x | 0.6548 | 0.0482 | +0.6066 |
| v2_m1 | spectralfed | 2.9086 | 0.4847 | 6.00x | 0.6548 | 0.0000 | +0.6548 |
| v2_m2 | celm | 1.9391 | 0.4847 | 4.00x | 0.6571 | 0.0482 | +0.6089 |
| v2_m2 | spectralfed | 1.9391 | 0.4847 | 4.00x | 0.6571 | 0.0000 | +0.6571 |
| v2_m3 | celm | 1.4543 | 0.4847 | 3.00x | 0.6966 | 0.0482 | +0.6484 |
| v2_m3 | spectralfed | 1.4543 | 0.4847 | 3.00x | 0.6966 | 0.0000 | +0.6966 |
| v2_m5 | celm | 0.9696 | 0.4847 | 2.00x | 0.6830 | 0.0482 | +0.6348 |
| v2_m5 | spectralfed | 0.9696 | 0.4847 | 2.00x | 0.6830 | 0.0000 | +0.6830 |
