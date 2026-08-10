# Archived FedPGO proxy cells (2026-07-30) — superseded 2026-07-31

These 20 result JSONs were produced by the **disclosed proxy** version of the
FedPGO baseline (`fedgraa/baselines_t5_fedpgo.py` as committed at
`c5c8ac8`/`30fd064`) — a reconstruction from this project's literature
review's mechanism summary, written while the paper's full text was still
access-blocked. They are archived here, not deleted, per decision **D2** of
[`docs/research_c/STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md`](../../../docs/research_c/STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md):

> Archive into `results/v2_frontier/_archive_fedpgo_proxy_2026-07-30/` with a
> dated README, not delete. Verified collision-free: `build_v2_frontier_table.py`'s
> glob is non-recursive and `check_stage_c_complete.py` verifies exact
> `cell.path`, so archived files can never contaminate seed counts.

On 2026-07-30 the paper's primary text became local at
[`docs/research_c/FedPGO/FedPGO.md`](../../../docs/research_c/FedPGO/FedPGO.md),
and on 2026-07-31 the proxy was replaced by a faithful implementation of the
primary text's mechanism (Eqs. 6–10; Algorithms 1–3) and the 20 cells were
rerun on Modal (plan §7). The rerun's cells live at the canonical paths
(`results/v2_frontier/{CIC_IDS_2017,UNSW_NB15}_fedpgo_g1.0_seed{0..9}.json`)
and the rebuilt frontier table/figure include the faithful FedPGO point.

The proxy's three named divergences from the primary text (plan §2):
same-round FedAvg mean in place of the Alg. 3 step-23 gradient memory;
`parallel + orthogonal_weight * orthogonal` in place of the Eq. 8
recomposition `alpha_i * g_par + (1 - alpha_i) * g_perp`; ReLU-clipped
cosine + orthogonal *fraction* in place of the Eq. 9 PGF score
`S_i = l1*cos(g_i,g_G) + l2*||g_perp|| + l3*Delta_L_val`.

Contents (22 files):

* `CIC_IDS_2017_fedpgo_g1.0_seed{0..9}.json`
* `UNSW_NB15_fedpgo_g1.0_seed{0..9}.json`
* `CIC_IDS_2017_fedpgo_g1.0_seed0_r2.json` / `UNSW_NB15_fedpgo_g1.0_seed0_r2.json`
  — the proxy-era 2-round smoke artifacts (the UNSW-NB15 one was tracked in
  git at `9d20ac9`; the CIC-IDS-2017 one was untracked), moved here with the
  cells so every proxy-era fedpgo artifact lives in one place. The
  2026-07-31 faithful rerun wrote a fresh CIC-IDS-2017 `_r2` smoke artifact
  at the canonical location (commit `25611ce`); the UNSW-NB15 `_r2` was
  never re-smoked and remains proxy-era here.
