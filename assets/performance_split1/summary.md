# Phase 4 RBM Split 1 Performance

## Inputs

- RBM log: `S_3D/phase4_logs/rbm_production_split1.log`
- Baseline log: `/home/mb3835/group_storage/mem_opt_gammasq_a8000p0_J0p5_Pth0p7071_Pph0p5_split1.log`
- RBM namespace: `Namespace(k=4, t=1, g_1=1.0, g_2=1.0, M_1=2000.0, M_2=2000.0, splits=5, split_idx=1, Pphi=0.5, Ptheta=0.7071, alpha=8000.0, NR=89, Nx=90, Ny=90, Nz=90, bo_spectrum=None, J=0.5, verbosity=5, iterations=10000, subspace=100, guess=None, evecs=None, save=None, potential='erf_coulomb', extent=None, backend='cupy', soc='full', Gammasq=True, phase4_rbm_diagnostics=False, phase4_rbm_svd_tol=1e-10, phase4_rbm_bank_size=4, phase4_rbm_store_size=8, phase4_rbm_polish_guess=True, phase4_stop_after_ps_solve=None, phase4_skip_expectations=False)`
- Baseline namespace: `Namespace(k=4, t=1, g_1=1.0, g_2=1.0, M_1=2000.0, M_2=2000.0, splits=5, split_idx=1, Pphi=0.5, Ptheta=0.7071, alpha=8000.0, NR=89, Nx=90, Ny=90, Nz=90, bo_spectrum=None, J=0.5, verbosity=5, iterations=5000, subspace=300, guess=None, evecs=None, save=None, potential='erf_coulomb', extent=None, backend='cupy', soc='full', Gammasq=True)`

## Headline

- PS solves compared: `1602`
- Baseline PS Davidson time: `25217.500 s`
- RBM-polished PS Davidson time: `2831.405 s`
- Time saved: `22386.095 s` (`6.22 h`)
- Total wall-time speedup over PS solves: `8.906x`
- Total cycle speedup over PS solves: `4.178x`
- Median per-solve wall-time speedup: `10.724x`
- Median per-solve cycle speedup: `4.941x`
- Baseline end-to-end `R for loop`: `25900.000 s`
- RBM end-to-end `R for loop`: `3800.000 s`
- End-to-end split-loop speedup: `6.816x`
- End-to-end time saved: `22100.000 s` (`6.14 h`)

## Correctness

- Max absolute final PS energy difference: `3.627e-13`
- All timed PS solves converged in both logs.

## Region Breakdown

| Region | Count | RBM cycles med. | Baseline cycles med. | RBM time med. (s) | Baseline time med. (s) | Time speedup med. |
|---|---:|---:|---:|---:|---:|---:|
| center seed | 18 | 117.5 | 96.0 | 10.200 | 18.200 | 1.80x |
| first lower | 18 | 94.0 | 84.0 | 8.785 | 16.000 | 1.81x |
| second lower | 18 | 12.0 | 84.0 | 1.190 | 16.250 | 13.23x |
| mature lower | 756 | 17.0 | 84.0 | 1.480 | 16.100 | 10.77x |
| first upper | 18 | 94.0 | 84.0 | 8.760 | 16.200 | 1.86x |
| second upper | 18 | 12.0 | 84.0 | 1.190 | 15.750 | 12.96x |
| mature upper | 756 | 17.0 | 84.0 | 1.490 | 16.100 | 10.74x |

## Artifacts

- `ps_solve_metrics.csv`
- `region_summary.csv`
- `rbm_residuals.csv`
- `cycles_vs_ps_solve_index.png`
- `wall_time_vs_ps_solve_index.png`
- `cumulative_ps_time.png`
- `cycles_vs_p_first_R.png`
- `wall_time_vs_p_first_R.png`
- `wall_time_speedup_histogram.png`
- `rbm_residuals_vs_solve_index.png`
