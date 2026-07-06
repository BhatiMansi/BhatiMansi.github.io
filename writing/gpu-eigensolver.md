---
layout: default
title: Matrix-Free Davidson on GPUs for a 2M-Dimensional Quantum Hamiltonian
permalink: /writing/gpu-eigensolver/
mathjax: true
---

# Matrix-Free Davidson on GPUs for a 2-Million-Dimensional Quantum Hamiltonian

By Mansi Bhati

Most molecular simulations begin with the Born-Oppenheimer (BO) approximation.
Because nuclei are much heavier than electrons, BO treats the nuclei as almost
frozen while solving the electronic quantum problem. This is the standard reason
chemists can talk about an electronic energy surface as a function of nuclear
geometry. The approximation is enormously useful, but it also removes part of the
fully quantum feedback between nuclear motion and electronic motion. The theory
behind this project asks whether some of that missing nuclear-electronic
correlation can be restored without paying the full cost of an exact
electron-nuclear calculation.

The benchmark is the smallest molecular system that still contains this coupling:
an $H_2^+$-like three-body problem with two nuclei and one electron. In the lab
frame, the three particles have nine spatial coordinates. Overall translation and
rotation do not change the internal energy, so the physical problem can be
reduced to three internal coordinates. That reduction is what makes the problem
possible to study exactly enough to validate the new phase-space theory.

Even after this reduction, the exact benchmark is computationally sharp. For
each sampled nuclear position and momentum `(R, P)`, we need the lowest few
eigenenergies of a slightly different Hamiltonian. Sampling the phase-space
surface therefore turns one exact quantum problem into thousands of related large
eigenvalue problems.

This writeup is the computational companion to the manuscript
<https://arxiv.org/abs/2605.27053>. It explains how the Hamiltonian's structure
changes the problem from "diagonalize an impossibly big matrix" into "apply a sparse,
structured operator many times," and why that distinction makes GPUs useful.

**TL;DR.** A calculation defined by a two-million-dimensional Hermitian
eigenproblem, repeated thousands of times, was profiled and optimized around the
actual GPU bottleneck: Davidson's basis machinery. That insight produced an
**8.9× production speedup with machine-precision-identical energies**. The key was
not a faster Hamiltonian application; it was shrinking the Davidson subspace and
replacing single-point warm starts with a reduced-basis projection over the affine
`P`-structure of the problem.

---

## Part 1 — Matrix-Free Eigenproblem Setup

The physics above leaves one concrete computational task at each
phase-space grid point `(R, P)`: diagonalize a large **Hermitian** matrix `H`.

Each of the three internal coordinates is discretized on a grid of 100 points. A
two-valued electron-spin index is also included, so a quantum state is a list of
complex amplitudes of length

```text
N = 2 · (100 × 100 × 100) = 2,000,000  ≈  2 million.
```

`H` is therefore a `2,000,000 × 2,000,000` matrix *in principle*. Four facts make
this problem both hard and exploitable:

**(1) `H` is far too large to store.** A dense `N × N` matrix has `N²` entries.
Here that is `4 × 10¹²` numbers; in double-precision complex (`complex128`, 16
bytes each) that is **~64 terabytes** — for a *single* grid point. Storing `H`
explicitly is off the table before diagonalization even begins.

**(2) Only the bottom of the spectrum is needed.** In this molecular problem the
physically relevant states are the ground state and a few low-lying excitations.
Computing all two million eigenvalues would solve a much larger problem than the
one physics asks for.

**(3) The solve is repeated thousands of times.** The phase-space surface is sampled on a
100 × 100 grid of `(R, P)` points, so this eigenproblem is solved ~**10,000 times**.
Throughput matters as much as the cost of any single solve. Fortunately, the
Hamiltonian changes smoothly as `(R, P)` changes, so neighboring phase-space
points have nearly the same low-energy eigenvectors. That smoothness becomes a
major algorithmic advantage later.

**(4) The matrix has helpful structure: it is sparse and partly diagonal.**
This is the quantum-chemistry silver lining. Local operators on a real-space grid
only couple a grid point to itself or to nearby grid points, so most possible
matrix entries are exactly zero. `H` is a sum of physical terms with very
different structure:

- The **potential energy** (electron–nucleus attraction, etc.) and the bulk of
  the **spin–orbit coupling** are *diagonal* or nearly so: on a real-space grid,
  the potential at a point only multiplies the amplitude at that same point.
- The **kinetic energy** is a *derivative* operator. On a grid, a derivative is
  approximated by a **finite-difference stencil** — a weighted combination of a
  point and its near neighbors. That couples each grid point only to a handful of
  others, so kinetic energy is *sparse* (mostly zeros) but lives **off the
  diagonal**.

So `H` is sparse, and a large chunk of it (the potential) sits on the diagonal. 


### 1.1 Matrix-Free Hamiltonian Application

If `H` cannot be stored, the useful operation is the action of `H` on any
single vector — given `x`, produce `y = H x` — without ever forming `H`. Each
physical term of `H` is applied directly to the state:

- The **potential** is applied as an *elementwise* multiply: multiply the
  amplitude at each grid point by the (precomputed) potential value there. That's
  one cheap pass over the array.
- The **kinetic energy** is applied as a derivative *along each axis*. The state
  is naturally a 4-dimensional array of shape `(spin, x, y, z) = (2, 100, 100,
  100)`. To take the second derivative along, say, the `x`-axis, multiply by a
  small `100 × 100` finite-difference matrix `D_x`, applied **independently for
  every combination of the other indices** `(spin, y, z)`.

Reshape the big array so the target axis is the "rows," and the operation becomes
one matrix–matrix multiply by the small `100 × 100` operator, batched over all the
other indices. The three kinetic directions are three such batched GEMMs.

> **Dimension check.** The *operators* are small — three `100 × 100`
> 1D derivative matrices, **not** a single `100 × 100 × 100` object. What's big is
> the *state* `x` (the ~2-million-entry array). The small operators are applied to a big
> array, one axis at a time. 

The dense matrix-multiply-like contractions and elementwise array math, both over
millions of complex numbers, are massively parallel and limited mainly by
**memory bandwidth** — the rate at which data can be streamed through the chip. A
modern GPU has far more memory bandwidth and arithmetic throughput than a CPU,
and this workload is large, regular, and data-parallel. Across ~10,000 solves, a
CPU would be bandwidth-starved.

This also explains why the implementation does not simply use a sparse-matrix
library. A reader might expect a sparse Hamiltonian to belong in NVIDIA
**cuSPARSE**, but cuSPARSE
operates on an *explicitly stored* sparse matrix, such as a CSR/COO list of
nonzero entries. That list is never built. The sparsity lives implicitly in the
tensor structure above: small derivative matrices applied along each axis, plus
diagonal elementwise terms. So the matvec is built from **dense** batched GEMMs
and elementwise kernels (the cuBLAS world), not from sparse kernels.

### 1.2 Matrix-Free Low-Energy Eigenvalues


To keep the dimensions concrete, momentarily ignore spin. Then `H` is a
`100³ × 100³` operator, a vector `x` is a `100 × 100 × 100` array, and a matvec
`Hx` produces another `100 × 100 × 100` array. Including spin only doubles the
leading dimension; it does not change the algorithmic picture.

**Idea 1 — Iterate, don't solve.** Since only `x ↦ H x` is available, the answer
is built by repeated multiplication by `H`, refining a guess each step. Methods
built this way are **iterative eigensolvers**.

**Idea 2 — Work inside a small subspace (the projection trick / Rayleigh–Ritz).**
The 2-million-dimensional matrix cannot be diagonalized directly, but a
small `m × m` matrix (say `m = 50`) can be diagonalized with a normal dense
routine. So:

1. Collect a handful of promising vectors `v₁, …, v_m` (each of length `N`) into
   the columns of a tall, skinny matrix `V` (`N × m`). Keep them orthonormal.
2. **Project** the giant problem onto that subspace: form the small matrix
   `H_eff = Vᵀ H V` (size `50 × 50` if `m = 50`). Building it costs one matvec
   per basis vector plus a tall-skinny GEMM — cheap compared with storing `H`.
3. Diagonalize the small `H_eff` exactly. Its eigenvalues `θ` approximate
   eigenvalues of `H`, and `V c` (where `c` is a small eigenvector) approximates
   the corresponding eigenvector of `H`.

This recipe — *approximate the eigenvectors of a big matrix by the best
combination available inside a chosen small subspace* — is the **Rayleigh–Ritz
procedure**, and the approximate pairs `(θ, Vc)` are called **Ritz pairs**. Every
method below is a different answer to the only remaining question: **which vectors
go into `V`?**

**Idea 3 — Measure error with the residual.** For a candidate eigenpair `(θ, u)`,
the **residual** `r = H u − θ u` measures how far `u` is from being a true
eigenvector. If `r = 0`, `u` is exact. The size `‖r‖` is the convergence
yardstick and, crucially, the signal for *how to improve the guess*.

**Idea 4 — A preconditioner turns the error into a correction.** To improve `u`
using its residual `r`, the ideal correction would solve
`(H − θ) Δ = r`, but that's another giant linear solve. A **preconditioner** `M`
is a *cheap, approximate* stand-in for `(H − θ)`: applying `M⁻¹` to the residual
gives an *approximate* correction `M⁻¹ r` that is folded back into the subspace. A
good preconditioner makes the iteration converge in far fewer steps; a useless one
(`M = I`) does nothing. The catch is that `M` must be cheap to invert — so the
craft is finding an `M` that is both close to `(H − θ)` *and* trivially
invertible. (The simplest choice, used below, is `M = diag(H) − θ`: just the
diagonal of `H`, which is trivial to invert because it's diagonal.)


### 1.3 Choosing an Iterative Eigensolver

The deciding question from §1.2 was *which vectors go into the subspace `V`?*
There are two grand strategies:

- **Powers of `H`** (the *Krylov* strategy): build `V` from `x, Hx, H²x, …`.
- **Preconditioned residuals** (the *Davidson* strategy): build `V` by repeatedly
  correcting the current estimate with `M⁻¹ r`.


**Dense diagonalization (LAPACK `*heevd`, NVIDIA cuSOLVER) — infeasible.**

It needs the full matrix in memory (`O(N²)` — about 64 TB here) and does
`O(N³)` arithmetic, because turning a dense matrix into diagonal form touches
every entry and combines every row with every other. For `N = 2 × 10⁶` that's
~`8 × 10¹⁸` floating-point operations. Even at an idealized 100 TFLOP/s, that is
around 22 hours of arithmetic before accounting for memory traffic, algorithmic
overheads, or the fact that the 64 TB matrix cannot fit on the device. In
practice this path is days-to-weeks expensive, and it computes two million
eigenvalues when only a few are needed.

**Sparse-direct / shift-invert — infeasible, but instructive.**
Iterative methods naturally find the
**extremal** eigenvalues first — the ones largest in magnitude — because
multiplying by `H` amplifies them most (Idea 1). But the desired eigenvalues are at the
*bottom* and not extremal in magnitude, so they're the *hardest* to reach.
**Shift-invert** flips this: instead of `H`, work with `(H − σI)⁻¹` for a shift
`σ` near the target energies. That matrix has the *same* eigenvectors as `H`, but
its eigenvalues are `1/(λ − σ)`, so the `λ` closest to `σ` become the *largest*,
extremal eigenvalues — now the easy ones to find. (This is why ARPACK/SciPy users
are told to use `which='LM', sigma=0` rather than asking for "smallest magnitude"
directly.)

The catch: applying `(H − σI)⁻¹` to a vector means *solving the linear system*
`(H − σI) z = b` at every iteration. Done directly, that requires a
**factorization** of `H − σI` (e.g. LU: writing it as a product of triangular
matrices solvable by substitution). Factorization is `O(N³)` and, for a
sparse matrix, suffers **fill-in** — the zeros turn into nonzeros, so the factors
need vastly more memory than the original sparse operator. With a matrix-free `H`,
there are no explicit entries to factor. So shift-invert, the textbook fast path
to small eigenvalues, is unavailable here. The bottom of the spectrum must be
found without factorization, which raises the bar on the preconditioner (§1.5).

**Krylov methods — Lanczos / ARPACK — matrix-free baseline.**
A **Krylov subspace** is `span{x, Hx, H²x, …, H^{m−1}x}` — the span of `x` and its
repeated images under `H`. It needs nothing but matvecs, so it's perfectly
matrix-free. The **Lanczos** method is the specialized, numerically careful way to
build an orthonormal basis of this subspace when `H` is Hermitian: a short
three-term recurrence generates each new basis vector from the previous two.
(Terminology: *Krylov* is the family; **Lanczos** is its Hermitian member, and
**Arnoldi** the general-matrix one. "ARPACK" is the famous library implementing a
restarted Arnoldi/Lanczos.) Lanczos then does Rayleigh–Ritz (Idea 2) on that
basis. Two limitations matter for us:

- **No preconditioner.** A Krylov subspace is rigidly determined by powers of `H`
  alone; there is no place to inject approximate physics such as "the diagonal
  potential is the dominant part." If the low-energy states are close together,
  repeated powers of `H` separate them only slowly, so Lanczos may need many
  matvecs before the individual low-lying eigenvectors are cleanly resolved.
- **Extremal bias.** As noted, plain Lanczos resolves the outermost eigenvalues
  first; the clustered bottom states converge last.

Lanczos is robust and assumption-free, which makes it the natural baseline — but
it leaves the structure of `H` (that diagonal-vs-kinetic split) completely
unused.

**Davidson**
Davidson keeps the Rayleigh–Ritz skeleton but changes the answer to "which
vectors go into `V`": instead of the next power `Hx`, it appends the
**preconditioned residual** `M⁻¹ r` of the current best estimate (Idea 4). The
subspace it builds is therefore *not* a Krylov subspace — it's adaptively bent
toward the eigenvectors using whatever knowledge of `H` is baked into `M`. That
single hook, the freedom to choose `M`, is the entire reason to prefer it here.
The mechanics are in §1.4, and the case for *why* that freedom helps is §1.5.

**Three close relatives.** These are variations on the Davidson
idea kept in reserve rather than used here:

- **Jacobi–Davidson.** A more robust cousin: rather than applying the simple
  correction `M⁻¹ r`, it asks for a correction direction that is explicitly
  orthogonal to the current eigenvector estimate. That prevents the algorithm
  from wasting effort by moving along a direction it already has, and it often
  behaves better for interior or tightly clustered eigenvalues. The price is that
  each outer iteration contains an inner iterative solve, so it is sturdier but
  heavier. It is a natural upgrade path if plain Davidson stalls.

- **LOBPCG** (Locally Optimal Block Preconditioned Conjugate Gradient). Davidson's
  subspace *grows* every iteration, so its memory grows too. LOBPCG instead keeps a
  **fixed-size** working set: just three blocks of `k` vectors each — the current
  `k` eigenvector estimates, the previous ones, and the `k` preconditioned
  residuals — and does a small Rayleigh–Ritz on those `3k` vectors each step.
  Because it never stores more than ~`3k` vectors of length `N`, its memory is
  `O(k·N)` — constant in the iteration count — versus Davidson's `O(m·N)` for a
  basis that grows to `m` vectors. Its edge here would be *memory*, not speed, but
  it's more sensitive to a mediocre preconditioner and can lose accuracy at the
  tight `10⁻¹²` tolerance used here. 

- **Chebyshev-filtered subspace iteration.** Instead of accumulating a subspace,
  repeatedly apply a polynomial of `H` (a Chebyshev polynomial) chosen to *amplify*
  the low-energy states and *suppress* the rest, then do one small Rayleigh–Ritz.
  It trades *extra cheap matvecs* for *less* of the expensive subspace bookkeeping
  — which, as Part 2's profiling shows, is exactly the trade this GPU workload
  rewards.
  It's the most interesting structural alternative after the RBM path developed
  below.

### 1.4 Davidson Algorithm: Project, Measure, Expand

Davidson ties together all four ideas from §1.2. It keeps an
orthonormal basis `V = [v₁, …, v_m]` of the current subspace and, each cycle, does
three things: **project**, **measure**, **expand**.

**Initial guess eigenvectors.** The subspace has to be seeded with an initial
guess. A cold start uses a crude guess, such as a few unit vectors near the grid
points of lowest potential energy. But recall property (3):
neighboring `(R, P)` points have nearly identical eigenvectors. In production,
the code therefore **warm-starts** by seeding the basis with the converged
eigenvectors from the previous grid point. That guess is already close to the
answer, so Davidson often needs only a few cycles to polish it.

Suppose the current basis has `m` vectors and the calculation needs the lowest
`k` eigenpairs.

**Step 1 — Project (Rayleigh–Ritz).** Form the small `m × m` matrix
`H_eff = Vᵀ H V` and diagonalize it with a standard dense routine:

$$
H_{\text{eff}} = V^{*} H V \in \mathbb{C}^{m\times m}, \qquad
H_{\text{eff}}\, c_n = \theta_n\, c_n .
$$

This is Idea 2. The projection gives `m` Ritz pairs; the lowest `k` are kept. The
Ritz value `θ_n` is the current estimate of eigenvalue `n`, and the Ritz vector
`u_n = V c_n`
(a combination of basis vectors) is the current estimate of the eigenvector.
This is a projection method: it asks for the best approximate eigenvectors inside
the subspace built so far, with residuals orthogonal to that same
subspace. That is the Galerkin/Rayleigh–Ritz principle in Saad's language, and it
is why the reduced-basis projection in Part 2 gives real variational eigenvalue
estimates rather than plausible-looking guesses.

*Why it's cheap:* the implementation caches `AV = H V` as the basis grows, so
once a vector is in the basis, `H` is never re-applied to it; `H_eff` is then
just `Vᵀ(AV)`, a small tall-skinny GEMM, and the `m × m` diagonalization is
microseconds.

**Step 2 — Measure (the residual).** For each kept Ritz pair, compute the residual

$$
r_n = H u_n - \theta_n u_n = (AV)c_n - \theta_n (V c_n).
$$

Note this also reuses the cached `AV` — no new matvec. If `‖r_n‖` is below
tolerance, root `n` has converged and needs no further work. *Why this is the
right yardstick:* `r_n = 0` exactly characterizes a true eigenpair (Idea 3), so
`‖r_n‖` directly measures remaining error and tells us which roots still need
attention.

**Step 3 — Expand (the Davidson correction).** For each unconverged root, turn its
residual into a new search direction with the preconditioner (Idea 4):

$$
t_n = M^{-1} r_n = \big(\operatorname{diag}(H) - \theta_n I\big)^{-1} r_n .
$$

Then orthonormalize `t_n` against everything already in `V` (so the basis stays
well-conditioned) and append it. The subspace grows from `m` to `m + k`, then the
algorithm returns to Step 1 with a richer subspace — and therefore a better
Rayleigh–Ritz estimate — next time.

*Why this particular correction?* The exact fix to the eigenvector would solve
`(H − θ) t = r`. That solve is too expensive, so Davidson replaces
`H` by just its diagonal: `t = (diag(H) − θ)⁻¹ r`, which is a trivial elementwise
divide. The bet is that `diag(H)` captures enough of `H` to point the correction
in roughly the right direction. **How good that bet is depends entirely on how
much of `H` actually lives on the diagonal** — which is the crux of §1.5.

This is the practical preconditioning rule from iterative linear algebra: a useful
`M` must be cheap to invert, nonsingular, and close enough to the operator that it
shrinks the spread of the spectrum. When that happens, the residual correction is
better aligned with the true error, and the number of Davidson cycles drops. When
it does not, the preconditioner is just extra arithmetic.

**One practical wrinkle — restarts.** The basis can't grow forever; storing `m`
vectors of length `N` is the dominant memory cost (the `O(m·N)` from §1.3). So
the implementation caps `m` at some `max_space`. When the basis hits the cap, it
**restarts**: throw
away the accumulated basis but keep the current best `k` Ritz vectors as the new
seed, and continue. This bounds memory at the price of occasionally discarding
some accumulated subspace information. The size of that cap turns out to be a
dominant performance parameter: Part 2 shows that shrinking it from 300 to 100
cuts single-solve wall time by ~43%.

### 1.5 Why Davidson Fits This Hamiltonian

Lanczos is the canonical unpreconditioned matrix-free method: once the starting
vector is chosen, the subspace is determined by `x, Hx, H²x, ...`. Davidson is the
canonical preconditioned version: it still uses Rayleigh–Ritz, but it grows the
subspace using `M⁻¹r`, so the algorithm can inject an approximate inverse of the
operator.

With a constant do-nothing preconditioner, the Davidson correction is just a
scaled residual, and the method behaves like an unpreconditioned Krylov method.
Davidson only helps when `M⁻¹` is a meaningfully better approximation to
`(H - θI)⁻¹` than a scalar multiple of the identity.

The classic quantum-chemistry case where Davidson works spectacularly is
configuration interaction (CI). In a CI basis, the diagonal entries are the
energies of individual electronic configurations, while the off-diagonal entries
are couplings between different configurations. Usually one or a few reference
configurations dominate the low-energy state, and the couplings to the many other
configurations are comparatively small. That makes the CI Hamiltonian strongly
diagonally dominant in the basis where the calculation is written, so
`diag(H) - θI` is a surprisingly good cheap model of `H - θI`. This is why
Davidson was invented by quantum chemists and why it became their default
large-CI eigensolver.

Diagonal dominance is not a mystical property of the molecule; it is a property
of the representation. The same Hamiltonian can look diagonally dominant in one
basis and much less so in another. On a real-space grid like ours, the kinetic
energy is a finite-difference stencil, so it explicitly spreads amplitude to
neighboring grid points. That weakens diagonal dominance relative to CI. On the
other hand, the position-dependent potential is exactly diagonal, and in the full
Hamiltonian it is large enough that the simple Jacobi preconditioner is still a
reasonable model. A useful mental check is Gershgorin's theorem: if the diagonal
entry in a row is large compared with the sum of the off-diagonal couplings in
that row, the eigenvalues stay close to the diagonal entries, and a diagonal
preconditioner has a real chance.

The reason to choose Davidson here has two parts. First, the full Hamiltonian has
enough diagonal structure that the cheap Jacobi correction is not absurd. Second,
Davidson leaves room to try better structure-aware corrections
when Jacobi is not enough. Part 2 tests exactly that idea with a Fast
Diagonalization preconditioner: because the kinetic operator is separable on a
tensor-product grid, its one-dimensional stencil matrices can be diagonalized
once and reused to apply an approximate inverse cheaply. That preconditioner
ended up not improving this problem, but Davidson gave enough flexibility to test
it.

Davidson also matches the other two pieces of structure in this calculation:
several nearby roots are needed at once, so a blocked method is natural; and the
code solves neighboring `(R, P)` points, so warm starts are extremely valuable.
Those features, plus preconditioner flexibility, are the through-line into Part 2.

### 1.6 Solver Comparison


| Method | Matrix-free? | Memory | Preconditionable? | Targets lowest few? | Robust to `1e-12`? | Verdict for this problem |
|---|---|---|---|---|---|---|
| Dense (LAPACK / cuSOLVER) | No (needs full `H`) | `O(N²)` ≈ 64 TB | n/a | computes *all* | yes | Out — impossible on memory, work, and output |
| Sparse-direct / shift-invert | No (needs factorization) | huge (fill-in) | exact (it *is* the inverse) | yes | yes | Out — no factorization for a matrix-free `H` |
| Lanczos / ARPACK (Krylov) | Yes | `O(m·N)` | **No** | poorly (extremal bias) | yes | Robust baseline; uses none of `H`'s structure |
| **Davidson** | **Yes** | `O(m·N)` | **Yes (the key feature)** | **Yes** | **Yes** | **Chosen** — preconditioner flexibility + blocking + warm starts |
| Jacobi–Davidson | Yes | `O(m·N)` | Yes (+ inner solve) | yes (esp. interior) | yes | Upgrade path if Davidson stalls |
| LOBPCG | Yes | `O(k·N)` (constant) | Yes (sensitive) | yes | fragile at `1e-12` | Held in reserve for its small memory footprint |
| Chebyshev filtering | Yes | `O(k·N)` | optional | yes | yes | Best *structural* alternative after RBM |

*(`N` ≈ 2 million is the state length; `m` is the Davidson basis size as it grows
to its cap `max_space`; `k` ≈ 4 is the number of eigenpairs wanted. Davidson's
`O(m·N)` basis is the memory cost that the restart in §1.4 exists to bound.)*

**Bottom line.** Davidson is the right default here for one reason above all: it is
the only mainstream eigensolver that is simultaneously **matrix-free**,
**preconditionable with an arbitrary structure-aware `M`**, **naturally blocked**,
and **warm-startable** — the exact four levers this problem provides. Diagonal
dominance makes the default Jacobi preconditioner plausible; flexibility is what
lets the calculation go beyond Jacobi when profiling says it should. Part 2 uses
that flexibility by measuring where the time actually goes and then changing the
algorithm around that measured cost profile.


## Part 2 — Profiling-Driven Optimization and Production Results

In Part 1 we concluded that Davidson was chosen to be the optimal algorithm for this problem structure. Next we look at ways to profile first, and then optimize these calculations.

### 2.1 Profiling Result: Basis Operations Dominate

In classical quantum chemistry, Davidson is usually motivated by the fact that
applying `H` is expensive. Standard implementations spend algorithmic effort to
reduce the number of matvecs and tolerate some bookkeeping around them. On the
GPU implementation here, the profile said the opposite. 

NVTX profiling for Davidson steps for different subspaces:

| Operation | Subspace 300 | Subspace 200 | Subspace 150 |
|---|---:|---:|---:|
| **Complete solve** | **45.84 s** | **28.80 s** | **27.91 s** |
| Hamiltonian applications | 0.68 s | 0.66 s | 0.71 s |
| Ritz vectors & residuals | 16.73 s | 2.88 s | 1.41 s |
| Convergence checks & host sync | 5.98 s | 3.52 s | 3.31 s |
| Orthonormalization | 2.57 s | 1.82 s | 1.78 s |
| Complex-copy kernels | 6.71 s | 4.68 s | 4.56 s |
| `cudaMalloc` API time | 16.75 s | 2.77 s | 1.54 s |


We see that for a subspace size 300, all ~106 Hamiltonian applications together
took **0.69 s out of a 45.84 s solve**, about **1.5%** of wall time. Each matvec was
only ~6.5 ms. The remaining time was Davidson's basis machinery: reconstructing
Ritz vectors and residuals from a large basis, doing big complex GEMMs,
orthogonalizing against the current basis, copying large arrays, allocating
temporaries, and synchronizing the CPU with the GPU. The single largest GPU kernel
was a large complex GEMM from the basis work, not from applying `H`.

One profiling caveat matters here. GPU work is asynchronous: CuPy can queue kernels
and return to Python before the GPU has finished them. An NVTX range that ends at a
synchronizing operation may inherit time from earlier queued work. For example, in the raw NVTX trace, `diagonalize Heff` range reported ~20 s regardless of subspace size, even though a
true `m × m` diagonalization for `m = 150–300` should not dominate a 45 s solve.
The reliable quantities are complete solve wall time, total GPU kernel time,
kernel identities, cycle counts, and controlled comparisons between runs. Single
NVTX ranges are useful clues, but only after checking them against the full solve.

The Hamiltonian applications row barely moves and is tiny since `H x` is a regular, bandwidth-friendly GPU operation. The expensive part is moving and combining many large basis vectors.

Therefore, a useful model is:

$$
\text{solve time} \approx (\text{number of cycles}) \times
(\text{per-cycle basis cost}).
$$

That gives two independent levers: shrink the per-cycle basis cost, and reduce the
number of cycles. The production speedup comes from doing both.

### 2.2 Optimization 1: Reduce Per-Cycle Basis Cost

The Davidson carries essentially two `max_space × N` arrays: the basis
`V` and its image `H V`. Large `max_space` values give the solver more room before
restart, but every cycle becomes more expensive because the Ritz reconstruction,
orthogonalization, copies, and temporary allocations all touch more vectors.

Sweeping the requested subspace showed that the old production value, 300, was too
large for this GPU cost model:

| Requested subspace | Cycles | Solve time | vs. 300 |
|---:|---:|---:|---:|
| 300 | 105 | 45.84 s | baseline |
| 200 | 106 | 28.80 s | -37% |
| 150 | 111 | 27.91 s | -39% |
| 125 | 114 | 26.5 s | -42% |
| 100 | 113 | 26.0 s | -43% |
| 75 | 122 | 26.9 s | -41% |

Going from 300 to 100 nearly halved the single-solve time while adding only eight
cycles. The supporting profiler rows above explain why: the Ritz-vector and
residual reconstruction collapsed, and `cudaMalloc` time fell by an order of
magnitude. The basis had been large enough to create allocator pressure and large
GEMM/copy costs. Around **100** vectors was the useful compromise: small enough to
make each cycle cheap, but not so small that convergence dropped.

### 2.3 Optimization 2: Improve the Initial Subspace

The second lever is cycle count. Neighboring phase-space points have very similar
low-energy eigenvectors, so a warm start from the previous `P` point already helps:
a cold solve took 376 cycles, while a previous-`P` warm start took 111 cycles on
the same point. That observation is more than a speed trick; it says the
eigenvectors along a `P` sweep live near a low-dimensional subspace.

The reduced-basis method (RBM) turns that observation into an algorithm. Instead
of starting from only the previous solution, store a small local reduced basis of
converged eigenvector blocks from recently solved `P` points. For the next `P`,
solve a tiny Rayleigh–Ritz problem in that reduced subspace, use the resulting
linear combination as Davidson's initial guess, and then let Davidson polish to
the usual residual tolerance.

This works especially well because the Hamiltonian is affine in the momentum
inside a `P` sweep:

$$
H(P) = A + p C,
$$

where `A` and `C` are fixed and only the scalar `p` changes. That structure is the
same reason reduced-basis ideas work for parametric eigenproblems: a few snapshots
can predict the next nearby eigenvector very accurately.

Davidson still performs the final residual check, so RBM cannot silently change
the answer. A bad reduced-basis guess only costs extra polish cycles. A good one
removes most of the solve.

### 2.4 Benchmark: 8.9× at the Same Tolerance

The production test used one full split: **1,602 Davidson solves** from the
phase-space sweep (`erf_coulomb` potential, full spin-orbit coupling,
`90×90×90` spatial grid, 89 internuclear points, lowest `k = 4` states). The same
points were solved three ways:

1. **Old baseline:** no RBM, Davidson subspace 300.
2. **Fresh control:** no RBM, Davidson subspace 100.
3. **RBM-polished:** subspace 100 plus reduced-basis initial guesses.

![Cumulative Davidson time across the split for the three configurations: 7.00 h, 4.12 h, and 0.79 h.](/assets/triple_cumulative_davidson_time.png)

The cumulative Davidson time over all 1,602 solves:

| Configuration | PS Davidson time | Speedup vs. previous | Speedup vs. baseline |
|---|---:|---:|---:|
| Old baseline (subspace 300) | **7.00 h** (25,218 s) | — | 1.00× |
| Fresh control (subspace 100) | **4.12 h** (14,819 s) | 1.70× | 1.70× |
| RBM-polished (subspace 100) | **0.79 h** (2,831 s) | 5.23× | **8.91×** |

This is the whole story in one table. Shrinking the subspace bought **1.70×** by
making each cycle cheaper. RBM then bought another **5.23×** by cutting the number
of cycles. The two factors multiply: `1.70 × 5.23 ≈ 8.9×`. On this single split,
the PS Davidson time fell from 7.00 h to 0.79 h, saving **6.2 GPU-hours**. The full
split loop, including Born–Oppenheimer solves and bookkeeping, improved by **6.8×**.

The speedup is not uniform, and that non-uniformity is exactly what the algorithm
predicts. RBM first has to build a local reduced basis. The first point on each
sweep branch has little or no history and therefore runs close to the ordinary
warm-start cost. After two snapshots, the reduced basis is accurate enough that
the bulk of the grid collapses to a small number of Davidson polish cycles.

![Davidson cycle-count heatmaps over the R/P grid for the three configurations; RBM collapses the bulk to ~17 cycles.](/assets/performance_split1/blog_figures/cycle_heatmaps_three_way.png)

The two no-RBM panels are a nearly uniform ~84 cycles everywhere. The RBM panel is
almost entirely dark (~17 cycles), except for two bright vertical stripes — the
center seed and the branch-reset column. The pattern reflects the algorithm:
full-cost solves occur only where the local reduced basis has not yet been formed;
the remaining grid points require only a small number of polish cycles.

### 2.5 Output: The Phase-Space Energy Surface

All of this exists to produce one object: the phase-space energy surface
`E(R, P)` that the new physics method needs. For this split, the ground and
first-excited surfaces were computed in **47 minutes instead of 7 hours**:

![Ground-state and excited-state phase-space energy surfaces over the populated R/P region for split 1.](/assets/energy_surfaces_gs_es.png)

(Only the `R` rows belonging to this split are populated; the full surface is the
union of all five splits.) This is the payoff of the entire numerical effort: the
surface that was previously a multi-day computation per parameter set is now cheap
enough to compute across the sweeps of spin-orbit strengths, angular momentum values, and mass ratio that validating the physics actually requires.

### 2.6 Technical Takeaways

Most of the performance gain came from a smaller Davidson basis and a much better
initial subspace.

- **Initial reduced-basis construction has a fixed cost.** The first solve on each
  branch always pays nearly the full Davidson cost because no local reduced basis
  exists yet. With ~90 such points out of 1,602 here, that cost is small, but it
  would matter more on a much shorter sweep.
- **The reduced basis is local.** The basis is reset at the branch jump rather than
  mixing snapshots from opposite sides of the sweep, because a local reduced
  subspace predicts far better than a global one. This is a deliberate
  accuracy-for-generality trade.
- **Chebyshev filtering remains interesting.** The profile says extra cheap
  matvecs may be worth spending if they reduce basis bookkeeping. RBM produced
  most of the performance gain with less implementation risk, but the profiling
  result points to Chebyshev-filtered subspace iteration as the natural next
  structural alternative.

