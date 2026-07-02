---
layout: default
title: GPU-Accelerated Cluster-Corrected Belief Propagation for Tensor Networks
permalink: /writing/tensor-network-gpu/
mathjax: true
---

# GPU-Accelerated Cluster-Corrected Belief Propagation for Tensor Networks
Mansi Bhati

*A hackathon project accelerating the tensor-network contraction algorithm developed by **Siddhant Midha** (Princeton Quantum Initiative) and **Frank Zhang** (ECE), with mentors **Lars Nyland** (NVIDIA) and **Congyue Cui** (Princeton), during the Open Hackathons program. Implementation in **Julia + CUDA.jl** on an **NVIDIA A100 80GB PCIe**.*

---

## 1. The problem: contracting tensor networks is (very) hard

Tensor networks are one of the most important representations in modern computational physics and chemistry. The idea is simple: replace a giant, exponentially large object — a partition function, a wavefunction, a probability distribution — with a *graph* in which each vertex holds a small tensor and each edge represents a shared index. The full object is recovered by **contracting** the network, i.e. multiplying all the tensors together and summing over every shared index.

They show up almost everywhere physicists care about many degrees of freedom at once:

- **Statistical mechanics.** The Ising model, the Potts model, and almost any classical lattice model can be written as a tensor network whose contraction is exactly the *partition function*, $$Z = \sum_{\text{states}} e^{-\beta H}$$. Why do we care about $$Z$$? Because once you know $$Z$$ you know essentially everything about the thermodynamics of the system: the free energy is $$F = -k_B T \log Z$$, correlation functions and magnetisations are derivatives of $$\log Z$$, and phase transitions show up as singularities in $$\log Z$$. Computing $$Z$$ is the whole game.
- **Quantum many-body physics.** Ground states and time evolution of strongly correlated electron systems, spin liquids, and superconductors are naturally represented by tensor-network ansätze like MPS, PEPS, and MERA. Contracting these networks is how one extracts expectation values, entanglement entropies, and dynamical correlators.
- **Quantum chemistry.** Electronic-structure methods such as DMRG, tensor-hyper-contraction, and coupled cluster with tensor decompositions turn intractable Fock-space sums into contractions of small-tensor networks whose topology reflects the molecule.
- **Quantum computing and error correction.** Simulating a quantum circuit is itself a tensor-network contraction: every gate is a tensor and every qubit-timeline is a bond. In quantum error correction, decoders need *syndrome probabilities* — given a set of parity-check measurement outcomes (the "syndrome"), what is the probability of the underlying error pattern, or, equivalently, what is the most likely error? Computing that probability is a marginalisation of a large tensor network over its bond indices, and it is the bottleneck of every modern surface-code-style decoder.

![Tensor network contraction](/assets/blog_images/fig_tensor_network.png)

Formally, on a graph $$G=(V,E)$$ with a tensor $$T_v$$ at each vertex, the contraction is

$$
Z \;=\; \sum_{\text{all bond indices}} \; \prod_{v \in V} T_v .
$$

### A concrete example: the 2D classical Ising model

To make "vertex", "edge", and "$$\#P$$-hard" concrete, take the **2D classical Ising model** on a periodic $$L \times L$$ square lattice. At each site sits a spin $$s_v \in \{+1,-1\}$$, and neighbouring spins interact through a Boltzmann weight $$e^{\beta s_u s_v}$$. The partition function is

$$
Z \;=\; \sum_{\{s_v\}} \prod_{(u,v)\in E} e^{\beta\, s_u s_v}.
$$

There is a standard trick to encode this exactly as a tensor network: at each site put a rank-4 tensor $$T_v$$ whose four indices correspond to the four adjacent bond variables, and on each edge put a $$2 \times 2$$ matrix $$e^{\beta \sigma \sigma'}$$. Contracting this network *is* $$Z$$. Here the physical meaning is immediate: **vertices are Ising spins**, **edges are nearest-neighbour interactions**, and the **bond dimension** $$\chi$$ counts the number of local states ($$\chi=2$$ for Ising).

The catch is that exact evaluation is `#P`-hard. The sum has $$2^{L^2}$$ terms — for a modest $$20 \times 20$$ lattice that is already $$2^{400} \approx 10^{120}$$ configurations, which no computer will ever enumerate. Even a smarter contraction schedule that leaves the outer sum implicit still runs into intermediate tensors whose size grows with the *treewidth* of $$G$$, and for 2D lattices the treewidth grows with $$L$$. So in practice we approximate.

This blog is about one such approximation — **Belief Propagation (BP) with cluster corrections**, developed by Sid and Frank [SM+FZ 2510.02290] — and how we ported the entire pipeline to a single GPU, ending up with roughly **1000× speedups** on both the message-passing step and the loop-contraction step.

---

## 2. The algorithm in one page

### 2.1 Belief propagation

The starting point is a classic trick from graphical-model inference: **if the underlying graph $$G$$ were a tree, the partition function factorises exactly**, and it can be computed by passing "messages" along the edges. On a tree, each vertex sends its neighbour a summary of everything that lives behind it, and after one sweep in each direction every vertex knows the exact contribution from the rest of the graph. That is the origin of the name *belief propagation*.

The BP recipe transplants this idea to graphs *with* loops as an approximation. On every undirected edge $$(i, j)$$ we keep **two directed messages**, $$\mu_{i \to j}$$ and $$\mu_{j \to i}$$. Each message is a length-$$\chi$$ vector ($$\chi$$ is the bond dimension of the network — 2 for Ising). The BP update recomputes each directed message by contracting the source vertex tensor $$T_v$$ against every *incoming* message *except* the one on the outgoing edge:

$$
\mu^{(k+1)}_{v \to w} \;\propto\; \sum_{\text{legs of } T_v} T_v \prod_{u \in \partial v \setminus \{w\}} \mu^{(k)}_{u \to v},
$$

and then renormalises. The intuition is straightforward: the message that $$v$$ sends to $$w$$ should summarise the whole graph *minus* $$w$$, viewed from $$v$$.

On a tree this converges in a single sweep in each direction and is exact. On a graph with cycles it becomes an **approximate fixed-point iteration**: you keep applying the update until the messages stop changing (below some tolerance), and the resulting messages define an approximate partition function

$$
Z_{\text{BP}} \;=\; \prod_v z_v \Big/ \prod_e z_e,
$$

where each $$z_v$$ and $$z_e$$ is a small local contraction against the converged messages. BP is very cheap — one iteration touches every edge once — and it is known to give an excellent answer whenever $$G$$ is "tree-like enough", i.e. whenever loops in the graph do not create too much correlated feedback. For 2D lattices near a phase transition, however, loops matter and BP alone is systematically wrong. That is where cluster corrections come in.

### 2.2 The cluster correction

Every deviation of the true $$\log Z$$ from $$\log Z_{\text{BP}}$$ can be written as a sum over **loops** in $$G$$ — subgraphs in which every included vertex has degree at least 2, so there are no dangling edges. Each loop $$\ell$$ contributes a small tensor contraction $$Z_\ell$$, evaluated by projecting the BP messages out of each edge (via projectors $$P_\perp$$) and contracting the resulting mini-network on the loop's support.

Single loops are only the leading-order correction. The full expansion groups loops into **clusters** — connected multisets of loops that share vertices — and weights them by their Ursell combinatorial coefficients $$\phi(C)$$. The end result is

$$
\log Z \;=\; \log Z_{\text{BP}} \;+\; \sum_{C \in \mathcal{C}} -\phi(C) \prod_{\ell \in C} Z_\ell,
$$

where

- $$Z_{\text{BP}}$$ is the mean-field-like baseline from the BP fixed point,
- each $$\ell$$ is a loop in $$G$$ (a subgraph with no dangling edges),
- $$Z_\ell$$ is the small tensor contraction supported on $$\ell$$, evaluated against the converged BP messages, and
- $$C$$ is a cluster (a multiset of loops touching each other) with combinatorial weight $$\phi(C)$$.

The more (and the larger) loops and clusters you include, the more accurate the approximation gets — at the cost of more work.

![Belief propagation with cluster corrections](/assets/blog_images/fig_belief_propagation.png)

### 2.3 Three computational stages

Getting from a raw lattice to $$\log Z$$ means going through **three** distinct computational stages, each with a very different flavour:

1. **Preprocessing (offline).** Enumerate all loops up to a chosen weight, translate them across the lattice, and group them into connected clusters. This is a combinatorial graph-search problem — no floating point at all. Expensive, but you only pay it once and cache the result to disk.
2. **Belief propagation (online).** Iteratively solve the fixed-point equation above for the messages living on the directed edges of $$G$$. Dense linear-algebra-flavoured, with a *lot* of independent work per iteration.
3. **Loop + cluster contraction (online).** Use the converged BP messages to compute one number $$Z_\ell$$ per unique loop, then combine those numbers according to each cluster. Thousands of small, oddly-shaped tensor contractions — the awkward middle child between BLAS and graph search.

The plots throughout the rest of this post live at each of these stages.

---

## 3. Stage 1 — Preprocessing: MPI-parallel loop and cluster enumeration

Enumeration is the "offline" stage. The exact steps we run for a periodic $$L \times L$$ square lattice are:

**Step 1: Enumerate all loops passing through the center site.**

1. Build the periodic $$L \times L$$ lattice.
2. Start from a fixed reference vertex (the center) and use **BFS** to grow partial edge sets one edge at a time, up to `max_loop_weight` edges. A state is recorded as a valid loop when every participating vertex has degree at least 2 (no dangling vertices) and the subgraph contains the reference vertex. Branches that cannot bring all degree-deficient vertices up to degree 2 within the remaining edge budget are pruned.
3. While searching, states are deduplicated using the D$$_4$$ symmetry of the square lattice, so rotated and reflected copies are not re-explored. (This is a square-lattice optimization; on a general tensor network the enumeration becomes more expensive.)

*Output:* every loop up to weight $$w$$ that passes through the center site.

**Step 2 — Translate those loops to cover the whole lattice.**

1. Translate the center-site loops over all lattice translations of the periodic $$L \times L$$ lattice, which generates all loops of the allowed weights anywhere on the lattice.
2. Deduplicate them and assign each unique loop an integer ID.

*Output:* the full deduplicated loop set, where each loop has an integer ID.

**Step 3 — Build the loop interaction graph, then grow clusters on the target site.**

1. Build the interaction graph whose nodes are loop IDs. Two loops are connected if they share at least one vertex. Then identify the loops whose vertex sets contain the target site (specified by the user).
2. Use **DFS** to grow clusters: starting from supported loops, extend the current cluster by adding loops adjacent in the interaction graph, while keeping total cluster weight at most `max_weight`. This guarantees every enumerated cluster is connected and contains at least one loop touching the target site. The DFS also allows repeated loop IDs, represented as loop multiplicities.

*Output:* all connected clusters touching the target site.

**Step 4 — Deduplicate clusters under translation.** For each cluster, compute a translation-invariant canonical signature and keep only one representative from each translation equivalence class.

**Step 5 — Save.** Persist the clusters, the loop list, the loop-interaction adjacency matrix, parameters, and metadata to disk so the online pipeline never has to redo this work.

Here is what BFS and DFS look like, as a reminder of why we pair them:

![DFS vs BFS](/assets/blog_images/Tensor_bandits_scrum_1_p6.png)

**BFS vs. DFS, briefly.** *Breadth-first search* explores the search tree one level at a time: enumerate all length-1 partial subgraphs first, then all length-2, and so on. It naturally lists candidates in order of size and is exhaustive for loops up to a given weight. *Depth-first search*, by contrast, plunges down one branch of the tree at a time and only backtracks when it dead-ends — a good fit for building larger objects incrementally.

Our pipeline uses a **hybrid**. Different stages naturally want different strategies:

- **Step 1 (loop enumeration)** wants BFS. We are asking "give me every loop up to weight $$w$$", and BFS delivers those in size-order for free while pruning branches whose remaining edge budget cannot repair every dangling vertex.
- **Step 3 (cluster growth)** wants DFS. A cluster is built one loop at a time, and the natural way to explore that space is to add-a-loop, recurse, then backtrack — classic DFS.

This BFS/DFS split is exactly what lets Step 1 be **parallelised cleanly across MPI ranks**. The coordinator rank runs a shallow BFS to generate partial-subgraph *seeds* (say, the first 3–4 edges of a candidate loop). Each worker rank then does a **DFS expansion** from its assigned seed, exploring all completions of that partial subgraph into a valid loop and reporting back canonical loop IDs. Because the seeds are disjoint slices of the BFS frontier, no two workers ever explore the same subtree, and merging is just a set union at the end. That is why the profile plot on the next page labels the worker time as "DFS expansion" — those workers really are doing DFS, they are just doing it from BFS-generated starting points.

### Why enumeration is the expensive step

Even though it is offline, this stage completely dominates the wall-clock cost of setting up a new problem size. A Julia + Nsight Systems profile for $$L=20$$, max weight $$w=10$$ makes that obvious:

![NVTX profile of the enumeration pipeline](/assets/blog_images/fig_profiler_loop.png)

`step 1: center site loop generation` alone takes **733.8 s**, i.e. **94.4 %** of the total enumeration time. Every other step (translation, interaction-graph build, DFS cluster growth, canonical dedup) is a rounding error next to it. So the honest speed-up target here is step 1.

### Parallelizing the enumeration with MPI

Step 1 is not a good fit for a GPU — the workload is combinatorial, branchy, memory-heavy, and largely integer-valued — but it *is* embarrassingly parallel in the right coordinates. Different BFS **seed subgraphs** grow into disjoint parts of the search tree, so if we split the seeds across MPI ranks, they can enumerate independently and merge canonical loop sets at the end.

We parameterize the MPI split as `1+K`: one *coordinator rank* that produces seeds by running a shallow BFS and hands them out, plus $$K$$ *worker ranks* that keep expanding those seeds into full loops. Increasing $$K$$ is what actually buys us speed.

The empirical scaling is close to ideal:

![One-site loop enumeration time vs. order and MPI parallelization factor](/assets/blog_images/fig_mpi_loop_enum.png)

On the left, the old serial enumeration explodes past order 8 (well over a thousand seconds at order 9), while the MPI versions stay near-flat. On the right, we compare MPI configurations against the `1+1` baseline: at max loop order 10, `1+10` reaches roughly a **10× speedup** — essentially perfect linear scaling — and `1+5` reaches about **5.5×**.

The rank-by-rank profile shows this is because the workers really do spend almost all their time in DFS expansion, not in MPI communication or bookkeeping:

![MPI profile category share by rank](/assets/blog_images/fig_mpi_profile.png)

Rank 0 (the coordinator) spends a big chunk in MPI receives, canonical parenting, and BFS seed generation — expected, since it's dispatching work — while ranks 1–5 (the workers) spend most of their time in DFS expansion and canonical-parent hashing. In other words: the workers are compute-bound, communication is a thin band, and there is no serialisation bottleneck. That is why we get near-ideal scaling in the plot above.

**Takeaway.** Preprocessing is a graph-enumeration problem, not a linear-algebra problem, so we treat it as MPI-parallel across ranks rather than GPU-parallel across threads. That already turns a job that would have taken over a thousand seconds into something that fits comfortably in a coffee break.

---

## 4. Stage 2 — Belief Propagation: one GPU thread per directed message

Once the loops and clusters are cached on disk, everything that follows is floating point and *very* parallel. This is where the GPU starts to earn its keep.

Recall the BP fixed-point update from §2.1: for each directed edge $$v \to w$$,

$$
\mu^{(k+1)}_{v \to w} \;\propto\; \sum_{\text{legs of } T_v} T_v \prod_{u \in \partial v \setminus \{w\}} \mu^{(k)}_{u \to v}.
$$

Three things about this update are golden for a GPU:

1. **Every directed message is independent within an iteration.** With $$2L^2$$ undirected edges on a periodic $$L \times L$$ lattice, that is roughly $$4L^2$$ fully independent updates per iteration — a big pile of ready-to-launch work.
2. **Each update is small and uniform.** For bond dimension $$\chi$$ and vertex degree $$d$$, it is a $$\chi^d$$-sized sum with a fixed control-flow shape — perfect for warp-level SIMD, no branch divergence.
3. **The work is memory-bandwidth-bound.** Inside a single message update, each thread reads one vertex tensor entry and $$(d-1)$$ message entries per combination, and performs only $$(d-1)$$ multiplies and 1 add per combination. That is an *arithmetic intensity* of roughly $$\mathcal{O}(1)$$ FLOPs per byte read from memory. Kernels like this are throttled by how fast you can pull operands out of DRAM, not by how fast the ALUs can multiply. That plays directly to the GPU's strengths: an A100 has ≈1.5 TB/s of HBM2 bandwidth, roughly an order of magnitude more than a modern CPU's DDR bandwidth, so the ceiling on this kernel is about 10× higher on the GPU before we even count CPU-side overhead.

### Our CUDA kernel

We wrote a single CUDA.jl kernel, `bp_update_kernel!`, which **assigns one thread to one directed message**. On a periodic square lattice, one launch is `~4L^2` threads over a 256-thread block grid:

```julia
@cuda threads=256 blocks=cld(2*nedges, 256) bp_update_kernel!(
    new_messages, old_messages, deltas,
    tensor_d, offsets_d, src_d, out_slot_d, slot_in_dir_d, deg_d,
    Float64(alpha), Int32(2*nedges), chi,
)
```

Inside each thread, the update is a tight loop over `chi^deg` combinations, and the accumulator lives in registers rather than global memory (via `StaticArrays.MVector`). That register-resident accumulator is important — it means the hot loop reads the vertex tensor once and streams messages in, instead of ping-ponging through global memory.

We use the standard synchronous BP schedule: every thread reads from `old_messages`, writes to `new_messages`, and after the whole launch the two arrays are swapped. That way each iteration is a single kernel launch with no synchronization inside it beyond the natural block-level barriers.

### The GPU speedup on message passing

The custom CUDA kernel behaves exactly the way this argument predicts:

![Custom CUDA kernel for message passing: runtime and speedup](/assets/blog_images/fig_custom_cuda_bp.png)

The top panel is raw runtime for one BP fixed-point solve on the dense synchronous schedule. CPU time is a smooth quadratic-looking curve rising from a few seconds at $$L=100$$ to ~130 s at $$L=700$$. The GPU curve barely leaves the x-axis. The bottom panel plots the ratio: **speedup saturates at ~38× around $$L\ge400$$**, exactly where we expect the GPU to be fully occupied (~640K threads at $$L=400$$) and CPU-side overhead has amortized away.

For the **general vertex-dimension** BP path (which supports arbitrary $$\chi$$ and degree, at the cost of doing more work per thread), the story is similar but the pre-asymptotic ramp is different:

![Flat CPU vs. GPU-general speedup and runtimes vs. system size](/assets/blog_images/fig_flat_bp_speedup.png)

Speedup climbs to around **30×** and holds there across $$N=1.5\times10^4$$ to $$3\times10^4$$ sites. Runtimes on the right make the absolute magnitudes concrete: at $$N=3\times10^4$$, the CPU spends ~280 ms per fixed-point solve while the GPU is at ~10 ms.

And, crucially, GPU BP agrees with CPU BP to numerical precision:

![Free-energy and message agreement between CPU and GPU BP](/assets/blog_images/fig_accuracy.png)

Both the free-energy difference $$\lvert\Delta f\rvert$$ and the max message difference are pinned around $$10^{-15}$$ across all system sizes tested — the two implementations are converging to the same fixed point, they are just doing it at very different speeds.

The **big picture** message-passing plot from the top-level summary tells the same story on a log scale, and pushes the total observed speedup to about **1000×** once you factor in warm-cache steady-state timings on the largest instances:

![~1000× speedup on message passing and on loop contractions](/assets/blog_images/fig_1000x_speedup.png)

---

## 5. Stage 3 — Loop and cluster contractions: one CUDA block per loop, dense arrays per cluster bucket

This is the stage that made us really work for the speedup — and, honestly, the most fun part of the project.

### Why loop contractions are structurally awkward

The BP kernel is pleasant because every thread does *the same* amount of work: `chi^deg` operations, uniform for a given lattice. Loop contractions are the opposite. The loop set for one problem is a bag of subgraphs with **wildly different shapes**:

![Inherent parallelizability: clusters, loop tensors, and cluster-corrected BP](/assets/blog_images/fig_parallelizability.png)

A single job on a $$20\times20$$ periodic Ising lattice with max cluster weight 12 involves, from our CSV run:

- **302 unique loops** used out of a preprocessed pool of many more,
- **110,400 loop instances** after translating those loops around the lattice,
- **329 unique clusters** yielding **131,600 cluster instances**,
- **2,088,000 total contraction steps** across all compiled loop programs,
- a peak intermediate rank of 8 legs.

Each loop is a tiny irregular tensor network in its own right — a mix of edge factors (built from the BP projectors $$P_\perp$$) and vertex factors (BP messages absorbed on the non-loop legs). Two loops of the same edge-count can still have completely different contraction shapes, so this workload does not batch cleanly into a single GEMM.

Naively, if you tried to evaluate each loop by summing over the states of all its selected edges you would pay `chi^(2E)` per loop — for Ising ($$\chi=2$$) that's already `2^24` at loop order 12. That's the old "full state-sum" approach and it does not scale. The correct answer is to keep the tensor-contraction *structure* on the GPU, not just the summation.

### Trick 1 — Compile every loop into a "contraction program" on the CPU

Our GPU backend `gpu_general_v1_loop_contribution_batch` treats the GPU as a tiny **interpreter**. The heavy planning is done once on the CPU:

1. Build the flat edge factors from $$P_\perp$$.
2. Build the flat vertex factors with external BP messages absorbed.
3. Relabel factor legs to compact local IDs.
4. Choose a greedy contraction path that minimizes output rank first, then an estimated flop count.
5. Assign every initial factor and intermediate tensor a fixed slot in one flat workspace.
6. Store each contraction step as metadata: left/right/output slots, output rank, common rank, and leg-position maps for the output and the contracted indices.

The result is a small "byte-code" per loop: a list of steps like *"multiply slot A and slot B along their common indices, write into slot C, output rank 4, common rank 2, use these leg permutations"*. The GPU never has to think about topology at runtime; it just executes the plan.

This CPU compile step is visible as a distinct NVTX range in our profile (`step 2 | CPU compile GPU loop programs`) and is one of the reasons we insist on caching it — for a fixed problem size, you compile once and re-use across all the online reruns.

### Trick 2 — One CUDA block per loop, threads split the output entries

On the GPU we then launch **one block per loop**. Within a block:

- Threads cooperate on each contraction step by **splitting the output tensor entries** across the block. Thread `t` computes entries `t, t + blockDim, t + 2·blockDim, …`.
- For each output entry, that thread walks the space of the contracted (common) indices and accumulates the scalar sum.
- Consecutive contraction steps are separated by `CUDA.sync_threads()` because later steps read intermediates written by earlier ones.

Pseudocode:

```text
CPU:
    for each loop:
        build edge + vertex factors
        choose contraction path
        emit contraction-step metadata

GPU:
    launch one block per loop

    parallel block for loop_id = 1..nloops:
        for step in compiled_program(loop_id):
            parallel threads split output entries:
                for entry assigned to this thread:
                    sum over common contracted indices
                    write output tensor entry
            sync_threads()
```

This is very different from the CPU path, where each loop is contracted serially, one loop after another, one step after another. The block-per-loop layout gives us two nested levels of parallelism at once: across loops (via blocks) and within each contraction step (via threads), which is exactly the shape the A100's SM occupancy wants.

**Why this beats the alternatives we tried:**

- The **"full state-sum" kernel** we started with pays `chi^(2E)` per loop, and even at Ising $$\chi=2$$ it becomes unusable at loop order $$\ge 12$$.
- A **serialized reference kernel** (only thread 1 of each block does the work) exists in the code but only for correctness validation — it has almost no within-loop parallelism.
- **Topology bucketing** (grouping loops with identical shape and doing one launch per group) is great when loops naturally batch, but on non-uniform workloads it either shatters into tiny groups or forces expensive shape padding.

The interpreted "one block per compiled program" design keeps a single launch over all loops, does not care about topology mismatch, and lets threads pull useful work out of each contraction step.

### Trick 3 — Batch cluster products as dense broadcasts/reductions

Once every unique loop has its value $$Z_\ell$$, the cluster correction is just

$$
\Delta = \sum_{C \in \mathcal{C}} -\phi(C) \prod_{\ell \in C} Z_\ell^{m_\ell(C)},
$$

where $$m_\ell(C)$$ is the multiplicity of loop $$\ell$$ in cluster $$C$$. Clusters differ in how many distinct loops they contain, but *within* a bucket of clusters that all have the same "K distinct loops", the product is a fixed-shape reduction:

- pack loop IDs into a $$K \times n_{\text{clusters}}$$ matrix `loop_id_mat`,
- pack multiplicities into `mult_mat` of the same shape,
- gather `Z_l_array[loop_id_mat[k,c]]` and raise it to `mult_mat[k,c]`,
- reduce down each column to get one number per cluster,
- multiply by $$-\phi(C)$$ and sum.

We don't write a custom kernel for this — we let **CUDA.jl's array broadcast and reduction** do the work. The dense-matrix layout means every one of those steps is a well-shaped GPU launch, and it composes cleanly with the loop contractions upstream.

### The three levels of GPU parallelism, side by side

To recap the whole pipeline:

```text
BP iteration:      one GPU thread per directed message
Loop contraction:  one CUDA block per loop, threads split each contraction step
Cluster products:  dense broadcast/reduction over same-K cluster buckets
```

Each level matches its workload:

- BP is uniform and massively parallel across edges → **thread-per-item**.
- Loop contractions are heterogeneous but each is a mini-DAG → **block-per-item with cooperative threads**.
- Clusters are again uniform *within a bucket* → **array broadcast/reduction**.

That layering is what lets a single GPU stay busy end-to-end.

### The loop-contraction speedup

Going back to the summary plot, the loop-contraction column is the right half:

![Loop contraction time vs. lattice size, CPU vs. GPU](/assets/blog_images/fig_1000x_speedup.png)

CPU loop-contraction time grows from a few hundred seconds at $$L=10$$ to close to a thousand seconds at $$L=18$$. GPU stays in the tens-to-hundreds of milliseconds range. On a log scale, that's approximately **1000× speedup**, mirroring the message-passing panel on the left.

The single $$L=20$$, $$\beta=0.34$$, max-weight-12 run in our CSV is a good concrete data point:

| Stage | CPU | GPU (v1) |
| --- | ---: | ---: |
| BP fixed point (172 iters) | 0.10 s | 15.24 s* |
| Loop contraction execution | **237.56 s** | **2.43 s** |
| Cluster product assembly | 0.32 s | 0.28 s |
| **Total (BP + loop + cluster)** | **238.79 s** | **40.41 s** |

The CPU / GPU cluster correction values agree to the last digit: $$-7.277221798866\times 10^{0}$$ in both. The **~100× speedup on the loop contractions alone** is what dominates the total.

<sup>* The GPU BP time here is dominated by first-run JIT compilation and CUDA context setup, since $$L=20$$ is small enough that a single CPU BP iteration is already sub-millisecond. On the large-$$L$$ runs ($$L=100$$ through $$L=700$$) that overhead is amortized and the BP-per-iteration numbers match the plots above — this is why we always separate steady-state kernel time from warm-up time in Nsight Systems.</sup>

And here is the online-stage profile as it looks in Nsight Systems, where step 5 (cluster contributions) is the fat red bar and BP fixed point is the small green sliver next to it:

![NVTX profile of the online pipeline](/assets/blog_images/fig_profiler_cluster.png)

That imbalance is exactly why we spent so much of the GPU work on the loop-contraction interpreter rather than on BP.

---

## 6. What we learned about using the GPU well

The A100 gave us the ~1000× headline, but the number itself is a byproduct of a few specific design choices. If we had to distill them:

1. **Match the parallelism grain to the workload.** BP wants thread-per-message. Loop contractions want block-per-loop with intra-block cooperation. Cluster reductions want dense array ops. There is no single "GPU-ify this" recipe; there is a hierarchy of them.
2. **Move all planning off the hot path.** For loop contractions, we compile a per-loop contraction program *on the CPU* and ship pure metadata to the GPU. The kernel becomes an interpreter, not a solver — and interpreters can be launched once for tens of thousands of tiny problems.
3. **Keep hot data on-chip.** The BP kernel keeps its accumulator in registers via `MVector`; the loop kernel keeps intermediates in a pre-allocated flat workspace addressed by fixed slots. We only touch global memory when we have to.
4. **Use CUDA.jl where it's already excellent.** Cluster products are just gather + power + column reduction, and CUDA.jl's array primitives are perfectly good at that. Every custom kernel we don't write is a bug we don't own.
5. **Parallelize the offline step differently.** Loop enumeration is combinatorial, so we parallelized it with **MPI** rather than trying to force it onto the GPU. The `1+K` design gives near-ideal linear scaling up to at least `1+10`.
6. **Profile relentlessly.** Every design decision above came out of an Nsight Systems + NVTX trace showing us that (a) enumeration step 1 was 94% of preprocessing, (b) loop contractions were 99% of the online CPU time at $$L=20$$, and (c) BP was borderline free once we had the CUDA kernel. Without that visibility we would have optimized the wrong thing.

---

## 7. What's next

- **Anderson / Krylov acceleration of BP** to shrink the iteration count in the near-critical regime, especially where the BP Jacobian's spectral radius approaches 1.
- **Spectral certificates** for the cluster expansion: reporting, per run, whether the observed loop weights decay fast enough that the cluster correction is trustworthy.
- **Runtime diagnostics** (BP tail contraction ratio, simple-cycle transfer-matrix eigenvalue ratios, cluster cancellation ratios) so the pipeline can flag red / yellow / green regimes automatically.
- **Higher-dimensional networks** beyond the periodic 2D square lattice, where the loop enumerator has to work harder because the D$$_4$$ symmetry no longer applies.
- **ITensor.jl integration** so the community can drop cluster-corrected BP into existing quantum-many-body pipelines with one function call.

---

## 8. TL;DR

- We built a **GPU-accelerated cluster-corrected belief propagation** pipeline for tensor-network contraction, in **Julia + CUDA.jl** on an **NVIDIA A100 80GB**.
- **Preprocessing** (loop and cluster enumeration) is dominated by center-site loop generation; we parallelize it with **MPI** and get near-ideal linear scaling up to `1+10` workers.
- **Belief propagation** uses **one GPU thread per directed message**, with register-resident accumulators. That gives us up to a **~1000× speedup** on large lattices (~38× at moderate size and asymptoting there per-iteration).
- **Loop contractions** use **one CUDA block per loop**, with each loop pre-compiled on the CPU into an interpreted contraction program. Threads within a block split the output-tensor entries of each contraction step. This is ~**100× faster** than the CPU on our $$L=20$$ benchmark and roughly **1000×** on the sweep across $$L$$.
- **Cluster products** are packed into dense K-per-cluster matrices and evaluated via CUDA.jl array broadcasts and reductions — no custom kernel needed.
- **Correctness:** CPU and GPU pipelines agree on both the BP fixed point (to $$10^{-15}$$) and the final cluster correction (to the last printed digit).

The take-home is not really "GPUs are fast" — everyone knows that — but rather: **once you separate an algorithm into its parallelism regimes (uniform vs. structured-heterogeneous vs. combinatorial), you can pick a different parallelization strategy for each regime**, and the GPU speedups fall out.
