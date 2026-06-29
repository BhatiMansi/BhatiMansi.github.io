---
layout: default
title: Diagonalizing a 2M-dimensional quantum Hamiltonian on a GPU
permalink: /writing/gpu-eigensolver/
---

<h1>Diagonalizing a 2-million-dimensional quantum Hamiltonian on a GPU</h1>

<p class="meta" style="color:var(--muted)">Matrix-free, GPU-accelerated eigensolver · Davidson iteration · NVTX / Nsight profiling</p>

<p class="lede">A matrix-free, GPU-accelerated eigensolver for a three-body quantum-scattering problem — Davidson iteration with warm starts and reduced-basis methods, profiled end-to-end, for an <span class="metric">8.9×</span> production speedup at machine precision.</p>

<!--
  This is a single standalone page, not a blog system — exactly what you asked for.

  Two ways to use it:
    1. Paste your full writeup below (it's plain Markdown — headings, code blocks,
       images all work). Drop figures in /assets/ and reference them with
       ![caption](/assets/figure.png).
    2. OR delete the body and just link out: change the "Read the full writeup"
       links on the home and projects pages to point at your existing article
       (Medium, lab blog, etc.), and remove this file.

  Suggested section skeleton kept below to get you moving.
-->

<h2>The problem</h2>
<p>[What the three-body quantum-scattering problem is, in two or three sentences a GPU/HPC reader can follow. Why the Hamiltonian reaches ~2 million dimensions, and why forming it explicitly is a non-starter.]</p>

<h2>Matrix-free approach</h2>
<p>[How you apply the Hamiltonian as an operator instead of storing it, and what that buys you in memory and bandwidth.]</p>

<h2>Davidson iteration, warm starts, reduced basis</h2>
<p>[Why Davidson for the lowest eigenpairs. How warm starts across the parameter sweep cut iteration counts. What the reduced-basis step removes from the per-solve cost.]</p>

<h2>Profiling: NVTX + Nsight</h2>
<p>[The bottlenecks you found, what the timeline looked like before vs after, and the specific changes that moved the needle.]</p>

<h2>Result</h2>
<p>[The <span class="metric">8.9×</span> speedup in context: what "production" means here, and how you confirmed machine-precision agreement with the baseline.]</p>

<p class="actions" style="margin-top:1.5rem"><a href="https://github.com/{{ site.github_username }}">View the code on GitHub →</a></p>
