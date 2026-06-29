---
layout: default
title: Home
permalink: /
---

<h1>Computational physics, numerical methods, and GPU computing.</h1>

<p class="lede">I build large-scale iterative solvers and squeeze real performance out of them on GPUs. My recent work centers on matrix-free eigensolvers, numerical linear algebra, and profiling-driven optimization for scientific computing.</p>

<p>I'm a computational physicist / computational chemistry researcher moving toward GPU / HPC and numerical linear algebra roles. I care about the part of scientific computing where the math, the hardware, and the wall-clock time all have to agree.</p>

<ul class="tags" aria-label="Areas of focus">
  <li>numerical linear algebra</li>
  <li>GPU computing</li>
  <li>iterative eigensolvers</li>
  <li>matrix-free / sparse methods</li>
  <li>performance profiling</li>
  <li>HPC</li>
</ul>

<div class="featured">
  <p class="eyebrow">Featured writeup</p>
  <h3>Diagonalizing a 2-million-dimensional quantum Hamiltonian on a GPU</h3>
  <p>A matrix-free, GPU-accelerated eigensolver for a three-body quantum-scattering problem: Davidson iteration with warm starts and reduced-basis methods, profiled end-to-end with NVTX and Nsight — for an <span class="metric">8.9×</span> production speedup at machine precision.</p>
  <p class="actions"><a href="{{ '/projects/' | relative_url }}">Read the writeup</a></p>
</div>

<h2>Start here</h2>

<div class="btn-row">
  <a class="btn" href="{{ '/projects/' | relative_url }}">Projects &amp; writing</a>
  <a class="btn secondary" href="{{ '/research/' | relative_url }}">Research</a>
  <a class="btn secondary" href="{{ site.cv_pdf | relative_url }}">Download CV (PDF)</a>
</div>

<p class="inline-links">
  <a href="https://github.com/{{ site.github_username }}">GitHub</a>
  <a href="https://www.linkedin.com/in/{{ site.linkedin_handle }}/">LinkedIn</a>
  <a href="mailto:{{ site.email }}">{{ site.email }}</a>
</p>
