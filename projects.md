---
layout: default
title: Projects
permalink: /projects/
---

<h1>Projects &amp; technical writing</h1>

<p>Selected work in GPU computing, numerical linear algebra, and high-performance scientific computing. Each entry links to a writeup, the code, or both.</p>

<div class="featured">
  <p class="eyebrow">Featured</p>
  <h3>Diagonalizing a 2-million-dimensional quantum Hamiltonian on a GPU</h3>
  <p>A matrix-free, GPU-accelerated eigensolver for a three-body quantum-scattering problem. The Hamiltonian is never formed explicitly — the solver applies it as an operator — which makes a 2-million-dimensional eigenproblem tractable on a single GPU.</p>
  <p>Key ingredients: <strong>Davidson iteration</strong> for the lowest eigenpairs, <strong>warm starts</strong> across a parameter sweep, and <strong>reduced-basis</strong> methods to shrink the work per solve. I profiled the whole pipeline with <strong>NVTX</strong> and <strong>Nsight</strong> to find and remove the real bottlenecks, landing an <span class="metric">8.9×</span> production speedup at machine precision.</p>
  <ul class="tags">
    <li>matrix-free</li>
    <li>Davidson</li>
    <li>CUDA</li>
    <li>NVTX / Nsight</li>
    <li>reduced basis</li>
    <li>eigensolvers</li>
  </ul>
  <p class="actions">
    <a href="{{ '/writing/gpu-eigensolver/' | relative_url }}">Read the full writeup</a>
    <a href="https://github.com/{{ site.github_username }}">View the code</a>
  </p>
</div>

<h2>More work</h2>

<div class="item">
  <h3>Tensor networks &amp; belief propagation — Princeton NVIDIA Hackathon</h3>
  <p class="meta">Princeton &times; NVIDIA Open Hackathon</p>
  <p>A GPU-accelerated approach to contracting tensor networks using belief-propagation-style message passing. [One or two sentences on what you built, the result, and what you learned — e.g. the speedup, the library you used, or the part you owned on the team.]</p>
  <ul class="tags">
    <li>tensor networks</li>
    <li>belief propagation</li>
    <li>GPU</li>
  </ul>
  <p class="actions">
    <a href="https://github.com/{{ site.github_username }}">Code / slides</a>
  </p>
</div>

<!-- Duplicate this block to add a project. Delete it if you don't need it. -->
<div class="item">
  <h3>[Project title]</h3>
  <p>[One or two sentences: what problem, what method, what result. Lead with the number or outcome that a GPU/HPC reader would care about.]</p>
  <ul class="tags">
    <li>tag</li>
    <li>tag</li>
  </ul>
  <p class="actions">
    <a href="#">Writeup</a>
    <a href="#">Code</a>
  </p>
</div>
