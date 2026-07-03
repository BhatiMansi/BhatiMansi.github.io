---
layout: default
title: Home
permalink: /
---

<section class="about-section" id="about">
  <div class="about-hero">
    <img class="profile-photo" src="{{ '/assets/fix_img.png' | relative_url }}" alt="Mansi Bhati">
    <div class="about-text">
      <h1 class="about-name">Mansi Bhati</h1>
      <p>I am a theoretical chemistry PhD student at Princeton, working in <a href="https://subotnikgroup.chemistry.princeton.edu/index.html">Prof. Joe Subotnik's group</a>, where I develop new quantum theories and computational methods to describe how electrons and atomic nuclei move together inside molecules. Most molecular simulations rely on the century-old Born–Oppenheimer approximation, which assumes that electrons adjust almost instantly to the motion of the nuclei—an enormously successful picture that nonetheless misses important effects when molecules rotate, vibrate, or exchange momentum between electrons and nuclei. My research goes beyond it by developing <strong>phase-space approaches to electronic structure</strong>, in which the electronic problem depends not only on nuclear positions but also on nuclear momenta. This perspective captures geometric forces, such as Coriolis and centrifugal effects in molecules, as well as angular momentum transfer between electrons and nuclei--effects that are often hidden or treated perturbatively in conventional molecular theory.</p>
      <p>Computationally, I build and optimize the numerical methods this theory relies on. This includes <strong>matrix-free Hamiltonian operators and GPU-accelerated iterative eigensolvers</strong>, combining numerical linear algebra with the physical structure of the problem to make calculations practical.</p>
      <p>More broadly, I enjoy working at the interface of theoretical chemistry, mathematical physics, and computational science. I draw on ideas from quantum geometry, numerical linear algebra, and high-performance computing to build more accurate and efficient descriptions of molecular motion.</p>
      <p class="inline-links">
        <span class="contact-emails"><span class="contact-plain">Email:</span> <a href="mailto:{{ site.email }}">{{ site.email }}</a> <span class="contact-plain">-</span> <a href="mailto:{{ site.email_princeton }}">{{ site.email_princeton }}</a></span>
      </p>
      <p class="inline-links">
        <a href="https://www.linkedin.com/in/{{ site.linkedin_handle }}/">LinkedIn</a>
        <a href="https://github.com/{{ site.github_username }}">GitHub</a>
        <a href="{{ site.scholar_url }}">Google Scholar</a>
      </p>
    </div>
  </div>
</section>

<h2 id="projects">Projects</h2>

<div class="item">
  <h3><a href="{{ '/writing/gpu-eigensolver/' | relative_url }}">Matrix-Free Davidson Eigensolvers on GPUs</a></h3>
  <p class="meta">Technical writeup · matrix-free Davidson · 8.9× GPU speedup</p>
</div>

<div class="item">
  <h3><a href="{{ '/writing/tensor-network-gpu/' | relative_url }}">GPU-Accelerated Tensor Network Contraction</a></h3>
  <p class="meta">Princeton × NVIDIA Open Hackathon · cluster-corrected belief propagation</p>
</div>

<div class="item">
  <h3><a href="{{ '/writing/quantum-geometry-notes.pdf' | relative_url }}">Quantum Geometry — Born–Oppenheimer Notes</a></h3>
  <p class="meta">Notes · PDF</p>
</div>

<div class="item">
  <h3><a href="{{ '/writing/seminar_slides.pdf' | relative_url }}">Third-Year Seminar Slides</a></h3>
  <p class="meta">Slides · PDF</p>
</div>

<h2 id="papers">Research Papers</h2>

<ul class="paper-list">
  <li>
    <strong>Bhati, M.</strong> Cofer-Shabica, D. V., Rawlinson, J. I., Littlejohn, R. G., Subotnik, J., & Bradbury, N. C. (2026)
    <a href="https://arxiv.org/abs/2605.27053">Electronic Structure in a Phase Space, non-Born-Oppenheimer Framework: Geometric Forces and Moody-Shapere-Wilczek Revisited.</a>
    <span class="paper-meta">arXiv:2605.27053</span>
  </li>
  <li>
    Peng, L., Duston, T., Bradbury, N., <strong>Bhati, M.</strong>, Tao, X., Rosen, M., Subotnik, J.E. (2026).
    A Conceptual Shift In Our Understanding of Degenerate Radical Spin Systems: Spin-Rotation Coupling.
    <span class="paper-meta">Accepted, <em>JACS</em></span>
  </li>
  <li>
    Tao, Z., <strong>Bhati, M.</strong>, Subotnik, J.E. (2026).
    <a href="https://pubs.aip.org/aip/aco/article/2/2/026101/3385850/Non-resonant-Raman-optical-activity-from-phase">Non-Resonant Raman Optical Activity As Explored Via Phase-Space Electronic Structure Theory.</a>
    <span class="paper-meta"><em>APL Computational Physics</em> 2, 026101</span>
  </li>
  <li>
    Peng, L., Qiu, T., Bradbury, N., Bian, X., <strong>Bhati, M.</strong>, Littlejohn, R., Kidwell, N.M., Subotnik, J.E. (2026).
    <a href="https://pubs.acs.org/doi/full/10.1021/acs.jpclett.5c03970">Phase Space Electronic Structure Theory: From Diatomic Lambda-Doubling to Macroscopic Einstein-de Haas.</a>
    <span class="paper-meta"><em>The Journal of Physical Chemistry Letters</em>, 17, 2799–2811</span>
  </li>
  <li>
    Bian, X., Duston, T., Bradbury, N., Tao, Z., <strong>Bhati, M.</strong>, Qiu, T., Wu, X., Wu, Y., Subotnik, J.E. (2026).
    <a href="https://arxiv.org/abs/2506.15994">The Phase-Space Way To Electronic Structure Theory and Subsequently Chemical Dynamics.</a>
    <span class="paper-meta"><em>Chemical Physics Reviews</em>, 7.1</span>
  </li>
  <li>
    <strong>Bhati, M.</strong>, Tao, Z., Bian, X., Rawlinson, J., Littlejohn, R., Subotnik, J.E. (2025).
    <a href="https://pubs.acs.org/doi/full/10.1021/acs.jpca.4c07904">A Phase-Space Electronic Hamiltonian for Molecules in a Static Magnetic Field. I: Conservation of Total Pseudomomentum and Angular Momentum.</a>
    <span class="paper-meta"><em>The Journal of Physical Chemistry A</em>, 129(20), 4555–4572</span>
  </li>
  <li>
    <strong>Bhati, M.</strong>, Tao, Z., Bian, X., Rawlinson, J., Littlejohn, R., Subotnik, J.E. (2025).
    <a href="https://pubs.acs.org/doi/full/10.1021/acs.jpca.4c07905">A Phase-Space Electronic Hamiltonian for Molecules in a Static Magnetic Field II: Quantum Chemistry Calculations with Gauge Invariant Atomic Orbitals.</a>
    <span class="paper-meta"><em>The Journal of Physical Chemistry A</em>, 129(20), 4573–4590</span>
  </li>
  <li>
    Duston, T., Tao, Z., Bian, X., <strong>Bhati, M.</strong>, Rawlinson, J., Littlejohn, R.G., Pei, Z., Shao, Y., Subotnik, J.E. (2024).
    <a href="https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c00662">A phase-space electronic Hamiltonian for vibrational circular dichroism.</a>
    <span class="paper-meta"><em>Journal of Chemical Theory and Computation</em>, 20(18), 7904–7921</span>
  </li>
  <li>
    Qiu, T., <strong>Bhati, M.</strong>, Tao, Z., Bian, X., Rawlinson, J., Littlejohn, R.G., Subotnik, J.E. (2024).
    <a href="https://pubs.aip.org/aip/jcp/article/160/12/124102/3278928">A simple one-electron expression for electron rotational factors.</a>
    <span class="paper-meta"><em>The Journal of Chemical Physics</em>, 160(12)</span>
  </li>
  <li>
    Tao, Z., Qiu, T., <strong>Bhati, M.</strong>, Bian, X., Duston, T., Rawlinson, J., Littlejohn, R.G., Subotnik, J.E. (2024).
    <a href="https://pubs.aip.org/aip/jcp/article/160/12/124101/3278929">Practical phase-space electronic Hamiltonians for ab initio dynamics.</a>
    <span class="paper-meta"><em>The Journal of Chemical Physics</em>, 160(12)</span>
  </li>
  <li>
    Pant, R., Verma, P.K., Rangi, C., Mondal, E., <strong>Bhati, M.</strong>, Srinivasan, V., Wüster, S. (2024).
    <a href="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.126903">Universal Measure for the Impact of Adiabaticity on Quantum Transitions.</a>
    <span class="paper-meta"><em>Physical Review Letters</em>, 132(12), 126903</span>
  </li>
  <li>
    Mukherjee, S., Kar, M., <strong>Bhati, M.</strong>, Gao, X., Barbatti, M. (2023).
    <a href="https://link.springer.com/article/10.1007/s00214-023-03020-w">On the short and long phosphorescence lifetimes of aromatic carbonyls.</a>
    <span class="paper-meta"><em>Theoretical Chemistry Accounts</em>, 142(9), 85</span>
  </li>
</ul>
