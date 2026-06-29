---
layout: default
title: Research
permalink: /research/
---

<h1>Research</h1>

<p>My research sits at the intersection of computational physics / chemistry and numerical methods — quantum-scattering problems, large-scale eigenproblems, and the algorithms and implementations that make them solvable at scale.</p>

{% if site.scholar_url != "" or site.orcid_url != "" %}
<p class="inline-links">
  {% if site.scholar_url != "" %}<a href="{{ site.scholar_url }}">Google Scholar</a>{% endif %}
  {% if site.orcid_url != "" %}<a href="{{ site.orcid_url }}">ORCID</a>{% endif %}
</p>
{% endif %}

<h2>Publications</h2>

<!-- One block per paper. Newest first. Keep it to: authors, title, venue/year, links. -->

<div class="item">
  <h3>[Paper title goes here]</h3>
  <p class="meta">[Author list, with <strong>your name in bold</strong>] · [Journal / Conference], [Year]</p>
  <p>[Optional one-line plain-language summary of the result — helpful for non-specialist recruiters.]</p>
  <p class="actions">
    <a href="#">PDF</a>
    <a href="#">DOI / arXiv</a>
    <a href="#">Code</a>
  </p>
</div>

<div class="item">
  <h3>[Paper title goes here]</h3>
  <p class="meta">[Author list] · [Journal / Conference], [Year]</p>
  <p class="actions">
    <a href="#">PDF</a>
    <a href="#">DOI / arXiv</a>
  </p>
</div>

<h2>Preprints &amp; in progress</h2>

<div class="item">
  <h3>[Working title]</h3>
  <p class="meta">[Status — e.g. in preparation, under review], [Year]</p>
  <p>[One line on the contribution.]</p>
</div>
