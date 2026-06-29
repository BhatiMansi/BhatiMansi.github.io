# Personal site — launch checklist

A minimal Jekyll site for GitHub Pages. No theme to choose, no build config to babysit, no JavaScript. GitHub builds it for you when you push.

## Launch in ~10 minutes

1. **Create the repo.** On GitHub, make a new **public** repo named exactly:
   `yourusername.github.io` (use your real GitHub username).
2. **Add these files.** Upload everything in this folder to the repo
   (drag-and-drop in the GitHub web UI is fine), or push with git.
3. **Personalize.** Open `_config.yml` and edit the top block once:
   name, tagline, email, GitHub handle, LinkedIn handle, and (optionally)
   Scholar/ORCID. Every page reads from here, so you only set it in one place.
4. **Add your CV.** Replace `assets/cv.pdf` with your real CV (keep the filename).
5. **Turn on Pages.** Repo → **Settings → Pages** → *Build and deployment* →
   Source: **Deploy from a branch** → Branch: **main** / **/(root)** → Save.
6. **Wait ~1 minute**, then visit `https://yourusername.github.io`. Done.

## Then fill in the content

- `index.md` — home. The big lines are written; tweak the wording to taste.
- `projects.md` — the GPU eigensolver is featured. Add the hackathon details and
  any other projects (duplicate a `.item` block per project).
- `writing/gpu-eigensolver.md` — paste your full writeup here, **or** delete this
  file and point the "Read the full writeup" links at your existing article.
- `research.md` — replace the placeholder entries with your real papers.
- `cv.md` — fill the experience / education rows (the PDF is the source of truth).

Anything in `[square brackets]` is a placeholder meant to be replaced.

## Optional: preview locally

You don't need this — GitHub builds the site — but if you want to see changes
before pushing and you have Ruby installed:

```bash
bundle install
bundle exec jekyll serve
# open http://localhost:4000
```

## Adding a custom domain later (optional)

Settings → Pages → Custom domain. Add a `CNAME` file with your domain, and set
the DNS records GitHub shows you. Not needed for v1.
