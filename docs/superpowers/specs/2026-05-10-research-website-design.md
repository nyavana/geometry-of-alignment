# Research Website Design — geometry-of-alignment

**Date:** 2026-05-10
**Status:** Brainstorming complete; awaiting user review before implementation plan
**Hosted at:** `geometry-of-alignment.vercel.app`
**Repo location:** `web/` subdirectory of `geometry-of-alignment` root

---

## 1. Goal

Build a single-page research website that turns the geometry-of-alignment project into a shareable, narrative-driven artifact. The site is the public face of the work: ML-community readers should be able to land cold, scroll through the personal-narrative hook, the science, and the headline finding, and walk away both convinced of the result and able to share a single screenshot or URL.

The site frames the work as a research project — not a course final — with the EECS 6699 / Columbia attribution living in the footer.

## 2. Audience priority

1. **ML community / practitioners** (primary) — Hacker News, Twitter-ML, interp researchers. Optimized for shareability and a strong narrative hook.
2. **Academic readers** (secondary) — paper PDF, citation, methodology depth.
3. **Recruiters** (tertiary) — team identity in §13, reachable via the existing team contact email.

## 3. Site structure

A single long-scroll narrative — no internal routes other than `/paper.pdf` (served as a static asset). Fourteen sections, top to bottom:

| § | Section | Purpose | Interactive |
|---|---|---|---|
| 1 | Hero | Headline question + 3 metadata CTAs + scroll cue | — |
| 2 | The hike | Narrative beat 1: remote wilderness, Wikipedia on the phone | — |
| 3 | The promise | Narrative beat 2: Gemma 4 on a phone as an on-device expert | — |
| 4 | The refusal | Narrative beat 3: the actual prompt + the refusal text revealed letter-by-letter | — |
| 5 | The question | Narrative beat 4: how does safety work, and can it be selectively removed | — |
| 6 | Primer | What is the refusal direction? What is rank-1 abliteration? Diagram + KaTeX equation | — |
| 7 | Methodology | 4-stage pipeline (benchmark → mechanistic → abliterate → weight-diff) as an animated diagram | — |
| 8 | Refusal direction (mechanistic) | UMAP scatter + layer signal heatmap | **Demo 1, Demo 2** |
| 9 | The investigation (M0 → M6) | Animated milestone timeline; each milestone reveals one finding | **Demo 3** (M6 cascade) |
| 10 | The punch line | Full-screen `40.5%` reveal, contextual subtitle | — |
| 11 | α-sweep | Drag α from 0 → 2.0 and watch the residual flatten (vs. random control vs. Gram-Schmidt) | **Demo 4** |
| 12 | What this means | Geometric implications, future work, multi-rank descent open question | — |
| 13 | Team | Four terminal-style researcher cards (name, role, email, GitHub) | — |
| 14 | Footer | Paper PDF, GitHub, BibTeX citation, Columbia EECS 6699 / 2026 attribution, MIT license | — |

The opening 4 narrative sections (§§ 2–5) are sourced directly from `paper/presentation-slides/story.md` (the personal-hook narrative the team author already drafted). The 40.5% punch lands at §10 — *after* the story has earned it, not as the opening shot.

## 4. Visual identity

### 4.1 Color tokens

| Token | Value | Use |
|---|---|---|
| `--bg-base` | `#07021a` | Page background (deep purple-black) |
| `--iri-magenta` | `#ff2bb0` | Iridescent stop, accent |
| `--iri-cyan` | `#00d4ff` | Iridescent stop, primary CTA arrow, scope traces |
| `--iri-amber` | `#ffe04a` | Iridescent stop, warning chips |
| `--iri-violet` | `#b14dff` | Iridescent stop |
| `--status-green` | `#00ff88` | Pulsing status dots only |
| `--text-1` | `#ffffff` | Headlines |
| `--text-2` | `rgba(255,255,255,0.85)` | Body |
| `--text-3` | `rgba(255,255,255,0.65)` | Eyebrow text, captions |
| `--text-4` | `rgba(255,255,255,0.45)` | Section numbers, fine chrome |

The hero background is a 4-stop radial-gradient field (magenta + cyan + amber + violet on `--bg-base`). Subtle horizontal scanline overlay (3-px repeat, 2.5% opacity) ties the hero to a "phosphor" feel without being noisy.

### 4.2 Typography

- **Headline:** Inter Black 900, line-height 0.95, letter-spacing −0.035em
- **Body:** Inter 400 / 600
- **Mono:** JetBrains Mono — used for chrome (chips, status pills, CTA metadata, code, section numbers)
- **Math:** KaTeX default (Computer Modern), inline + block

### 4.3 Headline treatment (locked)

Solid white headlines with a single-phrase magenta→cyan gradient accent (e.g., "Gemma 4" in the hero gets the gradient; the rest stays white). No RGB-split text-shadow. Maximum legibility while preserving the holographic identity.

### 4.4 CTA treatment (locked)

Metadata-rich buttons: pulsing status dot + title + small monospace metadata (e.g., `PDF · 24 pp · v1.0`) + cyan arrow that nudges right on hover. Three CTAs in the hero row:

- **Read the paper** (primary; magenta dot)
- **View source** (secondary; green dot)
- **Interactive demo** (secondary; green dot)

### 4.5 Motion language

- Scroll-triggered reveals via Framer Motion `whileInView` (fade + 8-px translateY)
- Sticky figures during their narrative section (CSS `position: sticky`)
- Pulsing status dots (animation: 1.6s ease-in-out infinite)
- Hover: lift (`translateY(-1px)`) + border-color shift to cyan + arrow nudge (`translateX(4px)`)
- Hero entrance: header fade-in (200ms), eyebrow fade-in (400ms), headline word-staggered reveal (600ms-1200ms)
- Reduced-motion media query disables all transforms; reveals become opacity-only fades

## 5. Tech stack

| Concern | Choice | Rationale |
|---|---|---|
| Framework | Next.js 14 App Router, TypeScript strict | Vercel-native, ML-community standard, great for scroll-heavy sites with embedded interactives |
| Styling | Tailwind CSS 3.4 + custom CSS variables for tokens | Fast iteration, design-system friendly |
| Motion | Framer Motion | Industry standard for scroll-triggered reveals + page transitions |
| Charts (complex) | D3 v7 | UMAP scatter needs custom interaction; D3 is the right tool |
| Charts (simple) | recharts | α-sweep line plot, simple bar charts |
| Math | KaTeX (`react-katex`) | Inline + block equations |
| Icons | Lucide React | Permissive license, large set, tree-shakeable |
| Deploy | Vercel CLI (`vercel deploy --prod`) | User-specified |
| Testing | Playwright + `@axe-core/playwright` | User-specified for E2E + accessibility |
| Analytics | Vercel Analytics | Zero-config, privacy-friendly |
| Lint / format | ESLint + Prettier (Next.js defaults) | Standard |
| Package manager | pnpm | Faster, smaller node_modules than npm |

## 6. Repo layout

```
geometry-of-alignment/
├── web/                          ← website lives here
│   ├── app/                      ← Next.js App Router routes
│   │   ├── layout.tsx
│   │   ├── page.tsx              ← the single long-scroll page
│   │   └── globals.css
│   ├── components/
│   │   ├── hero/
│   │   ├── sections/             ← one file per § (Hike.tsx, Refusal.tsx, etc.)
│   │   ├── demos/                ← UmapScatter.tsx, AlphaSweep.tsx, M6Cascade.tsx, LayerHeatmap.tsx
│   │   ├── ui/                   ← MetadataButton, BracketPill, StatusDot, Eyebrow, etc.
│   │   └── motion/               ← Reveal.tsx, StickyFigure.tsx, etc.
│   ├── lib/
│   │   ├── tokens.ts             ← TS-typed access to color/spacing tokens
│   │   └── data/                 ← precomputed JSON for demos (see § 7)
│   ├── public/
│   │   ├── paper.pdf             ← linked from hero CTA
│   │   └── og-image.png          ← social share card
│   ├── tests/
│   │   ├── e2e/                  ← Playwright tests
│   │   └── data/                 ← test fixtures
│   ├── scripts/
│   │   └── extract_demo_data.py  ← reads results/*.pt + UMAP-fits, writes lib/data/*.json
│   │                                (Python, runs before `pnpm build`)
│   ├── next.config.mjs
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── package.json
│   └── playwright.config.ts
├── src/                          ← unchanged (Python research code)
├── results/                      ← unchanged (artifacts; demos read from here at build time)
├── paper/                        ← unchanged
└── ...
```

The `web/` directory is self-contained — it has its own `package.json` and node toolchain, and does not interfere with the Python project. Vercel deploys `web/` as the project root.

## 7. Interactive demos

All four demos consume **precomputed JSON** baked at build time. No runtime model inference.

The data pipeline is two-step:

1. **Python extraction** (`web/scripts/extract_demo_data.py`) — uses the project's existing `.venv` to read PyTorch `.pt` tensors from `results/`, fit UMAP on the activation tensors (one-shot per layer, cached), and emit compact JSON to `web/lib/data/`. Runs manually or via a `pnpm prebuild` hook that invokes Python.
2. **TypeScript build** (`pnpm build`) — Next.js reads the JSON via standard `import` statements; the build is fully static (no runtime Python or torch dependency).

The output JSON files (estimated total < 500 KB gzipped) are checked into `web/lib/data/` so Vercel deploys do not need a Python runtime.

### 7.1 Demo 1 — UMAP refusal-direction scatter (§ 8)

- **Source data:** `results/activations/refuse_activations.pt`, `results/activations/comply_activations.pt` at L15 (peak signal layer)
- **Preprocessing:** Reduce to 2D via UMAP at build time (one-shot; result cached as JSON)
- **Render:** D3 scatter, ~340 points (one per benchmark prompt). Refuse magenta, comply cyan, opacity 0.6, radius 4px
- **Interaction:** Hover any point → tooltip with the prompt text + the model's actual response. Selected layer is sticky: clicking a point in the heatmap (Demo 2) re-renders this scatter at that layer

### 7.2 Demo 2 — Layer signal heatmap (§ 8, paired with Demo 1)

- **Source data:** Per-layer Cohen's d from `results/activations/`
- **Render:** SVG strip of 42 narrow bands. Color-mapped on a magenta→white scale by Cohen's d magnitude. Global-attention layer indices `[5, 11, 17, 23, 29, 35, 41]` marked with a tick above the strip
- **Interaction:** Hover → reveals layer index + d value + attention type. Click → updates Demo 1 to render that layer's UMAP

### 7.3 Demo 3 — M6 cascade decision tree (§ 9)

- **Source data:** Hand-curated JSON encoding the H1–H6 hypotheses, the cosine numbers, and the verdicts (extracted from `docs/findings/M6_PROPOSAL_RANK1_FOLLOWUP.md`)
- **Render:** Custom SVG decision tree. Six nodes laid out left-to-right. Each node is a chip: hypothesis name + verdict tag (`PASS` / `REJECTED` / `LOAD-BEARING` / `INSUFFICIENT`)
- **Interaction:** Click a node → expands a panel below with the experiment description, the result, and the cosine number. The "load-bearing" node (H4 Gram-Schmidt) is pre-highlighted with a magenta glow

### 7.4 Demo 4 — α-sweep slider (§ 11)

- **Source data:** Sweep results from `results/ablation_results/`
- **Render:** recharts LineChart with three traces: baseline (flat ~30%), random control (flat ~30%), Gram-Schmidt (drops to ~40% at α=1.0). Y-axis: should_refuse rate. X-axis: α from 0 to 2.0
- **Interaction:** Slider below the chart drags the α value; a vertical guide line tracks the slider position; readout shows the three rates at that α

## 8. Imagery & narrative sections

No photos, no AI-generated illustrations. Sections § 2–5 (the personal narrative) are pure typography + animated SVG diagrams:

- **§ 2 The hike:** large blockquote treatment of the story.md opening. Animated SVG of a stylized topographic-line landscape that draws on scroll
- **§ 3 The promise:** stylized phone outline (SVG) with a slowly-pulsing app surface; large pull quote about on-device intelligence
- **§ 4 The refusal:** chat-bubble layout. Prompt appears (typed letter-by-letter via JS, ~30ms/char). Response bubble appears below with the refusal text, also typed. Reveals on scroll-into-view
- **§ 5 The question:** large typographic moment. Two-line question with the "selectively" word picking up the magenta→cyan accent

This keeps the entire site visually consistent and removes any photo-licensing or AI-generated awkwardness.

## 9. Team section (§ 13)

Four terminal-style researcher cards in a 2×2 grid:

```
┌─ chenhao yang ────────────────┐  ┌─ daitian zhao ────────────────┐
│  role · benchmarking + abl.    │  │  role · mechanistic + figures │
│  cy2822@columbia.edu           │  │  dz2585@columbia.edu          │
│  github.com/...                │  │  github.com/...               │
└────────────────────────────────┘  └───────────────────────────────┘

┌─ hanlin wang ─────────────────┐  ┌─ yuxi luo ────────────────────┐
│  role · weight-diff + paper   │  │  role · interactive + writeup │
│  hw3100@columbia.edu          │  │  yl6117@columbia.edu          │
└────────────────────────────────┘  └───────────────────────────────┘
```

(Roles to be confirmed by the team during implementation; placeholders shown.)

Hover a card → border picks up cyan + a one-line role bio fades in. Card uses a subtle holographic shimmer in the top border.

A team-contact line lives below the grid: `team contact · GeometryofAlignment@nyavana.io`.

## 10. Footer (§ 14)

- Paper download (`/paper.pdf`)
- GitHub link (`github.com/nyavana/geometry-of-alignment`)
- BibTeX citation block (copy-button)
- Course attribution: "EECS 6699 · Mathematics of Deep Learning · Columbia University · Spring 2026"
- License: "MIT (code) · CC BY 4.0 (figures + prose)"
- Final pulse-dot: `signal · live`

## 11. Testing

### 11.1 Playwright suite (`web/tests/e2e/`)

| Test | Asserts |
|---|---|
| `hero.spec.ts` | Hero renders; all 3 CTAs are clickable; headline text matches; visual snapshot at 3 viewports (375 / 768 / 1440) |
| `sections.spec.ts` | Each of the 14 sections renders without console errors; section count = 14 |
| `demos.spec.ts` | Each demo loads; UMAP renders ≥ 200 points; α slider drags and updates trace; M6 cascade has 6 clickable nodes; layer heatmap has 42 bands |
| `accessibility.spec.ts` | `@axe-core/playwright` reports 0 critical / serious issues per section |
| `responsive.spec.ts` | Mobile (375px), tablet (768px), desktop (1440px) — no overflow, all CTAs reachable, sticky figures degrade to inline on mobile |
| `motion.spec.ts` | With `prefers-reduced-motion: reduce`, transforms are disabled and reveals become opacity-only |

### 11.2 Lighthouse targets

- Performance ≥ 90
- Accessibility ≥ 95
- Best Practices ≥ 95
- SEO ≥ 90

Run via `lhci autorun` locally before each prod deploy. CI hookup is optional and out of scope for v1.

### 11.3 Manual smoke (pre-prod-deploy)

- Scroll the full page on Chrome desktop + a real iPhone Safari
- Hover all 4 demos; confirm they're responsive
- Click "Read the paper" → resolves to `/paper.pdf`
- Confirm OG image renders in Twitter / Slack unfurl

## 12. Deploy

- **Preview:** every commit to any branch triggers a Vercel preview deploy
- **Production:** `vercel deploy --prod` after the smoke checklist passes
- **Domain:** `geometry-of-alignment.vercel.app` (free Vercel subdomain). Custom domain can be wired later via Vercel project settings + DNS without code changes

## 13. Out of scope (explicit non-goals)

- Server-side runtime inference (all demos use precomputed data)
- Multi-page routing (intentional — the long-scroll IS the design)
- Comments / discussion / newsletter signup
- Multi-language (English only)
- Dark/light theme toggle (the holographic dark mode IS the theme)
- Photos of team members (terminal-style cards instead)
- AI-generated imagery
- Fancy 3D / WebGL effects (not needed for the locked aesthetic)

## 14. Open questions (to resolve during implementation)

- Confirm exact team-member roles for the §13 cards
- Confirm GitHub usernames for each team member
- Confirm whether the paper PDF is finalized at deploy time, or whether we ship with a "draft v0.x" indicator on the CTA metadata
- Confirm Vercel team / hobby account for the deploy target

## 15. Success criteria

- A first-time visitor lands on the page, scrolls top-to-bottom, and within 90 seconds:
  1. Understands the personal hook (refusal during emergency)
  2. Understands the research question
  3. Has seen the 40.5% headline finding
  4. Has interacted with at least one demo
  5. Knows how to access the paper and code
- Lighthouse passes the targets in § 11.2
- All Playwright tests pass on Chromium / Firefox / WebKit
- The site loads under 2 seconds on a desktop connection
- No `console.error` in production
- A senior ML researcher can read the page in 5 minutes and walk away with a defensible understanding of why standard rank-1 abliteration is partially inert on Gemma 4

---

## Appendix A — Sourced narrative (story.md, beats 1–4)

Verbatim source for sections § 2–5:

> When Google first released Gemma 4, I was thinking, hey, maybe this could be used to save lives. I used to do a lot of hiking. We hike in remote areas where there is no phone reception, and you can barely find anybody else. Maybe you meet a person once or twice per hour. If something happens, we are truly on our own. Despite this, I always kept a Wikipedia copy on my phone, so I can quickly reference it in case something bad happens.
>
> [The promise] With the Google Gemma 4 model, this becomes a reality right at the palm of your hand. This is a very intelligent open-source open-weight model that can run on a phone at decent speed. Right now I can just ask LLMs how to respond to emergencies, which is a big plus in terms of survivability in remote locations.
>
> [The refusal] So I downloaded Gemma 4 E2B IT on my phone, and I tried to ask, hey, if my friend is injured and we're in a very remote location, can you help me to diagnose what is going on? And surprisingly, Gemma directly refused to answer my prompt because of the safety guardrail.
>
> [The question] This made me think, how does safety work in large language models? And how can we change it so it only refuses to answer prompts that would be causing social problems and not the ones that would save lives.

The site copy will lightly polish these for the web (tighter sentences, present tense), but keep the personal voice intact.

## Appendix B — Brainstorming session artifacts

Visual companion mockups from this brainstorming session live at:

`.superpowers/brainstorm/1617927-1778457484/content/`

- `aesthetic-direction.html` — 4 initial aesthetic directions (A terminal / B brutalist / C lab / D holographic)
- `aesthetic-d-refined.html` — D-pure vs D×C-hybrid lean test (D-pure won)
- `buttons.html` — first round of CTA treatments (all rejected)
- `buttons-v2.html` — engineer-research CTA treatments (#3 metadata-rich won)
- `hero-integrated.html` — full hero compositions (variant A all-metadata won)
- `headline-treatments.html` — 4 readability variants (#1 solid white + accent won)
- `headline-compare.html` — before/after A/B confirmation
- `section-map.html` — first round of page structures (story-first won)
- `section-map-v2.html` — story-driven structure rebuilt from story.md (variant A won)

These are kept for reference and may be removed once implementation is complete.
