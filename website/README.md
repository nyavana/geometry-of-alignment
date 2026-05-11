# Geometry of Alignment — Project Website

Single-page V1 site for the EECS 6699 (Mathematics of Deep Learning, Columbia, Spring 2026) project on refusal geometry in Gemma 4.

## Contents

- `index.html` — the entire site (vanilla HTML/CSS/JS, no build step)
- `results/figures/*.png` — six published figures referenced from `§11b`
- `vercel.json` — static-hosting config (clean URLs, image caching)

## Local preview

```bash
python -m http.server -d website 8000
# open http://localhost:8000
```

## Deploy

```bash
cd website
vercel --prod   # project: geometry-of-alignment
```

## Notes

- `paper.pdf` is linked from nav and hero CTA but not yet present — the link 404s until the v1.0 draft is committed.
- Real data wired: `results/ablation_results/sweep_results.json` (α-sweep n=20), `results/figures/selective_safety_table.md` (cosine numbers), and the published figures in `§11b`. UMAP scatter + per-layer heatmap demos use stylized synthetic data; the real `umap_layer_15.png` and `signal_vs_layer.png` are shown alongside in `§11b`.
- Source: original design at `https://api.anthropic.com/v1/design/h/yLxOcT5eLJIaYI0ym5Ugeg`.
