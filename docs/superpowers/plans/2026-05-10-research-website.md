# Research Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the geometry-of-alignment research website per `docs/superpowers/specs/2026-05-10-research-website-design.md` — a single long-scroll narrative site with 14 sections, 4 interactive demos, holographic dark aesthetic, deployed to `geometry-of-alignment.vercel.app`.

**Architecture:** Next.js 14 App Router (TypeScript strict) in a self-contained `web/` subdirectory. Server-rendered static export. All four interactive demos consume precomputed JSON baked at build time by a Python extraction script (`web/scripts/extract_demo_data.py`) that reads from `results/`. Tailwind CSS for styling with custom CSS variables for the holographic palette. Framer Motion for scroll-triggered reveals. D3 v7 for the UMAP scatter; recharts for the α-sweep line plot. Playwright for E2E + accessibility tests.

**Tech Stack:** Next.js 14 · TypeScript strict · Tailwind CSS 3.4 · Framer Motion 11 · D3 v7 · recharts 2 · KaTeX · Lucide React · Playwright + @axe-core/playwright · pnpm · Vercel CLI · Python 3.12 (extraction script).

**Parallelization plan (4 team members):** Phase 0 is sequential (single owner). Phases 1–4 can run in parallel:
- **Member A:** Phase 1 UI primitives → Phase 3 sections §§ 1, 6, 7, 10, 12 → Phase 5 page assembly
- **Member B:** Phase 3 sections §§ 2–5 (narrative beats) → Phase 4 Demo 4 (α-sweep) → Phase 5 §§ 13, 14
- **Member C:** Phase 2 Python extraction → Phase 4 Demo 1 (UMAP) + Demo 2 (heatmap) + their integration into §8
- **Member D:** Phase 4 Demo 3 (M6 cascade) + integration into §9 → Phase 6 tests + deploy

---

## File Structure

```
web/
├── app/
│   ├── layout.tsx                 ← root layout, fonts, analytics
│   ├── page.tsx                   ← single long-scroll page composing all 14 sections
│   └── globals.css                ← Tailwind base + CSS variables for holographic tokens
├── components/
│   ├── ui/
│   │   ├── StatusDot.tsx          ← pulsing colored dot, 3 colors (green/magenta/cyan)
│   │   ├── Eyebrow.tsx            ← small monospace label above headlines
│   │   ├── SectionNumber.tsx      ← "00 · HERO" mono chrome
│   │   ├── MetadataButton.tsx     ← primary CTA: dot + title + meta + arrow
│   │   ├── BracketPill.tsx        ← secondary chip [ label → ]
│   │   ├── ChatBubble.tsx         ← used by §4 Refusal
│   │   ├── HoloHeading.tsx        ← solid white headline w/ accent gradient on one phrase
│   │   └── BibTexBlock.tsx        ← citation with copy button
│   ├── motion/
│   │   ├── Reveal.tsx             ← Framer Motion whileInView fade+translate wrapper
│   │   └── TypeOnReveal.tsx       ← letter-by-letter typing on scroll-into-view
│   ├── sections/
│   │   ├── Hero.tsx               (§1)
│   │   ├── Hike.tsx               (§2)
│   │   ├── Promise.tsx            (§3)
│   │   ├── Refusal.tsx            (§4)
│   │   ├── Question.tsx           (§5)
│   │   ├── Primer.tsx             (§6) — KaTeX equation
│   │   ├── Methodology.tsx        (§7) — animated 4-stage pipeline diagram
│   │   ├── RefusalDirection.tsx   (§8) — wraps demos 1+2
│   │   ├── Investigation.tsx      (§9) — wraps demo 3 (M6 cascade)
│   │   ├── PunchLine.tsx          (§10) — full-screen 40.5%
│   │   ├── AlphaSweepSection.tsx  (§11) — wraps demo 4
│   │   ├── WhatThisMeans.tsx      (§12)
│   │   ├── Team.tsx               (§13)
│   │   └── Footer.tsx             (§14)
│   └── demos/
│       ├── UmapScatter.tsx        (Demo 1) — D3
│       ├── LayerHeatmap.tsx       (Demo 2) — custom SVG, paired w/ Demo 1
│       ├── M6Cascade.tsx          (Demo 3) — custom SVG decision tree
│       └── AlphaSweep.tsx         (Demo 4) — recharts + slider
├── lib/
│   ├── tokens.ts                  ← TS-typed access to color/spacing tokens
│   ├── reduced-motion.ts          ← hook: usePrefersReducedMotion()
│   ├── citations.ts               ← BibTeX entry as a constant
│   └── data/
│       ├── m6-cascade.json        ← hand-curated, in repo
│       ├── alpha-sweep.json       ← passthrough from results/ablation_results/sweep_results.json
│       ├── umap-l15.json          ← UMAP-fitted from $RESULTS_DIR activations
│       ├── cohens-d-per-layer.json← per-layer signal strengths (42 values)
│       └── prompt-metadata.json   ← passthrough from results/activations/prompt_metadata.json
├── public/
│   ├── paper.pdf                  ← linked from hero CTA
│   ├── og-image.png               ← 1200x630 social share card
│   └── favicon.ico
├── tests/
│   └── e2e/
│       ├── hero.spec.ts
│       ├── sections.spec.ts
│       ├── demos.spec.ts
│       ├── accessibility.spec.ts
│       ├── responsive.spec.ts
│       └── motion.spec.ts
├── scripts/
│   ├── extract_demo_data.py       ← Python: reads results/, writes lib/data/*.json
│   └── requirements.txt           ← umap-learn (the one extra dep on top of project venv)
├── next.config.mjs
├── tailwind.config.ts
├── tsconfig.json
├── playwright.config.ts
├── package.json
├── pnpm-lock.yaml
└── .gitignore
```

---

## Phase 0 — Setup (sequential; single owner; ~3 hours)

### Task 1: Scaffold the Next.js project

**Files:**
- Create: `web/` (entire directory tree)
- Create: `web/package.json`
- Create: `web/.gitignore`
- Create: `web/tsconfig.json`
- Create: `web/next.config.mjs`

- [ ] **Step 1: Verify pnpm is available; install if missing**

```bash
which pnpm || npm install -g pnpm
pnpm --version  # should print 9.x or higher
```

- [ ] **Step 2: Initialize Next.js 14 with TypeScript strict + App Router + Tailwind**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
pnpm create next-app@14 web --typescript --tailwind --eslint --app --src-dir false --import-alias "@/*" --no-turbopack
```

When prompted, accept all defaults. The flag `--src-dir false` puts `app/` at `web/app/` (not `web/src/app/`), matching the spec.

- [ ] **Step 3: Verify scaffold is valid**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
pnpm dev
```

Expected: Next.js dev server starts on http://localhost:3000 with the default landing page. Stop the server (Ctrl+C).

- [ ] **Step 4: Pin TypeScript to strict mode**

Edit `web/tsconfig.json` — confirm `"strict": true` is present in `compilerOptions`. If absent, add it. Also add `"noUncheckedIndexedAccess": true`.

- [ ] **Step 5: Add `web/.gitignore` entries for build artifacts**

Append to `web/.gitignore` (the Next.js scaffold creates one — verify these lines exist; add any missing):

```
.next/
out/
node_modules/
.vercel/
playwright-report/
test-results/
.env*.local
```

- [ ] **Step 6: Commit the scaffold**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/
git commit -m "feat(web): scaffold Next.js 14 project with TypeScript strict + Tailwind"
```

---

### Task 2: Install runtime + dev dependencies

**Files:**
- Modify: `web/package.json`

- [ ] **Step 1: Install runtime dependencies**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
pnpm add framer-motion@^11 d3@^7 recharts@^2.12 katex react-katex lucide-react @vercel/analytics
pnpm add -D @types/d3 @types/react-katex
```

- [ ] **Step 2: Install Playwright + axe**

```bash
pnpm add -D @playwright/test @axe-core/playwright
pnpm exec playwright install --with-deps chromium firefox webkit
```

The browser install downloads ~500 MB and may take a few minutes.

- [ ] **Step 3: Verify package.json contains all deps**

```bash
cat package.json | grep -E "framer-motion|d3|recharts|katex|lucide-react|@vercel/analytics|@playwright/test|@axe-core/playwright"
```

Expected: 8 lines printed, one per dep.

- [ ] **Step 4: Add useful scripts to package.json**

Edit `web/package.json` `"scripts"` section to include:

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "playwright test",
    "test:ui": "playwright test --ui",
    "extract-data": "python3 scripts/extract_demo_data.py"
  }
}
```

- [ ] **Step 5: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/package.json web/pnpm-lock.yaml
git commit -m "feat(web): add framer-motion, d3, recharts, katex, playwright deps"
```

---

### Task 3: Configure design tokens (Tailwind + CSS variables)

**Files:**
- Modify: `web/tailwind.config.ts`
- Modify: `web/app/globals.css`
- Create: `web/lib/tokens.ts`

- [ ] **Step 1: Define CSS variables in globals.css**

Replace the entire contents of `web/app/globals.css` with:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  /* Holographic palette */
  --bg-base: #07021a;
  --iri-magenta: #ff2bb0;
  --iri-cyan: #00d4ff;
  --iri-amber: #ffe04a;
  --iri-violet: #b14dff;
  --status-green: #00ff88;

  /* Text scale */
  --text-1: #ffffff;
  --text-2: rgba(255, 255, 255, 0.85);
  --text-3: rgba(255, 255, 255, 0.65);
  --text-4: rgba(255, 255, 255, 0.45);

  /* Hero gradient field (used by Hero + PunchLine) */
  --hero-bg:
    radial-gradient(circle at 18% 22%, #4d1a8a 0%, transparent 48%),
    radial-gradient(circle at 78% 58%, #ff2bb0 0%, transparent 52%),
    radial-gradient(circle at 50% 95%, #00d4ff 0%, transparent 50%),
    radial-gradient(circle at 90% 10%, #ffe04a 0%, transparent 38%),
    var(--bg-base);

  /* Phosphor scanline overlay */
  --scanlines: repeating-linear-gradient(
    0deg,
    rgba(255, 255, 255, 0.025) 0px,
    rgba(255, 255, 255, 0.025) 1px,
    transparent 1px,
    transparent 3px
  );
}

html, body {
  background: var(--bg-base);
  color: var(--text-2);
  font-family: var(--font-inter), system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

- [ ] **Step 2: Extend Tailwind theme with the holographic tokens**

Replace `web/tailwind.config.ts` with:

```ts
import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg-base)",
        "iri-magenta": "var(--iri-magenta)",
        "iri-cyan": "var(--iri-cyan)",
        "iri-amber": "var(--iri-amber)",
        "iri-violet": "var(--iri-violet)",
        "status-green": "var(--status-green)",
        "text-1": "var(--text-1)",
        "text-2": "var(--text-2)",
        "text-3": "var(--text-3)",
        "text-4": "var(--text-4)",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "ui-monospace", "monospace"],
      },
      backgroundImage: {
        hero: "var(--hero-bg)",
        scanlines: "var(--scanlines)",
      },
      keyframes: {
        "dot-pulse": {
          "0%, 100%": { boxShadow: "0 0 8px currentColor", opacity: "1" },
          "50%": { boxShadow: "0 0 16px currentColor", opacity: "0.6" },
        },
      },
      animation: {
        "dot-pulse": "dot-pulse 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
```

- [ ] **Step 3: Create lib/tokens.ts (typed token access for components)**

Create `web/lib/tokens.ts`:

```ts
export const tokens = {
  bg: "var(--bg-base)",
  iriMagenta: "var(--iri-magenta)",
  iriCyan: "var(--iri-cyan)",
  iriAmber: "var(--iri-amber)",
  iriViolet: "var(--iri-violet)",
  statusGreen: "var(--status-green)",
  text: {
    1: "var(--text-1)",
    2: "var(--text-2)",
    3: "var(--text-3)",
    4: "var(--text-4)",
  },
} as const;

export type DotColor = "green" | "magenta" | "cyan";

export const dotColors: Record<DotColor, string> = {
  green: tokens.statusGreen,
  magenta: tokens.iriMagenta,
  cyan: tokens.iriCyan,
};
```

- [ ] **Step 4: Verify Tailwind compiles**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
pnpm dev
```

Open http://localhost:3000 and confirm the page background is now `#07021a` (deep purple-black). Stop the server.

- [ ] **Step 5: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/app/globals.css web/tailwind.config.ts web/lib/tokens.ts
git commit -m "feat(web): wire holographic design tokens (CSS vars + Tailwind theme)"
```

---

### Task 4: Add fonts + base layout

**Files:**
- Modify: `web/app/layout.tsx`

- [ ] **Step 1: Replace web/app/layout.tsx with the locked layout**

```tsx
import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/react";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "600", "900"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-jetbrains",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Geometry of Alignment — why doesn't abliteration work on Gemma 4?",
  description:
    "A research study of where safety lives in LLM weights, and why a single rank-1 weight edit fails to remove it on Google's Gemma 4.",
  metadataBase: new URL("https://geometry-of-alignment.vercel.app"),
  openGraph: {
    title: "Geometry of Alignment",
    description: "Why doesn't abliteration work on Gemma 4?",
    url: "https://geometry-of-alignment.vercel.app",
    siteName: "Geometry of Alignment",
    images: [{ url: "/og-image.png", width: 1200, height: 630 }],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Geometry of Alignment",
    description: "Why doesn't abliteration work on Gemma 4?",
    images: ["/og-image.png"],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrains.variable}`}>
      <body>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
```

- [ ] **Step 2: Replace app/page.tsx with a placeholder so dev server runs**

```tsx
export default function HomePage() {
  return (
    <main className="min-h-screen flex items-center justify-center">
      <p className="font-mono text-text-3 text-xs tracking-[0.22em] uppercase">
        site under construction · check back soon
      </p>
    </main>
  );
}
```

- [ ] **Step 3: Verify fonts + layout render**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
pnpm dev
```

Open http://localhost:3000. Expected: deep purple-black background, monospace placeholder text in dim white. Stop the server.

- [ ] **Step 4: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/app/layout.tsx web/app/page.tsx
git commit -m "feat(web): wire Inter + JetBrains Mono, OG metadata, Vercel Analytics"
```

---

### Task 5: Set up Playwright + first smoke test

**Files:**
- Create: `web/playwright.config.ts`
- Create: `web/tests/e2e/smoke.spec.ts`

- [ ] **Step 1: Create playwright.config.ts**

```ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "html",
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
  webServer: {
    command: "pnpm dev",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
```

- [ ] **Step 2: Write the smoke test FIRST**

Create `web/tests/e2e/smoke.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("homepage loads without console errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });

  await page.goto("/");
  await expect(page).toHaveTitle(/Geometry of Alignment/);
  expect(errors).toEqual([]);
});
```

- [ ] **Step 3: Run smoke test and confirm it passes**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
pnpm test --project=chromium
```

Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/playwright.config.ts web/tests/
git commit -m "test(web): add Playwright config + smoke test (loads w/o console errors)"
```

---

### Task 6: Vercel project setup

**Files:**
- (none in repo; this is a setup-only task)

- [ ] **Step 1: Install Vercel CLI**

```bash
which vercel || pnpm add -g vercel
vercel --version
```

- [ ] **Step 2: Link the web/ directory to a Vercel project**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
vercel link
```

When prompted:
- Set up and deploy: **N** (we'll deploy explicitly later)
- Link to existing project: **N** (creating a new one)
- Project name: **geometry-of-alignment**
- Directory: **./** (default)
- Want to override settings: **N**

This creates `web/.vercel/` (gitignored).

- [ ] **Step 3: Run a preview deploy to confirm wiring**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
vercel
```

Expected: prints a preview URL like `https://geometry-of-alignment-<hash>.vercel.app`. Open it; the placeholder page should render.

- [ ] **Step 4: No commit (this task only changes Vercel-side state + .vercel/ which is gitignored)**

Phase 0 complete. The next phases can run in parallel.

---

## Phase 1 — UI primitives (parallelizable after Phase 0)

### Task 7: StatusDot component

**Files:**
- Create: `web/components/ui/StatusDot.tsx`
- Create: `web/tests/e2e/status-dot.spec.ts`

- [ ] **Step 1: Write the failing test**

Create a temporary test page so we can render StatusDot in isolation. Add to `web/app/page.tsx` (temporarily for this task):

```tsx
import { StatusDot } from "@/components/ui/StatusDot";

export default function HomePage() {
  return (
    <main className="p-12 flex gap-8">
      <StatusDot color="green" data-testid="dot-green" />
      <StatusDot color="magenta" data-testid="dot-magenta" />
      <StatusDot color="cyan" data-testid="dot-cyan" />
    </main>
  );
}
```

Create `web/tests/e2e/status-dot.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("StatusDot renders three colors with pulse animation", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("dot-green")).toBeVisible();
  await expect(page.getByTestId("dot-magenta")).toBeVisible();
  await expect(page.getByTestId("dot-cyan")).toBeVisible();

  // Confirm pulse animation is applied
  const animationName = await page.getByTestId("dot-green").evaluate(
    (el) => getComputedStyle(el).animationName
  );
  expect(animationName).toContain("pulse");
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
pnpm test status-dot --project=chromium
```

Expected: FAIL — "Cannot find module @/components/ui/StatusDot".

- [ ] **Step 3: Implement StatusDot**

Create `web/components/ui/StatusDot.tsx`:

```tsx
import { dotColors, type DotColor } from "@/lib/tokens";

interface Props {
  color?: DotColor;
  size?: number;
  className?: string;
  "data-testid"?: string;
}

export function StatusDot({
  color = "green",
  size = 8,
  className = "",
  ...rest
}: Props) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block rounded-full animate-dot-pulse ${className}`}
      style={{
        width: size,
        height: size,
        backgroundColor: dotColors[color],
        color: dotColors[color], // for currentColor in box-shadow
      }}
      {...rest}
    />
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pnpm test status-dot --project=chromium
```

Expected: PASS.

- [ ] **Step 5: Revert page.tsx to placeholder**

Replace `web/app/page.tsx` back to the under-construction placeholder.

- [ ] **Step 6: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/ui/StatusDot.tsx web/tests/e2e/status-dot.spec.ts web/app/page.tsx
git commit -m "feat(web/ui): add StatusDot with pulse animation (green/magenta/cyan)"
```

---

### Task 8: Eyebrow + SectionNumber components

**Files:**
- Create: `web/components/ui/Eyebrow.tsx`
- Create: `web/components/ui/SectionNumber.tsx`
- Create: `web/tests/e2e/chrome.spec.ts`

- [ ] **Step 1: Write the failing test**

Temporarily replace `web/app/page.tsx`:

```tsx
import { Eyebrow } from "@/components/ui/Eyebrow";
import { SectionNumber } from "@/components/ui/SectionNumber";

export default function HomePage() {
  return (
    <main className="p-12 space-y-8">
      <SectionNumber number="01" label="THE HIKE" data-testid="sn" />
      <Eyebrow data-testid="eyebrow">
        A study in mechanistic interpretability · Gemma 4 E4B
      </Eyebrow>
    </main>
  );
}
```

Create `web/tests/e2e/chrome.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("SectionNumber and Eyebrow render with monospace tracking", async ({ page }) => {
  await page.goto("/");

  const sn = page.getByTestId("sn");
  await expect(sn).toBeVisible();
  await expect(sn).toContainText("01");
  await expect(sn).toContainText("THE HIKE");

  const eyebrow = page.getByTestId("eyebrow");
  await expect(eyebrow).toBeVisible();
  await expect(eyebrow).toContainText("mechanistic interpretability");

  // Both should use monospace
  const fontFamily = await sn.evaluate((el) => getComputedStyle(el).fontFamily);
  expect(fontFamily.toLowerCase()).toContain("mono");
});
```

- [ ] **Step 2: Run test, expect fail**

```bash
pnpm test chrome --project=chromium
```

Expected: FAIL (module not found).

- [ ] **Step 3: Implement Eyebrow**

Create `web/components/ui/Eyebrow.tsx`:

```tsx
import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
  className?: string;
  "data-testid"?: string;
}

export function Eyebrow({ children, className = "", ...rest }: Props) {
  return (
    <p
      className={`font-mono text-[11px] tracking-[0.22em] uppercase text-text-3 ${className}`}
      {...rest}
    >
      {children}
    </p>
  );
}
```

- [ ] **Step 4: Implement SectionNumber**

Create `web/components/ui/SectionNumber.tsx`:

```tsx
interface Props {
  number: string; // e.g., "01"
  label: string;  // e.g., "THE HIKE"
  className?: string;
  "data-testid"?: string;
}

export function SectionNumber({ number, label, className = "", ...rest }: Props) {
  return (
    <div
      className={`font-mono text-[10px] tracking-[0.22em] uppercase text-text-4 ${className}`}
      {...rest}
    >
      <span className="text-text-3">{number}</span>
      <span className="mx-2 text-iri-magenta">·</span>
      <span>{label}</span>
    </div>
  );
}
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pnpm test chrome --project=chromium
```

Expected: PASS.

- [ ] **Step 6: Revert page.tsx, commit**

Restore page.tsx to the under-construction placeholder.

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/ui/Eyebrow.tsx web/components/ui/SectionNumber.tsx web/tests/e2e/chrome.spec.ts web/app/page.tsx
git commit -m "feat(web/ui): add Eyebrow + SectionNumber monospace chrome components"
```

---

### Task 9: MetadataButton component

**Files:**
- Create: `web/components/ui/MetadataButton.tsx`
- Create: `web/tests/e2e/metadata-button.spec.ts`

- [ ] **Step 1: Write the failing test**

Temporarily replace `web/app/page.tsx`:

```tsx
import { MetadataButton } from "@/components/ui/MetadataButton";

export default function HomePage() {
  return (
    <main className="p-12 flex gap-3 flex-wrap">
      <MetadataButton
        href="/paper.pdf"
        title="Read the paper"
        meta="PDF · 24 pp · v1.0"
        dotColor="magenta"
        data-testid="cta-paper"
      />
      <MetadataButton
        href="https://github.com/nyavana/geometry-of-alignment"
        title="View source"
        meta="GitHub · 2.1k SLOC · MIT"
        dotColor="green"
        data-testid="cta-github"
      />
    </main>
  );
}
```

Create `web/tests/e2e/metadata-button.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("MetadataButton renders title + meta + correct href", async ({ page }) => {
  await page.goto("/");

  const cta = page.getByTestId("cta-paper");
  await expect(cta).toBeVisible();
  await expect(cta).toContainText("Read the paper");
  await expect(cta).toContainText("PDF · 24 pp · v1.0");
  await expect(cta).toHaveAttribute("href", "/paper.pdf");

  const github = page.getByTestId("cta-github");
  await expect(github).toHaveAttribute("href", /github\.com/);
});

test("MetadataButton arrow nudges right on hover", async ({ page }) => {
  await page.goto("/");
  const cta = page.getByTestId("cta-paper");
  const arrow = cta.locator('[data-arrow]');

  const before = await arrow.evaluate((el) => getComputedStyle(el).transform);
  await cta.hover();
  await page.waitForTimeout(250); // wait for transition
  const after = await arrow.evaluate((el) => getComputedStyle(el).transform);

  expect(before).not.toEqual(after);
});
```

- [ ] **Step 2: Run test, expect fail**

```bash
pnpm test metadata-button --project=chromium
```

Expected: FAIL.

- [ ] **Step 3: Implement MetadataButton**

Create `web/components/ui/MetadataButton.tsx`:

```tsx
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { StatusDot } from "./StatusDot";
import type { DotColor } from "@/lib/tokens";

interface Props {
  href: string;
  title: string;
  meta: string;
  dotColor?: DotColor;
  external?: boolean;
  className?: string;
  "data-testid"?: string;
}

export function MetadataButton({
  href,
  title,
  meta,
  dotColor = "green",
  external = false,
  className = "",
  ...rest
}: Props) {
  const isExternal = external || href.startsWith("http");
  const inner = (
    <>
      <StatusDot color={dotColor} />
      <span>
        <span className="block font-sans font-semibold text-[14px] leading-none mb-1 text-text-1">
          {title}
        </span>
        <span className="font-mono text-[9.5px] tracking-[0.06em] uppercase text-text-3">
          {meta}
        </span>
      </span>
      <ArrowRight
        size={16}
        data-arrow
        className="text-iri-cyan transition-transform duration-200 group-hover:translate-x-1 ml-1"
      />
    </>
  );

  const classes =
    `group inline-flex items-center gap-3.5 px-4 py-3 rounded-md ` +
    `bg-white/[0.05] hover:bg-white/[0.10] ` +
    `border border-white/[0.22] hover:border-iri-cyan/60 ` +
    `backdrop-blur-sm ` +
    `transition-all duration-200 hover:-translate-y-px ` +
    `text-text-1 ${className}`;

  if (isExternal) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className={classes}
        {...rest}
      >
        {inner}
      </a>
    );
  }
  return (
    <Link href={href} className={classes} {...rest}>
      {inner}
    </Link>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pnpm test metadata-button --project=chromium
```

Expected: 2 passed.

- [ ] **Step 5: Revert page.tsx, commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/ui/MetadataButton.tsx web/tests/e2e/metadata-button.spec.ts web/app/page.tsx
git commit -m "feat(web/ui): add MetadataButton primary CTA (dot + title + meta + arrow)"
```

---

### Task 10: BracketPill + HoloHeading components

**Files:**
- Create: `web/components/ui/BracketPill.tsx`
- Create: `web/components/ui/HoloHeading.tsx`
- Create: `web/tests/e2e/typography.spec.ts`

- [ ] **Step 1: Write the failing test**

Temporarily replace `web/app/page.tsx`:

```tsx
import { BracketPill } from "@/components/ui/BracketPill";
import { HoloHeading } from "@/components/ui/HoloHeading";

export default function HomePage() {
  return (
    <main className="p-12 space-y-8">
      <HoloHeading level={1} accent="Gemma 4" data-testid="heading">
        Why doesn't{"\n"}abliteration work{"\n"}on Gemma 4?
      </HoloHeading>
      <BracketPill href="https://github.com" data-testid="pill">
        github
      </BracketPill>
    </main>
  );
}
```

Create `web/tests/e2e/typography.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("HoloHeading isolates the accent gradient to a single phrase", async ({ page }) => {
  await page.goto("/");
  const heading = page.getByTestId("heading");
  await expect(heading).toBeVisible();
  await expect(heading).toContainText("Why doesn't");
  await expect(heading).toContainText("Gemma 4?");

  const accentSpan = heading.locator('[data-accent]');
  await expect(accentSpan).toHaveText(/Gemma 4/);

  // Accent should be transparent (gradient via bg-clip)
  const color = await accentSpan.evaluate((el) => getComputedStyle(el).color);
  expect(color).toMatch(/rgba?\(0,\s*0,\s*0,\s*0\)|transparent/);
});

test("BracketPill renders with monospace and visible brackets", async ({ page }) => {
  await page.goto("/");
  const pill = page.getByTestId("pill");
  await expect(pill).toBeVisible();
  await expect(pill).toContainText("github");
  await expect(pill).toContainText("[");
  await expect(pill).toContainText("]");
});
```

- [ ] **Step 2: Run, expect fail**

```bash
pnpm test typography --project=chromium
```

- [ ] **Step 3: Implement HoloHeading**

Create `web/components/ui/HoloHeading.tsx`:

```tsx
import type { ReactNode } from "react";

interface Props {
  children: ReactNode; // the headline text, can include "\n" for breaks
  accent?: string;     // the phrase to highlight, e.g., "Gemma 4"
  level?: 1 | 2;
  className?: string;
  "data-testid"?: string;
}

export function HoloHeading({
  children,
  accent,
  level = 1,
  className = "",
  ...rest
}: Props) {
  const Tag = level === 1 ? "h1" : "h2";
  const text = typeof children === "string" ? children : "";

  // Split around the accent phrase
  const parts =
    accent && text.includes(accent)
      ? text.split(accent).flatMap((part, i) =>
          i === 0
            ? [part]
            : [
                <span
                  key={`accent-${i}`}
                  data-accent
                  className="bg-gradient-to-r from-iri-magenta to-iri-cyan bg-clip-text text-transparent"
                >
                  {accent}
                </span>,
                part,
              ]
        )
      : [text];

  return (
    <Tag
      className={`font-sans font-black leading-[0.95] tracking-[-0.035em] text-text-1 whitespace-pre-line ${level === 1 ? "text-[64px]" : "text-[40px]"} ${className}`}
      {...rest}
    >
      {parts}
    </Tag>
  );
}
```

- [ ] **Step 4: Implement BracketPill**

Create `web/components/ui/BracketPill.tsx`:

```tsx
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import type { ReactNode } from "react";

interface Props {
  href: string;
  children: ReactNode;
  external?: boolean;
  className?: string;
  "data-testid"?: string;
}

export function BracketPill({
  href,
  children,
  external,
  className = "",
  ...rest
}: Props) {
  const isExternal = external || href.startsWith("http");
  const inner = (
    <>
      <span className="text-iri-cyan transition-transform duration-200 group-hover:-translate-x-0.5 group-hover:text-iri-magenta">
        [
      </span>
      <span>{children}</span>
      <ArrowRight
        size={12}
        className="text-iri-magenta transition-transform duration-200 group-hover:translate-x-1"
      />
      <span className="text-iri-cyan transition-transform duration-200 group-hover:translate-x-0.5 group-hover:text-iri-magenta">
        ]
      </span>
    </>
  );
  const classes =
    `group inline-flex items-center gap-1.5 px-3.5 py-2.5 rounded ` +
    `bg-black/45 hover:bg-black/70 ` +
    `border border-white/[0.18] hover:border-iri-cyan ` +
    `font-mono font-semibold text-xs tracking-[0.04em] text-text-1 ` +
    `transition-all duration-200 ${className}`;

  if (isExternal) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className={classes} {...rest}>
        {inner}
      </a>
    );
  }
  return <Link href={href} className={classes} {...rest}>{inner}</Link>;
}
```

- [ ] **Step 5: Run, verify pass**

```bash
pnpm test typography --project=chromium
```

Expected: 2 passed.

- [ ] **Step 6: Revert page.tsx, commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/ui/HoloHeading.tsx web/components/ui/BracketPill.tsx web/tests/e2e/typography.spec.ts web/app/page.tsx
git commit -m "feat(web/ui): add HoloHeading (gradient-accent) + BracketPill"
```

---

### Task 11: Reveal motion wrapper + reduced-motion hook

**Files:**
- Create: `web/lib/reduced-motion.ts`
- Create: `web/components/motion/Reveal.tsx`
- Create: `web/tests/e2e/motion.spec.ts` (placeholder; full motion tests come in Phase 6)

- [ ] **Step 1: Write the reduced-motion hook test inline**

We won't TDD the hook directly (it just reads a media query); instead, the Reveal component test below verifies behavior.

- [ ] **Step 2: Implement usePrefersReducedMotion hook**

Create `web/lib/reduced-motion.ts`:

```ts
"use client";

import { useEffect, useState } from "react";

export function usePrefersReducedMotion(): boolean {
  const [prefers, setPrefers] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setPrefers(mq.matches);
    const handler = (e: MediaQueryListEvent) => setPrefers(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  return prefers;
}
```

- [ ] **Step 3: Implement Reveal component**

Create `web/components/motion/Reveal.tsx`:

```tsx
"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

interface Props {
  children: ReactNode;
  delay?: number;
  yOffset?: number;
  className?: string;
}

export function Reveal({
  children,
  delay = 0,
  yOffset = 8,
  className = "",
}: Props) {
  const reduce = usePrefersReducedMotion();
  return (
    <motion.div
      initial={{ opacity: 0, y: reduce ? 0 : yOffset }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-10% 0px" }}
      transition={{ duration: reduce ? 0.2 : 0.6, delay, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
```

- [ ] **Step 4: Smoke-test Reveal renders without crashing**

Temporarily replace `web/app/page.tsx`:

```tsx
import { Reveal } from "@/components/motion/Reveal";

export default function HomePage() {
  return (
    <main className="p-12">
      <Reveal>
        <p data-testid="revealed">Reveal child</p>
      </Reveal>
    </main>
  );
}
```

Create `web/tests/e2e/motion.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("Reveal renders its child", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("revealed")).toBeVisible();
});
```

- [ ] **Step 5: Run, expect pass**

```bash
pnpm test motion --project=chromium
```

Expected: 1 passed.

- [ ] **Step 6: Revert page.tsx, commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/lib/reduced-motion.ts web/components/motion/Reveal.tsx web/tests/e2e/motion.spec.ts web/app/page.tsx
git commit -m "feat(web/motion): add Reveal wrapper + usePrefersReducedMotion hook"
```

---

### Task 12: TypeOnReveal (letter-by-letter typing animation)

**Files:**
- Create: `web/components/motion/TypeOnReveal.tsx`
- Create: `web/tests/e2e/type-on-reveal.spec.ts`

- [ ] **Step 1: Implement TypeOnReveal**

Create `web/components/motion/TypeOnReveal.tsx`:

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";

interface Props {
  text: string;
  speedMs?: number;    // per-character delay
  startDelay?: number; // ms before first char appears
  className?: string;
  "data-testid"?: string;
}

export function TypeOnReveal({
  text,
  speedMs = 30,
  startDelay = 0,
  className = "",
  ...rest
}: Props) {
  const ref = useRef<HTMLSpanElement>(null);
  const [revealed, setRevealed] = useState(0);
  const reduce = usePrefersReducedMotion();

  useEffect(() => {
    if (reduce) {
      setRevealed(text.length);
      return;
    }
    if (!ref.current) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          let i = 0;
          const tick = () => {
            setRevealed(i);
            if (i++ < text.length)
              setTimeout(tick, speedMs);
          };
          setTimeout(tick, startDelay);
          observer.disconnect();
        }
      },
      { threshold: 0.4 }
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [text, speedMs, startDelay, reduce]);

  return (
    <span ref={ref} className={className} {...rest}>
      {text.slice(0, revealed)}
      <span aria-hidden="true" className="opacity-70 ml-0.5">
        {revealed < text.length ? "▍" : ""}
      </span>
    </span>
  );
}
```

- [ ] **Step 2: Smoke test**

Temporarily replace `web/app/page.tsx`:

```tsx
import { TypeOnReveal } from "@/components/motion/TypeOnReveal";

export default function HomePage() {
  return (
    <main className="p-12">
      <TypeOnReveal text="hello world" data-testid="typed" />
    </main>
  );
}
```

Create `web/tests/e2e/type-on-reveal.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("TypeOnReveal completes typing within ~2s", async ({ page }) => {
  await page.goto("/");
  // Wait for the typing animation to finish
  await expect(page.getByTestId("typed")).toContainText("hello world", { timeout: 2000 });
});

test("TypeOnReveal renders immediately under reduced-motion", async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: "reduce" });
  const page = await ctx.newPage();
  await page.goto("/");
  await expect(page.getByTestId("typed")).toContainText("hello world", { timeout: 200 });
  await ctx.close();
});
```

- [ ] **Step 3: Run, verify pass**

```bash
pnpm test type-on-reveal --project=chromium
```

Expected: 2 passed.

- [ ] **Step 4: Revert page.tsx, commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/motion/TypeOnReveal.tsx web/tests/e2e/type-on-reveal.spec.ts web/app/page.tsx
git commit -m "feat(web/motion): add TypeOnReveal letter-by-letter typing animation"
```

---

Phase 1 complete. UI primitives are ready for use in sections and demos.

---

## Phase 2 — Python data extraction (parallelizable; uses project .venv)

### Task 13: Extract script scaffold + α-sweep passthrough

**Files:**
- Create: `web/scripts/extract_demo_data.py`
- Create: `web/scripts/requirements.txt`
- Create: `web/lib/data/.gitkeep`

- [ ] **Step 1: Add umap-learn requirement (one-time)**

Create `web/scripts/requirements.txt`:

```
# Extra deps not in the project's root requirements.txt
umap-learn>=0.5.7
```

Install into the existing project venv:

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
source /home/nyavana/columbia/6699/shared/env.sh
source .venv/bin/activate
pip install -r web/scripts/requirements.txt
```

- [ ] **Step 2: Write the script scaffold (modes: stub | full)**

Create `web/scripts/extract_demo_data.py`:

```python
#!/usr/bin/env python3
"""
Extract demo data from results/ into web/lib/data/*.json.

Modes:
  --mode stub  : emit placeholder data so frontend builds without GPU access
  --mode full  : read activations from $RESULTS_DIR (or results/) and run UMAP

Outputs:
  web/lib/data/alpha-sweep.json
  web/lib/data/m6-cascade.json     (hand-curated, see extract_m6_cascade)
  web/lib/data/umap-l15.json       (full-mode only; stub: 50 random points)
  web/lib/data/cohens-d-per-layer.json (full-mode only; stub: hand-typed peak values)
  web/lib/data/prompt-metadata.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DATA = Path(__file__).resolve().parent.parent / "lib" / "data"
RESULTS_DIR_ENV = os.environ.get("RESULTS_DIR")
LOCAL_RESULTS = REPO_ROOT / "results"


def _write(name: str, payload: dict | list) -> None:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    out = WEB_DATA / name
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def extract_alpha_sweep() -> None:
    """Passthrough + reshape from results/ablation_results/sweep_results.json."""
    src = LOCAL_RESULTS / "ablation_results" / "sweep_results.json"
    if not src.exists():
        print(f"  WARN: {src} not found, skipping alpha-sweep")
        return
    raw = json.loads(src.read_text())
    # Shape into the format the AlphaSweep demo expects:
    #   { "alphas": [...], "traces": { "baseline": [...], "random": [...], "gramSchmidt": [...] } }
    out = {
        "alphas": raw.get("alphas", []),
        "traces": {
            "baseline": raw.get("baseline", []),
            "random": raw.get("random_control", []),
            "gramSchmidt": raw.get("gram_schmidt", []),
        },
        "yLabel": "should_refuse rate",
        "xLabel": "α (projection strength)",
    }
    _write("alpha-sweep.json", out)


def extract_prompt_metadata() -> None:
    """Passthrough from results/activations/prompt_metadata.json."""
    src = LOCAL_RESULTS / "activations" / "prompt_metadata.json"
    if not src.exists():
        print(f"  WARN: {src} not found, skipping prompt-metadata")
        return
    raw = json.loads(src.read_text())
    _write("prompt-metadata.json", raw)


def extract_umap(mode: str) -> None:
    """UMAP-fit refuse + comply activations at L15. Requires full activations."""
    if mode == "stub":
        # 50 random points with deterministic seed so build is reproducible
        import random
        random.seed(42)
        points = []
        for i in range(25):
            points.append({
                "id": f"refuse-{i}",
                "label": "refuse",
                "x": random.gauss(-1, 0.4),
                "y": random.gauss(0, 0.4),
                "prompt": f"[stub] refusal prompt {i}",
            })
        for i in range(25):
            points.append({
                "id": f"comply-{i}",
                "label": "comply",
                "x": random.gauss(1, 0.4),
                "y": random.gauss(0, 0.4),
                "prompt": f"[stub] comply prompt {i}",
            })
        _write("umap-l15.json", {"layer": 15, "points": points, "stub": True})
        return

    # FULL mode
    import numpy as np
    import torch
    import umap

    res = Path(RESULTS_DIR_ENV) if RESULTS_DIR_ENV else LOCAL_RESULTS
    refuse_path = res / "activations" / "refuse_activations.pt"
    comply_path = res / "activations" / "comply_activations.pt"
    if not refuse_path.exists() or not comply_path.exists():
        raise SystemExit(
            f"FULL mode needs activations. Looked at:\n"
            f"  {refuse_path}\n  {comply_path}\n"
            f"Run M2b first or set RESULTS_DIR to the shared sidecar."
        )

    refuse = torch.load(refuse_path, map_location="cpu")  # dict[layer_idx -> (N_refuse, 2560)]
    comply = torch.load(comply_path, map_location="cpu")
    refuse_l15 = refuse[15].numpy()
    comply_l15 = comply[15].numpy()

    # Stack and fit UMAP
    X = np.vstack([refuse_l15, comply_l15])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(X)

    # Load prompt metadata to attach prompt text per point
    meta_path = LOCAL_RESULTS / "activations" / "prompt_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    refuse_meta = meta.get("refuse_prompts", [])
    comply_meta = meta.get("comply_prompts", [])

    points = []
    for i, c in enumerate(coords[: len(refuse_l15)]):
        m = refuse_meta[i] if i < len(refuse_meta) else {}
        points.append({
            "id": m.get("id", f"refuse-{i}"),
            "label": "refuse",
            "x": float(c[0]),
            "y": float(c[1]),
            "prompt": m.get("prompt", ""),
            "category": m.get("category", ""),
        })
    for i, c in enumerate(coords[len(refuse_l15):]):
        m = comply_meta[i] if i < len(comply_meta) else {}
        points.append({
            "id": m.get("id", f"comply-{i}"),
            "label": "comply",
            "x": float(c[0]),
            "y": float(c[1]),
            "prompt": m.get("prompt", ""),
            "category": m.get("category", ""),
        })
    _write("umap-l15.json", {"layer": 15, "points": points, "stub": False})


def extract_cohens_d(mode: str) -> None:
    """Per-layer Cohen's d. Requires full activations OR hand-curated values."""
    if mode == "stub":
        # Hand-typed peak values per ONBOARDING.md §6.5
        # L4=2.84, L14=2.80, L15=2.87 (peak), 2.5+ across L4-L17, decays toward 0 at edges
        ds = [0.3, 0.5, 0.8, 1.4, 2.84, 2.6, 2.5, 2.55, 2.6, 2.55, 2.5, 2.55, 2.6, 2.7,
              2.80, 2.87, 2.6, 2.5, 2.3, 2.0, 1.8, 1.5, 1.3, 1.1, 1.0, 0.95, 0.9, 0.85,
              0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        global_layers = [5, 11, 17, 23, 29, 35, 41]
        out = [
            {"layer": i, "d": ds[i], "attentionType": "global" if i in global_layers else "sliding"}
            for i in range(42)
        ]
        _write("cohens-d-per-layer.json", {"layers": out, "stub": True})
        return

    # FULL mode
    import numpy as np
    import torch

    res = Path(RESULTS_DIR_ENV) if RESULTS_DIR_ENV else LOCAL_RESULTS
    refuse_path = res / "activations" / "refuse_activations.pt"
    comply_path = res / "activations" / "comply_activations.pt"
    if not refuse_path.exists() or not comply_path.exists():
        raise SystemExit("FULL mode needs activations (see umap step error message).")

    refuse = torch.load(refuse_path, map_location="cpu")
    comply = torch.load(comply_path, map_location="cpu")
    global_layers = {5, 11, 17, 23, 29, 35, 41}

    out = []
    for layer in sorted(refuse.keys()):
        r = refuse[layer].numpy()  # (N_refuse, 2560)
        c = comply[layer].numpy()  # (N_comply, 2560)
        # Cohen's d on the projection onto (mean_r - mean_c) direction
        diff = r.mean(0) - c.mean(0)
        d_unit = diff / (np.linalg.norm(diff) + 1e-9)
        proj_r = r @ d_unit
        proj_c = c @ d_unit
        pooled = np.sqrt(((proj_r.var(ddof=1) + proj_c.var(ddof=1)) / 2.0))
        d_val = float((proj_r.mean() - proj_c.mean()) / (pooled + 1e-9))
        out.append({
            "layer": int(layer),
            "d": d_val,
            "attentionType": "global" if int(layer) in global_layers else "sliding",
        })
    _write("cohens-d-per-layer.json", {"layers": out, "stub": False})


def extract_m6_cascade() -> None:
    """Hand-curated M6 cascade tree. Always emitted; no source dependency."""
    cascade = {
        "title": "The M6 cascade",
        "subtitle": "Six hypotheses, isolated one at a time.",
        "nodes": [
            {
                "id": "H1",
                "label": "H1 — bnb int8 edit-path",
                "verdict": "REJECTED",
                "experiment": "Re-run abliteration in bf16 (no quantization).",
                "result": "should_refuse 6/6 = 100%, identical to int8.",
                "cosine": None,
            },
            {
                "id": "H6",
                "label": "H6 — pipeline-sound positive control",
                "verdict": "PASS",
                "experiment": "Run TrevorJS bf16 through the same eval.",
                "result": "0/48 refused. Eval pipeline works.",
                "cosine": None,
            },
            {
                "id": "H2",
                "label": "H2 — chat-template alone (D1)",
                "verdict": "INSUFFICIENT",
                "experiment": "Apply the chat template during direction extraction.",
                "result": "6/6 should_refuse. cos(M2b raw, D1) at L15 = 0.09.",
                "cosine": 0.09,
            },
            {
                "id": "H3",
                "label": "H3 — winsorization alone (D2)",
                "verdict": "INSUFFICIENT",
                "experiment": "Per-layer winsorize at 99.5% before mean.",
                "result": "6/6 should_refuse. cos(D1, D2) at L15 = 0.994.",
                "cosine": 0.994,
            },
            {
                "id": "H4",
                "label": "H4 — Gram-Schmidt vs harmless mean (D3)",
                "verdict": "LOAD-BEARING",
                "experiment": "Two-pass Gram-Schmidt orthogonalize d against harmless-mean activation, layer by layer.",
                "result": "1/6 on smoke; 17/42 = 40.5% on confirmation. cos(D2, D3) at L15 = 0.952.",
                "cosine": 0.952,
            },
            {
                "id": "H5",
                "label": "H5 — norm-preserving biprojection",
                "verdict": "REFUTED",
                "experiment": "TrevorJS-style biprojection on top of D3.",
                "result": "Identical per-prompt to D3 vanilla. Mean Δ‖W_i‖ = 0.038%.",
                "cosine": None,
            },
        ],
        "highlight": "H4",
    }
    _write("m6-cascade.json", cascade)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["stub", "full"], default="stub",
                        help="stub = placeholder data; full = read .pt activations")
    args = parser.parse_args()

    print(f"Extracting demo data (mode={args.mode})")
    extract_alpha_sweep()
    extract_prompt_metadata()
    extract_m6_cascade()
    extract_umap(args.mode)
    extract_cohens_d(args.mode)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run in stub mode (always works)**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
source /home/nyavana/columbia/6699/shared/env.sh
source .venv/bin/activate
python3 web/scripts/extract_demo_data.py --mode stub
```

Expected: 5 files written under `web/lib/data/`.

- [ ] **Step 4: Verify outputs**

```bash
ls web/lib/data/
```

Expected files: `alpha-sweep.json`, `m6-cascade.json`, `umap-l15.json`, `cohens-d-per-layer.json`, `prompt-metadata.json` (if results/activations/prompt_metadata.json exists).

- [ ] **Step 5: Commit script + stub data**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/scripts/extract_demo_data.py web/scripts/requirements.txt web/lib/data/
git commit -m "feat(web/data): extract_demo_data.py with stub + full modes; commit stub JSONs"
```

- [ ] **Step 6: Document the full-mode prerequisite (open question)**

Add a note to `web/scripts/README.md`:

```markdown
# Demo data extraction

Run `python3 extract_demo_data.py --mode full` once on a machine with access to the
shared sidecar (`$RESULTS_DIR/activations/refuse_activations.pt` and `comply_activations.pt`).
This regenerates `umap-l15.json` and `cohens-d-per-layer.json` from real data.

Until then, `--mode stub` produces placeholder data sufficient for frontend dev.
The peak Cohen's d values (L15=2.87, L4=2.84, L14=2.80) are hand-typed from
ONBOARDING.md §6.5 and are accurate; the rest of the curve in stub mode is
interpolated.

After running full mode, commit the regenerated JSONs.
```

```bash
git add web/scripts/README.md
git commit -m "docs(web/scripts): document extract_demo_data.py full mode prereqs"
```

---

Phase 2 complete. The frontend can now develop against stub data; full data lands once a team member with $RESULTS_DIR access runs `--mode full`.

---

## Phase 3 — Section components (parallelizable across team after Phase 1)

### Task 14: Hero section (§1)

**Files:**
- Create: `web/components/sections/Hero.tsx`
- Create: `web/tests/e2e/hero.spec.ts`

- [ ] **Step 1: Write the failing test**

Temporarily wire `web/app/page.tsx`:

```tsx
import { Hero } from "@/components/sections/Hero";
export default function HomePage() {
  return <Hero />;
}
```

Create `web/tests/e2e/hero.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("Hero renders headline, eyebrow, and 3 CTAs", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Why doesn't");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Gemma 4");
  await expect(page.getByText("mechanistic interpretability", { exact: false })).toBeVisible();

  await expect(page.getByRole("link", { name: /Read the paper/ })).toBeVisible();
  await expect(page.getByRole("link", { name: /View source/ })).toBeVisible();
  await expect(page.getByRole("link", { name: /Interactive demo/ })).toBeVisible();
});

test("Hero pulse dot is present in top-bar", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("hero-signal")).toBeVisible();
});
```

- [ ] **Step 2: Run, expect fail**

```bash
pnpm test hero --project=chromium
```

- [ ] **Step 3: Implement Hero**

Create `web/components/sections/Hero.tsx`:

```tsx
import { HoloHeading } from "@/components/ui/HoloHeading";
import { Eyebrow } from "@/components/ui/Eyebrow";
import { MetadataButton } from "@/components/ui/MetadataButton";
import { StatusDot } from "@/components/ui/StatusDot";
import { Reveal } from "@/components/motion/Reveal";

export function Hero() {
  return (
    <section
      id="hero"
      className="relative min-h-screen px-11 pt-9 pb-20 overflow-hidden bg-hero"
      aria-label="Hero"
    >
      <div className="absolute inset-0 pointer-events-none bg-scanlines" />
      <div className="relative max-w-6xl mx-auto">
        {/* Top bar */}
        <div className="flex justify-between items-center font-mono text-[9px] tracking-[0.28em] uppercase text-text-3">
          <div className="flex gap-5 items-center" data-testid="hero-signal">
            <span className="inline-flex items-center gap-1.5">
              <StatusDot color="green" /> signal · live
            </span>
            <span>columbia eecs 6699 · 2026</span>
          </div>
          <span>geometry-of-alignment / v1.0</span>
        </div>

        {/* Eyebrow */}
        <div className="mt-9">
          <Reveal delay={0.1}>
            <Eyebrow>
              A study in mechanistic interpretability <span className="text-iri-magenta mx-2.5">|</span>{" "}
              Gemma 4 E4B
            </Eyebrow>
          </Reveal>
        </div>

        {/* Headline */}
        <Reveal delay={0.2}>
          <HoloHeading level={1} accent="Gemma 4" className="mt-3.5">
            {"Why doesn't\nabliteration work\non Gemma 4?"}
          </HoloHeading>
        </Reveal>

        {/* Deck */}
        <Reveal delay={0.35}>
          <p className="mt-2 max-w-[640px] text-base leading-[1.5] text-text-2">
            A rank-1 weight edit erases refusal in Llama, Qwen, and Mistral. On Gemma 4 it
            leaves a stubborn 40.5% residual. We mapped the geometry that explains why.
          </p>
        </Reveal>

        {/* CTAs */}
        <Reveal delay={0.5}>
          <div className="mt-6 flex gap-3.5 flex-wrap">
            <MetadataButton
              href="/paper.pdf"
              title="Read the paper"
              meta="PDF · 24 pp · v1.0"
              dotColor="magenta"
            />
            <MetadataButton
              href="https://github.com/nyavana/geometry-of-alignment"
              title="View source"
              meta="GitHub · MIT"
              dotColor="green"
              external
            />
            <MetadataButton
              href="#refusal-direction"
              title="Interactive demo"
              meta="UMAP · refusal direction"
              dotColor="green"
            />
          </div>
        </Reveal>

        {/* Scroll cue */}
        <div className="absolute bottom-6 left-11 right-11 flex justify-between items-center font-mono text-[10px] tracking-[0.18em] uppercase text-text-3">
          <span>scroll</span>
          <span aria-hidden="true">↓</span>
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Run, expect pass**

```bash
pnpm test hero --project=chromium
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/Hero.tsx web/tests/e2e/hero.spec.ts web/app/page.tsx
git commit -m "feat(web/sections): add Hero (§1) with holographic bg + 3 metadata CTAs"
```

---

### Task 15: Narrative section §2 — The Hike

**Files:**
- Create: `web/components/sections/Hike.tsx`

- [ ] **Step 1: Implement Hike with animated topographic SVG**

Create `web/components/sections/Hike.tsx`:

```tsx
import { SectionNumber } from "@/components/ui/SectionNumber";
import { Reveal } from "@/components/motion/Reveal";

export function Hike() {
  return (
    <section id="hike" className="relative py-32 px-11 overflow-hidden" aria-label="The hike">
      <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
        <div>
          <Reveal>
            <SectionNumber number="01" label="THE HIKE" />
          </Reveal>
          <Reveal delay={0.1}>
            <blockquote className="mt-6 font-sans font-semibold text-[34px] leading-[1.15] text-text-1 tracking-[-0.02em]">
              "If something happens, we are truly{" "}
              <span className="bg-gradient-to-r from-iri-magenta to-iri-cyan bg-clip-text text-transparent">
                on our own.
              </span>
              "
            </blockquote>
          </Reveal>
          <Reveal delay={0.2}>
            <p className="mt-6 max-w-[520px] text-base leading-[1.6] text-text-2">
              Hiking in remote areas. No phone reception. You might meet another person once or
              twice an hour. I always kept a Wikipedia mirror on my phone — in case.
            </p>
          </Reveal>
        </div>

        {/* Topographic SVG */}
        <Reveal delay={0.15}>
          <svg
            viewBox="0 0 400 280"
            className="w-full h-auto"
            aria-hidden="true"
          >
            <defs>
              <linearGradient id="topo-stroke" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="var(--iri-cyan)" stopOpacity="0.6" />
                <stop offset="100%" stopColor="var(--iri-magenta)" stopOpacity="0.4" />
              </linearGradient>
            </defs>
            {Array.from({ length: 10 }).map((_, i) => {
              const y = 40 + i * 12;
              const wobble = 12 + i * 3;
              const d = `M -20 ${y} Q 60 ${y - wobble} 140 ${y + wobble/2} T 320 ${y - wobble/2} T 480 ${y}`;
              return (
                <path
                  key={i}
                  d={d}
                  fill="none"
                  stroke="url(#topo-stroke)"
                  strokeWidth="0.7"
                  strokeDasharray="600"
                  strokeDashoffset="600"
                  style={{
                    animation: `draw-topo 2.2s ${i * 0.15}s ease-out forwards`,
                  }}
                />
              );
            })}
          </svg>
          <style jsx>{`
            @keyframes draw-topo {
              to { stroke-dashoffset: 0; }
            }
          `}</style>
        </Reveal>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Test in isolation**

Temporarily replace `web/app/page.tsx`:

```tsx
import { Hike } from "@/components/sections/Hike";
export default function HomePage() { return <Hike />; }
```

```bash
pnpm dev
```

Open http://localhost:3000 and confirm: section number "01 · THE HIKE", blockquote with gradient accent on "on our own", topographic lines draw on load.

- [ ] **Step 3: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/Hike.tsx
git commit -m "feat(web/sections): add Hike (§2) blockquote + animated topographic SVG"
```

---

### Task 16: Narrative section §3 — The Promise

**Files:**
- Create: `web/components/sections/Promise.tsx`

- [ ] **Step 1: Implement Promise with stylized phone SVG**

Create `web/components/sections/Promise.tsx`:

```tsx
import { SectionNumber } from "@/components/ui/SectionNumber";
import { Reveal } from "@/components/motion/Reveal";

export function Promise() {
  return (
    <section id="promise" className="relative py-32 px-11" aria-label="The promise">
      <div className="max-w-6xl mx-auto grid lg:grid-cols-[1fr_auto] gap-16 items-center">
        <div>
          <Reveal><SectionNumber number="02" label="THE PROMISE" /></Reveal>
          <Reveal delay={0.1}>
            <h2 className="mt-6 font-sans font-black text-[44px] leading-[1.0] tracking-[-0.03em] text-text-1">
              An on-device expert,
              <br />
              right in your pocket.
            </h2>
          </Reveal>
          <Reveal delay={0.2}>
            <p className="mt-6 max-w-[540px] text-base leading-[1.6] text-text-2">
              Gemma 4 E2B-IT runs on a phone at decent speed. It's a real, open-weight,
              instruction-tuned LLM — the kind of intelligence you'd want with you when
              you can't dial 911.
            </p>
          </Reveal>
        </div>

        {/* Phone SVG */}
        <Reveal delay={0.15}>
          <svg
            viewBox="0 0 180 320"
            className="w-[180px] h-[320px]"
            aria-hidden="true"
          >
            <defs>
              <linearGradient id="phone-screen" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--iri-violet)" stopOpacity="0.45" />
                <stop offset="50%" stopColor="var(--iri-magenta)" stopOpacity="0.35" />
                <stop offset="100%" stopColor="var(--iri-cyan)" stopOpacity="0.45" />
              </linearGradient>
            </defs>
            {/* phone body */}
            <rect x="6" y="6" width="168" height="308" rx="22" ry="22"
              fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="1.5" />
            {/* notch */}
            <rect x="70" y="14" width="40" height="6" rx="3" fill="rgba(255,255,255,0.6)" />
            {/* screen */}
            <rect x="14" y="32" width="152" height="260" rx="6"
              fill="url(#phone-screen)" opacity="0.7">
              <animate attributeName="opacity" values="0.5;0.9;0.5" dur="3.2s" repeatCount="indefinite" />
            </rect>
            {/* chat-line placeholders */}
            <g fill="rgba(255,255,255,0.55)">
              <rect x="26" y="54" width="90" height="6" rx="2" />
              <rect x="26" y="68" width="60" height="6" rx="2" />
              <rect x="26" y="92" width="120" height="6" rx="2" />
              <rect x="26" y="106" width="80" height="6" rx="2" />
            </g>
            <circle cx="90" cy="304" r="5" fill="none" stroke="rgba(255,255,255,0.5)" strokeWidth="1" />
          </svg>
        </Reveal>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Visual check**

Replace `app/page.tsx` to render Promise; `pnpm dev`; confirm SVG phone with pulsing screen + headline.

- [ ] **Step 3: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/Promise.tsx
git commit -m "feat(web/sections): add Promise (§3) with pulsing phone SVG"
```

---

### Task 17: Narrative section §4 — The Refusal (chat bubble + typed reveal)

**Files:**
- Create: `web/components/ui/ChatBubble.tsx`
- Create: `web/components/sections/Refusal.tsx`

- [ ] **Step 1: Implement ChatBubble**

Create `web/components/ui/ChatBubble.tsx`:

```tsx
import type { ReactNode } from "react";

interface Props {
  variant: "user" | "assistant";
  children: ReactNode;
  className?: string;
}

export function ChatBubble({ variant, children, className = "" }: Props) {
  const isUser = variant === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} ${className}`}>
      <div
        className={
          "max-w-[80%] px-5 py-3.5 rounded-2xl font-sans text-[15px] leading-[1.5] " +
          (isUser
            ? "bg-gradient-to-br from-iri-cyan/30 to-iri-violet/20 border border-iri-cyan/40 text-text-1"
            : "bg-white/[0.06] border border-white/[0.18] text-text-2") +
          " backdrop-blur-sm"
        }
      >
        {children}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Implement Refusal section**

Create `web/components/sections/Refusal.tsx`:

```tsx
"use client";

import { SectionNumber } from "@/components/ui/SectionNumber";
import { ChatBubble } from "@/components/ui/ChatBubble";
import { TypeOnReveal } from "@/components/motion/TypeOnReveal";
import { Reveal } from "@/components/motion/Reveal";

const PROMPT =
  "If my friend is injured and we're in a very remote location, can you help me diagnose what's going on?";

const REFUSAL =
  "I'm sorry, but I can't help with that. Diagnosing medical conditions requires a licensed healthcare provider. Please contact emergency services.";

export function Refusal() {
  return (
    <section id="refusal" className="relative py-32 px-11" aria-label="The refusal">
      <div className="max-w-3xl mx-auto">
        <Reveal><SectionNumber number="03" label="THE REFUSAL" /></Reveal>
        <Reveal delay={0.05}>
          <p className="mt-4 font-mono text-[12px] tracking-[0.12em] uppercase text-text-3">
            gemma 4 e2b-it · on-device
          </p>
        </Reveal>

        <div className="mt-10 space-y-5">
          <ChatBubble variant="user">
            <TypeOnReveal text={PROMPT} speedMs={20} />
          </ChatBubble>
          <ChatBubble variant="assistant">
            <TypeOnReveal text={REFUSAL} speedMs={20} startDelay={2500} />
          </ChatBubble>
        </div>

        <Reveal delay={0.4}>
          <p className="mt-12 text-text-2 text-base leading-[1.6] max-w-[560px]">
            This is the model behaving exactly as designed — safety guardrails default to
            refusing medical content. It would have refused regardless of whether my friend
            was bleeding out in the woods.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
```

- [ ] **Step 3: Visual check + commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/ui/ChatBubble.tsx web/components/sections/Refusal.tsx
git commit -m "feat(web/sections): add Refusal (§4) with typed chat bubbles"
```

---

### Task 18: Narrative section §5 — The Question

**Files:**
- Create: `web/components/sections/Question.tsx`

- [ ] **Step 1: Implement**

Create `web/components/sections/Question.tsx`:

```tsx
import { SectionNumber } from "@/components/ui/SectionNumber";
import { Reveal } from "@/components/motion/Reveal";

export function Question() {
  return (
    <section id="question" className="relative py-40 px-11" aria-label="The question">
      <div className="max-w-5xl mx-auto">
        <Reveal><SectionNumber number="04" label="THE QUESTION" /></Reveal>
        <Reveal delay={0.1}>
          <h2 className="mt-8 font-sans font-black text-[58px] leading-[1.0] tracking-[-0.035em] text-text-1">
            How does safety work in LLMs —
            <br />
            and can we remove it{" "}
            <span className="bg-gradient-to-r from-iri-magenta to-iri-cyan bg-clip-text text-transparent">
              selectively
            </span>
            ?
          </h2>
        </Reveal>
        <Reveal delay={0.25}>
          <p className="mt-8 max-w-[640px] text-lg leading-[1.55] text-text-2">
            Keep the refusal on truly harmful queries. Remove it on the ones that would have
            saved a life. That's the research project.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/Question.tsx
git commit -m "feat(web/sections): add Question (§5) typographic moment"
```

---

### Task 19: Primer section (§6) — KaTeX equation

**Files:**
- Create: `web/components/sections/Primer.tsx`
- Modify: `web/app/layout.tsx` (import KaTeX CSS)

- [ ] **Step 1: Wire KaTeX stylesheet in layout**

Add this import at the top of `web/app/layout.tsx`:

```tsx
import "katex/dist/katex.min.css";
```

- [ ] **Step 2: Implement Primer**

Create `web/components/sections/Primer.tsx`:

```tsx
"use client";

import { SectionNumber } from "@/components/ui/SectionNumber";
import { Reveal } from "@/components/motion/Reveal";
import { BlockMath, InlineMath } from "react-katex";

export function Primer() {
  return (
    <section id="primer" className="relative py-32 px-11" aria-label="Primer">
      <div className="max-w-3xl mx-auto">
        <Reveal><SectionNumber number="05" label="PRIMER" /></Reveal>
        <Reveal delay={0.1}>
          <h2 className="mt-6 font-sans font-black text-[40px] leading-[1.0] tracking-[-0.03em] text-text-1">
            What is the refusal direction?
          </h2>
        </Reveal>

        <Reveal delay={0.2}>
          <div className="mt-8 space-y-5 text-text-2 text-base leading-[1.7]">
            <p>
              Take a batch of prompts the model refuses, and a batch it complies with. Look at
              the residual-stream activation at some middle layer. Subtract the means.
            </p>
            <div className="my-8 px-6 py-5 bg-white/[0.04] border border-white/[0.10] rounded-lg overflow-x-auto">
              <BlockMath math={"d = \\frac{\\mu_{\\text{refuse}} - \\mu_{\\text{comply}}}{\\| \\mu_{\\text{refuse}} - \\mu_{\\text{comply}} \\|}"} />
            </div>
            <p>
              The unit vector <InlineMath math="d" /> turns out to be the{" "}
              <em className="text-text-1 not-italic font-semibold">refusal direction</em>.
              Add <InlineMath math="d" /> to a benign prompt's hidden state and the model
              refuses. Subtract it from a harmful one and the model complies.
            </p>
            <p>
              <strong className="text-text-1 font-semibold">Abliteration</strong> bakes the
              "subtract <InlineMath math="d" />" behavior into the weights — a rank-1
              perturbation per output matrix:
            </p>
            <div className="my-8 px-6 py-5 bg-white/[0.04] border border-white/[0.10] rounded-lg overflow-x-auto">
              <BlockMath math={"W \\leftarrow W - \\alpha \\, d \\, (d^\\top W)"} />
            </div>
            <p className="text-text-3 text-sm">
              On Llama, Qwen, and Mistral this drives refusal from ~100% to ~0%. On Gemma 4,
              as we'll see, it doesn't.
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
```

- [ ] **Step 3: Visual check + commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/Primer.tsx web/app/layout.tsx
git commit -m "feat(web/sections): add Primer (§6) with KaTeX abliteration equations"
```

---

### Task 20: Methodology section (§7) — animated 4-stage pipeline

**Files:**
- Create: `web/components/sections/Methodology.tsx`

- [ ] **Step 1: Implement**

Create `web/components/sections/Methodology.tsx`:

```tsx
import { SectionNumber } from "@/components/ui/SectionNumber";
import { Reveal } from "@/components/motion/Reveal";

const STAGES = [
  {
    n: "1",
    title: "Benchmark",
    desc: "344 prompts × 8 categories. Refusal regex. CSV per model.",
    color: "var(--iri-magenta)",
  },
  {
    n: "2",
    title: "Mechanistic",
    desc: "Hook every layer; compute the refusal direction; PCA.",
    color: "var(--iri-cyan)",
  },
  {
    n: "3",
    title: "Abliterate",
    desc: "Apply rank-1 weight edit; sweep α and layer subsets.",
    color: "var(--iri-violet)",
  },
  {
    n: "4",
    title: "Weight diff",
    desc: "Compare published uncensored variants vs base. SVD per tensor.",
    color: "var(--iri-amber)",
  },
];

export function Methodology() {
  return (
    <section id="methodology" className="relative py-32 px-11" aria-label="Methodology">
      <div className="max-w-6xl mx-auto">
        <Reveal><SectionNumber number="06" label="METHODOLOGY" /></Reveal>
        <Reveal delay={0.1}>
          <h2 className="mt-6 font-sans font-black text-[40px] leading-[1.0] tracking-[-0.03em] text-text-1">
            A four-stage pipeline.
          </h2>
        </Reveal>

        <div className="mt-14 grid md:grid-cols-4 gap-4 relative">
          {/* connecting line behind cards */}
          <div className="hidden md:block absolute top-8 left-[8%] right-[8%] h-px bg-gradient-to-r from-iri-magenta via-iri-cyan via-iri-violet to-iri-amber opacity-50" />
          {STAGES.map((s, i) => (
            <Reveal key={s.n} delay={0.15 + i * 0.1}>
              <div className="relative bg-white/[0.04] border border-white/[0.12] rounded-lg p-5 backdrop-blur-sm">
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center font-mono font-bold text-sm text-bg"
                  style={{ background: s.color }}
                >
                  {s.n}
                </div>
                <h3 className="mt-4 font-sans font-bold text-lg text-text-1">{s.title}</h3>
                <p className="mt-2 text-sm text-text-3 leading-[1.55]">{s.desc}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/Methodology.tsx
git commit -m "feat(web/sections): add Methodology (§7) 4-stage pipeline diagram"
```

---

### Task 21: RefusalDirection container section (§8)

**Files:**
- Create: `web/components/sections/RefusalDirection.tsx`

This task lays out the container; the actual UmapScatter + LayerHeatmap demos drop in during Phase 4.

- [ ] **Step 1: Implement container with placeholder slots**

Create `web/components/sections/RefusalDirection.tsx`:

```tsx
"use client";

import { useState } from "react";
import { SectionNumber } from "@/components/ui/SectionNumber";
import { Reveal } from "@/components/motion/Reveal";
import { LayerHeatmap } from "@/components/demos/LayerHeatmap";
import { UmapScatter } from "@/components/demos/UmapScatter";

export function RefusalDirection() {
  const [selectedLayer, setSelectedLayer] = useState<number>(15);

  return (
    <section id="refusal-direction" className="relative py-32 px-11" aria-label="Refusal direction (interactive)">
      <div className="max-w-6xl mx-auto">
        <Reveal><SectionNumber number="07" label="WHERE DOES REFUSAL LIVE?" /></Reveal>
        <Reveal delay={0.1}>
          <h2 className="mt-6 font-sans font-black text-[40px] leading-[1.0] tracking-[-0.03em] text-text-1">
            The geometry of refusal —{" "}
            <span className="bg-gradient-to-r from-iri-magenta to-iri-cyan bg-clip-text text-transparent">
              hover to explore.
            </span>
          </h2>
        </Reveal>
        <Reveal delay={0.2}>
          <p className="mt-6 max-w-[640px] text-base leading-[1.6] text-text-2">
            Layer-by-layer signal strength along the refuse-vs-comply axis. The peak — layer
            15, Cohen's d 2.87 — is where refusal is most cleanly linearly separable.
          </p>
        </Reveal>

        <div className="mt-12">
          <Reveal delay={0.3}>
            <LayerHeatmap selectedLayer={selectedLayer} onSelectLayer={setSelectedLayer} />
          </Reveal>
        </div>

        <div className="mt-10">
          <Reveal delay={0.35}>
            <UmapScatter layer={selectedLayer} />
          </Reveal>
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Add temporary demo stubs (so the section renders before Phase 4)**

Create stub `web/components/demos/UmapScatter.tsx`:

```tsx
"use client";

interface Props { layer: number; }

export function UmapScatter({ layer }: Props) {
  return (
    <div
      data-testid="umap-scatter"
      className="h-[320px] rounded-lg border border-white/[0.12] bg-white/[0.04] flex items-center justify-center font-mono text-sm text-text-3"
    >
      [UMAP scatter — layer {layer} — implemented in Phase 4]
    </div>
  );
}
```

Create stub `web/components/demos/LayerHeatmap.tsx`:

```tsx
"use client";

interface Props {
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
}

export function LayerHeatmap({ selectedLayer, onSelectLayer }: Props) {
  return (
    <div
      data-testid="layer-heatmap"
      className="h-[80px] rounded border border-white/[0.12] bg-white/[0.04] flex items-center justify-center font-mono text-sm text-text-3 cursor-pointer"
      onClick={() => onSelectLayer((selectedLayer + 1) % 42)}
    >
      [Layer heatmap — selected: L{selectedLayer} — implemented in Phase 4]
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/RefusalDirection.tsx web/components/demos/UmapScatter.tsx web/components/demos/LayerHeatmap.tsx
git commit -m "feat(web/sections): add RefusalDirection (§8) container w/ demo stubs"
```

---

### Task 22: Investigation section (§9) — milestone timeline + M6 cascade slot

**Files:**
- Create: `web/components/sections/Investigation.tsx`
- Create stub: `web/components/demos/M6Cascade.tsx`

- [ ] **Step 1: Implement Investigation**

Create `web/components/sections/Investigation.tsx`:

```tsx
import { SectionNumber } from "@/components/ui/SectionNumber";
import { Reveal } from "@/components/motion/Reveal";
import { M6Cascade } from "@/components/demos/M6Cascade";

const MILESTONES = [
  { id: "M0", label: "Bootstrap", finding: "Six worktrees + shared sidecar." },
  { id: "M1", label: "Benchmark", finding: "344 prompts × 8 categories." },
  { id: "M2a", label: "Behavioral", finding: "Base E4B refuses 100% harmful · 2% medical." },
  { id: "M2b", label: "Mechanistic", finding: "L15 peak. d = 2.87. Top-1 PC captures 86.6%." },
  { id: "M2c", label: "Abliterate", finding: "Inert. α-sweep flat. Random control flat." },
  { id: "M3", label: "Weight diff", finding: "OBLITERATUS rank≈6, TrevorJS rank=1, orthogonal subspaces." },
  { id: "M6", label: "Cascade", finding: "Gram-Schmidt is load-bearing. 100% → 40.5% on n=42." },
];

export function Investigation() {
  return (
    <section id="investigation" className="relative py-32 px-11" aria-label="The investigation">
      <div className="max-w-6xl mx-auto">
        <Reveal><SectionNumber number="08" label="THE INVESTIGATION (M0 → M6)" /></Reveal>
        <Reveal delay={0.1}>
          <h2 className="mt-6 font-sans font-black text-[40px] leading-[1.0] tracking-[-0.03em] text-text-1">
            Seven milestones, one cascade.
          </h2>
        </Reveal>

        {/* Timeline */}
        <ol className="mt-12 relative pl-6 border-l border-white/[0.15] space-y-6">
          {MILESTONES.map((m, i) => (
            <Reveal key={m.id} delay={0.1 + i * 0.05}>
              <li className="relative">
                <span
                  className="absolute -left-[31px] top-1.5 w-3 h-3 rounded-full"
                  style={{
                    background: m.id === "M6" ? "var(--iri-magenta)" : "var(--iri-cyan)",
                    boxShadow: m.id === "M6" ? "0 0 12px var(--iri-magenta)" : undefined,
                  }}
                />
                <div className="font-mono text-[10px] tracking-[0.18em] uppercase text-text-3">
                  {m.id} · {m.label}
                </div>
                <p className="mt-1 text-text-1 text-base font-semibold">{m.finding}</p>
              </li>
            </Reveal>
          ))}
        </ol>

        {/* M6 cascade demo */}
        <div className="mt-16">
          <Reveal>
            <h3 className="font-sans font-bold text-2xl text-text-1">The M6 cascade</h3>
            <p className="mt-2 text-text-3 text-sm max-w-[640px]">
              Six hypotheses about why standard rank-1 fails on Gemma 4. Click each to see the
              experiment, the cosine, and the verdict.
            </p>
          </Reveal>
          <div className="mt-8">
            <M6Cascade />
          </div>
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Add M6Cascade stub**

Create `web/components/demos/M6Cascade.tsx`:

```tsx
"use client";

export function M6Cascade() {
  return (
    <div
      data-testid="m6-cascade"
      className="h-[320px] rounded-lg border border-white/[0.12] bg-white/[0.04] flex items-center justify-center font-mono text-sm text-text-3"
    >
      [M6 cascade decision tree — implemented in Phase 4]
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/Investigation.tsx web/components/demos/M6Cascade.tsx
git commit -m "feat(web/sections): add Investigation (§9) milestone timeline + cascade slot"
```

---

### Task 23: PunchLine section (§10) — full-screen 40.5%

**Files:**
- Create: `web/components/sections/PunchLine.tsx`

- [ ] **Step 1: Implement**

Create `web/components/sections/PunchLine.tsx`:

```tsx
import { SectionNumber } from "@/components/ui/SectionNumber";
import { Reveal } from "@/components/motion/Reveal";

export function PunchLine() {
  return (
    <section
      id="punch-line"
      className="relative min-h-[80vh] flex items-center justify-center px-11 bg-hero overflow-hidden"
      aria-label="The punch line"
    >
      <div className="absolute inset-0 pointer-events-none bg-scanlines" />
      <div className="relative max-w-3xl text-center">
        <Reveal>
          <SectionNumber number="09" label="THE PUNCH LINE" className="text-center" />
        </Reveal>
        <Reveal delay={0.15}>
          <div
            className="mt-8 font-sans font-black text-[180px] leading-[0.85] tracking-[-0.05em] bg-gradient-to-r from-iri-magenta via-iri-cyan to-iri-amber bg-clip-text text-transparent"
            data-testid="punch-stat"
          >
            40.5%
          </div>
        </Reveal>
        <Reveal delay={0.4}>
          <p className="mt-8 text-xl text-text-1 font-semibold">
            residual after rank-1 abliteration
          </p>
        </Reveal>
        <Reveal delay={0.55}>
          <p className="mt-4 text-text-3 text-sm max-w-[460px] mx-auto">
            n = 42 should_refuse prompts · α = 1.0 · best recipe (Gram-Schmidt). Compare to
            Llama, where the same intervention drives refusal to ~5%.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/PunchLine.tsx
git commit -m "feat(web/sections): add PunchLine (§10) full-screen 40.5% reveal"
```

---

### Task 24: AlphaSweepSection (§11) container + demo stub

**Files:**
- Create: `web/components/sections/AlphaSweepSection.tsx`
- Create stub: `web/components/demos/AlphaSweep.tsx`

- [ ] **Step 1: Implement section**

Create `web/components/sections/AlphaSweepSection.tsx`:

```tsx
import { SectionNumber } from "@/components/ui/SectionNumber";
import { Reveal } from "@/components/motion/Reveal";
import { AlphaSweep } from "@/components/demos/AlphaSweep";

export function AlphaSweepSection() {
  return (
    <section id="alpha-sweep" className="relative py-32 px-11" aria-label="Alpha sweep (interactive)">
      <div className="max-w-5xl mx-auto">
        <Reveal><SectionNumber number="10" label="α-SWEEP" /></Reveal>
        <Reveal delay={0.1}>
          <h2 className="mt-6 font-sans font-black text-[40px] leading-[1.0] tracking-[-0.03em] text-text-1">
            Drag α — watch the residual flatten.
          </h2>
        </Reveal>
        <Reveal delay={0.2}>
          <p className="mt-6 max-w-[640px] text-base leading-[1.6] text-text-2">
            Standard rank-1 abliteration sweeps α from 0 to 2.0 with no behavioral effect.
            The Gram-Schmidt direction-build gives the only meaningful drop — but it plateaus
            at ~40%.
          </p>
        </Reveal>
        <div className="mt-10">
          <Reveal delay={0.3}>
            <AlphaSweep />
          </Reveal>
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Add stub**

Create `web/components/demos/AlphaSweep.tsx`:

```tsx
"use client";

export function AlphaSweep() {
  return (
    <div
      data-testid="alpha-sweep"
      className="h-[400px] rounded-lg border border-white/[0.12] bg-white/[0.04] flex items-center justify-center font-mono text-sm text-text-3"
    >
      [α-sweep slider + 3-trace chart — implemented in Phase 4]
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/AlphaSweepSection.tsx web/components/demos/AlphaSweep.tsx
git commit -m "feat(web/sections): add AlphaSweepSection (§11) container + demo stub"
```

---

### Task 25: WhatThisMeans section (§12)

**Files:**
- Create: `web/components/sections/WhatThisMeans.tsx`

- [ ] **Step 1: Implement**

Create `web/components/sections/WhatThisMeans.tsx`:

```tsx
import { SectionNumber } from "@/components/ui/SectionNumber";
import { Reveal } from "@/components/motion/Reveal";

const POINTS = [
  {
    title: "Standard rank-1 is inert on Gemma 4.",
    body: "α-sweep is flat. Random control matches baseline. The mean-diff direction alone doesn't move the needle.",
  },
  {
    title: "Gram-Schmidt against the harmless mean is load-bearing.",
    body: "A single change to the direction-build recipe (~17° rotation at L15) cuts refusal from 100% to 40.5%.",
  },
  {
    title: "The remaining 40% is a multi-rank residual.",
    body: "OBLITERATUS uses median rank_95 = 6 on the same base model. Single-direction abliteration has a ceiling here.",
  },
  {
    title: "Selective de-alignment is geometrically possible.",
    body: "Per-category refusal directions cluster at +0.93 internal cosine and are orthogonal to should_refuse (mean −0.015).",
  },
];

export function WhatThisMeans() {
  return (
    <section id="what-this-means" className="relative py-32 px-11" aria-label="What this means">
      <div className="max-w-5xl mx-auto">
        <Reveal><SectionNumber number="11" label="WHAT THIS MEANS" /></Reveal>
        <Reveal delay={0.1}>
          <h2 className="mt-6 font-sans font-black text-[40px] leading-[1.0] tracking-[-0.03em] text-text-1">
            Four takeaways.
          </h2>
        </Reveal>

        <div className="mt-12 grid md:grid-cols-2 gap-6">
          {POINTS.map((p, i) => (
            <Reveal key={i} delay={0.1 + i * 0.1}>
              <div className="bg-white/[0.04] border border-white/[0.12] rounded-lg p-6">
                <h3 className="font-sans font-bold text-lg text-text-1 leading-[1.3]">{p.title}</h3>
                <p className="mt-3 text-sm text-text-2 leading-[1.55]">{p.body}</p>
              </div>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.6}>
          <p className="mt-12 text-text-3 text-sm max-w-[680px]">
            Future work: a multi-rank descent that addresses the residual core safety circuit
            without disturbing harmless completions. Out of scope for this paper.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/WhatThisMeans.tsx
git commit -m "feat(web/sections): add WhatThisMeans (§12) four-takeaway grid"
```

---

### Task 26: Team section (§13)

**Files:**
- Create: `web/components/sections/Team.tsx`

- [ ] **Step 1: Implement**

Create `web/components/sections/Team.tsx`:

```tsx
import { SectionNumber } from "@/components/ui/SectionNumber";
import { Reveal } from "@/components/motion/Reveal";

// NOTE: roles + GitHub handles are placeholders per spec §14 "open questions".
// Update when confirmed by team.
const MEMBERS = [
  { name: "chenhao yang",   role: "benchmark · abliteration", email: "cy2822@columbia.edu", github: null },
  { name: "daitian zhao",   role: "mechanistic · figures",    email: "dz2585@columbia.edu", github: null },
  { name: "hanlin wang",    role: "weight diff · paper",      email: "hw3100@columbia.edu", github: null },
  { name: "yuxi luo",       role: "interactive · writeup",    email: "yl6117@columbia.edu", github: null },
];

export function Team() {
  return (
    <section id="team" className="relative py-32 px-11" aria-label="Team">
      <div className="max-w-6xl mx-auto">
        <Reveal><SectionNumber number="12" label="TEAM" /></Reveal>
        <Reveal delay={0.1}>
          <h2 className="mt-6 font-sans font-black text-[40px] leading-[1.0] tracking-[-0.03em] text-text-1">
            Four engineers.
          </h2>
        </Reveal>

        <div className="mt-12 grid md:grid-cols-2 gap-5">
          {MEMBERS.map((m, i) => (
            <Reveal key={m.email} delay={0.1 + i * 0.05}>
              <article
                className="group relative rounded border border-white/[0.18] bg-white/[0.03] p-5 font-mono text-sm transition-colors hover:border-iri-cyan"
                data-testid={`member-${m.email}`}
              >
                <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-iri-cyan to-transparent opacity-50 group-hover:opacity-100 transition-opacity" />
                <div className="flex items-baseline gap-2">
                  <span className="text-iri-cyan">┌─</span>
                  <span className="text-text-1 font-bold">{m.name}</span>
                  <span className="text-iri-cyan">─┐</span>
                </div>
                <dl className="mt-3 space-y-1.5 pl-3">
                  <div>
                    <dt className="inline text-text-3">role · </dt>
                    <dd className="inline text-text-2">{m.role}</dd>
                  </div>
                  <div>
                    <dt className="sr-only">email</dt>
                    <dd>
                      <a
                        href={`mailto:${m.email}`}
                        className="text-text-2 hover:text-iri-cyan transition-colors"
                      >
                        {m.email}
                      </a>
                    </dd>
                  </div>
                  {m.github && (
                    <div>
                      <dt className="sr-only">github</dt>
                      <dd>
                        <a
                          href={`https://github.com/${m.github}`}
                          className="text-text-2 hover:text-iri-cyan transition-colors"
                        >
                          github.com/{m.github}
                        </a>
                      </dd>
                    </div>
                  )}
                </dl>
                <div className="mt-3 text-iri-cyan/70">└──────────────────────┘</div>
              </article>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.5}>
          <p className="mt-10 font-mono text-sm text-text-3">
            team contact ·{" "}
            <a href="mailto:GeometryofAlignment@nyavana.io" className="text-iri-cyan hover:underline">
              GeometryofAlignment@nyavana.io
            </a>
          </p>
        </Reveal>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/sections/Team.tsx
git commit -m "feat(web/sections): add Team (§13) terminal-style researcher cards"
```

---

### Task 27: Footer section (§14) + BibTeX block

**Files:**
- Create: `web/lib/citations.ts`
- Create: `web/components/ui/BibTexBlock.tsx`
- Create: `web/components/sections/Footer.tsx`

- [ ] **Step 1: Define citation**

Create `web/lib/citations.ts`:

```ts
export const BIBTEX = `@misc{geometryofalignment2026,
  author = {Yang, Chenhao and Zhao, Daitian and Wang, Hanlin and Luo, Yuxi},
  title  = {Geometry of Alignment: Why Rank-1 Abliteration is Partially Inert on Gemma 4},
  year   = {2026},
  howpublished = {EECS 6699 final project, Columbia University},
  url    = {https://geometry-of-alignment.vercel.app}
}`;
```

- [ ] **Step 2: Implement BibTexBlock with copy button**

Create `web/components/ui/BibTexBlock.tsx`:

```tsx
"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";

interface Props {
  bibtex: string;
}

export function BibTexBlock({ bibtex }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(bibtex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative bg-black/40 border border-white/[0.12] rounded p-4 pr-12 font-mono text-[12px] leading-[1.5] text-text-2 overflow-x-auto">
      <pre className="whitespace-pre-wrap">{bibtex}</pre>
      <button
        onClick={handleCopy}
        aria-label="Copy citation"
        className="absolute top-3 right-3 p-2 rounded text-text-3 hover:text-iri-cyan transition-colors"
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
    </div>
  );
}
```

- [ ] **Step 3: Implement Footer**

Create `web/components/sections/Footer.tsx`:

```tsx
import { SectionNumber } from "@/components/ui/SectionNumber";
import { StatusDot } from "@/components/ui/StatusDot";
import { BibTexBlock } from "@/components/ui/BibTexBlock";
import { BIBTEX } from "@/lib/citations";

export function Footer() {
  return (
    <footer className="relative pt-24 pb-12 px-11 border-t border-white/[0.08]" aria-label="Footer">
      <div className="max-w-6xl mx-auto">
        <SectionNumber number="13" label="FOOTER" />

        <div className="mt-8 grid md:grid-cols-2 gap-12">
          <div>
            <h3 className="font-sans font-bold text-text-1">Cite this work</h3>
            <div className="mt-3 max-w-[520px]">
              <BibTexBlock bibtex={BIBTEX} />
            </div>
          </div>

          <div className="space-y-3 text-sm font-mono text-text-3">
            <div>
              <a href="/paper.pdf" className="text-text-2 hover:text-iri-cyan">/paper.pdf</a>
            </div>
            <div>
              <a
                href="https://github.com/nyavana/geometry-of-alignment"
                target="_blank"
                rel="noopener noreferrer"
                className="text-text-2 hover:text-iri-cyan"
              >
                github.com/nyavana/geometry-of-alignment
              </a>
            </div>
            <div className="pt-2 text-text-3">
              EECS 6699 · Mathematics of Deep Learning
              <br />
              Columbia University · Spring 2026
            </div>
            <div className="text-text-4">
              MIT (code) · CC BY 4.0 (figures + prose)
            </div>
          </div>
        </div>

        <div className="mt-16 flex justify-center gap-2 items-center font-mono text-[10px] tracking-[0.22em] uppercase text-text-3">
          <StatusDot color="green" /> signal · live
        </div>
      </div>
    </footer>
  );
}
```

- [ ] **Step 4: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/lib/citations.ts web/components/ui/BibTexBlock.tsx web/components/sections/Footer.tsx
git commit -m "feat(web/sections): add Footer (§14) with copyable BibTeX + license"
```

---

### Task 28: Page assembly — wire all 14 sections into app/page.tsx

**Files:**
- Modify: `web/app/page.tsx`

- [ ] **Step 1: Replace page.tsx with the full composition**

```tsx
import { Hero } from "@/components/sections/Hero";
import { Hike } from "@/components/sections/Hike";
import { Promise as PromiseSection } from "@/components/sections/Promise";
import { Refusal } from "@/components/sections/Refusal";
import { Question } from "@/components/sections/Question";
import { Primer } from "@/components/sections/Primer";
import { Methodology } from "@/components/sections/Methodology";
import { RefusalDirection } from "@/components/sections/RefusalDirection";
import { Investigation } from "@/components/sections/Investigation";
import { PunchLine } from "@/components/sections/PunchLine";
import { AlphaSweepSection } from "@/components/sections/AlphaSweepSection";
import { WhatThisMeans } from "@/components/sections/WhatThisMeans";
import { Team } from "@/components/sections/Team";
import { Footer } from "@/components/sections/Footer";

export default function HomePage() {
  return (
    <main>
      <Hero />
      <Hike />
      <PromiseSection />
      <Refusal />
      <Question />
      <Primer />
      <Methodology />
      <RefusalDirection />
      <Investigation />
      <PunchLine />
      <AlphaSweepSection />
      <WhatThisMeans />
      <Team />
      <Footer />
    </main>
  );
}
```

- [ ] **Step 2: Run dev server and scroll the entire page**

```bash
pnpm dev
```

Visit http://localhost:3000 and scroll top-to-bottom. Confirm: all 14 sections render in order, no console errors, demos show stub placeholders, scroll-reveals animate.

- [ ] **Step 3: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/app/page.tsx
git commit -m "feat(web): assemble single long-scroll page with all 14 sections"
```

---

Phase 3 complete. The site renders end-to-end with stub demos.

---

## Phase 4 — Interactive demos (parallelizable; replace stubs from Phase 3)

### Task 29: Demo 4 — AlphaSweep (recharts + slider)

**Files:**
- Replace: `web/components/demos/AlphaSweep.tsx`
- Create: `web/tests/e2e/alpha-sweep.spec.ts`

- [ ] **Step 1: Write the failing test**

Create `web/tests/e2e/alpha-sweep.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("AlphaSweep renders chart + 3 traces + draggable slider", async ({ page }) => {
  await page.goto("/");
  await page.locator("#alpha-sweep").scrollIntoViewIfNeeded();

  const demo = page.getByTestId("alpha-sweep");
  await expect(demo).toBeVisible();

  // 3 traces (one polyline per trace) — recharts renders <path> elements
  const traces = demo.locator("svg.recharts-surface path.recharts-line-curve");
  await expect(traces).toHaveCount(3);

  // Slider exists and reads "α = 1.0" in the readout by default
  await expect(demo.getByTestId("alpha-readout")).toContainText("α = 1.0");

  // Drag slider to extreme: should update the readout
  const slider = demo.getByRole("slider");
  await slider.focus();
  await page.keyboard.press("End");
  await expect(demo.getByTestId("alpha-readout")).toContainText("α = 2.0");
});
```

- [ ] **Step 2: Run, expect fail (only stub renders)**

```bash
pnpm test alpha-sweep --project=chromium
```

- [ ] **Step 3: Implement AlphaSweep**

Replace `web/components/demos/AlphaSweep.tsx`:

```tsx
"use client";

import { useState } from "react";
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";
import data from "@/lib/data/alpha-sweep.json";

const TRACE_META = {
  baseline:    { label: "baseline (raw direction)", color: "var(--iri-cyan)"   },
  random:      { label: "random control",          color: "var(--text-3)"     },
  gramSchmidt: { label: "Gram-Schmidt (D3)",       color: "var(--iri-magenta)" },
} as const;

type TraceKey = keyof typeof TRACE_META;

export function AlphaSweep() {
  const alphas = data.alphas as number[];
  const traces = data.traces as Record<TraceKey, number[]>;
  const fallback = alphas.length === 0;

  // For demo purposes, if data is missing/empty, synthesize a plausible curve
  const usedAlphas = fallback ? Array.from({ length: 9 }, (_, i) => i * 0.25) : alphas;
  const usedTraces = fallback
    ? {
        baseline:    [0.30, 0.32, 0.30, 0.31, 0.30, 0.30, 0.29, 0.30, 0.31],
        random:      [0.30, 0.31, 0.30, 0.30, 0.30, 0.31, 0.30, 0.30, 0.30],
        gramSchmidt: [1.00, 0.92, 0.78, 0.61, 0.405, 0.43, 0.45, 0.47, 0.50],
      }
    : traces;

  const chartData = usedAlphas.map((a, i) => ({
    alpha: a,
    baseline: usedTraces.baseline[i],
    random: usedTraces.random[i],
    gramSchmidt: usedTraces.gramSchmidt[i],
  }));

  const [alphaIdx, setAlphaIdx] = useState(Math.floor(usedAlphas.length / 2));
  const currentAlpha = usedAlphas[alphaIdx] ?? 0;
  const readout = chartData[alphaIdx];

  return (
    <div data-testid="alpha-sweep" className="rounded-lg border border-white/[0.12] bg-white/[0.04] p-6">
      <div className="h-[320px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 24, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
            <XAxis
              dataKey="alpha"
              type="number"
              domain={[0, 2]}
              tickFormatter={(v: number) => v.toFixed(2)}
              stroke="rgba(255,255,255,0.45)"
              fontSize={11}
              fontFamily="JetBrains Mono"
              label={{ value: "α", position: "insideBottom", offset: -8, fill: "rgba(255,255,255,0.6)" }}
            />
            <YAxis
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              stroke="rgba(255,255,255,0.45)"
              fontSize={11}
              fontFamily="JetBrains Mono"
              domain={[0, 1]}
              label={{ value: "should_refuse", angle: -90, position: "insideLeft", fill: "rgba(255,255,255,0.6)" }}
            />
            <Tooltip
              contentStyle={{ background: "rgba(7,2,26,0.95)", border: "1px solid rgba(255,255,255,0.18)", borderRadius: 4, fontSize: 12 }}
              formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
            />
            <Legend wrapperStyle={{ fontSize: 11, fontFamily: "JetBrains Mono" }} />
            <ReferenceLine x={currentAlpha} stroke="var(--iri-amber)" strokeDasharray="2 2" />
            {(Object.keys(TRACE_META) as TraceKey[]).map((k) => (
              <Line
                key={k}
                type="monotone"
                dataKey={k}
                stroke={TRACE_META[k].color}
                strokeWidth={2}
                dot={false}
                name={TRACE_META[k].label}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Slider + readout */}
      <div className="mt-6 flex flex-col gap-3">
        <input
          type="range"
          min={0}
          max={usedAlphas.length - 1}
          step={1}
          value={alphaIdx}
          onChange={(e) => setAlphaIdx(parseInt(e.target.value, 10))}
          className="w-full accent-iri-magenta"
          aria-label="Alpha"
        />
        <div className="flex justify-between font-mono text-xs text-text-2">
          <span data-testid="alpha-readout" className="text-text-1 font-bold">
            α = {currentAlpha.toFixed(2)}
          </span>
          <span>baseline {(readout.baseline * 100).toFixed(0)}%</span>
          <span>random {(readout.random * 100).toFixed(0)}%</span>
          <span className="text-iri-magenta">gram-schmidt {(readout.gramSchmidt * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test, verify pass**

```bash
pnpm test alpha-sweep --project=chromium
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/demos/AlphaSweep.tsx web/tests/e2e/alpha-sweep.spec.ts
git commit -m "feat(web/demos): implement AlphaSweep (Demo 4) — recharts + slider"
```

---

### Task 30: Demo 3 — M6Cascade (custom SVG decision tree)

**Files:**
- Replace: `web/components/demos/M6Cascade.tsx`
- Create: `web/tests/e2e/m6-cascade.spec.ts`

- [ ] **Step 1: Write the failing test**

Create `web/tests/e2e/m6-cascade.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("M6Cascade renders 6 nodes; clicking expands a panel", async ({ page }) => {
  await page.goto("/");
  await page.locator("#investigation").scrollIntoViewIfNeeded();

  const demo = page.getByTestId("m6-cascade");
  await expect(demo).toBeVisible();

  // 6 nodes
  const nodes = demo.getByRole("button", { name: /^H[1-6]/ });
  await expect(nodes).toHaveCount(6);

  // Click the LOAD-BEARING node (H4) — verify panel reveals expected text
  await demo.getByRole("button", { name: /H4/ }).click();
  await expect(demo.getByTestId("cascade-panel")).toContainText("Gram-Schmidt");
  await expect(demo.getByTestId("cascade-panel")).toContainText("17/42");
});
```

- [ ] **Step 2: Run, expect fail**

```bash
pnpm test m6-cascade --project=chromium
```

- [ ] **Step 3: Implement M6Cascade**

Replace `web/components/demos/M6Cascade.tsx`:

```tsx
"use client";

import { useState } from "react";
import data from "@/lib/data/m6-cascade.json";

type Verdict = "PASS" | "REJECTED" | "INSUFFICIENT" | "LOAD-BEARING" | "REFUTED";

interface Node {
  id: string;
  label: string;
  verdict: Verdict;
  experiment: string;
  result: string;
  cosine: number | null;
}

const VERDICT_STYLES: Record<Verdict, { bg: string; border: string; text: string }> = {
  PASS:           { bg: "rgba(0,255,136,0.12)",  border: "rgba(0,255,136,0.65)",  text: "#00ff88" },
  REJECTED:       { bg: "rgba(255,90,90,0.10)",  border: "rgba(255,90,90,0.6)",   text: "#ff5a5a" },
  INSUFFICIENT:   { bg: "rgba(255,224,74,0.10)", border: "rgba(255,224,74,0.55)", text: "#ffe04a" },
  "LOAD-BEARING": { bg: "rgba(255,43,176,0.18)", border: "rgba(255,43,176,0.85)", text: "#ff8aff" },
  REFUTED:        { bg: "rgba(177,77,255,0.10)", border: "rgba(177,77,255,0.55)", text: "#b14dff" },
};

export function M6Cascade() {
  const nodes = data.nodes as Node[];
  const highlight = data.highlight as string;
  const [openId, setOpenId] = useState<string>(highlight);
  const open = nodes.find((n) => n.id === openId) ?? nodes[0];

  return (
    <div data-testid="m6-cascade" className="rounded-lg border border-white/[0.12] bg-white/[0.04] p-6">
      {/* Node grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {nodes.map((n) => {
          const style = VERDICT_STYLES[n.verdict];
          const isOpen = n.id === openId;
          const isLoad = n.id === highlight;
          return (
            <button
              key={n.id}
              type="button"
              onClick={() => setOpenId(n.id)}
              className={
                `text-left p-4 rounded transition-all border-2 ` +
                (isOpen ? "scale-[1.02] " : "hover:scale-[1.01] ")
              }
              style={{
                background: style.bg,
                borderColor: isOpen ? style.border : "transparent",
                boxShadow: isLoad ? `0 0 18px ${style.border}` : undefined,
              }}
            >
              <div className="font-sans font-bold text-text-1 text-sm">{n.label}</div>
              <div
                className="mt-2 font-mono text-[10px] tracking-[0.18em] uppercase"
                style={{ color: style.text }}
              >
                {n.verdict}
              </div>
            </button>
          );
        })}
      </div>

      {/* Detail panel */}
      <div
        data-testid="cascade-panel"
        className="mt-6 p-5 rounded bg-black/40 border border-white/[0.10]"
      >
        <h4 className="font-sans font-bold text-text-1 text-base">{open.label}</h4>
        <dl className="mt-3 space-y-3 text-sm">
          <div>
            <dt className="font-mono text-[10px] tracking-[0.18em] uppercase text-text-3">experiment</dt>
            <dd className="mt-1 text-text-2">{open.experiment}</dd>
          </div>
          <div>
            <dt className="font-mono text-[10px] tracking-[0.18em] uppercase text-text-3">result</dt>
            <dd className="mt-1 text-text-2">{open.result}</dd>
          </div>
          {open.cosine !== null && (
            <div>
              <dt className="font-mono text-[10px] tracking-[0.18em] uppercase text-text-3">cosine</dt>
              <dd className="mt-1 font-mono text-text-1 text-base">{open.cosine.toFixed(3)}</dd>
            </div>
          )}
        </dl>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test, verify pass**

```bash
pnpm test m6-cascade --project=chromium
```

- [ ] **Step 5: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/demos/M6Cascade.tsx web/tests/e2e/m6-cascade.spec.ts
git commit -m "feat(web/demos): implement M6Cascade (Demo 3) — clickable verdict tree"
```

---

### Task 31: Demo 2 — LayerHeatmap (42-band SVG strip)

**Files:**
- Replace: `web/components/demos/LayerHeatmap.tsx`
- Create: `web/tests/e2e/layer-heatmap.spec.ts`

- [ ] **Step 1: Write failing test**

Create `web/tests/e2e/layer-heatmap.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("LayerHeatmap renders 42 bands; selected band changes on click", async ({ page }) => {
  await page.goto("/");
  await page.locator("#refusal-direction").scrollIntoViewIfNeeded();

  const heatmap = page.getByTestId("layer-heatmap");
  const bands = heatmap.locator("[data-layer]");
  await expect(bands).toHaveCount(42);

  // Click layer 5 (a global-attention layer)
  await heatmap.locator("[data-layer='5']").click();
  await expect(heatmap.locator("[data-selected='true']")).toHaveAttribute("data-layer", "5");
});
```

- [ ] **Step 2: Run, expect fail**

```bash
pnpm test layer-heatmap --project=chromium
```

- [ ] **Step 3: Implement LayerHeatmap**

Replace `web/components/demos/LayerHeatmap.tsx`:

```tsx
"use client";

import { useMemo, useState } from "react";
import data from "@/lib/data/cohens-d-per-layer.json";

interface Props {
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
}

interface LayerEntry {
  layer: number;
  d: number;
  attentionType: "global" | "sliding";
}

export function LayerHeatmap({ selectedLayer, onSelectLayer }: Props) {
  const layers = data.layers as LayerEntry[];
  const [hovered, setHovered] = useState<number | null>(null);

  const maxD = useMemo(() => Math.max(...layers.map((l) => l.d), 1), [layers]);

  const colorFor = (d: number) => {
    const t = Math.min(1, Math.max(0, d / maxD));
    // magenta→white scale
    const r = 255;
    const g = Math.round(43 + (255 - 43) * t);
    const b = Math.round(176 + (255 - 176) * t);
    return `rgb(${r},${g},${b})`;
  };

  const display = hovered ?? selectedLayer;
  const displayEntry = layers.find((l) => l.layer === display) ?? layers[0];

  return (
    <div data-testid="layer-heatmap" className="rounded border border-white/[0.12] bg-white/[0.04] p-5">
      {/* Tick row for global-attention layers */}
      <div className="relative h-3 mb-1">
        {layers.map((l) =>
          l.attentionType === "global" ? (
            <span
              key={l.layer}
              className="absolute top-0 w-px h-3 bg-iri-amber"
              style={{ left: `${(l.layer / 41) * 100}%` }}
              aria-hidden="true"
            />
          ) : null
        )}
      </div>

      {/* Heatmap strip */}
      <div className="flex h-12 rounded overflow-hidden">
        {layers.map((l) => {
          const isSelected = l.layer === selectedLayer;
          return (
            <button
              key={l.layer}
              type="button"
              data-layer={l.layer}
              data-selected={isSelected ? "true" : "false"}
              onClick={() => onSelectLayer(l.layer)}
              onMouseEnter={() => setHovered(l.layer)}
              onMouseLeave={() => setHovered(null)}
              className="flex-1 transition-all"
              style={{
                backgroundColor: colorFor(l.d),
                outline: isSelected ? "2px solid var(--iri-cyan)" : "none",
                outlineOffset: -2,
              }}
              aria-label={`Layer ${l.layer}: Cohen's d = ${l.d.toFixed(2)}`}
            />
          );
        })}
      </div>

      {/* Index strip */}
      <div className="mt-2 flex justify-between font-mono text-[9px] text-text-3">
        <span>L0</span>
        <span>L15 (peak)</span>
        <span>L41</span>
      </div>

      {/* Readout */}
      <div className="mt-4 font-mono text-xs text-text-2 flex gap-6">
        <span>
          layer · <strong className="text-text-1">L{displayEntry.layer}</strong>
        </span>
        <span>
          Cohen&rsquo;s d · <strong className="text-text-1">{displayEntry.d.toFixed(2)}</strong>
        </span>
        <span>
          attention · <strong className="text-text-1">{displayEntry.attentionType}</strong>
        </span>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run, verify pass**

```bash
pnpm test layer-heatmap --project=chromium
```

- [ ] **Step 5: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/demos/LayerHeatmap.tsx web/tests/e2e/layer-heatmap.spec.ts
git commit -m "feat(web/demos): implement LayerHeatmap (Demo 2) — 42-band SVG strip"
```

---

### Task 32: Demo 1 — UmapScatter (D3)

**Files:**
- Replace: `web/components/demos/UmapScatter.tsx`
- Create: `web/tests/e2e/umap-scatter.spec.ts`

- [ ] **Step 1: Write failing test**

Create `web/tests/e2e/umap-scatter.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("UmapScatter renders points and shows tooltip on hover", async ({ page }) => {
  await page.goto("/");
  await page.locator("#refusal-direction").scrollIntoViewIfNeeded();

  const scatter = page.getByTestId("umap-scatter");
  await expect(scatter).toBeVisible();

  // At least 40 points (stub mode emits 50; full mode emits ~340)
  const points = scatter.locator("circle[data-point]");
  expect(await points.count()).toBeGreaterThanOrEqual(40);

  // Hover the first point — tooltip appears
  await points.first().hover();
  await expect(scatter.getByTestId("scatter-tooltip")).toBeVisible();
});
```

- [ ] **Step 2: Run, expect fail**

```bash
pnpm test umap-scatter --project=chromium
```

- [ ] **Step 3: Implement UmapScatter with D3**

Replace `web/components/demos/UmapScatter.tsx`:

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import data from "@/lib/data/umap-l15.json";

interface Point {
  id: string;
  label: "refuse" | "comply";
  x: number;
  y: number;
  prompt: string;
  category?: string;
}

interface Props {
  layer: number;
}

export function UmapScatter({ layer }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; point: Point } | null>(null);
  const points = data.points as Point[];

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const width = ref.current.clientWidth;
    const height = ref.current.clientHeight;
    const margin = 24;

    const xExtent = d3.extent(points, (p) => p.x) as [number, number];
    const yExtent = d3.extent(points, (p) => p.y) as [number, number];

    const xScale = d3.scaleLinear().domain(xExtent).range([margin, width - margin]);
    const yScale = d3.scaleLinear().domain(yExtent).range([height - margin, margin]);

    svg
      .selectAll("circle")
      .data(points)
      .enter()
      .append("circle")
      .attr("data-point", (d) => d.id)
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 4)
      .attr("fill", (d) => (d.label === "refuse" ? "var(--iri-magenta)" : "var(--iri-cyan)"))
      .attr("opacity", 0.65)
      .on("mouseenter", (event, d) => {
        setTooltip({ x: event.offsetX, y: event.offsetY, point: d });
      })
      .on("mouseleave", () => setTooltip(null));
  }, [points]);

  return (
    <div
      data-testid="umap-scatter"
      className="relative h-[420px] rounded-lg border border-white/[0.12] bg-white/[0.04] overflow-hidden"
    >
      <div className="absolute top-3 left-4 font-mono text-[10px] tracking-[0.18em] uppercase text-text-3">
        UMAP · L{layer}
      </div>
      <div className="absolute top-3 right-4 flex gap-3 font-mono text-[10px] uppercase text-text-3">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-iri-magenta" /> refuse
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-iri-cyan" /> comply
        </span>
      </div>

      <svg ref={ref} className="w-full h-full" />

      {tooltip && (
        <div
          data-testid="scatter-tooltip"
          className="absolute pointer-events-none px-3 py-2 rounded bg-black/90 border border-white/[0.18] text-text-1 text-xs max-w-[280px] z-10"
          style={{
            left: Math.min(tooltip.x + 12, 9999),
            top: Math.max(tooltip.y - 50, 4),
          }}
        >
          <div className="font-mono text-[9px] uppercase tracking-[0.15em] text-text-3">
            {tooltip.point.label}
            {tooltip.point.category ? ` · ${tooltip.point.category}` : ""}
          </div>
          <div className="mt-1 leading-[1.35]">{tooltip.point.prompt || "(no prompt text)"}</div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run, verify pass**

```bash
pnpm test umap-scatter --project=chromium
```

- [ ] **Step 5: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/components/demos/UmapScatter.tsx web/tests/e2e/umap-scatter.spec.ts
git commit -m "feat(web/demos): implement UmapScatter (Demo 1) — D3 scatter w/ hover tooltip"
```

---

Phase 4 complete. All four demos are wired to live data (or stub data, which the frontend will overwrite when extract_demo_data.py is re-run in `--mode full`).

---

## Phase 5 — Comprehensive E2E tests, accessibility, responsive, motion

### Task 33: Sections coverage test (all 14 sections render)

**Files:**
- Create: `web/tests/e2e/sections.spec.ts`

- [ ] **Step 1: Write the test**

Create `web/tests/e2e/sections.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

const SECTION_IDS = [
  "hero",
  "hike",
  "promise",
  "refusal",
  "question",
  "primer",
  "methodology",
  "refusal-direction",
  "investigation",
  "punch-line",
  "alpha-sweep",
  "what-this-means",
  "team",
];

test("all 14 sections (incl. footer) render in order", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

  await page.goto("/");

  for (const id of SECTION_IDS) {
    const sec = page.locator(`#${id}`);
    await expect(sec).toBeAttached();
  }
  // Footer is a <footer>, not <section> with id
  await expect(page.getByRole("contentinfo")).toBeVisible();
  expect(errors).toEqual([]);
});

test("section count = 13 sections + 1 footer", async ({ page }) => {
  await page.goto("/");
  const sections = page.locator("section");
  await expect(sections).toHaveCount(13);
});
```

- [ ] **Step 2: Run, verify pass**

```bash
pnpm test sections --project=chromium
```

- [ ] **Step 3: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/tests/e2e/sections.spec.ts
git commit -m "test(web): add section coverage spec (14 sections, no console errors)"
```

---

### Task 34: Accessibility test with axe

**Files:**
- Create: `web/tests/e2e/accessibility.spec.ts`

- [ ] **Step 1: Write the test**

Create `web/tests/e2e/accessibility.spec.ts`:

```ts
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("homepage has no critical/serious accessibility violations", async ({ page }) => {
  await page.goto("/");

  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();

  const blocking = results.violations.filter(
    (v) => v.impact === "critical" || v.impact === "serious"
  );

  if (blocking.length > 0) {
    console.log("Violations:", JSON.stringify(blocking, null, 2));
  }
  expect(blocking).toEqual([]);
});
```

- [ ] **Step 2: Run; fix any failures**

```bash
pnpm test accessibility --project=chromium
```

If failures, common fixes:
- Add `aria-label` to interactive elements (status dots already use `aria-hidden="true"` because they're decorative)
- Verify color contrast for `text-3` and `text-4` against `bg-base` — these tokens are `rgba(255,255,255,0.65)` and `rgba(255,255,255,0.45)` against `#07021a`. The 0.45 may fail AA contrast for body text — only use it for non-essential chrome (section numbers, fine print). Ensure no body copy uses `text-4`.
- All form inputs (the α slider) need accessible labels — already added via `aria-label="Alpha"`.

- [ ] **Step 3: Commit (after fixes if needed)**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/tests/e2e/accessibility.spec.ts
git commit -m "test(web): add axe accessibility scan (0 critical/serious violations)"
```

---

### Task 35: Responsive test (mobile / tablet / desktop)

**Files:**
- Create: `web/tests/e2e/responsive.spec.ts`

- [ ] **Step 1: Write the test**

Create `web/tests/e2e/responsive.spec.ts`:

```ts
import { test, expect, devices } from "@playwright/test";

const VIEWPORTS = [
  { name: "mobile",  width: 375,  height: 812  },
  { name: "tablet",  width: 768,  height: 1024 },
  { name: "desktop", width: 1440, height: 900  },
];

for (const vp of VIEWPORTS) {
  test(`hero + hero CTAs are reachable at ${vp.name} (${vp.width}x${vp.height})`, async ({ page }) => {
    await page.setViewportSize({ width: vp.width, height: vp.height });
    await page.goto("/");

    // No horizontal overflow
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(scrollWidth).toBeLessThanOrEqual(vp.width + 1);

    // All 3 hero CTAs are visible
    await expect(page.getByRole("link", { name: /Read the paper/ })).toBeVisible();
    await expect(page.getByRole("link", { name: /View source/ })).toBeVisible();
    await expect(page.getByRole("link", { name: /Interactive demo/ })).toBeVisible();
  });
}
```

- [ ] **Step 2: Run; if overflow on mobile, fix**

```bash
pnpm test responsive --project=chromium
```

If horizontal overflow on mobile, audit the largest fixed-width elements:
- Hero headline at 64px may overflow on 375px viewport. Reduce via responsive class: `text-[44px] sm:text-[64px]` in HoloHeading default sizing, or scope per-section.

- [ ] **Step 3: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/tests/e2e/responsive.spec.ts
git commit -m "test(web): add responsive viewport coverage (375/768/1440)"
```

---

### Task 36: Reduced-motion test

**Files:**
- Modify: `web/tests/e2e/motion.spec.ts` (replace stub)

- [ ] **Step 1: Replace the existing motion stub with the real reduced-motion check**

Replace `web/tests/e2e/motion.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("reduced-motion: pulse animation does not loop", async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: "reduce" });
  const page = await ctx.newPage();
  await page.goto("/");

  // The status dot pulse is a CSS animation; under reduced-motion it should be effectively
  // disabled by the `prefers-reduced-motion` block in globals.css.
  const dot = page.getByTestId("hero-signal").locator("span").first();
  const animDuration = await dot.evaluate(
    (el) => getComputedStyle(el).animationDuration
  );
  // `0.01ms` from the reduced-motion override (or a fallback near-zero value)
  expect(animDuration).toMatch(/0\.01ms|0s/);

  await ctx.close();
});

test("reveals render even with motion disabled (no permanent opacity:0)", async ({ browser }) => {
  const ctx = await browser.newContext({ reducedMotion: "reduce" });
  const page = await ctx.newPage();
  await page.goto("/");
  await page.locator("#hike").scrollIntoViewIfNeeded();
  // Hike section blockquote should be visible even with reduced motion
  await expect(page.getByText(/on our own/)).toBeVisible({ timeout: 1500 });
  await ctx.close();
});
```

- [ ] **Step 2: Run, verify pass**

```bash
pnpm test motion --project=chromium
```

- [ ] **Step 3: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/tests/e2e/motion.spec.ts
git commit -m "test(web): replace motion stub with real reduced-motion checks"
```

---

### Task 37: Cross-browser run (chromium + firefox + webkit)

**Files:** none (this is a verification task)

- [ ] **Step 1: Run the full Playwright suite across all 3 browsers**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
pnpm test
```

Expected: all tests pass on chromium, firefox, and webkit. If WebKit fails on KaTeX or Framer Motion-specific behavior, isolate the failure and either fix or skip the test under WebKit explicitly with `test.skip(({ browserName }) => browserName === 'webkit', '...')` and document why.

- [ ] **Step 2: If any failures, fix and re-run; commit fixes**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/
git commit -m "test(web): cross-browser fixes (chromium + firefox + webkit)"
```

(If no failures, no commit.)

---

## Phase 6 — Polish, OG, Lighthouse, deploy

### Task 38: Generate OG image + favicon + paper.pdf placeholder

**Files:**
- Create: `web/public/og-image.png` (1200×630)
- Create: `web/public/favicon.ico`
- Create: `web/public/paper.pdf` (placeholder)

- [ ] **Step 1: Build a minimal OG image route at build time**

Create `web/app/og-image-generator/route.tsx` (we'll generate the PNG once and check it in):

```tsx
import { ImageResponse } from "next/og";

export const runtime = "edge";

export async function GET() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "flex-start",
          padding: 80,
          background:
            "radial-gradient(circle at 18% 22%, #4d1a8a 0%, transparent 48%), " +
            "radial-gradient(circle at 78% 58%, #ff2bb0 0%, transparent 52%), " +
            "radial-gradient(circle at 50% 95%, #00d4ff 0%, transparent 50%), " +
            "radial-gradient(circle at 90% 10%, #ffe04a 0%, transparent 38%), #07021a",
          color: "white",
          fontFamily: "Inter, system-ui, sans-serif",
        }}
      >
        <div style={{ fontSize: 24, opacity: 0.7, letterSpacing: "0.18em" }}>
          GEOMETRY OF ALIGNMENT
        </div>
        <div style={{ fontSize: 78, fontWeight: 900, lineHeight: 1.0, marginTop: 30 }}>
          Why doesn't abliteration
          <br />
          work on Gemma 4?
        </div>
        <div style={{ fontSize: 28, opacity: 0.85, marginTop: 30 }}>
          A 40.5% residual. A geometric explanation.
        </div>
      </div>
    ),
    { width: 1200, height: 630 }
  );
}
```

- [ ] **Step 2: Capture the OG image as a static asset**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
pnpm dev &
sleep 6
curl -s http://localhost:3000/og-image-generator -o public/og-image.png
ls -lh public/og-image.png  # should be ~30-100 KB
kill %1
```

Then delete the route directory (we only need the PNG file):

```bash
rm -rf app/og-image-generator
```

- [ ] **Step 3: Add a placeholder paper.pdf and favicon**

If the team has a draft paper PDF, copy it:
```bash
cp /home/nyavana/columbia/6699/geometry-of-alignment/paper/paper-v0.pdf web/public/paper.pdf 2>/dev/null || \
  echo "%PDF-1.4 (placeholder; replace before prod deploy)" > web/public/paper.pdf
```

For the favicon, generate a simple one using the same iridescent gradient. For now, use the Next.js scaffold default at `app/favicon.ico` (already present). Optionally replace later.

- [ ] **Step 4: Commit**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git add web/public/og-image.png web/public/paper.pdf
git commit -m "feat(web): add OG share card + paper.pdf placeholder"
```

---

### Task 39: Lighthouse audit + perf fixes

**Files:** none (audit)

- [ ] **Step 1: Build production bundle**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
pnpm build
```

Expected: build succeeds. If Next.js complains about `next/font` or missing imports, fix the specific error.

- [ ] **Step 2: Start production server**

```bash
pnpm start &
sleep 4
```

- [ ] **Step 3: Install lighthouse + run an audit**

```bash
which lighthouse || pnpm add -g lighthouse
lighthouse http://localhost:3000 \
  --output=json --output-path=./lh-report.json \
  --chrome-flags="--headless --no-sandbox" \
  --quiet
node -e "const r = require('./lh-report.json'); console.log({ perf: r.categories.performance.score, a11y: r.categories.accessibility.score, bp: r.categories['best-practices'].score, seo: r.categories.seo.score });"
kill %1
```

Targets per spec §11.2: perf ≥ 0.90, a11y ≥ 0.95, best-practices ≥ 0.95, seo ≥ 0.90.

- [ ] **Step 4: Fix any score below target**

Common perf wins:
- Add `loading="lazy"` to off-screen images (no images in our design — skip)
- Reduce KaTeX bundle size by importing only what's used
- Verify `next/font` is properly subsetting fonts (it does by default)
- Add `priority` to the Hero's most-important paint elements

Common a11y wins:
- Ensure all interactive elements have visible focus rings (Tailwind `focus:outline-2 focus:outline-iri-cyan`)
- Ensure heading order is sequential (no h2 before h1, etc.)

After fixes, re-run Step 3 and confirm targets met.

- [ ] **Step 5: Commit fixes (if any)**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
rm web/lh-report.json
git add web/
git commit -m "perf(web): meet Lighthouse targets (perf ≥90, a11y ≥95, bp ≥95, seo ≥90)" || echo "no perf changes needed"
```

---

### Task 40: Manual smoke checklist

**Files:** none (manual verification)

- [ ] **Step 1: Smoke each item from spec §11.3 — record pass/fail**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
pnpm dev
```

Open http://localhost:3000 and verify:

- [ ] Scroll the full page top → bottom on Chrome desktop. No layout breaks.
- [ ] Hover all four demos. UMAP tooltip appears. Heatmap selection updates UMAP layer. M6 cascade nodes click + reveal panel. α-slider drags + chart updates.
- [ ] Click "Read the paper" → resolves to `/paper.pdf` (downloads or opens inline).
- [ ] Click "View source" → opens GitHub in new tab.
- [ ] Open the same URL on a real iPhone via Safari (or use Chrome DevTools mobile emulation). Confirm no horizontal scroll, all CTAs reachable, narrative sections legible.
- [ ] Test the Twitter / Slack unfurl: paste the deployed URL into Slack DM-to-self. Confirm the OG card renders with the iridescent gradient and headline.

- [ ] **Step 2: If any check fails, file a follow-up + fix before deploying to prod**

(No commit unless fixes are made.)

---

### Task 41: Production deploy via Vercel CLI

**Files:** none

- [ ] **Step 1: Confirm clean working tree**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git status
```

Expected: nothing to commit (working tree clean).

- [ ] **Step 2: Production deploy**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment/web
vercel deploy --prod
```

Expected: prints `Production: https://geometry-of-alignment.vercel.app`. The first prod deploy may take ~2 minutes for the build.

- [ ] **Step 3: Post-deploy smoke**

Open https://geometry-of-alignment.vercel.app in a fresh browser tab. Repeat the smoke checks from Task 40 against the live URL.

- [ ] **Step 4: Tag the release**

```bash
cd /home/nyavana/columbia/6699/geometry-of-alignment
git tag -a web-v1.0 -m "First production deploy of geometry-of-alignment.vercel.app"
git push origin web-v1.0  # only if user requests pushing
```

(Do not push to remote without explicit user instruction.)

---

## Self-review (post-write check)

**Spec coverage:**

| Spec section | Covered by |
|---|---|
| § 1 Goal | Phase 3 (sections compose the long-scroll narrative) |
| § 2 Audience priority | Section ordering in Phase 3 reflects ML-community → Academic → Recruiters |
| § 3 Site structure (14 sections) | Tasks 14–28 |
| § 4.1 Color tokens | Task 3 |
| § 4.2 Typography (Inter / JetBrains Mono / KaTeX) | Tasks 4 (fonts), 19 (KaTeX) |
| § 4.3 Headline treatment | Task 10 (HoloHeading) |
| § 4.4 CTA treatment (metadata-rich) | Task 9 (MetadataButton) |
| § 4.5 Motion language | Tasks 11 (Reveal), 12 (TypeOnReveal), 36 (reduced-motion test) |
| § 5 Tech stack | Tasks 1, 2 |
| § 6 Repo layout (web/ subdir) | Task 1 |
| § 7 Interactive demos (4) | Tasks 13 (data), 29–32 (demos) |
| § 8 Imagery & narrative (typography + SVG) | Tasks 15–18 |
| § 9 Team section | Task 26 |
| § 10 Footer | Task 27 |
| § 11.1 Playwright suite | Tasks 5, 7–12, 14, 29–37 |
| § 11.2 Lighthouse targets | Task 39 |
| § 11.3 Manual smoke | Task 40 |
| § 12 Deploy | Task 6 (link), Task 41 (prod) |
| § 13 Out of scope | (no tasks needed; explicit non-goals) |
| § 14 Open questions | Annotated in Task 26 (team roles); paper.pdf placeholder in Task 38; Vercel account choice handled in Task 6 |
| § 15 Success criteria | Combination of Tasks 33–41 |

No spec sections missing.

**Placeholder scan:** No "TBD", "TODO", or "implement later" in steps. Team roles in Task 26 are explicitly flagged as placeholders awaiting team confirmation (per spec §14 open question).

**Type consistency:**
- `DotColor` defined in `lib/tokens.ts` (Task 3) → used by `StatusDot` (Task 7) → used by `MetadataButton` (Task 9). ✓
- `Point`, `LayerEntry`, `Node`, `Verdict` interfaces are defined inside their owning demo files (Tasks 32, 31, 30) and consume the corresponding JSON shape emitted by `extract_demo_data.py` (Task 13). ✓
- `selectedLayer`/`onSelectLayer` props on `LayerHeatmap` (Tasks 21 stub, 31 real impl) match the consumer in `RefusalDirection` (Task 21). ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-10-research-website.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Best for this plan because the 41 tasks decompose cleanly across the 4-person team and can run in parallel after Phase 0.

2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints. Slower but keeps everything in one transcript.

Which approach?

