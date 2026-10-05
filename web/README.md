# Decode Research website

A local, static research page. Use Node.js 24 (`nvm use`), then run commands from this directory:

```sh
npm install --cache .npm-cache
npm run dev
```

Open http://127.0.0.1:3000. To check and preview the static production build:

```sh
npm run test
npm run typecheck
npm run lint
npm run build
npm run preview
```

`next build` writes the static site to `out/`. `preview` serves those files locally; no application backend is required. No remote fonts or runtime APIs are used.

## Data flow

`lib/research.ts` reads `../experiments/*/report.json`, optional patches, current candidate and reference source, and harness configuration. `components/research-page.tsx` prepares the page at build time. `app/globals.css` contains the visual design.

History is paginated into ten chronological summaries per static page (`/` and `/history/2/`, etc.). Only summary fields cross the client boundary; full proposals, reviews, evaluations, and patches are excluded from those pages. Opening a round fetches its individually exported `/rounds/<round>/` document and inserts the report content in place. No API or backend is involved. Reports remain cached in the disclosure for the life of that page; failed loads offer retry and a direct report link. Without JavaScript, the same direct link provides the full report.

All accepted rounds have the same status styling, independent of gain. The latest accepted round is identified by a current-baseline label. Cumulative progress always uses the complete history; after several acceptances its default view is compact, with the complete accepted sequence available in a disclosure.

The current implementation section uses reports only for the latest accepted round, cumulative throughput, and evaluation/confirmation links. Implementation details come from the actual candidate and reference files, including a build-time unified diff of their `inference.py` files. Nonempty candidate Python files are discovered recursively and exposed as additional source disclosures; empty package files, cache directories, and non-Python assets are omitted. There is no proposal-derived mechanism description, code-pattern detection, or round-specific diagram to maintain. Accepted history does not imply that every past mechanism remains active.

Reports are ordered by round. Reviewed proposals take precedence over researcher proposals. The headline multiplies only accepted `confirmed_gain` values, keeping full precision until display. Each accepted gain is the lower of its two confirmation ratios against that round's baseline. The compounded record is not an independent end-to-end benchmark. Missing measurements remain unavailable; malformed reports and missing accepted gains fail the build.

Mechanism and lesson summaries are deterministic transformations of report evidence. The recorded status remains authoritative, including when an implementation limitation leaves the mechanism unresolved. Full report prose and tested patches remain available in disclosures.

A new report appears on the next build. There is no manually maintained experiment dataset, ingestion service, deployment configuration, or modification to research code.
