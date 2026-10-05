import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(web, "out");

// Deliberately report categories, never the matching content.
/** @type {[string, RegExp][]} */
const patterns = [
  ["email address", /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i],
  ["private key", /-----BEGIN (?:[A-Z ]*PRIVATE KEY|PGP PRIVATE KEY BLOCK)-----/],
  ["access token", /\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{20,})\b/],
  ["credential in URL", /https?:\/\/[^\s/:]+:[^\s/@]+@/i],
  ["personal home path", /(?:\/Users\/[^/\s"'<>]+|\/home\/(?!user(?:\/|\b)|test(?:\/|\b)|runner(?:\/|\b))[^/\s"'<>]+|[A-Z]:\\Users\\[^\\\s"'<>]+)/i],
  ["assigned credential", /["']?\b(?:password|passwd|api[_-]?key|access[_-]?token|secret[_-]?key|client[_-]?secret|vercel[_-]?token)\b["']?\s*[:=]\s*["'][^"'\s][^"'\r\n]*["']/i],
];

export function privacyFindings(content) {
  // Check text as well as common HTML/JSON representations emitted by Next.
  const decoded = content.replace(/&#(?:64|x40);/gi, "@").replace(/&quot;|&#34;/g, '"')
    .replace(/&#39;|&apos;/g, "'").replace(/\\u([0-9a-f]{4})/gi, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
  return patterns.filter(([, pattern]) => pattern.test(decoded)).map(([category]) => category);
}

export function sensitiveFilename(file) {
  return /(?:^|\/)(?:\.env(?:\..*)?|\.npmrc|credentials[^/]*\.json)$|\.(?:pem|key|p12|pfx)$/i.test(file)
    || /(?:^|\/)\.vercel\//.test(file);
}

async function walk(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await walk(file));
    else if (entry.isFile()) files.push(file);
    // Never follow symlinks into unrelated local files.
  }
  return files;
}

async function check() {
  const files = await walk(output);
  let count = 0;
  for (const file of files) {
    const relative = path.relative(output, file).split(path.sep).join("/");
    const content = await readFile(file);
    const findings = sensitiveFilename(relative) ? ["credential file"] : [];
    if (!content.includes(0)) findings.push(...privacyFindings(content.toString("utf8")));
    if (findings.length) {
      count++;
      const safePath = privacyFindings(relative).length ? "[private filename]" : relative;
      console.error(`${safePath}: ${findings.join(", ")}`);
    }
  }
  if (count) throw new Error(`Privacy check blocked ${count} exported file(s). Remove private content before publishing.`);
  console.log(`Privacy check passed (${files.length} exported files).`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  check().catch(error => {
    // Do not print exceptions containing source snippets, tokens, or absolute paths.
    console.error(error.message.startsWith("Privacy check blocked") ? error.message : "Privacy check could not complete; build blocked.");
    process.exitCode = 1;
  });
}
