import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import CodeBlock from "../components/code-block";
import { Prose } from "../components/report";

function textContent(html: string) {
  const entities: Record<string, string> = { "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#x27;": "'" };
  return html.replace(/<[^>]*>/g, "").replace(/&(?:amp|lt|gt|quot|#x27);/g, entity => entities[entity]);
}

test("Python highlighting preserves source whitespace and escapes HTML", () => {
  const code = '\ndef decode(x):\n\t# A comment\n\treturn "<script>alert(1)</script>"\n';
  const html = renderToStaticMarkup(createElement(CodeBlock, { code, language: "python" }));
  assert.match(html, /hljs-keyword/);
  assert.match(html, /hljs-comment/);
  assert.match(html, /hljs-string/);
  assert.equal(textContent(html), code);
  assert.ok(!html.includes("<script>"));
});

test("diff highlighting preserves signs and plain text has no inferred language", () => {
  const code = "@@ -1 +1 @@\n-return old\n+return new\n";
  const html = renderToStaticMarkup(createElement(CodeBlock, { code, language: "diff" }));
  assert.match(html, /hljs-addition/);
  assert.match(html, /hljs-deletion/);
  assert.equal(textContent(html), code);
  for (const language of [undefined, "unknown-language"]) {
    const plain = renderToStaticMarkup(createElement(CodeBlock, { code, language }));
    assert.ok(!plain.includes("hljs-"));
    assert.equal(textContent(plain), code);
  }
});

test("Markdown fences share highlighting while inline and unlabeled code stay plain", () => {
  const text = "Inline `return`.\n\n```py\nreturn 1\n```\n\n```\nreturn 2\n```\n\n```unknown\nreturn 3\n```";
  const html = renderToStaticMarkup(createElement(Prose, { text }));
  assert.match(html, /Inline <code>return<\/code>/);
  const blocks = html.match(/<pre\b[^>]*>[\s\S]*?<\/pre>/g)!;
  assert.equal(blocks.length, 3);
  assert.match(blocks[0], /hljs-keyword/);
  assert.ok(!blocks[1].includes("hljs-"));
  assert.ok(!blocks[2].includes("hljs-"));
  assert.deepEqual(blocks.map(textContent), ["return 1\n", "return 2\n", "return 3\n"]);
});
