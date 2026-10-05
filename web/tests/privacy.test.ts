import assert from "node:assert/strict";
import test from "node:test";
import { privacyFindings, sensitiveFilename } from "../scripts/check-privacy.mjs";

test("privacy check detects private content without returning its value", () => {
  const email = ["person", "example.test"].join("@");
  const token = "ghp_" + "a".repeat(36);
  const samples = [
    [email, "email address"],
    [email.replace("@", "&#64;"), "email address"],
    [email.replace("@", "\\u0040"), "email address"],
    [token, "access token"],
    [["-----BEGIN", "PRIVATE KEY-----"].join(" "), "private key"],
    [["https://", "person", ":", "password", "@", "example.test/"].join(""), "credential in URL"],
    [["/Users", "person", "project"].join("/"), "personal home path"],
    ["password = " + JSON.stringify("sensitive-value"), "assigned credential"],
    ["password = " + JSON.stringify("short"), "assigned credential"],
    ["password = " + JSON.stringify("a pass phrase"), "assigned credential"],
  ];
  for (const [content, category] of samples) {
    assert.ok(privacyFindings(content).includes(category));
    assert.ok(!privacyFindings(content).some(finding => finding.includes(content)));
  }
});

test("privacy check allows research identifiers but rejects credential filenames", () => {
  assert.deepEqual(privacyFindings("MODEL_REVISION = abc123; confirmed_gain = 1.234; https://github.com/ackord/decode-research"), []);
  for (const file of ["web/.env.local", "web/.vercel/project.json", "web/server.key", "web/.npmrc"]) {
    assert.equal(sensitiveFilename(file), true);
  }
  assert.equal(sensitiveFilename("experiments/0006/report.json"), false);
});
