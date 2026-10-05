import type { NextConfig } from "next";

const config: NextConfig = {
  output: "export",
  trailingSlash: true,
  // All content is prepared locally; the exported page needs no server.
  reactStrictMode: true,
};

export default config;
