import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Decode Research",
  description: "Autonomous research on decode throughput for a fixed language model, with exact greedy-token evaluation and recorded experiment results.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
