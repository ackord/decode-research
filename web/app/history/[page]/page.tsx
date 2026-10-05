import { notFound } from "next/navigation";
import ResearchPage from "../../../components/research-page";
import { HISTORY_PAGE_SIZE } from "../../../lib/history";
import { loadResearch } from "../../../lib/research";

export const dynamicParams = false;

export async function generateStaticParams() {
  const { experiments } = await loadResearch();
  return Array.from({ length: Math.max(1, Math.ceil(experiments.length / HISTORY_PAGE_SIZE)) }, (_, index) => ({ page: String(index + 1) }));
}

export default async function HistoryPage({ params }: { params: Promise<{ page: string }> }) {
  const { page } = await params;
  if (!/^[1-9]\d*$/.test(page)) notFound();
  return <ResearchPage historyPage={Number(page)} />;
}
