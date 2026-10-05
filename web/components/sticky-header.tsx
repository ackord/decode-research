"use client";

import { useLayoutEffect, useRef, type ReactNode } from "react";

export default function StickyHeader({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLElement>(null);

  useLayoutEffect(() => {
    const header = ref.current;
    if (!header) return;
    const measure = () => {
      document.documentElement.style.setProperty("--sticky-header-height", `${header.getBoundingClientRect().height}px`);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(header);

    // Native fragment navigation may happen before hydration. Apply the
    // measured offset once for direct links and paginated history navigation.
    const frame = requestAnimationFrame(() => {
      if (window.location.hash) document.getElementById(window.location.hash.slice(1))?.scrollIntoView({ block: "start" });
    });
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }, []);

  return <header className="site-header" ref={ref}>{children}</header>;
}
