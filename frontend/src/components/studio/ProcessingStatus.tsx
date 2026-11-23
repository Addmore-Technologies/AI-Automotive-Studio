import type { JobProgressInfo } from "@/services/studioApi";

const STAGES = [
  { key: "uploading", label: "Media validation & upload" },
  { key: "initializing", label: "Initializing vision models" },
  { key: "detecting", label: "Vehicle detection (YOLOv8)" },
  { key: "segmenting", label: "Vehicle segmentation & cutout" },
  { key: "rendering", label: "Studio compositing & lighting" },
  { key: "completed", label: "Final render & video encoding" },
];

export function ProcessingStatus({
  kind,
  progress,
}: {
  kind: "image" | "video";
  progress?: JobProgressInfo;
}) {
  const pct = Math.min(100, Math.max(0, Math.round(progress?.progressPct ?? (kind === "video" ? 5 : 20))));

  // Determine active stage index
  let activeIndex = 0;
  if (kind === "video") {
    const stage = progress?.stage;
    if (stage === "uploading") activeIndex = 0;
    else if (stage === "initializing") activeIndex = 1;
    else if (stage === "processing_frames") {
      // Rotate through detection and segmentation as frames progress
      activeIndex = pct > 50 ? 3 : 2;
    } else if (stage === "rendering" || stage === "audio_muxing") activeIndex = 4;
    else if (stage === "completed" || pct >= 100) activeIndex = 5;
    else {
      if (pct < 10) activeIndex = 0;
      else if (pct < 20) activeIndex = 1;
      else if (pct < 60) activeIndex = 2;
      else if (pct < 90) activeIndex = 3;
      else if (pct < 100) activeIndex = 4;
      else activeIndex = 5;
    }
  } else {
    if (pct < 25) activeIndex = 0;
    else if (pct < 50) activeIndex = 1;
    else if (pct < 90) activeIndex = 4;
    else activeIndex = 5;
  }

  const currentMessage =
    progress?.message ||
    (kind === "video"
      ? "Processing video frames into studio environment..."
      : "Placing vehicle inside the fixed LED studio...");

  return (
    <div className="border-t border-border py-8" role="status" aria-live="polite">
      {/* Header with status and progress percentage */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
          </span>
          <span className="label-mono text-foreground">Processing</span>
        </div>
        <div className="flex items-center gap-4">
          {progress?.currentFrame && progress?.totalFrames ? (
            <span className="label-mono text-xs text-muted-foreground">
              Frame {progress.currentFrame} / {progress.totalFrames}
            </span>
          ) : null}
          <span className="label-mono text-foreground font-semibold">
            {pct}%
          </span>
        </div>
      </div>

      {/* Real progress bar */}
      <div className="relative mt-4 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full bg-primary transition-all duration-300 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Live backend message */}
      <p className="mt-4 text-sm font-medium text-foreground">
        {currentMessage}
      </p>

      {/* Stage indicators */}
      <ol className="mt-6 grid gap-x-8 sm:grid-cols-2">
        {STAGES.map((s, i) => {
          const isDone = i < activeIndex || pct >= 100;
          const isCurrent = i === activeIndex && pct < 100;

          return (
            <li
              key={s.key}
              className={`flex items-center justify-between border-b border-border py-3 pr-3 transition-colors ${
                isCurrent
                  ? "bg-secondary/40 font-medium"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span
                  className={`label-mono text-xs ${
                    isDone
                      ? "text-primary"
                      : isCurrent
                      ? "text-foreground font-semibold"
                      : "text-muted-foreground"
                  }`}
                >
                  {isDone ? "✓" : String(i + 1).padStart(2, "0")}
                </span>
                <span
                  className={`label-mono text-xs ${
                    isDone
                      ? "text-foreground"
                      : isCurrent
                      ? "text-primary font-semibold"
                      : "text-muted-foreground"
                  }`}
                >
                  {s.label}
                </span>
              </div>
              {isCurrent && (
                <span className="label-mono text-[10px] uppercase text-primary animate-pulse">
                  Active
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
