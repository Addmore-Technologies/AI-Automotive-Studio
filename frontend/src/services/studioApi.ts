/**
 * Studio backend integration.
 *
 * The backend API contract has NOT been agreed yet. Everything here is driven by env vars
 * so no endpoint path or response shape is hard-coded as fact:
 *   VITE_STUDIO_API_URL        base URL of the processing backend (required to enable processing)
 *   VITE_STUDIO_IMAGE_ENDPOINT path for image processing (default "/process/image" — placeholder)
 *   VITE_STUDIO_VIDEO_ENDPOINT path for video processing (default "/process/video" — placeholder)
 *
 * Assumed placeholder contract (adjust in `parseResult` once the backend is defined):
 *   POST multipart/form-data with field "file"
 *   -> JSON { status?: string; result_url?: string; output_format?: string; message?: string; error?: string }
 */

export type MediaKind = "image" | "video";

export interface JobProgressInfo {
  jobId?: string;
  status: "pending" | "processing" | "completed" | "failed";
  progressPct: number;
  stage?: string;
  message?: string;
  currentFrame?: number;
  totalFrames?: number;
}

export interface StudioResult {
  kind: MediaKind;
  resultUrl: string | null;
  status: string;
  outputFormat: string | null;
  message: string | null;
  fileName: string;
}

export class StudioApiError extends Error {
  constructor(
    message: string,
    public code: "not_configured" | "no_vehicle" | "network" | "server",
  ) {
    super(message);
  }
}

const BASE = (import.meta.env["VITE_STUDIO_API_URL"] as string | undefined)?.replace(/\/$/, "");
const IMAGE_PATH = (import.meta.env["VITE_STUDIO_IMAGE_ENDPOINT"] as string | undefined) ?? "/process/image";
const VIDEO_PATH = (import.meta.env["VITE_STUDIO_VIDEO_ENDPOINT"] as string | undefined) ?? "/process/video";

export const isBackendConfigured = Boolean(BASE);

export const ACCEPT: Record<MediaKind, { ext: string[]; mime: string[]; label: string }> = {
  image: { ext: ["jpg", "jpeg", "png"], mime: ["image/jpeg", "image/png"], label: "JPG · JPEG · PNG" },
  video: { ext: ["mp4", "mov"], mime: ["video/mp4", "video/quicktime"], label: "MP4 · MOV" },
};

export function validateFile(file: File, kind: MediaKind): string | null {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!ACCEPT[kind].ext.includes(ext)) {
    return `Unsupported format ".${ext || "?"}". Accepted: ${ACCEPT[kind].label}.`;
  }
  return null;
}

function parseResult(kind: MediaKind, fileName: string, data: Record<string, unknown>): StudioResult {
  const url = (data["result_url"] ?? data["url"] ?? null) as string | null;
  const absolute = url && BASE && url.startsWith("/") ? `${BASE}${url}` : url;
  return {
    kind,
    fileName,
    resultUrl: absolute,
    status: String(data["status"] ?? "completed"),
    outputFormat: (data["output_format"] as string) ?? null,
    message: (data["message"] as string) ?? null,
  };
}

/** Cancellable delay helper using AbortSignal. */
function abortableSleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

export async function processMedia(
  file: File,
  kind: MediaKind,
  signal?: AbortSignal,
  onProgress?: (progress: JobProgressInfo) => void,
): Promise<StudioResult> {
  if (!BASE) {
    throw new StudioApiError(
      "The processing backend is not connected yet. Set VITE_STUDIO_API_URL to enable studio rendering.",
      "not_configured",
    );
  }

  const body = new FormData();
  body.append("file", file);

  if (kind === "image") {
    onProgress?.({
      status: "processing",
      progressPct: 15,
      stage: "uploading",
      message: "Uploading car image to studio backend...",
    });

    let res: Response;
    try {
      res = await fetch(`${BASE}${IMAGE_PATH}`, { method: "POST", body, signal: signal ?? null });
    } catch (e) {
      if ((e as Error).name === "AbortError") throw e;
      throw new StudioApiError("Could not reach the processing backend. Check your connection and try again.", "network");
    }

    onProgress?.({
      status: "processing",
      progressPct: 60,
      stage: "rendering",
      message: "Compositing vehicle into approved studio lighting...",
    });

    let data: Record<string, unknown> = {};
    try {
      data = await res.json();
    } catch {
      /* non-JSON body */
    }

    const errText = String(data["error"] ?? data["detail"] ?? data["message"] ?? "");
    if (!res.ok || data["error"]) {
      if (/no vehicle|vehicle not (found|detected)/i.test(errText)) {
        throw new StudioApiError("No vehicle was detected in this media. Try a clearer photo where the full vehicle is visible.", "no_vehicle");
      }
      throw new StudioApiError(errText || `Processing failed (status ${res.status}).`, "server");
    }

    onProgress?.({
      status: "completed",
      progressPct: 100,
      stage: "completed",
      message: "Studio render completed.",
    });

    return parseResult(kind, file.name, data);
  }

  // --- Video Processing: Asynchronous Workflow with Real Progress Polling ---
  onProgress?.({
    status: "pending",
    progressPct: 3,
    stage: "uploading",
    message: "Uploading video to studio backend...",
  });

  const uploadEndpoint = `${BASE}${VIDEO_PATH}${VIDEO_PATH.includes("?") ? "&" : "?"}async_mode=true`;

  let postRes: Response;
  try {
    postRes = await fetch(uploadEndpoint, { method: "POST", body, signal: signal ?? null });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new StudioApiError("Could not reach the processing backend. Check your connection and try again.", "network");
  }

  let postData: Record<string, unknown> = {};
  try {
    postData = await postRes.json();
  } catch {
    /* non-JSON body */
  }

  const postErrText = String(postData["error"] ?? postData["detail"] ?? postData["message"] ?? "");
  if (!postRes.ok || postData["error"]) {
    if (/no vehicle|vehicle not (found|detected)/i.test(postErrText)) {
      throw new StudioApiError("No vehicle was detected in this video. Try a clearer video where the vehicle is visible.", "no_vehicle");
    }
    throw new StudioApiError(postErrText || `Video upload failed (status ${postRes.status}).`, "server");
  }

  // If backend responded synchronously with a completed result
  if (postData["status"] === "completed" && postData["result_url"]) {
    onProgress?.({
      status: "completed",
      progressPct: 100,
      stage: "completed",
      message: "Video processing completed.",
    });
    return parseResult(kind, file.name, postData);
  }

  // Extract Job ID for polling
  const jobId = (postData["job_id"] ?? postData["id"]) as string | undefined;
  if (!jobId) {
    throw new StudioApiError("Backend did not return a valid job ID for video processing.", "server");
  }

  onProgress?.({
    jobId,
    status: "processing",
    progressPct: Number(postData["progress_pct"] ?? 5),
    stage: String(postData["stage"] ?? "initializing"),
    message: String(postData["message"] ?? "Initializing AI vision models..."),
    currentFrame: postData["current_frame"] as number | undefined,
    totalFrames: postData["total_frames"] as number | undefined,
  });

  // Poll job status until completion or failure
  const POLL_INTERVAL_MS = 800;
  while (true) {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    await abortableSleep(POLL_INTERVAL_MS, signal);
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");

    let pollRes: Response;
    try {
      pollRes = await fetch(`${BASE}/api/jobs/${jobId}`, { signal: signal ?? null });
    } catch (e) {
      if ((e as Error).name === "AbortError") throw e;
      throw new StudioApiError("Connection to backend interrupted during job polling.", "network");
    }

    let jobData: Record<string, unknown> = {};
    try {
      jobData = await pollRes.json();
    } catch {
      throw new StudioApiError("Invalid status response from server.", "server");
    }

    if (!pollRes.ok) {
      const errDetail = typeof jobData["detail"] === "object" ? (jobData["detail"] as any)?.error : jobData["detail"];
      const pollErr = String(jobData["error"] ?? errDetail ?? `Status check failed (status ${pollRes.status})`);
      throw new StudioApiError(pollErr, "server");
    }

    const jobStatus = String(jobData["status"] ?? "processing");
    const progressPct = Number(jobData["progress_pct"] ?? 0);
    const message = (jobData["message"] as string) ?? undefined;
    const stage = (jobData["stage"] as string) ?? undefined;
    const currentFrame = (jobData["current_frame"] as number) ?? undefined;
    const totalFrames = (jobData["total_frames"] as number) ?? undefined;

    onProgress?.({
      jobId,
      status: jobStatus as any,
      progressPct,
      stage,
      message,
      currentFrame,
      totalFrames,
    });

    if (jobStatus === "completed") {
      return parseResult(kind, file.name, jobData);
    }

    if (jobStatus === "failed") {
      const errText = String(jobData["error"] ?? jobData["message"] ?? "Video processing failed.");
      if (/no vehicle|vehicle not (found|detected)/i.test(errText)) {
        throw new StudioApiError("No vehicle was detected in this video. Try a clearer video where the vehicle is clearly visible.", "no_vehicle");
      }
      throw new StudioApiError(errText || "Video processing failed.", "server");
    }
  }
}

export function formatBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}
