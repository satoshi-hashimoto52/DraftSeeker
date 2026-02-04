export type UploadResponse = {
  image_id: string;
  width: number;
  height: number;
};

export type DetectResult = {
  class_name: string;
  score: number;
  bbox: { x: number; y: number; w: number; h: number };
  template_name: string;
  scale: number;
  contour?: { x: number; y: number }[];
};

export type DetectPointResponse = {
  results: DetectResult[];
};

export type Candidate = {
  id: string;
  class_name: string;
  score: number;
  bbox: { x: number; y: number; w: number; h: number };
  template: string;
  scale: number;
  segPolygon?: { x: number; y: number }[];
  segMethod?: "sam" | "fallback";
  source?: "template" | "manual";
};

export type Annotation = {
  id: string;
  class_name: string;
  bbox: { x: number; y: number; w: number; h: number };
  source: "template" | "manual" | "sam";
  created_at: string;
  segPolygon?: { x: number; y: number }[];
  originalSegPolygon?: { x: number; y: number }[];
  segMethod?: "sam" | "fallback";
};

export function toCandidates(res: DetectPointResponse): Candidate[] {
  const now = Date.now();
  return (res.results || []).map((r, idx) => ({
    id: `${now}-${Math.random()}-${idx}`,
    class_name: r.class_name,
    score: r.score,
    bbox: r.bbox,
    template: r.template_name,
    scale: r.scale,
  }));
}

export type SegmentCandidateRequest = {
  image_id: string;
  bbox: { x: number; y: number; w: number; h: number };
  click?: { x: number; y: number } | null;
  expand?: number;
  simplify_eps?: number;
};

export type SegmentCandidateResponse = {
  ok: boolean;
  polygon?: { x: number; y: number }[];
  bbox?: { x: number; y: number; w: number; h: number };
  meta?: { device: "mps" | "cpu"; method: "sam" | "fallback"; area: number };
  error?: string;
};

export async function segmentCandidate(
  params: SegmentCandidateRequest
): Promise<SegmentCandidateResponse> {
  const res = await fetch(`${API_BASE}/segment/candidate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Segment failed");
  }

  return (await res.json()) as SegmentCandidateResponse;
}

export type ExportAnnotation = {
  class_name: string;
  bbox: { x: number; y: number; w: number; h: number };
  segPolygon?: { x: number; y: number }[];
};

export type ExportYoloResponse = {
  ok: boolean;
  saved_path?: string;
  text_preview?: string;
  error?: string;
};

export async function exportYolo(params: {
  project: string;
  image_id: string;
  annotations: ExportAnnotation[];
}): Promise<ExportYoloResponse> {
  const res = await fetch(`${API_BASE}/export/yolo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Export failed");
  }

  return (await res.json()) as ExportYoloResponse;
}

const API_BASE = "http://127.0.0.1:8000";

export async function uploadImage(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/image/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Upload failed");
  }

  return (await res.json()) as UploadResponse;
}

export async function fetchProjects(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/projects`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Projects fetch failed");
  }
  return (await res.json()) as string[];
}

export type ProjectTemplates = {
  name: string;
  classes: { class_name: string; count: number }[];
};

export async function fetchTemplates(): Promise<ProjectTemplates[]> {
  const res = await fetch(`${API_BASE}/templates`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Templates fetch failed");
  }
  return (await res.json()) as ProjectTemplates[];
}

export async function detectPoint(params: {
  image_id: string;
  project: string;
  x: number;
  y: number;
  roi_size: number;
  scale_min: number;
  scale_max: number;
  scale_steps: number;
  score_threshold?: number;
  iou_threshold?: number;
  topk: number;
  template_off?: boolean;
}): Promise<DetectPointResponse> {
  const res = await fetch(`${API_BASE}/detect/point`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Detect failed");
  }

  return (await res.json()) as DetectPointResponse;
}
