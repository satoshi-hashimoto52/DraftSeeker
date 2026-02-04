import React, { useEffect, useMemo, useRef, useState } from "react";

import {
  Annotation,
  Candidate,
  detectPoint,
  exportYolo,
  fetchProjects,
  segmentCandidate,
  toCandidates,
  uploadImage,
} from "./api";
import ImageCanvas, { ImageCanvasHandle } from "./components/ImageCanvas";
import CandidateList from "./components/CandidateList";
import { normalizeToHex } from "./utils/color";

const DEFAULT_ROI_SIZE = 200;
const DEFAULT_TOPK = 3;
const DEFAULT_SCALE_MIN = 0.5;
const DEFAULT_SCALE_MAX = 1.5;
const DEFAULT_SCALE_STEPS = 12;

export default function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageId, setImageId] = useState<string | null>(null);
  const [projects, setProjects] = useState<string[]>([]);
  const [project, setProject] = useState<string>("");
  const [roiSize, setRoiSize] = useState<number>(DEFAULT_ROI_SIZE);
  const [topk, setTopk] = useState<number>(DEFAULT_TOPK);
  const [scaleMin, setScaleMin] = useState<number>(DEFAULT_SCALE_MIN);
  const [scaleMax, setScaleMax] = useState<number>(DEFAULT_SCALE_MAX);
  const [scaleSteps, setScaleSteps] = useState<number>(DEFAULT_SCALE_STEPS);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null);
  const [colorMap, setColorMap] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [showCandidates, setShowCandidates] = useState<boolean>(true);
  const [showAnnotations, setShowAnnotations] = useState<boolean>(true);
  const canvasRef = useRef<ImageCanvasHandle | null>(null);
  const [lastClick, setLastClick] = useState<{ x: number; y: number } | null>(null);
  const [exportPreview, setExportPreview] = useState<string | null>(null);

  const selectedCandidate = useMemo(() => {
    if (!selectedCandidateId) return null;
    return candidates.find((c) => c.id === selectedCandidateId) || null;
  }, [candidates, selectedCandidateId]);

  useEffect(() => {
    let mounted = true;
    fetchProjects()
      .then((list) => {
        if (!mounted) return;
        setProjects(list);
        if (!project && list.length > 0) {
          setProject(list[0]);
        }
      })
      .catch((err) => {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : "Projects fetch failed");
      });
    return () => {
      mounted = false;
    };
  }, [project]);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(null);
    setBusy(true);
    try {
      const res = await uploadImage(file);
      setImageId(res.image_id);
      setImageUrl(URL.createObjectURL(file));
      setCandidates([]);
      setSelectedCandidateId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const handleClickPoint = async (x: number, y: number) => {
    if (!imageId || !project) return;
    setError(null);
    setNotice(null);
    setBusy(true);
    setLastClick({ x, y });
    try {
      const res = await detectPoint({
        image_id: imageId,
        project,
        x,
        y,
        roi_size: roiSize,
        scale_min: scaleMin,
        scale_max: scaleMax,
        scale_steps: scaleSteps,
        topk,
      });
      const nextCandidates = toCandidates(res);
      setCandidates(nextCandidates);
      setSelectedCandidateId(nextCandidates.length > 0 ? nextCandidates[0].id : null);
      setColorMap((prev) => {
        const next = { ...prev };
        nextCandidates.forEach((r) => {
          if (!next[r.class_name]) {
            next[r.class_name] = normalizeToHex(randomColor(r.class_name));
          }
        });
        return next;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detect failed");
    } finally {
      setBusy(false);
    }
  };

  const handleConfirmCandidate = () => {
    if (!selectedCandidate) return;
    const createdAt = new Date().toISOString();
    const source = selectedCandidate.segPolygon ? "sam" : "template";
    setAnnotations((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${Math.random()}`,
        class_name: selectedCandidate.class_name,
        bbox: selectedCandidate.bbox,
        source,
        created_at: createdAt,
        segPolygon: selectedCandidate.segPolygon,
      },
    ]);
    setNotice(`${selectedCandidate.class_name} を確定しました`);
    if (candidates.length > 0) {
      const index = candidates.findIndex((c) => c.id === selectedCandidate.id);
      if (index >= 0) {
        const nextIndex = (index + 1) % candidates.length;
        setSelectedCandidateId(candidates[nextIndex].id);
      }
    }
  };

  const handleRejectCandidate = () => {
    if (!selectedCandidate) return;
    const index = candidates.findIndex((c) => c.id === selectedCandidate.id);
    const next = candidates.filter((c) => c.id !== selectedCandidate.id);
    setCandidates(next);
    if (next.length === 0) {
      setSelectedCandidateId(null);
      return;
    }
    const nextIndex = index < next.length ? index : next.length - 1;
    setSelectedCandidateId(next[nextIndex].id);
  };

  const handleNextCandidate = () => {
    if (candidates.length === 0) return;
    const index = selectedCandidateId
      ? candidates.findIndex((c) => c.id === selectedCandidateId)
      : -1;
    const nextIndex = index >= 0 ? (index + 1) % candidates.length : 0;
    setSelectedCandidateId(candidates[nextIndex].id);
  };

  const handleSegCandidate = async () => {
    if (!selectedCandidate || !imageId) return;
    setError(null);
    setNotice(null);
    setBusy(true);
    try {
      const res = await segmentCandidate({
        image_id: imageId,
        bbox: selectedCandidate.bbox,
        click: lastClick,
      });
      if (!res.ok || !res.polygon) {
        setError(res.error || "Segmentation failed");
        return;
      }
      setCandidates((prev) =>
        prev.map((c) =>
          c.id === selectedCandidate.id ? { ...c, segPolygon: res.polygon } : c
        )
      );
      setNotice(`${selectedCandidate.class_name} のSegを生成しました`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Segmentation failed");
    } finally {
      setBusy(false);
    }
  };

  const handleExportYolo = async () => {
    if (!imageId || !project) return;
    setError(null);
    setNotice(null);
    setBusy(true);
    try {
      const res = await exportYolo({
        project,
        image_id: imageId,
        annotations: annotations.map((a) => ({
          class_name: a.class_name,
          bbox: a.bbox,
          segPolygon: a.segPolygon,
        })),
      });
      if (!res.ok) {
        setError(res.error || "Export failed");
        return;
      }
      setExportPreview(res.text_preview || "");
      setNotice(`Exported: ${res.saved_path || ""}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setBusy(false);
    }
  };

  const handleSelectAnnotation = (annotation: Annotation) => {
    setSelectedAnnotationId(annotation.id);
    const centerX = annotation.bbox.x + annotation.bbox.w / 2;
    const centerY = annotation.bbox.y + annotation.bbox.h / 2;
    canvasRef.current?.panTo(centerX, centerY);
  };

  return (
    <div style={{ fontFamily: "\"IBM Plex Sans\", system-ui, sans-serif" }}>
      <div style={{ padding: "16px 20px", borderBottom: "1px solid #eee" }}>
        <div style={{ fontSize: 18, fontWeight: 700 }}>Annotator MVP</div>
        <div style={{ fontSize: 12, color: "#666" }}>画像クリックでテンプレ照合</div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 320px",
          gap: 20,
          padding: 20,
        }}
      >
        <div>
          <div style={{ marginBottom: 12, display: "flex", gap: 12, flexWrap: "wrap" }}>
            <label style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <input type="file" accept="image/*" onChange={handleFileChange} />
              <span style={{ fontSize: 13 }}>画像アップロード</span>
            </label>
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 12 }}>プロジェクト</span>
              <select
                value={project}
                onChange={(e) => setProject(e.target.value)}
                style={{ minWidth: 140 }}
              >
                {projects.length === 0 && <option value="">(none)</option>}
                {projects.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 12 }}>ROIサイズ</span>
              <input
                type="number"
                min={10}
                value={roiSize}
                step={10}
                onChange={(e) => setRoiSize(Number(e.target.value))}
                style={{ width: 80 }}
              />
            </label>
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 12 }}>上位件数</span>
              <input
                type="number"
                min={1}
                value={topk}
                onChange={(e) => setTopk(Number(e.target.value))}
                style={{ width: 60 }}
              />
            </label>
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 12 }}>最小スケール</span>
              <input
                type="number"
                step={0.1}
                min={0.1}
                value={scaleMin}
                onChange={(e) => setScaleMin(Number(e.target.value))}
                style={{ width: 70 }}
              />
            </label>
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 12 }}>最大スケール</span>
              <input
                type="number"
                step={0.1}
                min={0.1}
                value={scaleMax}
                onChange={(e) => setScaleMax(Number(e.target.value))}
                style={{ width: 70 }}
              />
            </label>
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 12 }}>スケール分割数</span>
              <input
                type="number"
                min={1}
                value={scaleSteps}
                onChange={(e) => setScaleSteps(Number(e.target.value))}
                style={{ width: 60 }}
              />
            </label>
          </div>

          {error && (
            <div style={{ marginBottom: 12, color: "#b00020" }}>Error: {error}</div>
          )}
          {notice && (
            <div style={{ marginBottom: 12, color: "#1b5e20", fontSize: 12 }}>{notice}</div>
          )}

          <ImageCanvas
            ref={canvasRef}
            imageUrl={imageUrl}
            candidates={candidates}
            selectedCandidateId={selectedCandidateId}
            annotations={annotations}
            selectedAnnotationId={selectedAnnotationId}
            colorMap={colorMap}
            showCandidates={showCandidates}
            showAnnotations={showAnnotations}
            onClickPoint={handleClickPoint}
          />
          {busy && <div style={{ marginTop: 10, color: "#666" }}>処理中...</div>}
        </div>

        <div style={{ borderLeft: "1px solid #eee", paddingLeft: 16 }}>
          <div style={{ marginBottom: 12 }}>
            <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
              <input
                type="checkbox"
                checked={showCandidates}
                onChange={(e) => setShowCandidates(e.target.checked)}
              />
              <span style={{ fontSize: 12 }}>未確定候補を表示</span>
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input
                type="checkbox"
                checked={showAnnotations}
                onChange={(e) => setShowAnnotations(e.target.checked)}
              />
              <span style={{ fontSize: 12 }}>確定アノテーションを表示</span>
            </label>
          </div>
          <div style={{ marginBottom: 18 }}>
            <CandidateList
              candidates={candidates}
              selectedCandidateId={selectedCandidateId}
              onSelect={setSelectedCandidateId}
              colorMap={colorMap}
            />
            <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
              <button
                type="button"
                onClick={handleConfirmCandidate}
                disabled={!selectedCandidate}
                style={{ padding: "8px 10px", fontSize: 12, cursor: "pointer" }}
              >
                ✔ この候補を確定
              </button>
              <button
                type="button"
                onClick={handleRejectCandidate}
                disabled={!selectedCandidate}
                style={{ padding: "8px 10px", fontSize: 12, cursor: "pointer" }}
              >
                ✖ 破棄（候補から除外）
              </button>
              <button
                type="button"
                onClick={handleNextCandidate}
                disabled={candidates.length === 0}
                style={{ padding: "8px 10px", fontSize: 12, cursor: "pointer" }}
              >
                ▶ 次の候補へ
              </button>
              <button
                type="button"
                onClick={handleSegCandidate}
                disabled={!selectedCandidate}
                style={{ padding: "8px 10px", fontSize: 12, cursor: "pointer" }}
              >
                🧩 Seg生成（SAM）
              </button>
            </div>
          </div>

          <div style={{ marginBottom: 18 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>
              確定アノテーション（合計 {annotations.length}件）
            </div>
            <div style={{ fontSize: 12, color: "#666", marginBottom: 8 }}>
              {annotations.length === 0
                ? "内訳: なし"
                : Object.entries(
                    annotations.reduce<Record<string, number>>((acc, a) => {
                      acc[a.class_name] = (acc[a.class_name] || 0) + 1;
                      return acc;
                    }, {})
                  )
                    .map(([name, count]) => `${name}: ${count}`)
                    .join(" / ")}
            </div>
            {annotations.length === 0 && (
              <div style={{ color: "#666" }}>確定アノテはまだありません。</div>
            )}
            {annotations.map((a) => (
              <div
                key={a.id}
                style={{
                  padding: "8px 10px",
                  marginBottom: 8,
                  border: "1px solid #e3e3e3",
                  borderRadius: 6,
                  background: selectedAnnotationId === a.id ? "#eef6ff" : "#fff",
                  cursor: "pointer",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: 8,
                }}
                onClick={() => handleSelectAnnotation(a)}
              >
                <div>
                  <div style={{ fontWeight: 600 }}>{a.class_name}</div>
                  <div style={{ fontSize: 12, color: "#666" }}>
                    bbox: ({a.bbox.x}, {a.bbox.y}, {a.bbox.w}, {a.bbox.h})
                  </div>
                </div>
                <button
                  type="button"
                  aria-label="delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    setAnnotations((prev) => prev.filter((item) => item.id !== a.id));
                    if (selectedAnnotationId === a.id) {
                      setSelectedAnnotationId(null);
                    }
                  }}
                  style={{
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                    fontSize: 16,
                  }}
                >
                  🗑
                </button>
              </div>
            ))}
          </div>

          <div style={{ marginBottom: 18 }}>
            <button
              type="button"
              onClick={handleExportYolo}
              disabled={annotations.length === 0 || !imageId}
              style={{ padding: "8px 10px", fontSize: 12, cursor: "pointer" }}
            >
              ⤴ YOLOエクスポート
            </button>
            {exportPreview !== null && (
              <pre
                style={{
                  marginTop: 8,
                  padding: 8,
                  background: "#f6f6f6",
                  border: "1px solid #e3e3e3",
                  borderRadius: 6,
                  fontSize: 12,
                  whiteSpace: "pre-wrap",
                }}
              >
                {exportPreview}
              </pre>
            )}
          </div>
          {Object.keys(colorMap).length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>シリーズ配色</div>
              {Object.entries(colorMap).map(([name, color]) => {
                const hexColor = normalizeToHex(color);
                return (
                <div
                  key={name}
                  style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}
                >
                  <input
                    type="color"
                    value={hexColor}
                    onChange={(e) =>
                      setColorMap((prev) => ({ ...prev, [name]: e.target.value }))
                    }
                  />
                  <span style={{ fontSize: 12 }}>{name}</span>
                </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function randomColor(seed: string): string {
  let hash = 0;
  for (let i = 0; i < seed.length; i += 1) {
    hash = (hash << 5) - hash + seed.charCodeAt(i);
    hash |= 0;
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 70%, 50%)`;
}
