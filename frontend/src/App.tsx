import React, { useEffect, useMemo, useRef, useState } from "react";

import {
  Annotation,
  Candidate,
  detectPoint,
  exportYolo,
  fetchProjects,
  fetchTemplates,
  segmentCandidate,
  toCandidates,
  uploadImage,
} from "./api";
import ImageCanvas, { ImageCanvasHandle } from "./components/ImageCanvas";
import CandidateList from "./components/CandidateList";
import { normalizeToHex } from "./utils/color";
import { clampToImage, simplifyPolygon } from "./utils/polygon";

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
  const [classOptions, setClassOptions] = useState<string[]>([]);
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
  const [segEditMode, setSegEditMode] = useState<boolean>(false);
  const [showSegVertices, setShowSegVertices] = useState<boolean>(true);
  const [selectedVertexIndex, setSelectedVertexIndex] = useState<number | null>(null);
  const [segUndoStack, setSegUndoStack] = useState<{ x: number; y: number }[][]>([]);
  const [segSimplifyEps, setSegSimplifyEps] = useState<number>(2);
  const [imageSize, setImageSize] = useState<{ w: number; h: number } | null>(null);
  const [isCreatingManualBBox, setIsCreatingManualBBox] = useState<boolean>(false);

  const selectedCandidate = useMemo(() => {
    if (!selectedCandidateId) return null;
    return candidates.find((c) => c.id === selectedCandidateId) || null;
  }, [candidates, selectedCandidateId]);

  const isManualSelected = useMemo(
    () => selectedCandidate?.source === "manual",
    [selectedCandidate]
  );
  const manualClassMissing = useMemo(
    () => isManualSelected && !selectedCandidate?.class_name,
    [isManualSelected, selectedCandidate]
  );

  const selectedAnnotation = useMemo(() => {
    if (!selectedAnnotationId) return null;
    return annotations.find((a) => a.id === selectedAnnotationId) || null;
  }, [annotations, selectedAnnotationId]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!selectedCandidate || segEditMode) return;
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;

      let dx = 0;
      let dy = 0;
      const step = event.shiftKey ? 10 : 1;
      if (event.key === "ArrowLeft") dx = -step;
      if (event.key === "ArrowRight") dx = step;
      if (event.key === "ArrowUp") dy = -step;
      if (event.key === "ArrowDown") dy = step;
      if (dx === 0 && dy === 0) return;

      event.preventDefault();
      setCandidates((prev) =>
        prev.map((c) => {
          if (c.id !== selectedCandidate.id) return c;
          let nextX = c.bbox.x + dx;
          let nextY = c.bbox.y + dy;
          if (imageSize) {
            nextX = Math.min(imageSize.w - c.bbox.w, Math.max(0, nextX));
            nextY = Math.min(imageSize.h - c.bbox.h, Math.max(0, nextY));
          }
          return {
            ...c,
            bbox: { ...c.bbox, x: nextX, y: nextY },
          };
        })
      );
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedCandidate, segEditMode, imageSize]);

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
    fetchTemplates()
      .then((list) => {
        if (!mounted) return;
        const selected = list.find((p) => p.name === project) || list[0];
        const classes = selected ? selected.classes.map((c) => c.class_name) : [];
        setClassOptions(classes);
      })
      .catch((err) => {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : "Templates fetch failed");
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
      setImageSize({ w: res.width, h: res.height });
      setCandidates([]);
      setSelectedCandidateId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const handleClickPoint = async (x: number, y: number) => {
    if (isCreatingManualBBox) return;
    if (manualClassMissing) return;
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
    if (selectedCandidate.source === "manual" && !selectedCandidate.class_name) {
      setError("手動候補はクラスを選択してください");
      return;
    }
    const createdAt = new Date().toISOString();
    const source =
      selectedCandidate.source === "manual"
        ? "manual"
        : selectedCandidate.segPolygon
          ? "sam"
          : "template";
    const segPolygon = selectedCandidate.segPolygon
      ? selectedCandidate.segPolygon.map((p) => ({ ...p }))
      : undefined;
    const segMethod = selectedCandidate.segMethod;
    setAnnotations((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${Math.random()}`,
        class_name: selectedCandidate.class_name,
        bbox: selectedCandidate.bbox,
        source,
        created_at: createdAt,
        segPolygon,
        originalSegPolygon: segPolygon ? segPolygon.map((p) => ({ ...p })) : undefined,
        segMethod,
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
      let nextPolygon = res.polygon;
      if (imageSize) {
        nextPolygon = clampToImage(nextPolygon, imageSize.w, imageSize.h);
      }
      nextPolygon = simplifyPolygon(nextPolygon, segSimplifyEps);
      setCandidates((prev) =>
        prev.map((c) =>
          c.id === selectedCandidate.id
            ? { ...c, segPolygon: nextPolygon, segMethod: res.meta?.method }
            : c
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
    setSegEditMode(false);
    setSelectedVertexIndex(null);
    setSegUndoStack([]);
    setShowSegVertices(true);
    const centerX = annotation.bbox.x + annotation.bbox.w / 2;
    const centerY = annotation.bbox.y + annotation.bbox.h / 2;
    canvasRef.current?.panTo(centerX, centerY);
  };

  const handleSegUndo = () => {
    if (segUndoStack.length === 0 || !selectedAnnotation) return;
    const last = segUndoStack[segUndoStack.length - 1];
    setSegUndoStack((prev) => prev.slice(0, -1));
    setAnnotations((prev) =>
      prev.map((a) =>
        a.id === selectedAnnotation.id ? { ...a, segPolygon: last.map((p) => ({ ...p })) } : a
      )
    );
  };

  const handleSegReset = () => {
    if (!selectedAnnotation?.originalSegPolygon) return;
    const reset = selectedAnnotation.originalSegPolygon.map((p) => ({ ...p }));
    setSegUndoStack([]);
    setAnnotations((prev) =>
      prev.map((a) => (a.id === selectedAnnotation.id ? { ...a, segPolygon: reset } : a))
    );
  };

  const applySegSimplify = () => {
    if (!selectedAnnotation?.segPolygon) return;
    let next = selectedAnnotation.segPolygon;
    if (imageSize) {
      next = clampToImage(next, imageSize.w, imageSize.h);
    }
    next = simplifyPolygon(next, segSimplifyEps);
    setAnnotations((prev) =>
      prev.map((a) => (a.id === selectedAnnotation.id ? { ...a, segPolygon: next } : a))
    );
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
            editablePolygon={segEditMode ? selectedAnnotation?.segPolygon || null : null}
            editMode={segEditMode}
            showVertices={showSegVertices}
            selectedVertexIndex={selectedVertexIndex}
            onSelectVertex={setSelectedVertexIndex}
            onUpdateEditablePolygon={(next) => {
              if (!selectedAnnotation) return;
              setAnnotations((prev) =>
                prev.map((a) => (a.id === selectedAnnotation.id ? { ...a, segPolygon: next } : a))
              );
            }}
            onVertexDragStart={() => {
              if (!selectedAnnotation?.segPolygon) return;
              setSegUndoStack((prev) => [
                ...prev,
                selectedAnnotation.segPolygon!.map((p) => ({ ...p })),
              ]);
            }}
            onClickPoint={handleClickPoint}
            onCreateManualBBox={(bbox) => {
              const id = `${Date.now()}-${Math.random()}`;
              const manualCandidate: Candidate = {
                id,
                class_name: "",
                score: 1.0,
                bbox,
                template: "manual",
                scale: 1.0,
                source: "manual",
              };
              setCandidates((prev) => [...prev, manualCandidate]);
              setSelectedCandidateId(id);
            }}
            onManualCreateStateChange={setIsCreatingManualBBox}
            onResizeSelectedBBox={(bbox) => {
              if (!selectedCandidateId) return;
              setCandidates((prev) =>
                prev.map((c) => (c.id === selectedCandidateId ? { ...c, bbox } : c))
              );
            }}
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
                disabled={!selectedCandidate || manualClassMissing}
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
            {isManualSelected && (
              <div style={{ marginTop: 10 }}>
                <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 12, minWidth: 60 }}>クラス選択</span>
                  <select
                    value={selectedCandidate?.class_name || ""}
                    onChange={(e) => {
                      const nextClass = e.target.value;
                      setCandidates((prev) =>
                        prev.map((c) =>
                          c.id === selectedCandidate?.id ? { ...c, class_name: nextClass } : c
                        )
                      );
                      if (nextClass) {
                        setColorMap((prev) => {
                          if (prev[nextClass]) return prev;
                          return { ...prev, [nextClass]: normalizeToHex(randomColor(nextClass)) };
                        });
                      }
                    }}
                    style={{ minWidth: 160 }}
                  >
                    <option value="">クラスを選択</option>
                    {classOptions.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            )}
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
                  <div style={{ fontWeight: 600, display: "flex", alignItems: "center", gap: 6 }}>
                    <span>{a.class_name}</span>
                    {a.segPolygon && a.segMethod && (
                      <span
                        style={{
                          fontSize: 10,
                          padding: "2px 6px",
                          borderRadius: 10,
                          background: a.segMethod === "sam" ? "#2e7d32" : "#888",
                          color: "#fff",
                        }}
                      >
                        {a.segMethod.toUpperCase()}
                      </span>
                    )}
                  </div>
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

          {selectedAnnotation?.segPolygon && (
            <div style={{ marginBottom: 18 }}>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>Seg編集</div>
              <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                <input
                  type="checkbox"
                  checked={segEditMode}
                  onChange={(e) => {
                    const next = e.target.checked;
                    if (!next && segEditMode) {
                      applySegSimplify();
                    }
                    setSegEditMode(next);
                  }}
                />
                <span style={{ fontSize: 12 }}>編集モードON/OFF</span>
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
                <input
                  type="checkbox"
                  checked={showSegVertices}
                  onChange={(e) => setShowSegVertices(e.target.checked)}
                  disabled={!segEditMode}
                />
                <span style={{ fontSize: 12 }}>頂点を表示</span>
              </label>
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ fontSize: 12, minWidth: 70 }}>簡略化</span>
                <input
                  type="range"
                  min={1}
                  max={10}
                  step={1}
                  value={segSimplifyEps}
                  onChange={(e) => setSegSimplifyEps(Number(e.target.value))}
                  disabled={!segEditMode}
                />
                <span style={{ fontSize: 12 }}>{segSimplifyEps}</span>
              </label>
              <div style={{ display: "flex", gap: 8 }}>
                <button
                  type="button"
                  onClick={handleSegUndo}
                  disabled={!segEditMode || segUndoStack.length === 0}
                  style={{ padding: "6px 10px", fontSize: 12, cursor: "pointer" }}
                >
                  Undo
                </button>
                <button
                  type="button"
                  onClick={handleSegReset}
                  disabled={!segEditMode || !selectedAnnotation.originalSegPolygon}
                  style={{ padding: "6px 10px", fontSize: 12, cursor: "pointer" }}
                >
                  Reset
                </button>
              </div>
            </div>
          )}

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
