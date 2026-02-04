import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

import { Annotation, Candidate } from "../api";

type Props = {
  imageUrl: string | null;
  candidates: Candidate[];
  selectedCandidateId: string | null;
  annotations: Annotation[];
  selectedAnnotationId: string | null;
  colorMap: Record<string, string>;
  showCandidates: boolean;
  showAnnotations: boolean;
  editablePolygon: { x: number; y: number }[] | null;
  editMode: boolean;
  showVertices: boolean;
  selectedVertexIndex: number | null;
  onSelectVertex: (index: number | null) => void;
  onUpdateEditablePolygon: (next: { x: number; y: number }[]) => void;
  onVertexDragStart: () => void;
  onClickPoint: (x: number, y: number) => void;
  onCreateManualBBox: (bbox: { x: number; y: number; w: number; h: number }) => void;
  onManualCreateStateChange: (active: boolean) => void;
  onResizeSelectedBBox: (bbox: { x: number; y: number; w: number; h: number }) => void;
};

export type ImageCanvasHandle = {
  panTo: (x: number, y: number) => void;
};

export default forwardRef<ImageCanvasHandle, Props>(function ImageCanvas(
  {
    imageUrl,
    candidates,
    selectedCandidateId,
    annotations,
    selectedAnnotationId,
    colorMap,
  showCandidates,
  showAnnotations,
  editablePolygon,
  editMode,
  showVertices,
  selectedVertexIndex,
  onSelectVertex,
  onUpdateEditablePolygon,
  onVertexDragStart,
  onClickPoint,
  onCreateManualBBox,
  onManualCreateStateChange,
  onResizeSelectedBBox,
}: Props,
  ref
) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const panRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const dragRef = useRef<{ active: boolean; vertexIndex: number | null }>({
    active: false,
    vertexIndex: null,
  });
  const manualDragRef = useRef<{
    active: boolean;
    start: { x: number; y: number } | null;
    current: { x: number; y: number } | null;
  }>({ active: false, start: null, current: null });
  const suppressNextClickRef = useRef<boolean>(false);
  const [cursorStyle, setCursorStyle] = useState<string>("default");
  const resizeDragRef = useRef<{
    active: boolean;
    handle: "tl" | "tr" | "bl" | "br" | null;
    origin: { x: number; y: number; w: number; h: number } | null;
  }>({ active: false, handle: null, origin: null });
  const moveDragRef = useRef<{
    active: boolean;
    origin: { x: number; y: number; w: number; h: number } | null;
    start: { x: number; y: number } | null;
  }>({ active: false, origin: null, start: null });
  const [manualPreview, setManualPreview] = useState<{
    start: { x: number; y: number };
    current: { x: number; y: number };
  } | null>(null);

  useEffect(() => {
    panRef.current = { x: 0, y: 0 };
    setPanOffset({ x: 0, y: 0 });
  }, [imageUrl]);

  useImperativeHandle(ref, () => ({
    panTo: (x: number, y: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const offsetX = Math.round(canvas.width / 2 - x);
      const offsetY = Math.round(canvas.height / 2 - y);
      panRef.current = { x: offsetX, y: offsetY };
      setPanOffset({ x: offsetX, y: offsetY });
    },
  }));

  useEffect(() => {
    if (!imageUrl || !canvasRef.current) return;

    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, panOffset.x, panOffset.y);

      const baseLine = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) * 0.003));
      const drawLabel = (x: number, y: number, text: string, color: string, alpha = 1) => {
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.font = "12px \"IBM Plex Sans\", system-ui, sans-serif";
        const paddingX = 4;
        const paddingY = 2;
        const metrics = ctx.measureText(text);
        const labelW = Math.ceil(metrics.width + paddingX * 2);
        const labelH = 16;
        const bx = Math.max(0, x + panOffset.x);
        const by = Math.max(0, y + panOffset.y - labelH - 2);
        ctx.fillStyle = "#ffffff";
        ctx.globalAlpha = alpha * 0.9;
        ctx.fillRect(bx, by, labelW, labelH);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(bx, by, labelW, labelH);
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.fillText(text, bx + paddingX, by + 12);
        ctx.restore();
      };

      const drawBox = (
        x: number,
        y: number,
        w: number,
        h: number,
        color: string,
        lineWidth: number,
        dashed: boolean,
        alpha: number
      ) => {
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash(dashed ? [6, 4] : []);
        ctx.strokeRect(x + panOffset.x, y + panOffset.y, w, h);
        ctx.restore();
      };

      const drawPolygon = (
        points: { x: number; y: number }[],
        color: string,
        lineWidth: number,
        alpha: number
      ) => {
        if (!points || points.length === 0) return;
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(points[0].x + panOffset.x, points[0].y + panOffset.y);
        for (let i = 1; i < points.length; i += 1) {
          ctx.lineTo(points[i].x + panOffset.x, points[i].y + panOffset.y);
        }
        ctx.closePath();
        ctx.stroke();
        ctx.restore();
      };

      const drawVertices = (
        points: { x: number; y: number }[],
        color: string,
        selectedIndex: number | null
      ) => {
        const radius = 5;
        points.forEach((pt, idx) => {
          ctx.save();
          ctx.fillStyle = idx === selectedIndex ? "#ffffff" : color;
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(pt.x + panOffset.x, pt.y + panOffset.y, radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.restore();
        });
      };

      if (showCandidates) {
        candidates.forEach((c) => {
          const isSelected = c.id === selectedCandidateId;
          const color = colorMap[c.class_name] || "#ff2b2b";
          drawBox(
          c.bbox.x,
          c.bbox.y,
          c.bbox.w,
          c.bbox.h,
          color,
          isSelected ? baseLine * 1.8 : baseLine,
          !isSelected,
          isSelected ? 0.9 : 0.35
        );
        if (c.segPolygon) {
          drawPolygon(
            c.segPolygon,
            color,
            isSelected ? baseLine * 2.2 : baseLine * 1.6,
            isSelected ? 0.95 : 0.6
          );
        }
          drawLabel(c.bbox.x, c.bbox.y, c.class_name, color, isSelected ? 0.9 : 0.45);
        });
      }

      const selected = selectedCandidateId
        ? candidates.find((c) => c.id === selectedCandidateId) || null
        : null;
      if (selected && selected.source === "manual") {
        const size = Math.max(4, Math.round(baseLine * 2.0));
        const color = "#ff9800";
        const points = [
          { x: selected.bbox.x, y: selected.bbox.y },
          { x: selected.bbox.x + selected.bbox.w, y: selected.bbox.y },
          { x: selected.bbox.x, y: selected.bbox.y + selected.bbox.h },
          { x: selected.bbox.x + selected.bbox.w, y: selected.bbox.y + selected.bbox.h },
        ];
        points.forEach((pt) => {
          ctx.save();
          ctx.fillStyle = "#ffffff";
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.fillRect(pt.x + panOffset.x - size / 2, pt.y + panOffset.y - size / 2, size, size);
          ctx.strokeRect(
            pt.x + panOffset.x - size / 2,
            pt.y + panOffset.y - size / 2,
            size,
            size
          );
          ctx.restore();
        });
      }

      if (showAnnotations) {
        annotations.forEach((a) => {
          const isSelected = a.id === selectedAnnotationId;
          const color = colorMap[a.class_name] || "#ff2b2b";
          const lineWidth = isSelected ? baseLine * 2.4 : baseLine * 1.8;
          drawBox(a.bbox.x, a.bbox.y, a.bbox.w, a.bbox.h, color, lineWidth, false, 1);
          if (a.segPolygon) {
            drawPolygon(
              a.segPolygon,
              color,
              isSelected ? baseLine * 2.6 : baseLine * 2.0,
              1
            );
          }
          if (isSelected) {
            const size = Math.max(4, Math.round(baseLine * 2.2));
          ctx.save();
          ctx.fillStyle = color;
          ctx.fillRect(a.bbox.x + panOffset.x - size, a.bbox.y + panOffset.y - size, size, size);
          ctx.fillRect(
            a.bbox.x + panOffset.x + a.bbox.w,
            a.bbox.y + panOffset.y - size,
            size,
            size
          );
          ctx.fillRect(
            a.bbox.x + panOffset.x - size,
            a.bbox.y + panOffset.y + a.bbox.h,
            size,
            size
          );
          ctx.fillRect(
            a.bbox.x + panOffset.x + a.bbox.w,
            a.bbox.y + panOffset.y + a.bbox.h,
            size,
            size
          );
          ctx.restore();
        }
          drawLabel(a.bbox.x, a.bbox.y, a.class_name, color, 1);
        });
      }

      if (editMode && editablePolygon && editablePolygon.length > 0) {
        const color = "#1a73e8";
        drawPolygon(editablePolygon, color, baseLine * 2.4, 1);
        if (showVertices) {
          drawVertices(editablePolygon, color, selectedVertexIndex);
        }
      }

      if (manualPreview) {
        const x0 = manualPreview.start.x;
        const y0 = manualPreview.start.y;
        const x1 = manualPreview.current.x;
        const y1 = manualPreview.current.y;
        const left = Math.min(x0, x1);
        const top = Math.min(y0, y1);
        const w = Math.abs(x1 - x0);
        const h = Math.abs(y1 - y0);
        ctx.save();
        ctx.strokeStyle = "#ff9800";
        ctx.fillStyle = "rgba(255, 152, 0, 0.15)";
        ctx.lineWidth = baseLine * 1.6;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(left + panOffset.x, top + panOffset.y, w, h);
        ctx.fillRect(left + panOffset.x, top + panOffset.y, w, h);
        ctx.restore();
      }
    };
    img.src = imageUrl;
  }, [
    imageUrl,
    candidates,
    selectedCandidateId,
    annotations,
    selectedAnnotationId,
    colorMap,
    showCandidates,
    showAnnotations,
    panOffset,
    editablePolygon,
    editMode,
    showVertices,
    selectedVertexIndex,
    manualPreview,
  ]);

  const getImageCoords = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const rawX = (event.clientX - rect.left) * scaleX;
    const rawY = (event.clientY - rect.top) * scaleY;
    const x = Math.round(rawX - panRef.current.x);
    const y = Math.round(rawY - panRef.current.y);
    return { x, y };
  };

  const findVertexIndex = (x: number, y: number) => {
    if (!editablePolygon || editablePolygon.length === 0) return null;
    const radius = 8;
    for (let i = 0; i < editablePolygon.length; i += 1) {
      const pt = editablePolygon[i];
      const dx = pt.x - x;
      const dy = pt.y - y;
      if (dx * dx + dy * dy <= radius * radius) {
        return i;
      }
    }
    return null;
  };

  const getSelectedCandidate = () => {
    if (!selectedCandidateId) return null;
    return candidates.find((c) => c.id === selectedCandidateId) || null;
  };

  const isManualClassMissing = () => {
    const selected = getSelectedCandidate();
    return !!selected && selected.source === "manual" && !selected.class_name;
  };

  const findResizeHandle = (x: number, y: number) => {
    const selected = getSelectedCandidate();
    if (!selected || selected.source !== "manual") return null;
    const canvas = canvasRef.current;
    const size = canvas ? Math.max(10, Math.round(Math.min(canvas.width, canvas.height) * 0.01)) : 12;
    const handles = [
      { key: "tl" as const, x: selected.bbox.x, y: selected.bbox.y },
      { key: "tr" as const, x: selected.bbox.x + selected.bbox.w, y: selected.bbox.y },
      { key: "bl" as const, x: selected.bbox.x, y: selected.bbox.y + selected.bbox.h },
      { key: "br" as const, x: selected.bbox.x + selected.bbox.w, y: selected.bbox.y + selected.bbox.h },
    ];
    for (const h of handles) {
      const dx = h.x - x;
      const dy = h.y - y;
      if (dx * dx + dy * dy <= size * size) return h.key;
    }
    return null;
  };

  const findMoveEdge = (x: number, y: number) => {
    const selected = getSelectedCandidate();
    if (!selected || selected.source !== "manual") return false;
    if (!isManualClassMissing()) return false;
    const canvas = canvasRef.current;
    const tolerance = canvas
      ? Math.max(6, Math.round(Math.min(canvas.width, canvas.height) * 0.006))
      : 8;
    const left = selected.bbox.x;
    const right = selected.bbox.x + selected.bbox.w;
    const top = selected.bbox.y;
    const bottom = selected.bbox.y + selected.bbox.h;
    const nearLeft = Math.abs(x - left) <= tolerance && y >= top && y <= bottom;
    const nearRight = Math.abs(x - right) <= tolerance && y >= top && y <= bottom;
    const nearTop = Math.abs(y - top) <= tolerance && x >= left && x <= right;
    const nearBottom = Math.abs(y - bottom) <= tolerance && x >= left && x <= right;
    return nearLeft || nearRight || nearTop || nearBottom;
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!editMode) {
      const coords = getImageCoords(event);
      if (coords) {
        const handle = findResizeHandle(coords.x, coords.y);
        if (handle) {
          const selected = getSelectedCandidate();
          if (selected) {
            resizeDragRef.current = {
              active: true,
              handle,
              origin: { ...selected.bbox },
            };
            return;
          }
        }
        if (findMoveEdge(coords.x, coords.y)) {
          const selected = getSelectedCandidate();
          if (selected) {
            moveDragRef.current = {
              active: true,
              origin: { ...selected.bbox },
              start: coords,
            };
            suppressNextClickRef.current = true;
            return;
          }
        }
      }
    }
    if (!editMode || !showVertices || !editablePolygon) return;
    const coords = getImageCoords(event);
    if (!coords) return;
    const index = findVertexIndex(coords.x, coords.y);
    if (index === null) return;
    dragRef.current = { active: true, vertexIndex: index };
    onSelectVertex(index);
    onVertexDragStart();
  };

  const updateCursorByHandle = (x: number, y: number) => {
    if (findMoveEdge(x, y)) {
      setCursorStyle("move");
      return;
    }
    const handle = findResizeHandle(x, y);
    if (handle === "tl" || handle === "br") {
      setCursorStyle("nwse-resize");
      return;
    }
    if (handle === "tr" || handle === "bl") {
      setCursorStyle("nesw-resize");
      return;
    }
    if (manualDragRef.current.active) {
      setCursorStyle("crosshair");
      return;
    }
    setCursorStyle(imageUrl ? "crosshair" : "default");
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (moveDragRef.current.active && moveDragRef.current.origin && moveDragRef.current.start) {
      const coords = getImageCoords(event);
      if (!coords) return;
      const dx = coords.x - moveDragRef.current.start.x;
      const dy = coords.y - moveDragRef.current.start.y;
      const next = {
        x: Math.round(moveDragRef.current.origin.x + dx),
        y: Math.round(moveDragRef.current.origin.y + dy),
        w: moveDragRef.current.origin.w,
        h: moveDragRef.current.origin.h,
      };
      onResizeSelectedBBox(next);
      return;
    }
    if (resizeDragRef.current.active && resizeDragRef.current.origin) {
      const coords = getImageCoords(event);
      if (!coords) return;
      const { origin, handle } = resizeDragRef.current;
      let x0 = origin.x;
      let y0 = origin.y;
      let x1 = origin.x + origin.w;
      let y1 = origin.y + origin.h;
      if (handle === "tl" || handle === "bl") x0 = coords.x;
      if (handle === "tr" || handle === "br") x1 = coords.x;
      if (handle === "tl" || handle === "tr") y0 = coords.y;
      if (handle === "bl" || handle === "br") y1 = coords.y;
      const left = Math.min(x0, x1);
      const top = Math.min(y0, y1);
      const w = Math.max(2, Math.abs(x1 - x0));
      const h = Math.max(2, Math.abs(y1 - y0));
      onResizeSelectedBBox({ x: Math.round(left), y: Math.round(top), w, h });
      return;
    }
    if (!editMode) {
      const coords = getImageCoords(event);
      if (coords) updateCursorByHandle(coords.x, coords.y);
    }
    if (manualDragRef.current.active) {
      const coords = getImageCoords(event);
      if (!coords || !manualDragRef.current.start) return;
      manualDragRef.current.current = coords;
      setManualPreview({ start: manualDragRef.current.start, current: coords });
      return;
    }
    if (!dragRef.current.active || dragRef.current.vertexIndex === null) return;
    const coords = getImageCoords(event);
    if (!coords || !editablePolygon) return;
    const next = editablePolygon.map((pt, idx) =>
      idx === dragRef.current.vertexIndex ? { x: coords.x, y: coords.y } : pt
    );
    onUpdateEditablePolygon(next);
  };

  const handleMouseUp = () => {
    if (moveDragRef.current.active) {
      moveDragRef.current = { active: false, origin: null, start: null };
      return;
    }
    if (resizeDragRef.current.active) {
      resizeDragRef.current = { active: false, handle: null, origin: null };
      suppressNextClickRef.current = true;
      return;
    }
    if (manualDragRef.current.active && manualDragRef.current.start && manualDragRef.current.current) {
      const start = manualDragRef.current.start;
      const current = manualDragRef.current.current;
      const left = Math.min(start.x, current.x);
      const top = Math.min(start.y, current.y);
      const w = Math.abs(current.x - start.x);
      const h = Math.abs(current.y - start.y);
      manualDragRef.current = { active: false, start: null, current: null };
      setManualPreview(null);
      if (w >= 2 && h >= 2) {
        onCreateManualBBox({ x: left, y: top, w, h });
        suppressNextClickRef.current = true;
      }
      onManualCreateStateChange(false);
      return;
    }
    if (dragRef.current.active) {
      dragRef.current = { active: false, vertexIndex: null };
    }
  };

  const handleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (suppressNextClickRef.current) {
      suppressNextClickRef.current = false;
      return;
    }
    if (manualDragRef.current.active) return;
    if (editMode) return;
    const coords = getImageCoords(event);
    if (!coords) return;
    onClickPoint(coords.x, coords.y);
  };

  const handleMouseDownCapture = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (editMode) return;
    if (!event.shiftKey) return;
    const coords = getImageCoords(event);
    if (!coords) return;
    manualDragRef.current = { active: true, start: coords, current: coords };
    setManualPreview({ start: coords, current: coords });
    onManualCreateStateChange(true);
  };

  return (
    <div style={{ width: "100%" }}>
      <canvas
        ref={canvasRef}
        onClick={imageUrl ? handleClick : undefined}
        onMouseDown={imageUrl ? handleMouseDown : undefined}
        onMouseDownCapture={imageUrl ? handleMouseDownCapture : undefined}
        onMouseMove={imageUrl ? handleMouseMove : undefined}
        onMouseUp={imageUrl ? handleMouseUp : undefined}
        onMouseLeave={(event) => {
          if (imageUrl) handleMouseUp();
          setCursorStyle(imageUrl ? "crosshair" : "default");
        }}
        style={{
          width: "100%",
          border: "1px solid #ddd",
          background: "#fafafa",
          cursor: cursorStyle,
        }}
      />
      {!imageUrl && (
        <div style={{ padding: "12px 0", color: "#666" }}>画像をアップロードしてください。</div>
      )}
    </div>
  );
});
