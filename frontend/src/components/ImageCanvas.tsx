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
  onClickPoint: (x: number, y: number) => void;
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
    onClickPoint,
  }: Props,
  ref
) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const panRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

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
  ]);

  const handleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const rawX = (event.clientX - rect.left) * scaleX;
    const rawY = (event.clientY - rect.top) * scaleY;
    const x = Math.round(rawX - panRef.current.x);
    const y = Math.round(rawY - panRef.current.y);
    onClickPoint(x, y);
  };

  return (
    <div style={{ width: "100%" }}>
      <canvas
        ref={canvasRef}
        onClick={imageUrl ? handleClick : undefined}
        style={{
          width: "100%",
          border: "1px solid #ddd",
          background: "#fafafa",
          cursor: imageUrl ? "crosshair" : "default",
        }}
      />
      {!imageUrl && (
        <div style={{ padding: "12px 0", color: "#666" }}>画像をアップロードしてください。</div>
      )}
    </div>
  );
});
