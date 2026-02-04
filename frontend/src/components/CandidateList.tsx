import React from "react";

import { Candidate } from "../api";

type Props = {
  candidates: Candidate[];
  selectedCandidateId: string | null;
  onSelect: (id: string) => void;
  colorMap: Record<string, string>;
};

export default function CandidateList({
  candidates,
  selectedCandidateId,
  onSelect,
  colorMap,
}: Props) {
  return (
    <div>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>候補 (TopK)</div>
      {candidates.length === 0 && (
        <div style={{ color: "#666" }}>候補はまだありません。</div>
      )}
      {candidates.map((c, idx) => (
        <div
          key={c.id}
          style={{
            padding: "8px 10px",
            marginBottom: 8,
            border: "1px solid #e3e3e3",
            borderRadius: 6,
            background: selectedCandidateId === c.id ? "#fff3f3" : "#fff",
            cursor: "pointer",
          }}
          onClick={() => onSelect(c.id)}
        >
          <div style={{ fontWeight: 600, display: "flex", alignItems: "center", gap: 8 }}>
            <span
              style={{
                display: "inline-block",
                width: 10,
                height: 10,
                borderRadius: 2,
                background: colorMap[c.class_name] || "#999",
              }}
            />
            {idx + 1}. {c.class_name}
          </div>
          <div style={{ fontSize: 12, color: "#444" }}>score: {c.score.toFixed(4)}</div>
          <div style={{ fontSize: 12, color: "#666" }}>template: {c.template}</div>
          <div style={{ fontSize: 12, color: "#666" }}>scale: {c.scale.toFixed(3)}</div>
          <div style={{ fontSize: 12, color: "#666" }}>
            bbox: ({c.bbox.x}, {c.bbox.y}, {c.bbox.w}, {c.bbox.h})
          </div>
        </div>
      ))}
    </div>
  );
}
