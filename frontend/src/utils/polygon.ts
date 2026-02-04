type Point = { x: number; y: number };

function distanceToLine(p: Point, a: Point, b: Point): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (dx === 0 && dy === 0) {
    const px = p.x - a.x;
    const py = p.y - a.y;
    return Math.sqrt(px * px + py * py);
  }
  const t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy);
  const clamped = Math.max(0, Math.min(1, t));
  const projX = a.x + clamped * dx;
  const projY = a.y + clamped * dy;
  const vx = p.x - projX;
  const vy = p.y - projY;
  return Math.sqrt(vx * vx + vy * vy);
}

function rdp(points: Point[], epsilon: number): Point[] {
  if (points.length <= 2) return points.slice();
  let maxDist = -1;
  let index = -1;
  const start = points[0];
  const end = points[points.length - 1];
  for (let i = 1; i < points.length - 1; i += 1) {
    const dist = distanceToLine(points[i], start, end);
    if (dist > maxDist) {
      maxDist = dist;
      index = i;
    }
  }
  if (maxDist > epsilon && index !== -1) {
    const left = rdp(points.slice(0, index + 1), epsilon);
    const right = rdp(points.slice(index), epsilon);
    return left.slice(0, -1).concat(right);
  }
  return [start, end];
}

export function simplifyPolygon(points: Point[], epsilon: number): Point[] {
  if (!points || points.length < 3) return points.slice();
  const eps = Math.max(0, epsilon);
  return rdp(points, eps);
}

export function clampToImage(points: Point[], width: number, height: number): Point[] {
  return points.map((p) => ({
    x: Math.min(width - 1, Math.max(0, Math.round(p.x))),
    y: Math.min(height - 1, Math.max(0, Math.round(p.y))),
  }));
}
