export function isHexColor(value: string): boolean {
  return /^#[0-9a-fA-F]{6}$/.test(value.trim());
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r1 = 0;
  let g1 = 0;
  let b1 = 0;
  if (hp >= 0 && hp < 1) {
    r1 = c;
    g1 = x;
  } else if (hp >= 1 && hp < 2) {
    r1 = x;
    g1 = c;
  } else if (hp >= 2 && hp < 3) {
    g1 = c;
    b1 = x;
  } else if (hp >= 3 && hp < 4) {
    g1 = x;
    b1 = c;
  } else if (hp >= 4 && hp < 5) {
    r1 = x;
    b1 = c;
  } else if (hp >= 5 && hp < 6) {
    r1 = c;
    b1 = x;
  }
  const m = l - c / 2;
  const r = Math.round((r1 + m) * 255);
  const g = Math.round((g1 + m) * 255);
  const b = Math.round((b1 + m) * 255);
  return [r, g, b];
}

export function hslToHex(hslString: string): string {
  const match = hslString
    .trim()
    .match(/^hsl\(\s*([0-9.]+)\s*,\s*([0-9.]+)%\s*,\s*([0-9.]+)%\s*\)$/i);
  if (!match) return "#000000";
  const h = Number(match[1]) % 360;
  const s = Math.min(100, Math.max(0, Number(match[2]))) / 100;
  const l = Math.min(100, Math.max(0, Number(match[3]))) / 100;
  const [r, g, b] = hslToRgb(h, s, l);
  return (
    "#" +
    r.toString(16).padStart(2, "0") +
    g.toString(16).padStart(2, "0") +
    b.toString(16).padStart(2, "0")
  );
}

export function normalizeToHex(color: string): string {
  if (!color) return "#000000";
  if (isHexColor(color)) return color;
  if (color.trim().toLowerCase().startsWith("hsl(")) {
    return hslToHex(color);
  }
  return "#000000";
}
