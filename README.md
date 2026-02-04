# Draft Seeker Annotator

CAD 図面画像向けのアノテーション支援 Web アプリです。  
OpenCV のテンプレートマッチと SAM（必要時のみ）＋フォールバック輪郭抽出を組み合わせ、  
候補生成から YOLO / YOLO-seg 出力までを一通りサポートします。

## できること

- クリック点ベースのテンプレ照合（ROI）
- マルチスケール対応
- クラス別 NMS / TopK 候補
- 候補の確定 / 破棄などの手動修正 UI
- SAM によるセグメンテーション（オンデマンド）
- SAM不使用時のフォールバック（輪郭抽出）
- YOLO / YOLO-seg エクスポート

## ディレクトリ構成

```
draft_seeker/
  backend/   FastAPI (API・推論)
  frontend/  React + Vite (UI)
  data/      画像・テンプレ・出力
  docs/      ドキュメント
```

### テンプレ配置

`data/templates_root` は2階層構成です。1階層目がプロジェクト名、2階層目がクラス名です。

```
data/templates_root/
  project_a/
    roof_fan/
      0.jpeg
      1.jpeg
    door_w/
      0.jpeg
```

## 動作環境

- Backend: FastAPI (Python)
- Frontend: React + Vite
- SAM: CPU / Apple MPS（macOS M1 想定）

## 起動手順

バックエンド:

```bash
cd /Users/hashimoto/vscode/_project/draft_seeker/backend
/Users/hashimoto/vscode/_project/draft_seeker/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

フロントエンド:

```bash
cd /Users/hashimoto/vscode/_project/draft_seeker/frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

## SAM セットアップ

```bash
pip install torch torchvision segment-anything
```

チェックポイントは `SAM_CHECKPOINT` 環境変数、または `backend/app/config.py` の
`SAM_CHECKPOINT` で指定します。

## エクスポート仕様（YOLO / YOLO-seg）

- 1画像 = 1 txt
- `segPolygon` があれば YOLO-seg（`class_id x1 y1 x2 y2 ...`）
- `segPolygon` が無ければ YOLO bbox（`class_id cx cy w h`）
- 全座標は 0〜1 正規化
- class_id は `data/runs/<project>/classes.json` に保存・固定

## 注意事項

- SAM は常時 ON にせず、必要時のみ実行する設計です
- 図面特有の注記・寸法文字は対象外（テンプレから外す前提）

詳細な API は `docs/api.md` を参照してください。
