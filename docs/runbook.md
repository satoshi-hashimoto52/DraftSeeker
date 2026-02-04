# Runbook

運用・保守向けの手順と判断基準をまとめています。

## 依存インストール

バックエンド:
```bash
cd /Users/hashimoto/vscode/_project/draft_seeker
/Users/hashimoto/vscode/_project/draft_seeker/.venv/bin/pip install -r backend/requirements.txt
```

フロントエンド:
```bash
cd /Users/hashimoto/vscode/_project/draft_seeker/frontend
npm install
```

## 起動

バックエンド:
```bash
source .venv/bin/activate
cd /backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

フロントエンド:
```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

## classes.json と class_id の固定ルール

- `/export/yolo` 実行時に `data/runs/<project>/classes.json` を生成
- class_id は **プロジェクト配下クラス名のソート順**
- 同一プロジェクト内でテンプレ構成が変わる場合は
  classes.json を更新する前提で学習側も同期する

## テンプレ追加時の注意点

- クラス名の揺れがあると class_id が変わるため名称を固定する
- 似すぎたテンプレが多いと誤検出が増える
- まず少数・代表的なテンプレで精度を見て追加する

## SAM セグメンテーション運用ルール

なぜ:
- 常時 SAM を回すと速度低下・安定性低下に直結する

推奨:
- テンプレ数を減らして候補精度を高める
- SAM は ROI 限定で使う（候補1件に対して実行）

## よくあるトラブルと対処

- CORS エラー:
  - `backend/app/main.py` の `allow_origins` に
    `http://127.0.0.1:5173` と `http://localhost:5173` を入れる
- Failed to fetch:
  - backend が起動しているか
  - `API_BASE` が正しいか
  - ポートが競合していないか
- RUNS_DIR 未定義:
  - `backend/app/storage.py` から `RUNS_DIR` を import する実装に統一
- color picker 警告（hsl → hex）:
  - `frontend/src/utils/color.ts` の `normalizeToHex` を通す
  - `<input type="color">` の `value` は必ず `#rrggbb` にする
