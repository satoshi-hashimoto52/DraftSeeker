# Draft Seeker Annotator (MVP Step1)

図面アノテ支援WebアプリのMVP Step1です。図面を表示し、クリック点周辺ROIでテンプレ照合し、候補BBoxを返します。

## 構成

- `backend/` FastAPI
- `frontend/` React + Vite
- `data/` 画像とテンプレ

## ディレクトリ構造

```
DraftSeeker/
  .venv/
  backend/
  data/
    images/
    runs/
    templates_root/
  frontend/
```

## テンプレ配置

`templates_root` は2階層構成です。1階層目がプロジェクト名、2階層目がクラス名になります。

```
data/templates_root/
  project_a/
    図形1/
      0.jpeg
      1.jpeg
    図形2/
      0.jpeg
```

## 起動

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

## SAMセットアップメモ

SAMを使う場合は、`backend/requirements.txt` に加えて以下をインストールしてください。

```bash
pip install torch torchvision segment-anything
```

チェックポイントは `SAM_CHECKPOINT` 環境変数、または `backend/app/config.py` の
`SAM_CHECKPOINT` を指定してください。

## 動作確認

1. `http://127.0.0.1:8000/projects` でプロジェクト一覧を取得
2. フロントで画像アップロード
3. 画像クリックで候補TopKとBBoxを確認

## API概要

- `GET /projects` プロジェクト一覧
- `GET /templates` プロジェクト配下クラス一覧
- `POST /image/upload` 画像アップロード
- `POST /detect/point` クリック検出

詳細は `docs/api.md` を参照してください。
