# Minimax-H3-lipsync-mv

[English](README.en.md) | 日本語

1枚のキャラクター参照画像と楽曲、またはテキスト原稿から、MiniMax-H3 Ref2VAを使って
リップシンク付きMV／読み上げ動画を生成するローカルWebアプリケーションです。

TXTを入力した場合はAivisSpeech Engineで1文ずつ音声合成し、結合した読み上げ音声を
Vocal Lockへ直接渡します。映像シナリオは入力原稿の内容を元に作成します。

## 固定仕様

- MiniMax-H3 Ref2VA only
- 動画サイズは投稿先と画質で選択 / 24fps（下の「動画サイズ」参照）
- TURBO ON
- Vocal Lock ON
- H3の構成はGPUのVRAMから自動選択（下の「GPUと速度」参照）
- 同一参照画像のprefix cache ON
- 31Bマルチモーダル解析で参照画像の顔・髪・服装・スタイルを固定
- 歌唱区間では正面または3/4向きのmedium close-upと可視口元を強制
- H3公式Ref2VAの6セクション記法（`subject_definitions` → `summary` →
  `retention_analysis` → `detailed_description` → `overall_soundscape` →
  `non_diegetic_music`）で各クリップのプロンプトを生成
- 参照画像はキーフレームではなく`<Subject 1>`の人物定義として使用し、Vocal Lock用の
  分離ボーカルは`<Audio 1>: fully_copy`として`<Subject 1> (S1)`の歌唱へ関連付け
- 歌詞を推測して`<d>`へ書かず、`<Audio 1>`そのものへ口を合わせる
- E4Bなどの音声理解LLMは使用しない
- 1シーン5～10秒。均等割りではなく、音響特徴から得た無音、ボーカル音量谷、
  ブレス候補、セクション、ビート、小節を評価して曲全体を動的計画法で分割
  （[ボーカルのシーン分割仕様](docs/vocal-scene-segmentation.md)）
- 最終音声はH3生成音声ではなく入力原曲

## 動画サイズ

「投稿先」と「画質」を選ぶと、縦横比に合ったサイズで生成します。

| 投稿先 | 高画質 | 標準 | 高速 |
|---|---|---|---|
| YouTube・X（横長 16:9） | 1344×768 | 1024×576 | 768×448 |
| TikTok・YouTubeショート・Instagramリール（縦長 9:16） | 768×1344 | 576×1024 | 448×768 |
| Instagram フィード（縦 4:5） | 768×960 | 640×800 | 512×640 |
| 正方形（Instagram・X 1:1） | 960×960 | 768×768 | 576×576 |
| 従来サイズ（4:3 / 3:4） | 1024×768 | 768×576 | 512×384 |

縦長と正方形では、シーンのプロンプトに「被写体を中央に置く」構図の指示を自動で加えます
（横長前提の構図のまま縦長で生成すると、被写体が画面の端に押し出されることがあるため）。

## GPUと速度

`H3_PRESET=auto`（既定）では、計算に使うGPUのVRAMからH3の構成を選びます。

| GPUのVRAM | 使う構成 | 10秒シーンで収まるサイズの目安 |
|---|---|---|
| 90GB以上 | `96gb-int8` | 全サイズ |
| 40GB以上 | `ref2va-only-32gb`（W4A8量子化） | 全サイズ |
| 28GB以上（RTX 5090など） | `ref2va-only-32gb` | 標準まで（約59万画素） |
| それ未満（RTX 4090など） | `ref2va-only-24gb` | 高速まで（約34万画素） |

選んだサイズがGPUに対して重すぎる場合は、画質欄の下に警告を表示します。

連続生成を速くするために、次の2つを行います。

- **シーンの先行投入**: 次のシーンを先に投入し、前のシーンのデコードと次のシーンの生成を
  並行させる
- **2枚目のGPUでのデコード**: 2枚目のGPUがあれば、デコードをそちらで行う
  （`H3_DECODE_GPU=auto`）

RTX PRO 5000（48GB）+ RTX PRO 4000（24GB）での実測は、1024×768・約10秒のシーンで
**シーン間隔89秒**です（シーンを1本ずつ処理した場合は131秒）。

参照画像は短辺1024pxに縮小してH3へ渡します（`H3_REF_SHORT_EDGE=1024`）。2048pxと比べて
生成が約15%速く、同じseedで比べた画質は同等でした。顔のツヤ（テカリ）は量子化やこの設定
ではなく参照画像の肌の質感で決まるため、マットで均一な照明の画像を推奨します。

| 環境変数 | 既定 | 内容 |
|---|---|---|
| `H3_PRESET` | `auto` | H3の構成。`auto`以外を指定するとそのまま使う |
| `H3_GPUS` | `0` | 計算に使うGPU |
| `H3_DECODE_GPU` | `auto` | デコード用の2枚目のGPU（`auto` / `off` / GPU番号） |
| `H3_REF_SHORT_EDGE` | `1024` | 参照画像の短辺。`0`でバックエンド既定（2048） |

## サンプル生成動画

[サンプル動画を再生・ダウンロード（MP4、約39MB）](examples/minimax-h3-lipsync-mv-sample.mp4)

- 再生時間: 90秒
- 映像: H.264 / 1024×768 / 24fps
- 音声: AAC

## セットアップ

Python 3.12、ffmpeg、ffprobe、稼働中のdiffusers-movie-server gateway、
DemucsサービスまたはDemucs入りPython環境が必要です。読み上げ動画には
AivisSpeech Engine（既定 `http://127.0.0.1:10101`）も必要です。

```bash
cd Minimax-H3-lipsync-mv
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
# .envのLLM接続先、モデルID、必要に応じてDemucs用Pythonを設定
./run.sh
```

ブラウザで `http://127.0.0.1:8780` を開きます。

## 外部サービス

| 用途 | 既定値 |
|---|---|
| シナリオ・H3プロンプト | `http://127.0.0.1:64650/v1` |
| Demucs | `http://127.0.0.1:8889` |
| H3 gateway | `http://127.0.0.1:8630` |
| AivisSpeech Engine | `http://127.0.0.1:10101`（まお／ノーマル） |

接続先やモデルIDは `.env` で変更できます。APIキーはログやジョブJSONへ保存しません。

## データ構造

```text
data/jobs/<job-id>/
├── input/
├── analysis/
│   ├── boundary_analysis.json
│   ├── scene_plan.json
│   └── scenario.json
├── scenes/
├── job.json
└── output.mp4
```

解析JSONには、選択された境界時刻、候補種別、スコア、歌唱比率を保存します。
生成に失敗して再実行した場合、入力・プロンプト・H3設定のハッシュが一致する
完成済みシーンだけを再利用します。人物固定やリップシンク規則が変更された場合は
古いシーンを誤って再利用せず、再生成します。

## API

- `POST /api/jobs` — character/song/conceptをmultipart送信
- `GET /api/jobs/{id}` — 状態取得
- `GET /api/jobs/{id}/events` — SSE進捗
- `POST /api/jobs/{id}/cancel` — キャンセル要求
- `GET /api/jobs/{id}/output` — インライン再生
- `GET /api/jobs/{id}/download` — ダウンロード
- `GET /api/health` — LLM接続確認
- `GET /api/h3/capacity` — このGPUで収まる画素数の目安

H3 gatewayにはジョブ途中キャンセルAPIがないため、キャンセル要求はシーンの区切りで
確定します。すでに投入済みのシーンはバックエンドで最後まで生成されますが、結果は使われません。

## MiniMax-H3参照元

- 公式ソースコード・ドキュメント: [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)
- 公式モデルカード・モデル重み: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- 公式プロンプト記述ガイド: [MiniMax-H3 Prompt Writing Skills](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills)

本リポジトリにMiniMax-H3のモデル重みは含まれません。利用時は公式モデルカード、
ライセンス、利用条件を確認してください。

## ライセンス

本リポジトリのソースコードは[Apache License 2.0](LICENSE)で提供します。

このライセンスは本リポジトリのソースコードにのみ適用されます。MiniMax-H3のモデル重みは
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
に従い、その他のモデル、LoRA、ライブラリなどの依存コンポーネントには各配布元の
ライセンスが適用されます。

## テスト

```bash
.venv/bin/pytest -q
.venv/bin/ruff check app tests
```
