# Minimax-H3-lipsync-mv

1枚のキャラクター参照画像と1本の楽曲から、MiniMax-H3 Ref2VAだけを使って
リップシンク付きMVを生成するローカルWebアプリケーションです。

## 固定仕様

- MiniMax-H3 Ref2VA only
- 1024×768 / 24fps
- TURBO ON
- Vocal Lock ON
- `96gb-int8` / GPU 0
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

## サンプル生成動画

[サンプル動画を再生・ダウンロード（MP4、約39MB）](examples/minimax-h3-lipsync-mv-sample.mp4)

- 再生時間: 90秒
- 映像: H.264 / 1024×768 / 24fps
- 音声: AAC

## セットアップ

Python 3.12、ffmpeg、ffprobe、稼働中のdiffusers-movie-server gateway、
DemucsサービスまたはDemucs入りPython環境が必要です。

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

H3 gatewayにはジョブ途中キャンセルAPIがないため、キャンセル要求は現在のH3シーン終了後、
次のシーンへ進む前に確定します。

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
