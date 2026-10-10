# Minimax-H3-lipsync-mv

English | [日本語](README.md)

A local web application that uses MiniMax-H3 Ref2VA to generate lip-synced music videos
and narration videos from a single character reference image plus either a song or a
text script.

When you provide a TXT script, AivisSpeech Engine synthesizes it one sentence at a time,
and the joined narration audio is passed straight to Vocal Lock. The visual scenario is
written from the content of the script.

## Fixed specifications

- MiniMax-H3 Ref2VA only
- Video size chosen by destination platform and quality / 24 fps (see "Video size" below)
- TURBO on
- Vocal Lock on
- H3 configuration selected automatically from GPU VRAM (see "GPU and speed" below)
- Prefix cache on for the same reference image
- A 31B multimodal analysis pins the reference image's face, hair, outfit and style
- Singing intervals force a front or three-quarter medium close-up with a visible mouth
- Each clip's prompt follows the official H3 Ref2VA six-section format
  (`subject_definitions` → `summary` → `retention_analysis` → `detailed_description` →
  `overall_soundscape` → `non_diegetic_music`)
- The reference image is used as the `<Subject 1>` character definition, not as a
  keyframe; the separated vocal for Vocal Lock is linked to `<Subject 1> (S1)`'s
  performance as `<Audio 1>: fully_copy`
- Lyrics are never guessed into `<d>`; the mouth follows `<Audio 1>` itself
- No audio-understanding LLM (such as E4B) is used
- Each scene is 5–10 seconds. Instead of equal splits, the whole song is segmented with
  dynamic programming over silences, vocal-level valleys, breath candidates, sections,
  beats and bars ([vocal scene segmentation spec, Japanese](docs/vocal-scene-segmentation.md))
- The final audio is the original input track, not H3's generated audio

## Video size

Pick a destination and a quality, and the video is generated at a size that matches the
platform's aspect ratio.

| Destination | High quality | Standard | Fast |
|---|---|---|---|
| YouTube / X (landscape 16:9) | 1344×768 | 1024×576 | 768×448 |
| TikTok / YouTube Shorts / Instagram Reels (vertical 9:16) | 768×1344 | 576×1024 | 448×768 |
| Instagram feed (portrait 4:5) | 768×960 | 640×800 | 512×640 |
| Square (Instagram / X 1:1) | 960×960 | 768×768 | 576×576 |
| Classic (4:3 / 3:4) | 1024×768 | 768×576 | 512×384 |

Official standard sizes are also available (the quality selector is disabled for them).

| Official size | Sizes | Source |
|---|---|---|
| H3 official | 1344×768 / 768×1344 / 768×768 | Short edge 768, up to 768×1344 pixels (defaults of the diffusers H3 implementation) |
| LTX-2.5 official | 960×544 / 544×960 | Stage-1 size in the official model card |

LTX-2.5's final official output is 1920×1088, twice the stage-1 size, but generating that
size directly with H3 does not fit a 48 GB-class GPU, so the stage-1 size 960×544 is
offered instead.

For vertical and square output, a composition instruction that centers the subject is
added to each scene prompt automatically. Without it, a scene composed for landscape can
push the subject to the edge of a vertical frame.

## GPU and speed

With `H3_PRESET=auto` (the default), the H3 configuration is chosen from the VRAM of the
compute GPU.

| GPU VRAM | Configuration | Sizes that fit a 10-second scene |
|---|---|---|
| 90 GB or more | `96gb-int8` | All sizes |
| 40 GB or more | `ref2va-only-32gb` (W4A8 quantized) | All sizes |
| 28 GB or more (e.g. RTX 5090) | `ref2va-only-32gb` | Up to Standard (about 0.59 MP) |
| Less (e.g. RTX 4090) | `ref2va-only-24gb` | Up to Fast (about 0.34 MP) |

If the selected size is too heavy for the GPU, a warning appears under the quality
selector.

Two things speed up continuous generation:

- **Submitting scenes ahead**: the next scene is submitted before the previous one
  finishes, so the previous scene's decode overlaps the next scene's generation
- **Decoding on a second GPU**: if a second GPU is present, decoding runs there
  (`H3_DECODE_GPU=auto`)

**A 12 GB-class card is recommended as the second GPU.** Decoding a 1024×768,
10-second scene used 7.9 GB on the second GPU. An 8 GB-class card should fit when you
stick to the Fast sizes, but this has not been tested on real hardware. Decode memory for
the High quality sizes (such as 1344×768) has not been measured, so even a 12 GB card has
less headroom there.

Measured on an RTX PRO 5000 (48 GB) + RTX PRO 4000 (24 GB) with 1024×768 scenes of about
10 seconds: **one scene every 89 seconds** (131 seconds when scenes are processed one at
a time).

The reference image is downscaled to a short edge of 1024 px before it is sent to H3
(`H3_REF_SHORT_EDGE=1024`). Compared with 2048 px this is about 15% faster, and
same-seed comparisons showed equal image quality. Facial shine is determined by the skin
texture of the reference image, not by quantization or this setting, so a reference with
matte skin and even lighting is recommended.

| Environment variable | Default | Meaning |
|---|---|---|
| `H3_PRESET` | `auto` | H3 configuration. Any value other than `auto` is used as is |
| `H3_GPUS` | `0` | GPU used for computation |
| `H3_DECODE_GPU` | `auto` | Second GPU for decoding (`auto` / `off` / GPU index) |
| `H3_REF_SHORT_EDGE` | `1024` | Short edge of the reference image. `0` uses the backend default (2048) |

## Sample video

[Play / download the sample video (MP4, about 39 MB)](examples/minimax-h3-lipsync-mv-sample.mp4)

- Length: 90 seconds
- Video: H.264 / 1024×768 / 24 fps
- Audio: AAC

## Setup

Requires Python 3.12, ffmpeg, ffprobe, a running diffusers-movie-server gateway, and
either a Demucs service or a Python environment with Demucs installed. Narration videos
also need AivisSpeech Engine (default `http://127.0.0.1:10101`).

```bash
cd Minimax-H3-lipsync-mv
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
# Set the LLM endpoint, model IDs, and if needed the Python used for Demucs in .env
./run.sh
```

Open `http://127.0.0.1:8780` in a browser.

## External services

| Purpose | Default |
|---|---|
| Scenario / H3 prompts | `http://127.0.0.1:64650/v1` |
| Demucs | `http://127.0.0.1:8889` |
| H3 gateway | `http://127.0.0.1:8630` |
| AivisSpeech Engine | `http://127.0.0.1:10101` (Mao / Normal) |

Endpoints and model IDs can be changed in `.env`. API keys are never written to logs or
job JSON.

## Data layout

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

The analysis JSON records the selected boundary times, candidate types, scores and
singing ratios. When a failed job is re-run, only finished scenes whose input, prompt and
H3 settings hashes match are reused. If the identity or lip-sync rules change, old scenes
are regenerated rather than reused by mistake.

## API

- `POST /api/jobs` — submit character / song / concept as multipart
- `GET /api/jobs/{id}` — job status
- `GET /api/jobs/{id}/events` — SSE progress
- `POST /api/jobs/{id}/cancel` — request cancellation
- `GET /api/jobs/{id}/output` — inline playback
- `GET /api/jobs/{id}/download` — download
- `GET /api/health` — LLM connectivity check
- `GET /api/h3/capacity` — estimated pixel budget for this GPU

The H3 gateway has no API to cancel a running job, so a cancellation takes effect at a
scene boundary. A scene that was already submitted runs to completion on the backend, but
its result is discarded.

## MiniMax-H3 references

- Official source code and documentation: [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)
- Official model card and weights: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Official prompt writing guide: [MiniMax-H3 Prompt Writing Skills](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills)

This repository does not include MiniMax-H3 model weights. Check the official model card,
license and terms of use before using them.

## License

The source code in this repository is provided under the [Apache License 2.0](LICENSE).

That license covers only the source code in this repository. The MiniMax-H3 model
weights are governed by the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
and other dependencies such as models, LoRAs and libraries are governed by the licenses
of their respective distributors.

## Tests

```bash
.venv/bin/pytest -q
.venv/bin/ruff check app tests
```
