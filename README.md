# Audio-Transcription-Tools

Python scripts for transcribing media with OpenAI Whisper and trimming media using recognized word timestamps.

| Task | Script | Guide |
| --- | --- | --- |
| Write TXT transcripts or SRT subtitles | `_Transcriber/batch_transcribe-status-video-ffmpeg.py` | [Transcriber guide](_Transcriber/whisper-batch-transcribe.md) |
| Concatenate intervals around recognized words | `_Media Trimmer/whisper-media-trimmer.py` | [Trimmer guide](_Media%20Trimmer/media-trimmer-docs.md) |

## Setup

Use a virtual environment with a Python version supported by your installed Whisper and PyTorch releases. The transcriber uses `Path.with_stem`, which requires Python 3.9 or later; that is a code-level minimum, not a tested dependency compatibility guarantee. This repository has no pinned dependency set or tested platform matrix.

Install [FFmpeg](https://ffmpeg.org/download.html) separately, with both `ffmpeg` and `ffprobe` on PATH. After activating your virtual environment:

```bash
python -m pip install openai-whisper torch pydub numpy tqdm
ffmpeg -version
ffprobe -version
```

The package name is `openai-whisper`; the Python import is `whisper`. The trimmer imports pydub and NumPy even though its processing path uses FFmpeg. Follow [Whisper's setup instructions](https://github.com/openai/whisper#setup) and [PyTorch's installation instructions](https://pytorch.org/get-started/locally/) for platform-specific installation. Whisper model weights may need to download on first use.

## First run

Run these commands from the repository root, replacing `sample-media` with a directory containing **copies of a few short recordings**. Both scripts scan subdirectories by default.

```bash
python "_Transcriber/batch_transcribe-status-video-ffmpeg.py" "sample-media" --model base --format both --workers 1
python "_Media Trimmer/whisper-media-trimmer.py" "sample-media" --model base --padding 250 --merge-threshold 1000 --workers 1
```

Add `--language en` only for English material; omit it for automatic language detection. Use each script's `--help` for its accepted arguments.

Outputs are written beside the inputs. Read the guides before rerunning: trimming can overwrite existing outputs and reprocess previously trimmed files; transcription has output collisions and a nonfunctional `--force` flag. Start with one worker because concurrent inference and temporary-file behavior have not been validated.

## Choosing an approach

The trimmer derives intervals from Whisper transcription; it does not implement a dedicated voice activity detector (VAD) or a loudness threshold. Silence, non-speech sound, and unrecognized speech are different things. Review the output for missing words, repeated boundaries, and audio/video synchronization.

See [speech and silence research](docs/speech-and-silence.md) for separate FFmpeg examples, VAD references, and the disposition of incomplete research. Cutting changes the timeline; this repository does not export a mapping back to original timestamps.

## Documentation status

Usage and defaults were checked against source at commit `1b40e701ec09eb6e6d0809ba32a862d289c6e6ab`. This does not certify end-to-end model inference or every listed media format. [Documentation review and closure checklist](docs/documentation-review.md) records corrections and remaining confirmations for [issue #2](https://github.com/Hypercub3/Audio-Transcription-Tools/issues/2).

Thank you Claude. Existing documentation has been consolidated against the executable scripts; the exact Claude-generated source still needs maintainer confirmation.

## License

See [LICENSE](LICENSE) for the repository's MIT license. Dependencies retain their own licenses.
