# Whisper Media Trimmer

## What the script does

`whisper-media-trimmer.py` transcribes a temporary 16 kHz mono WAV with Whisper's `word_timestamps=True`. It sorts recognized words, groups words separated by at most `--merge-threshold`, adds `--padding` around each group, and uses FFmpeg to extract and concatenate those intervals from the original media.

This is transcription-derived trimming. It is not a standalone VAD, does not use a dB/RMS threshold, and does not guarantee that every retained interval contains only speech or that every spoken word is retained. It cuts time out; it has no mute-in-place mode.

See [setup](../README.md#setup) before running. Both `ffmpeg` and `ffprobe` are required; the startup check only checks FFmpeg.

## Usage

From the repository root:

```bash
python "_Media Trimmer/whisper-media-trimmer.py" "sample-media" --workers 1
python "_Media Trimmer/whisper-media-trimmer.py" "sample-media" --language en --model base --padding 250 --merge-threshold 1000 --workers 1
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `input_dir` | Required | Directory to scan recursively |
| `--model` | `base` | One of `tiny`, `base`, `small`, `medium`, `large` |
| `--padding` | `250` | Milliseconds added before and after each grouped interval, clamped to file bounds |
| `--merge-threshold` | `1000` | Maximum gap in milliseconds between recognized words to join into one interval |
| `--workers` | `1` | Thread-pool workers sharing one model |
| `--language` | Unset | Language code passed to Whisper; unset allows detection |

Use nonnegative padding and merge thresholds and at least one worker. The parser accepts integers without enforcing those ranges.

Increasing the merge threshold preserves more gaps inside groups. Padding extends group boundaries; neither option sets a fixed final pause duration. Groups are merged **before** padding. If padding makes separate groups overlap, that overlap can be repeated in the output. For example, two groups separated by 400 ms with a 250 ms merge threshold remain separate, but 250 ms padding on each side overlaps by 100 ms.

The Windows `whisper-media-trimmer.bat` uses English, model `base`, padding 250 ms, and **merge threshold 250 ms**, differing from the Python default. It refers to the parent of the current working directory and a relative script path. Run it from `_Media Trimmer` only when that parent is the intended media directory, or use the explicit Python command above.

## Inputs and outputs

The extension allowlist in `SUPPORTED_FORMATS` is:

- Audio: `.mp3 .wav .m4a .flac .ogg .aac .wma .aiff .alac .ac3 .amr .ape .au .mka .pcm .ra .wv .opus`.
- Video: `.mp4 .avi .mkv .mov .wmv .flv .webm .m4v .mpg .mpeg .3gp .3g2 .ogv .ts .mts .m2ts`.

An accepted extension is not a tested codec/container guarantee. Decoding, duration probing, stream copying, and concatenation must all work with the installed FFmpeg build. Raw PCM, for example, may require format parameters that this script does not provide.

`meeting.mp4` produces `meeting_trimmed.mp4` in the same directory. The source is not intentionally modified. The scan skips directories, zero-byte files, and files whose own name starts with a dot; it does not explicitly exclude hidden parent directories.

**Rerun behavior:** the scan does not exclude `*_trimmed` files or skip sources with an existing output. FFmpeg uses `-y`, so a rerun can overwrite `meeting_trimmed.mp4` and create `meeting_trimmed_trimmed.mp4`. Keep generated media outside the input tree before another run.

## Quality and failure behavior

- Extraction and concatenation both request `-c copy`; there is no re-encoding fallback. This avoids codec re-encoding when successful, but does not guarantee exact cuts, preserved metadata, or all input streams. There is no explicit `-map` for all streams.
- Check cuts and audio/video sync, especially with inter-frame video codecs. Stream copy is not a guarantee of frame-accurate speech boundaries.
- If Whisper returns no word intervals, no new trimmed file is made and the file counts as unsuccessful. Transcription errors also return an empty interval list, so inspect earlier logs to distinguish an error from no recognized speech. An older output may still exist.
- Temporary WAVs and segment files use per-file temporary directories. Disk demand depends on duration and segment count; no fixed storage or output-size ratio has been established.
- Logs/progress go to the console. The script does not configure a persistent log file. Inspect the summary and output playback; a zero process exit code alone is not a success guarantee.

## Troubleshooting and validation

Check `ffmpeg -version` and `ffprobe -version` in the same environment used to run Python. For import failures, install dependencies into that interpreter. For memory errors, keep `--workers 1` and try a smaller model. For empty output or missed words, inspect the source audio and earlier errors before changing padding.

Before batch use, listen to a short result, compare boundaries against the original, and check video sync. Confirm behavior with quiet speech, background music, short pauses, and no-speech recordings. Resource estimates and accuracy claims from older drafts were removed because they were not backed by repository tests.

See [research and alternatives](../docs/speech-and-silence.md) and [remaining confirmations](../docs/documentation-review.md).
