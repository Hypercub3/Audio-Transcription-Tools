# Whisper Batch Transcriber

`batch_transcribe-status-video-ffmpeg.py` scans a directory and writes Whisper transcripts beside each eligible media file. See [setup](../README.md#setup). It uses CUDA when `torch.cuda.is_available()` is true, otherwise CPU.

## Usage

From the repository root:

```bash
python "_Transcriber/batch_transcribe-status-video-ffmpeg.py" "sample-media" --model base --format both --workers 1
python "_Transcriber/batch_transcribe-status-video-ffmpeg.py" "sample-media" --language en --format srt --no-recursive
```

| Argument | Default | Actual behavior |
| --- | --- | --- |
| `input_dir` | `.` | Input directory, relative to the working directory |
| `--model` | `base` | `tiny`, `base`, `small`, `medium`, or `large` |
| `--language` | Unset | Whisper language code; unset allows detection |
| `--format` | `txt` | `txt`, `srt`, or `both` |
| `--workers` | `1` | Number of workers; use at least one |
| `--no-recursive` | Off | Limit scanning to the input directory |
| `--force` | Off | **Parsed but never read by the processing code; has no effect** |

Do not rely on `--force` to regenerate existing results. To rerun, first move the relevant generated transcript files to a backup location outside their original paths.

## Accepted extensions

- Audio: `.mp3 .wav .m4a .flac .ogg .aac .wma .aiff .aifc .aif`.
- Video: `.mp4 .mov .mkv .avi .wmv .flv`.

These are the source-code allowlist, not a guarantee for every codec/container. Unlike the trimmer, the transcriber does not include WebM or Opus extensions. Videos are converted to temporary 16 kHz mono PCM WAV files before transcription; audio paths are passed to Whisper directly.

## Output and reruns

For `meeting.mp4`, outputs are `meeting_transcription.txt` and/or `meeting_transcription.srt` beside the input. TXT contains transcript text; SRT uses Whisper segment timestamps, not word-level timestamps. Logs are appended to `transcription_log.txt` in the current working directory.

A file is skipped only when **all requested output formats already exist**. With `--format both`, if TXT exists and SRT is missing, the script transcribes again and writes both, overwriting the existing TXT.

Media with the same stem in one directory (for example, `meeting.mp3` and `meeting.mp4`) share output names. Rename copies to unique stems before processing.

Previously trimmed media are also eligible for transcription. If transcription follows trimming, subtitle times refer to the shortened file, not the original. The scripts do not export a timestamp mapping.

## Windows launchers

Run launchers from `_Transcriber` only if its parent is the desired input directory. They use relative script paths and `%cd%\..`, rather than locating themselves independently.

| Launcher | Format | Language | Model |
| --- | --- | --- | --- |
| `batch_transcribe-status-video-ffmpeg.bat` | both | en | medium |
| `batch_transcribe-status-video-ffmpeg-txt.bat` | txt | en | medium |
| `batch_transcribe-status-video-ffmpeg-srt.bat` | srt | en | medium |

## Known limits

Use one worker for the initial validation. Workers share the same model, and video temporary filenames use only the source stem in the system temporary directory. Same-stem videos in different folders, or concurrent runs, can collide.

The SRT formatter uses `timedelta.seconds`, which wraps after 24 hours, and rounds milliseconds without carrying a result of 1000 to the next second. Long recordings and rounding boundaries need correction or explicit testing before relying on their subtitles.

Inspect the failure summary and logs. Some errors are caught and reported without a nonzero process exit status. Confirm the requested files exist and contain expected text; check subtitles against playback. No end-to-end transcription or supported-platform matrix is certified by this documentation update.

The former embedded copy of the full Python script has been removed to prevent drift. The executable [Python source](batch_transcribe-status-video-ffmpeg.py) is authoritative. See the [closure checklist](../docs/documentation-review.md) for unresolved items.
