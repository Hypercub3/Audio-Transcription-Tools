# Documentation review for issue #2

## Scope and evidence

[Documentation issue #2](https://github.com/Hypercub3/Audio-Transcription-Tools/issues/2) asks to improve understanding of the scripts and merge Claude-generated documentation. At this review, it is open, unassigned, and has no comments or issue events. No open pull request was returned before this update.

Source baseline: [commit 1b40e701ec09eb6e6d0809ba32a862d289c6e6ab](https://github.com/Hypercub3/Audio-Transcription-Tools/commit/1b40e701ec09eb6e6d0809ba32a862d289c6e6ab). Reviewed both Python scripts, all four Windows launchers, the README, both trimmer guides, and the transcriber guide. The baseline tree has no dependency lockfile, test suite, or AGENTS.md.

Research input: the Google Doc titled “remove non speech parts of audio github.” Its incomplete examples and AI accuracy disclaimer require independent checks. The public upstream references and accepted corrections are captured in [speech and silence research](speech-and-silence.md).

The repository thanks Claude, but neither the issue nor the reviewed files identifies the exact Claude draft/version. This update consolidates the existing guides; it does **not** establish that all requested Claude material has been merged.

## Corrections made

| Previous documentation problem | Resolution |
| --- | --- |
| Commands used nonexistent `media_trimmer.py` or `batch_transcribe.py` names | Use actual repository paths and quote the directory containing a space |
| Trimmer guides disagreed on padding and omitted merge behavior | Use source defaults: 250 ms padding, 1000 ms merge threshold; explain the launcher's 250 ms threshold |
| Two competing trimmer guides | Consolidate into `media-trimmer-docs.md`; leave a redirect at the double-extension path |
| Documentation duplicated the full transcriber implementation | Link to executable source instead |
| `--force` advertised as functional | Mark it as parsed but unused |
| Quality, storage ratios, CPU-based worker counts, and memory estimates presented without tests | Remove unsupported guarantees; start with one worker and require representative evaluation |
| “Non-speech” and silence treated interchangeably | Distinguish Whisper timestamps, VAD, and level-based detection |
| License statements generalized dependency licensing | Refer to repository LICENSE and each dependency's own license |

## Verification performed

- Compared argument names, defaults, model choices, and extension lists directly with source.
- Checked discovery and output naming with isolated source-function checks, including existing-output skipping, dotted stems, and re-inclusion of trimmed files.
- Checked the word-grouping behavior with a fake transcription result; demonstrated the documented padding-overlap case without loading Whisper.
- Exercised the documented FFmpeg filters with FFmpeg 6.1.1-3ubuntu5 on a 16 kHz mono PCM WAV containing two tones and three silence intervals. Detection found three intervals; removal shortened 5.000 seconds to 2.93925 seconds. A rerun logged that the output already existed and left its SHA-256 unchanged. Residual silence is expected; this is not a claim of exact speech boundaries.
- Checked Markdown relative links and documentation-only changes.

These checks do not install or load Whisper, validate dependency combinations, certify inference accuracy, test GPU/concurrent inference, or validate every media format.

## Confirm before closing #2

- [ ] Maintainer identifies the exact Claude-generated material requested by the issue and confirms the consolidation covers it, or supplies the missing draft for reconciliation.
- [ ] Record at least one working OS, Python, Whisper, PyTorch, and FFmpeg/ffprobe version combination; verify installation from a fresh environment and update the setup guide with that evidence.
- [ ] Run both documented Python commands on representative copied audio and video; check TXT/SRT content, speech retention, boundary quality, and audio/video sync. Record expected outputs and known failures.
- [ ] Test no-speech, quiet speech, background music, short pauses, and a decoding/probing failure; confirm logs and rerun behavior are understandable.
- [ ] Maintainer decides whether the known defects below must be fixed before closing the documentation issue, or tracked separately with links and the limitations retained.
- [ ] Review the documentation changes and confirm the issue's scope is met. This update intentionally does not auto-close #2.

### Code behavior requiring a maintainer decision

| Area | What needs confirmation or a separate fix |
| --- | --- |
| Transcriber force/reruns | Wire `--force` into discovery; decide whether requesting a missing format should overwrite an existing one |
| Output identity | Prevent same-stem collisions; preserve the confirmed handling of dotted stems |
| Trimmer reruns | Decide how to exclude generated files and whether existing outputs should be overwritten |
| Timing | Merge intervals again after padding if overlaps should not repeat; validate stream-copy boundaries and sync |
| Video streams/formats | Confirm intended audio track selection, subtitle/metadata handling, and supported codec/container combinations |
| Parallel execution | Validate shared-model use; make transcriber video temporary paths unique across files and runs |
| Subtitle formatting | Fix/check millisecond carry and timestamps at or above 24 hours |
| Error handling | Decide input-directory/parameter validation and nonzero exit-status requirements; check ffprobe availability at startup |
| Launchers | Confirm intended parent-directory scope and whether paths should resolve relative to the batch file |
