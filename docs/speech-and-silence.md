# Speech, silence, and preprocessing research

## Verified distinctions

| Approach | What determines an interval | Repository status |
| --- | --- | --- |
| Whisper transcription-derived trimming | Recognized word start/end times, followed by grouping and padding | Implemented in the media trimmer |
| Voice activity detection (VAD) | A model predicts speech activity independently of producing a transcript | Not implemented; [Silero VAD](https://github.com/snakers4/silero-vad) is an external reference |
| Level-based silence detection | Signal level and duration thresholds | Not implemented; FFmpeg examples below are independent commands |

Silence is not equivalent to non-speech. Music and noise can be loud without containing speech; quiet speech can fall below a level threshold. A VAD can also misclassify intervals. Evaluate the actual recordings before choosing an approach.

Cutting and concatenating intervals shortens media and changes timestamps. Muting intervals while retaining their duration is a different operation. Neither repository script provides a mute-in-place workflow or a mapping from trimmed timestamps back to the original.

## Standalone FFmpeg examples

These are audio-only alternatives, not options accepted by the Python scripts. Start with a copy of an input WAV and a new output filename. The threshold and durations below are illustrative, not tuned speech settings.

Detect low-level intervals lasting at least 0.5 seconds without writing edited audio:

```bash
ffmpeg -hide_banner -i "input.wav" -af "silencedetect=noise=-40dB:d=0.5" -f null -
```

Remove leading silence and qualifying later silence periods, writing PCM WAV:

```bash
ffmpeg -n -i "input.wav" -af "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-40dB:stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB" -c:a pcm_s16le "output-trimmed.wav"
```

The negative `stop_periods` restarts removal so it can handle silence inside the recording. `start_duration` specifies the required non-silent duration to start retaining audio; `stop_duration` specifies the qualifying silence duration. Filtering decodes and re-encodes audio; it cannot be combined with audio stream copy. `-n` refuses to overwrite an existing output.

Detection and removal use different detection behavior, so their boundaries need not match exactly. Listen for lost quiet words and clipped consonants. Do not apply this audio-shortening command to a video and assume the original video will stay synchronized.

References: official [silencedetect](https://ffmpeg.org/ffmpeg-filters.html#silencedetect), [silenceremove](https://ffmpeg.org/ffmpeg-filters.html#silenceremove), and local `ffmpeg -h filter=silenceremove`. The examples were exercised on synthetic tone/silence audio for syntax and duration behavior; that does not validate speech accuracy.

## Useful external references

These are research leads, not installed dependencies or supported repository integrations. Consult upstream documentation for versions, installation, and licensing before adopting them.

| Reference | Verified scope / treatment |
| --- | --- |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Dedicated speech activity model; upstream shows timestamp extraction and documents supported sample rates. No integration or benchmark is claimed here. |
| [audio-purifier](https://github.com/huynguyengl99/audio-purifier) | Upstream describes silent-part removal and exporting new files, inspired by faster-whisper. The research omitted the model name; no specific model or accuracy claim is adopted. |
| [unsilence](https://github.com/lagmoellertim/unsilence) | Audio/video silence processing; upstream documents separate speeds for audible and silent intervals. Its basic example speeds silent intervals up rather than simply deleting them. |
| [audio-slicer](https://github.com/openvpi/audio-slicer) | RMS-based silence detection and audio slicing. It is a silence-based example, not evidence that all listed tools use VAD. |
| [JovannMC/silence-remover](https://github.com/JovannMC/silence-remover) | Upstream describes trimming the beginning and end. The research's broader claim of internal-silence removal is not adopted. |

Additional leads retained for future evaluation, without endorsing their examples or current compatibility: [silence-detector-remover](https://github.com/abdulvahapmutlu/silence-detector-remover), [InnovativeCoder/Remove_Silence_from_audio](https://github.com/InnovativeCoder/Remove_Silence_from_audio), [NeuralFalconYT/Remove-Silence-From-Audio](https://github.com/NeuralFalconYT/Remove-Silence-From-Audio), and the [timestamp/muting discussion in Silero VAD](https://github.com/snakers4/silero-vad/discussions/441). These leads were not implementation-tested for this update.

## Disposition of incomplete research

The source research, titled “remove non speech parts of audio github,” explicitly warns “AI responses may include mistakes.” Treat it as a list of leads, not executable documentation or evidence of repository behavior.

- Corrected the opening conflation of non-speech with silence and the claim that most listed tools use VAD.
- Excluded the incomplete “libraries like or” and unnamed VAD/model placeholders. No missing library or model identity has been guessed.
- Replaced the missing FFmpeg command with the separate, checked examples above, clearly labeled as alternatives.
- Retained useful upstream references, while leaving forum answers, anonymous snippets, videos, and promotional articles out of the setup path.
- Did not add silence removal, VAD, speed changes, or muting as repository features. Those would require implementation and evaluation.

See [documentation review](documentation-review.md) for repository-specific confirmations.
