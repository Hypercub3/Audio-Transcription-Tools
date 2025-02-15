# Media Speech Trimmer Documentation

## Overview

The Media Speech Trimmer is a Python script that uses OpenAI's Whisper model to detect and extract speech segments from audio and video files. It removes sections without speech, creating trimmed versions that only contain speaking portions.

## Requirements

### System Requirements
- Python 3.8 or higher
- FFmpeg installation
- Multi-core processor recommended
- RAM: 
  - Minimum 4GB for base model
  - 8GB+ recommended for larger models
- Storage: Free space at least 2x the size of source media

### Required Python Packages
```bash
# Core dependencies
pip install openai-whisper
pip install torch
pip install pydub
pip install numpy

# Note: FFmpeg must be installed separately
```

## Installation

1. Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from ffmpeg.org and add to PATH
```

2. Install Python dependencies:
```bash
pip install openai-whisper torch pydub numpy
```

## Supported Formats

### Audio Formats
- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)
- AAC (.aac)
- WMA (.wma)
- AIFF (.aiff)
- ALAC (.alac)
- AC3 (.ac3)
- AMR (.amr)
- APE (.ape)
- AU (.au)
- MKA (.mka)
- PCM (.pcm)
- RA (.ra)
- WavPack (.wv)
- Opus (.opus)

### Video Formats
- MP4 (.mp4)
- AVI (.avi)
- MKV (.mkv)
- MOV (.mov)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)
- M4V (.m4v)
- MPG/MPEG (.mpg, .mpeg)
- 3GP (.3gp, .3g2)
- OGV (.ogv)
- Transport Streams (.ts, .mts, .m2ts)

## Usage

### Basic Usage
```bash
python media_trimmer.py /path/to/media/directory
```

### Advanced Usage
```bash
python media_trimmer.py /path/to/media/directory \
    --model base \
    --padding 500 \
    --workers 4 \
    --language en
```

### Command Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|----------|
| `input_dir` | Directory containing media files | Required | Any valid path |
| `--model` | Whisper model size | `base` | `tiny`, `base`, `small`, `medium`, `large` |
| `--padding` | Milliseconds to keep before/after speech | `500` | Any positive integer |
| `--workers` | Number of parallel processing workers | `1` | Any positive integer |
| `--language` | ISO 639-1 language code | None | `en`, `fr`, `de`, etc. |

Note: When no language is specified, Whisper will attempt to auto-detect the language.

## Progress Display

The script provides clear text-based progress information for each file:

1. File Processing Start:
```
Processing: example.mp3
```

2. Operation Status:
```
⟳ Extracting/loading audio from example.mp3
⟳ Detecting speech in example.mp3
⟳ Trimming example.mp3
```

3. Completion Status:
```
✓ Completed: example.mp3
```
or
```
✗ Failed: example.mp3
```

4. Overall Progress:
```
Progress: 1/6 files processed
```

### Status Symbols
- ⟳ : Operation in progress
- ✓ : Successfully completed
- ✗ : Operation failed/skipped

## Processing Workflow

1. Initialization:
   - Load Whisper model
   - Scan directory for media files
   - Initialize processing queue

2. Per-File Processing:
   - File type detection
   - Audio extraction (for videos)
   - Speech detection
   - Content trimming
   - Output file creation

3. Completion:
   - Success/failure count
   - Processing summary
   - Cleanup of temporary files

## Performance Considerations

### Memory Usage
Memory requirements vary by component:
- Video processing: ~500MB-1GB per worker
- Audio processing: ~50-100MB per worker
- Whisper models:
  - tiny: ~150MB
  - base: ~500MB
  - small: ~1GB
  - medium: ~2.5GB
  - large: ~5GB

### Processing Time
Factors affecting speed:
- Media duration
- Video resolution
- Whisper model size
- Number of workers
- CPU/GPU availability

### Storage Requirements
- Source files: Original size
- Temporary files: ~500MB per worker
- Output files: 30-70% of original size
- Required free space: 2x total source size

## Best Practices

### Directory Organization

The script preserves the original directory structure and creates trimmed files alongside the originals:

```
input_directory/
├── subfolder1/
│   ├── video1.mp4
│   ├── video1_trimmed.mp4           # Created by script
│   ├── audio1.mp3
│   └── audio1_trimmed.mp3           # Created by script
└── subfolder2/
    ├── lecture.mp4
    ├── lecture_trimmed.mp4          # Created by script
    ├── interview.wav
    └── interview_trimmed.wav        # Created by script
```

Each processed file is created in the same directory as its source file, with "_trimmed" added to the filename. This:
- Maintains the original organization
- Keeps related files together
- Makes it easy to find processed versions
- Preserves the original files

### Processing Strategy
1. Test Run:
```bash
# Process a few files first
python media_trimmer.py /test/directory --workers 1
```

2. Batch Processing:
```bash
# Process larger batches
python media_trimmer.py /main/directory --workers 4
```

3. Large Collections:
```bash
# Process overnight
nohup python media_trimmer.py /large/directory --workers 4 > processing.log 2>&1 &
```

### Optimizing Performance
1. Worker Count:
   - CPU cores ÷ 2 for video
   - CPU cores × 1 for audio
   - Reduce if memory limited

2. Model Selection:
   - tiny: Fast, basic accuracy
   - base: Good balance
   - large: Best quality, slower

3. Padding Values:
   - 500ms: Default
   - 750ms: More context
   - 250ms: Tighter trimming

## Troubleshooting

### Common Issues

1. FFmpeg not found:
```
Error: FFmpeg is not installed or not found in PATH
Solution: Install FFmpeg and verify PATH
```

2. Memory errors:
```
Error: Memory error during processing
Solution: Reduce worker count or use smaller model
```

3. Processing errors:
```
Error: Failed to process [filename]
Solution: Check file permissions and format
```

4. No speech detected:
```
Error: No speech segments found
Solution: Verify audio quality and speech content
```

### Quality Checks

1. Pre-processing:
   - Verify file integrity
   - Check audio levels
   - Ensure sufficient space
   - Test with sample files

2. Post-processing:
   - Compare durations
   - Verify audio quality
   - Check speech continuity
   - Validate file sizes

## Examples

### Process English Audio Files
```bash
python media_trimmer.py /path/to/audio \
    --model base \
    --language en \
    --workers 4
```

### Process High-Quality Video
```bash
python media_trimmer.py /path/to/video \
    --model medium \
    --padding 750 \
    --workers 2
```

Example output:
```
Processing: lecture.mp4
⟳ Extracting/loading audio from lecture.mp4
⟳ Detecting speech in lecture.mp4
⟳ Trimming lecture.mp4
✓ Completed: lecture.mp4

Progress: 1/3 files processed
```

### Mixed Media Processing
```bash
python media_trimmer.py /path/to/media \
    --model small \
    --workers 3 \
    --padding 500
```

## Support

For issues and updates:
1. Check FFmpeg installation
2. Verify Python dependencies
3. Test with smaller batches
4. Review log output
5. Check system resources

## License

This project is licensed under the MIT License. See the LICENSE file for details.