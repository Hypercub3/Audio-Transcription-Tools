# Whisper Media Trimmer Documentation

## Overview
Whisper Media Trimmer is a Python script that uses OpenAI's Whisper model to detect and extract speech segments from audio and video files. It removes non-speech portions while maintaining the original quality of the media.

## Features
- Supports both audio and video files
- Maintains original media quality using FFmpeg's stream copy
- Configurable padding around speech segments
- Adjustable threshold for merging nearby speech segments
- Parallel processing support
- Progress tracking and detailed logging
- Language-specific processing support

## Requirements

### Python Dependencies
- whisper
- pydub
- numpy
- torch
- tqdm

### System Dependencies
- FFmpeg (must be installed and available in system PATH)

## Installation

1. Install FFmpeg for your operating system:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # macOS
   brew install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. Install Python dependencies:
   ```bash
   pip install openai-whisper pydub numpy torch tqdm
   ```

## Usage

### Basic Usage
```bash
python whisper-media-trimmer.py input_directory
```

### Advanced Usage
```bash
python whisper-media-trimmer.py input_directory \
    --model base \
    --padding 250 \
    --merge-threshold 1000 \
    --workers 4 \
    --language en
```

### Command Line Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|----------|
| `input_dir` | Directory containing media files | (Required) | Any valid path |
| `--model` | Whisper model size | "base" | tiny, base, small, medium, large |
| `--padding` | Milliseconds of padding around speech | 250 | Any positive integer |
| `--merge-threshold` | Maximum gap (ms) between words to merge | 1000 | Any positive integer |
| `--workers` | Number of parallel workers | 1 | Any positive integer |
| `--language` | ISO language code | None | e.g., "en", "fr", "de" |

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
- And more...

### Video Formats
- MP4 (.mp4)
- AVI (.avi)
- MKV (.mkv)
- MOV (.mov)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)
- And more...

## Parameters Explained

### Model Size
The `--model` parameter affects accuracy and processing speed:
- `tiny`: Fastest but least accurate
- `base`: Good balance of speed and accuracy
- `small`: More accurate than base, slower
- `medium`: High accuracy, slower
- `large`: Most accurate, slowest

### Padding
The `--padding` parameter (in milliseconds) adds extra time before and after each speech segment:
- Lower values (e.g., 100ms): Tighter trimming, possible word clipping
- Default (250ms): Good balance
- Higher values (e.g., 500ms): More natural transitions, larger file size

### Merge Threshold
The `--merge-threshold` parameter (in milliseconds) determines when to merge nearby speech segments:
- Lower values (e.g., 500ms): More aggressive splitting
- Default (1000ms): Natural sentence breaks
- Higher values (e.g., 2000ms): More conservative splitting

## Output
- Processed files are saved in the same directory as the input
- Output filename format: `original_name_trimmed.extension`
- Maintains original codec and quality
- Progress is displayed in terminal
- Detailed logs are generated

## Common Issues and Solutions

### FFmpeg Not Found
```
Error: FFmpeg is not installed or not found in PATH
Solution: Install FFmpeg and ensure it's in your system PATH
```

### GPU Memory Issues
```
Error: CUDA out of memory
Solution: Use a smaller model size or process files sequentially (--workers 1)
```

### File Permission Errors
```
Error: Permission denied
Solution: Ensure write permissions in the output directory
```

## Performance Tips

1. **Model Selection**
   - For quick testing: Use "tiny" model
   - For production: Use "base" or "small" model
   - For highest quality: Use "medium" or "large" model

2. **Parallel Processing**
   - CPU-bound: Set --workers to CPU core count
   - GPU-bound: Use lower --workers value
   - Memory-limited: Use --workers 1

3. **Memory Usage**
   - Larger models require more RAM
   - Video files need more memory than audio
   - Consider using smaller models for batch processing

## Examples

1. **Basic Processing**
   ```bash
   python whisper-media-trimmer.py ~/media/podcasts
   ```

2. **High Quality Processing**
   ```bash
   python whisper-media-trimmer.py ~/media/interviews \
       --model medium \
       --padding 300 \
       --merge-threshold 1200
   ```

3. **Batch Processing**
   ```bash
   python whisper-media-trimmer.py ~/media/lectures \
       --model base \
       --workers 4
   ```

4. **Language-Specific Processing**
   ```bash
   python whisper-media-trimmer.py ~/media/french_videos \
       --language fr \
       --model small
   ```

## Contributing
Contributions are welcome! Please ensure:
1. Code follows existing style
2. New features include documentation
3. Changes maintain backward compatibility
4. Tests pass (if available)

## License
This project uses the OpenAI Whisper model and FFmpeg.
- Whisper is licensed under MIT
- FFmpeg is licensed under LGPL 2.1+

Please ensure compliance with all relevant licenses when using this tool.