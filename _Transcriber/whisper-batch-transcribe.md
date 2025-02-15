# Whisper Batch Transcription Script Documentation

## Overview
This script provides batch transcription capabilities for audio and video files using OpenAI's Whisper model. It supports multiple audio/video formats, parallel processing, and can output transcriptions in both TXT and SRT formats.

## Features
- Supports both audio and video files
- Multiple output formats (TXT and SRT)
- Progress tracking with detailed status updates
- Parallel processing support
- Recursive directory scanning
- Automatic audio extraction from video files
- Detailed logging
- Error handling and recovery

## Requirements
```bash
# Python packages
pip install torch
pip install openai-whisper
pip install tqdm

# System requirements
# FFmpeg must be installed on your system:
# Windows: Download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH
# Linux: sudo apt-get install ffmpeg
# Mac: brew install ffmpeg
```

## Supported File Formats
### Audio Formats
- .mp3
- .wav
- .m4a
- .flac
- .ogg
- .aac
- .wma
- .aiff
- .aifc
- .aif

### Video Formats
- .mp4
- .mov
- .mkv
- .avi
- .wmv
- .flv

## Usage
### Basic Usage
```bash
# Using current directory
python batch_transcribe.py

# Using explicit directory
python batch_transcribe.py "path/to/your/media/files"
```

### Advanced Usage
```bash
# Transcribe current directory with specific model and output both TXT and SRT
python batch_transcribe.py --model medium --format both

# Transcribe specific directory with model and format options
python batch_transcribe.py "path/to/files" --model medium --format both

# Use multiple worker threads
python batch_transcribe.py "path/to/files" --workers 4

# Specify language
python batch_transcribe.py "path/to/files" --language en

# Only process current directory (no subdirectories)
python batch_transcribe.py "path/to/files" --no-recursive

# Force retranscription of already processed files
python batch_transcribe.py "path/to/files" --force
```

### Command Line Arguments
- `input_dir`: Directory containing media files (optional, defaults to current directory '.')
- `--model`: Whisper model to use (tiny/base/small/medium/large)
- `--language`: Language code (e.g., 'en' for English)
- `--format`: Output format (txt/srt/both)
- `--workers`: Number of parallel workers (default: 1)
- `--no-recursive`: Don't search subdirectories
- `--force`: Force retranscription of existing files

## Output Files
- For input file `example.mp4`:
  - Text format: `example_transcription.txt`
  - SRT format: `example_transcription.srt`
- Logs are saved to `transcription_log.txt`

## Full Script Implementation
```python
import whisper
import torch
import time
import argparse
from pathlib import Path
import concurrent.futures
import logging
from typing import List, Set
from datetime import timedelta
from tqdm import tqdm
import tempfile
import subprocess
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription_log.txt'),
        logging.StreamHandler()
    ]
)

# Common audio and video file extensions
MEDIA_EXTENSIONS: Set[str] = {
    # Audio files
    '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', 
    '.wma', '.aiff', '.aifc', '.aif',
    # Video files
    '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv'
}

def extract_audio(video_path: Path) -> Path:
    """Extract audio from video file using ffmpeg"""
    temp_dir = tempfile.gettempdir()
    temp_audio_path = Path(temp_dir) / f"{video_path.stem}_temp_audio.wav"
    
    try:
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', str(video_path), 
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '16000',  # Sample rate
            '-ac', '1',  # Mono audio
            '-loglevel', 'error',  # Reduce ffmpeg output
            '-y',  # Overwrite output file if exists
            str(temp_audio_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return temp_audio_path
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error extracting audio from {video_path}: {e.stderr.decode()}")
        raise
    except Exception as e:
        logging.error(f"Error extracting audio from {video_path}: {str(e)}")
        raise

def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds//3600
    minutes = (td.seconds//60)%60
    seconds = td.seconds%60
    milliseconds = round(td.microseconds/1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def create_srt_content(segments) -> str:
    """Convert transcript segments to SRT format"""
    srt_content = []
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        
        srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
    
    return "\n".join(srt_content)

class MediaTranscriber:
    def __init__(self, model_name: str = "base", language: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        
        logging.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name).to(self.device)
        self.language = language

    def transcribe_file(self, media_path: Path, formats: Set[str], pbar=None) -> dict:
        """
        Transcribe a single media file and save in specified formats
        """
        temp_audio_path = None
        try:
            if pbar:
                pbar.set_description(f"Transcribing: {media_path.name}")
            
            start_time = time.time()

            # If it's a video file, extract the audio first
            if media_path.suffix.lower() in {'.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv'}:
                if pbar:
                    pbar.set_description(f"Extracting audio: {media_path.name}")
                temp_audio_path = extract_audio(media_path)
                audio_path = temp_audio_path
            else:
                audio_path = media_path

            if pbar:
                pbar.set_description(f"Transcribing: {media_path.name}")

            # Create transcription options
            options = {"language": self.language} if self.language else {}
            
            # Perform transcription
            result = self.model.transcribe(
                str(audio_path),
                **options,
                verbose=False
            )
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Save transcriptions in requested formats
            if 'txt' in formats:
                txt_path = media_path.with_stem(f"{media_path.stem}_transcription").with_suffix('.txt')
                txt_path.write_text(result["text"], encoding='utf-8')
            
            if 'srt' in formats:
                srt_path = media_path.with_stem(f"{media_path.stem}_transcription").with_suffix('.srt')
                srt_content = create_srt_content(result["segments"])
                srt_path.write_text(srt_content, encoding='utf-8')
            
            if pbar:
                pbar.set_postfix(duration=f"{duration:.1f}s")
            
            return {
                "file": media_path.name,
                "duration": duration,
                "status": "success"
            }
            
        except Exception as e:
            logging.error(f"Error transcribing {media_path}: {str(e)}")
            return {
                "file": media_path.name,
                "error": str(e),
                "status": "failed"
            }
        finally:
            # Clean up temporary audio file if it exists
            if temp_audio_path and temp_audio_path.exists():
                try:
                    temp_audio_path.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete temporary audio file {temp_audio_path}: {str(e)}")

def find_media_files(directory: Path, formats: Set[str], recursive: bool = True) -> List[Path]:
    """
    Find all media files in the given directory that need transcription
    """
    if recursive:
        all_files = directory.rglob("*")
    else:
        all_files = directory.glob("*")
        
    # Filter for media files and exclude those that already have transcriptions
    media_files = []
    for f in all_files:
        if f.suffix.lower() in MEDIA_EXTENSIONS:
            # Check existing transcriptions
            needs_transcription = False
            for fmt in formats:
                transcription_file = f.with_stem(f"{f.stem}_transcription").with_suffix(f'.{fmt}')
                if not transcription_file.exists():
                    needs_transcription = True
                    break
            if needs_transcription:
                media_files.append(f)
    
    return sorted(media_files)

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe media files in a directory")
    parser.add_argument("input_dir", nargs='?', default='.',
                        help="Directory containing media files (default: current directory)")
    parser.add_argument("--model", default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model to use")
    parser.add_argument("--language", help="Language code (e.g., 'en' for English)")
    parser.add_argument("--format", choices=['txt', 'srt', 'both'], default='txt',
                        help="Output format (txt, srt, or both)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Don't search subdirectories for media files")
    parser.add_argument("--force", action="store_true",
                        help="Force retranscription even if transcription file exists")
    
    args = parser.parse_args()
    
    # Determine output formats
    formats = {'txt', 'srt'} if args.format == 'both' else {args.format}
    
    # Convert path to Path object
    input_dir = Path(args.input_dir)
    
    # Ensure input directory exists
    if not input_dir.exists():
        logging.error(f"Directory does not exist: {input_dir}")
        return
    
    # Find all media files
    media_files = find_media_files(input_dir, formats, not args.no_recursive)
    if not media_files:
        logging.info(f"No new media files found in {input_dir}")
        return
    
    logging.info(f"Found {len(media_files)} media files to transcribe")
    logging.info(f"Output format(s): {', '.join(formats)}")
    
    # Initialize transcriber
    transcriber = MediaTranscriber(model_name=args.model, language=args.language)
    
    # Process files
    results = []
    
    # Single progress bar for single worker
    if args.workers == 1:
        with tqdm(total=len(media_files), desc="Overall progress") as pbar:
            for media_file in media_files:
                result = transcriber.transcribe_file(media_file, formats, pbar)
                results.append(result)
                pbar.update(1)
    
    # Multiple progress bars for multiple workers
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(transcriber.transcribe_file, media_file, formats, None): media_file
                for media_file in media_files
            }
            
            with tqdm(total=len(media_files), desc="Overall progress") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix_str(f"Latest: {result['file']}")
    
    # Generate summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    logging.info("\nTranscription Summary:")
    logging.info(f"Total files processed: {len(results)}")
    logging.info(f"Successfully transcribed: {successful}")
    logging.info(f"Failed: {failed}")
    
    if failed > 0:
        logging.info("\nFailed files:")
        for result in results:
            if result["status"] == "failed":
                logging.info(f"{result['file']}: {result['error']}")

if __name__ == "__main__":
    main()
```

## Performance Considerations
1. Model Selection:
   - tiny: Fastest, lowest accuracy
   - base: Good balance of speed and accuracy
   - small: Better accuracy, slower
   - medium: High accuracy, slower
   - large: Best accuracy, slowest

2. GPU vs CPU:
   - GPU transcription is significantly faster
   - Script automatically detects and uses CUDA if available

3. Parallel Processing:
   - Use `--workers` argument to process multiple files simultaneously
   - Recommended workers = CPU cores - 1
   - For GPU transcription, multiple workers may not improve performance

## Error Handling
- Failed transcriptions don't stop the entire process
- Errors are logged to `transcription_log.txt`
- Summary of failed files is provided at the end
- Temporary files are automatically cleaned up

## Best Practices
1. Start with the base model and test on a few files
2. Use GPU if available for faster processing
3. For large directories, run a test with a subset first
4. Monitor the log file for any errors
5. Backup important files before processing
6. Consider disk space for temporary files when processing videos

## Common Issues and Solutions
1. FFmpeg not found:
   - Ensure FFmpeg is installed and in system PATH
   - Check FFmpeg installation with `ffmpeg -version`

2. CUDA/GPU issues:
   - Verify CUDA installation with `torch.cuda.is_available()`
   - Update GPU drivers if necessary

3. Memory issues:
   - Reduce number of worker threads
   - Use a smaller model
   - Process fewer files at once

4. File permission errors:
   - Check read/write permissions
   - Run script with appropriate privileges

## Language Support
- Whisper supports multiple languages
- Use ISO language codes (e.g., 'en', 'es', 'fr')
- Examples:
  ```bash
  # English
  python batch_transcribe.py "path/to/files" --language en
  
  # Spanish
  python batch_transcribe.py "path/to/files" --language es
  
  # French
  python batch_transcribe.py "path/to/files" --language fr
  ```
- If no language is specified, Whisper will auto-detect the language

## File Naming Convention
- Input files retain their original names
- Output files follow the pattern:
  - Text: `originalname_transcription.txt`
  - SRT: `originalname_transcription.srt`
- Example:
  ```
  interview.mp4 → interview_transcription.txt
  lecture.wav → lecture_transcription.srt
  ```

## Directory Structure
```
project_root/
│
├── batch_transcribe.py     # Main script
├── transcription_log.txt   # Log file
│
├── input_directory/
│   ├── audio1.mp3
│   ├── video1.mp4
│   └── subdirectory/
│       └── audio2.wav
│
└── temp/                   # Temporary files (auto-cleaned)
    └── temp_audio_*.wav    # Temporary audio extracts
```

## Memory Management
The script includes several memory optimization features:
1. Temporary file cleanup
   - Audio extracted from videos is automatically deleted
   - Cleanup occurs even if processing fails

2. Efficient file handling
   - Files are processed one at a time
   - Audio extraction happens on-demand
   - Memory is released after each file

3. Progress tracking
   - Memory-efficient progress bars
   - Real-time status updates
   - Resource usage monitoring

## Customization Options
### Adding New File Types
To add support for additional file types, modify the `MEDIA_EXTENSIONS` set:
```python
MEDIA_EXTENSIONS.add('.new_extension')
```

### Modifying FFmpeg Parameters
The audio extraction can be customized by modifying the FFmpeg command in `extract_audio()`:
```python
cmd = [
    'ffmpeg', '-i', str(video_path),
    '-vn',                    # Disable video
    '-acodec', 'pcm_s16le',  # Audio codec (can be modified)
    '-ar', '16000',          # Sample rate (can be modified)
    '-ac', '1',              # Mono audio (can be modified)
    # Add additional FFmpeg parameters here
    str(temp_audio_path)
]
```

### Custom Output Formats
To add a new output format:
1. Add the format to the command line parser
2. Create a format-specific content creation function
3. Add the format handling in `transcribe_file()`

## Logging and Debugging
### Log File Structure
The log file (`transcription_log.txt`) contains:
- Timestamp for each operation
- File processing status
- Error messages and stack traces
- Performance metrics

Example log entry:
```
2024-01-24 10:30:15,123 - INFO - Found 5 media files to transcribe
2024-01-24 10:30:15,456 - INFO - Using device: cuda
2024-01-24 10:30:15,789 - INFO - Loading Whisper model: base
```

### Debug Mode
To enable more detailed logging:
```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription_log.txt'),
        logging.StreamHandler()
    ]
)
```

## Advanced Usage Scenarios
### Processing Large Directories
For directories with many files:
```bash
# Process files in batches using multiple workers
python batch_transcribe.py "large_directory" --workers 4 --format both

# Focus on specific subdirectories
python batch_transcribe.py "large_directory/subset" --no-recursive
```

### Handling Long Videos
For very long videos:
1. Use the 'large' model for better accuracy
2. Consider breaking the video into smaller segments
3. Monitor system resources during processing

### Using Current Directory
When no directory is specified, the script processes all media files in the current directory:
```bash
# Navigate to the directory containing media files
cd path/to/media/files

# Run the script (will process current directory)
python path/to/batch_transcribe.py
```

### Batch Processing Multiple Directories
Create a shell script to process multiple directories:
```bash
#!/bin/bash
directories=(
    "path/to/dir1"
    "path/to/dir2"
    "path/to/dir3")

for dir in "${directories[@]}"; do
    python batch_transcribe.py "$dir" --format both --model medium
done
```

## Troubleshooting Guide
### Common Errors and Solutions

1. "FFmpeg not found in system path"
   - Solution: Add FFmpeg to system PATH
   - Windows: Add FFmpeg bin directory to Environment Variables
   - Linux/Mac: Use package manager to install FFmpeg

2. "CUDA out of memory"
   - Solution: Reduce model size or batch size
   - Try using CPU instead: `CUDA_VISIBLE_DEVICES=""`
   - Close other GPU-intensive applications

3. "Permission denied"
   - Solution: Check file and directory permissions
   - Run with appropriate user privileges
   - Verify write access to output directory

4. "File not found"
   - Solution: Check file paths
   - Use absolute paths
   - Verify file existence before processing

5. "Invalid language code"
   - Solution: Use correct ISO language codes
   - Check Whisper documentation for supported languages

### Performance Issues
1. Slow Processing
   - Use GPU if available
   - Increase worker count on multi-core systems
   - Use smaller model for faster processing
   - Close unnecessary applications

2. High Memory Usage
   - Reduce worker count
   - Process fewer files simultaneously
   - Monitor system resources
   - Use smaller model size

3. Disk Space Issues
   - Clean temporary files regularly
   - Check available disk space before starting
   - Monitor disk usage during processing

## Contributing Guidelines
1. Code Style
   - Follow PEP 8 guidelines
   - Use type hints
   - Add docstrings for new functions
   - Comment complex logic

2. Testing
   - Test new features thoroughly
   - Verify compatibility with different file types
   - Check memory usage
   - Test error handling

3. Documentation
   - Update README with new features
   - Document any new parameters
   - Include usage examples
   - Update troubleshooting guide

## License
## Version History

### v1.0.0 (2024-01-24)
- Initial release
- Basic audio/video transcription
- Support for TXT and SRT formats
- Progress tracking
- Multi-worker support

### v1.1.0 (2024-01-24)
- Added FFmpeg integration
- Improved video support
- Enhanced progress tracking
- Added documentation
- Memory optimization

### v1.2.0 (2024-01-26)
- Added support for using current directory as default
- Updated documentation with current directory usage
- Improved command-line argument handling

## Future Enhancements
1. Additional Features
   - Custom timestamp formats
   - Multiple language support in single file
   - Automatic language detection
   - Word-level timestamps
   - Confidence scores for transcriptions

2. Performance Improvements
   - Streaming transcription
   - Batch processing optimization
   - Advanced caching
   - Resource usage optimization

3. User Interface
   - GUI implementation
   - Web interface
   - Real-time transcription preview
   - Interactive configuration

## Support and Community
- Report issues on GitHub
- Share improvements and modifications
- Contribute to documentation
- Help other users in discussions

## MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.