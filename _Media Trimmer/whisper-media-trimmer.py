import whisper
from pydub import AudioSegment
import numpy as np
import torch
import argparse
from pathlib import Path
import concurrent.futures
import logging
import subprocess
import json
import tempfile
import shutil
import os
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Supported media formats
SUPPORTED_FORMATS = {
    # Audio formats
    'audio': {
        '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.aiff',
        '.alac', '.ac3', '.amr', '.ape', '.au', '.mka', '.pcm', '.ra',
        '.wv', '.opus'
    },
    # Video formats
    'video': {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
        '.mpg', '.mpeg', '.3gp', '.3g2', '.ogv', '.ts', '.mts', '.m2ts'
    }
}

def get_media_type(file_path):
    """Determine if the file is audio or video."""
    suffix = Path(file_path).suffix.lower()
    if suffix in SUPPORTED_FORMATS['audio']:
        return 'audio'
    elif suffix in SUPPORTED_FORMATS['video']:
        return 'video'
    return None

def get_media_duration(file_path):
    """Get duration of media file using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', str(file_path)
    ]
    try:
        output = subprocess.check_output(cmd)
        data = json.loads(output)
        return float(data['format']['duration'])
    except Exception as e:
        logging.error(f"Error getting duration for {file_path}: {str(e)}")
        return None

def extract_audio(file_path, output_path):
    """Extract audio from video file."""
    cmd = [
        'ffmpeg', '-i', str(file_path),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # Convert to WAV
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',  # Mono
        '-y',  # Overwrite output
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        return False

def trim_video(input_path, output_path, segments, padding_ms=500):
    """Trim video keeping only speech segments."""
    if not segments:
        logging.warning(f"No speech segments found in {input_path}")
        return False

    # Create a temporary file for the filter complex
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        filter_parts = []
        for i, segment in enumerate(segments):
            start = max(0, segment["start"] - padding_ms) / 1000
            duration = (min(get_media_duration(input_path) * 1000, 
                          segment["end"] + padding_ms) - 
                      max(0, segment["start"] - padding_ms)) / 1000
            filter_parts.append(f"between(t,{start},{start+duration})")
        
        filter_complex = '+'.join(filter_parts)
        f.write(filter_complex)
        filter_file = f.name

    try:
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-vf', f"select='({filter_complex})',setpts=N/FRAME_RATE/TB",
            '-af', f"aselect='({filter_complex})',asetpts=N/SR/TB",
            '-y', str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.unlink(filter_file)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        os.unlink(filter_file)
        return False

def get_speech_segments(model, audio_path, language=None):
    """Use Whisper to detect speech segments in the audio."""
    try:
        result = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language=language if language else None
        )
        
        segments = []
        current_segment = None
        
        for segment in result["segments"]:
            for word in segment["words"]:
                start_time = word["start"] * 1000
                end_time = word["end"] * 1000
                
                if current_segment is None:
                    current_segment = {"start": start_time, "end": end_time}
                elif end_time - current_segment["end"] > 1000:
                    segments.append(current_segment)
                    current_segment = {"start": start_time, "end": end_time}
                else:
                    current_segment["end"] = end_time
        
        if current_segment is not None:
            segments.append(current_segment)
        
        return segments
    except Exception as e:
        logging.error(f"Error processing {audio_path}: {str(e)}")
        return []

def process_file(args):
    """Process a single media file."""
    input_path, model, padding_ms, language, _ = args  # Progress parameter is ignored
    try:
        print(f"\nProcessing: {input_path.name}")
        # Create output path
        output_path = input_path.parent / f"{input_path.stem}_trimmed{input_path.suffix}"
        
        # Determine media type
        media_type = get_media_type(input_path)
        if not media_type:
            print(f"✗ Skipped unsupported format: {input_path.name}")
            return False, f"Unsupported format: {input_path}"

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio = Path(temp_dir) / "temp_audio.wav"
            
            # Extract audio if it's a video, or load audio directly
            print(f"⟳ Extracting/loading audio from {input_path.name}")
            if media_type == 'video':
                if not extract_audio(input_path, temp_audio):
                    print(f"✗ Failed to extract audio from {input_path.name}")
                    return False, f"Failed to extract audio from {input_path}"
                audio_path = temp_audio
            else:
                audio_path = input_path

            print(f"⟳ Detecting speech in {input_path.name}")
            segments = get_speech_segments(model, audio_path, language)
            
            if not segments:
                print(f"✗ No speech segments found in {input_path.name}")
                return False, f"No speech segments found in {input_path}"

            print(f"⟳ Trimming {input_path.name}")
            # Process based on media type
            if media_type == 'video':
                success = trim_video(input_path, output_path, segments, padding_ms)
            else:
                # For audio files
                audio = AudioSegment.from_file(input_path)
                output_audio = AudioSegment.empty()
                
                for segment in segments:
                    start = max(0, segment["start"] - padding_ms)
                    end = min(len(audio), segment["end"] + padding_ms)
                    output_audio += audio[start:end]
                
                output_audio.export(output_path, format=output_path.suffix[1:])
                success = True

            if success:
                print(f"✓ Completed: {input_path.name}")
            else:
                print(f"✗ Failed: {input_path.name}")

            return success, input_path

    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {str(e)}")
        return False, f"Error processing {input_path}: {str(e)}"

def find_media_files(directory):
    """Recursively find all supported media files in directory."""
    media_files = []
    skipped_files = []
    
    for path in Path(directory).rglob('*'):
        if path.is_dir() or path.name.startswith('.'):
            continue
            
        media_type = get_media_type(path)
        if media_type:
            if path.stat().st_size > 0:
                media_files.append(path)
            else:
                skipped_files.append((path, "Empty file"))
        elif path.suffix.lower():
            skipped_files.append((path, "Unsupported format"))
    
    if skipped_files:
        logging.info("Skipped files:")
        for file, reason in skipped_files:
            logging.info(f"  {file}: {reason}")
    
    return media_files

def main():
    parser = argparse.ArgumentParser(description="Remove non-speech parts from media files")
    parser.add_argument("input_dir", type=str, help="Input directory containing media files")
    parser.add_argument("--model", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size to use")
    parser.add_argument("--padding", type=int, default=500,
                        help="Padding in milliseconds around speech segments")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--language", type=str, default=None,
                        help="ISO 639-1 language code (e.g., 'en' for English)")
    args = parser.parse_args()
    
    # Check for ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("FFmpeg is not installed or not found in PATH. Please install FFmpeg first.")
        return

    # Load Whisper model
    logging.info(f"Loading Whisper {args.model} model...")
    model = whisper.load_model(args.model)
    
    # Find all media files
    input_dir = Path(args.input_dir)
    media_files = find_media_files(input_dir)
    
    if not media_files:
        logging.error(f"No supported media files found in {input_dir}")
        return
    
    logging.info(f"Found {len(media_files)} media files to process")
    print(f"\nStarting processing of {len(media_files)} files...")
    
    # Process files with progress tracking
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Create process arguments
        process_args = [(f, model, args.padding, args.language, None) for f in media_files]
        
        # Process files
        for i, (success, result) in enumerate(executor.map(process_file, process_args), 1):
            if success:
                success_count += 1
            else:
                logging.error(result)
            print(f"\nProgress: {i}/{len(media_files)} files processed")
    
    print(f"\nProcessing complete: {success_count}/{len(media_files)} files successfully processed")

if __name__ == "__main__":
    main()