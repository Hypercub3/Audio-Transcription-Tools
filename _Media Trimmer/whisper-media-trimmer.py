import whisper
from pydub import AudioSegment
import numpy as np
import torch
import argparse
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import logging
import subprocess
import json
import tempfile
import shutil
import os

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
    """
    Determine if the file is audio or video based on its extension.
    
    Args:
        file_path: Path to the media file
        
    Returns:
        str: 'audio', 'video', or None if unsupported
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in SUPPORTED_FORMATS['audio']:
        return 'audio'
    elif suffix in SUPPORTED_FORMATS['video']:
        return 'video'
    return None

def get_media_duration(file_path):
    """
    Get duration of media file using ffprobe.
    
    Args:
        file_path: Path to the media file
        
    Returns:
        float: Duration in seconds, or None if error
    """
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
    """
    Extract audio from video file using FFmpeg.
    
    Args:
        file_path: Path to input video file
        output_path: Path for output audio file
        
    Returns:
        bool: True if successful, False otherwise
    """
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

def get_speech_segments(model, audio_path, language=None):
    """
    Use Whisper to detect speech segments in the audio, with improved handling
    of overlapping segments and proper chronological ordering.
    
    Args:
        model: Loaded Whisper model
        audio_path: Path to audio file
        language: Optional ISO language code
        
    Returns:
        list: List of dicts containing start and end times in milliseconds
    """
    try:
        # Transcribe audio with word-level timestamps
        result = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language=language if language else None
        )
        
        # Extract and flatten all words from all segments
        all_words = []
        for segment in result["segments"]:
            all_words.extend(segment["words"])
        
        # Sort words chronologically by start time
        all_words.sort(key=lambda x: x["start"])
        
        segments = []
        current_segment = None
        
        # Process words in chronological order
        for word in all_words:
            # Convert times to milliseconds
            start_time = word["start"] * 1000
            end_time = word["end"] * 1000
            
            if current_segment is None:
                # Start first segment
                current_segment = {"start": start_time, "end": end_time}
            elif start_time <= current_segment["end"] + 1000:
                # Word starts before or within 1 second of current segment
                # Extend current segment
                current_segment["end"] = max(current_segment["end"], end_time)
            else:
                # Gap is too large, start new segment
                segments.append(current_segment)
                current_segment = {"start": start_time, "end": end_time}
        
        # Add the final segment
        if current_segment is not None:
            segments.append(current_segment)
        
        # Merge any remaining overlapping segments
        if segments:
            merged = [segments[0]]
            for segment in segments[1:]:
                if segment["start"] <= merged[-1]["end"]:
                    # Merge overlapping segments
                    merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
                else:
                    # No overlap, add as new segment
                    merged.append(segment)
            segments = merged
        
        return segments
    except Exception as e:
        logging.error(f"Error processing {audio_path}: {str(e)}")
        return []

def trim_video(input_path, output_path, segments, padding_ms=250):
    """
    Trim video keeping only speech segments using FFmpeg.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        segments: List of speech segments
        padding_ms: Padding in milliseconds around segments
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not segments:
        logging.warning(f"No speech segments found in {input_path}")
        return False

    # Create a temporary file for the filter complex
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        filter_parts = []
        for i, segment in enumerate(segments):
            # Add padding while respecting file boundaries
            start = max(0, segment["start"] - padding_ms) / 1000
            duration = (min(get_media_duration(input_path) * 1000, 
                          segment["end"] + padding_ms) - 
                      max(0, segment["start"] - padding_ms)) / 1000
            filter_parts.append(f"between(t,{start},{start+duration})")
        
        filter_complex = '+'.join(filter_parts)
        f.write(filter_complex)
        filter_file = f.name

    try:
        # Apply the filter complex using FFmpeg
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

def process_file(args):
    """
    Process a single media file to remove non-speech segments.
    
    Args:
        args: Tuple containing (input_path, model, padding_ms, language)
        
    Returns:
        tuple: (success: bool, result: str)
    """
    input_path, model, padding_ms, language = args
    try:
        # Create output path
        output_path = input_path.parent / f"{input_path.stem}_trimmed{input_path.suffix}"
        
        # Determine media type
        media_type = get_media_type(input_path)
        if not media_type:
            return False, f"Unsupported format: {input_path}"

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio = Path(temp_dir) / "temp_audio.wav"
            
            # Extract audio if video, or use audio directly
            if media_type == 'video':
                if not extract_audio(input_path, temp_audio):
                    return False, f"Failed to extract audio from {input_path}"
                audio_path = temp_audio
            else:
                audio_path = input_path

            # Get speech segments
            segments = get_speech_segments(model, audio_path, language)
            
            if not segments:
                return False, f"No speech segments found in {input_path}"

            # Process based on media type
            if media_type == 'video':
                success = trim_video(input_path, output_path, segments, padding_ms)
            else:
                # For audio files, use pydub
                audio = AudioSegment.from_file(input_path)
                output_audio = AudioSegment.empty()
                
                for segment in segments:
                    start = max(0, segment["start"] - padding_ms)
                    end = min(len(audio), segment["end"] + padding_ms)
                    output_audio += audio[start:end]
                
                output_audio.export(output_path, format=output_path.suffix[1:])
                success = True

            return success, input_path

    except Exception as e:
        return False, f"Error processing {input_path}: {str(e)}"

def find_media_files(directory):
    """
    Recursively find all supported media files in directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        list: List of Path objects for supported media files
    """
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
    """Main function to handle command line arguments and process files."""
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
    
    # Process files with progress bar
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        process_args = [(f, model, args.padding, args.language) for f in media_files]
        
        for success, result in tqdm(
            executor.map(process_file, process_args),
            total=len(media_files),
            desc="Processing files"
        ):
            if success:
                success_count += 1
            else:
                logging.error(result)
    
    logging.info(f"Processing complete: {success_count}/{len(media_files)} files successfully processed")

if __name__ == "__main__":
    main()