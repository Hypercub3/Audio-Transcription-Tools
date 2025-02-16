#!/usr/bin/env python3

"""
Whisper Media Trimmer - A tool to remove non-speech segments from media files.

This script uses OpenAI's Whisper model to detect speech in audio/video files
and creates a new file containing only the speech segments with configurable
padding and merge thresholds.

Dependencies:
    - whisper
    - pydub
    - numpy
    - torch
    - tqdm
    - ffmpeg (system dependency)

Usage:
    python whisper-media-trimmer.py input_dir [options]

Options:
    --model: Whisper model size (tiny/base/small/medium/large)
    --padding: Milliseconds of audio to keep around speech segments
    --merge-threshold: Maximum gap between words to be considered same segment
    --workers: Number of parallel workers
    --language: ISO language code (e.g., 'en' for English)
"""

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

# Set up logging with timestamp and level
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Define supported media formats
SUPPORTED_FORMATS = {
    # Audio formats supported by ffmpeg
    'audio': {
        '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.aiff',
        '.alac', '.ac3', '.amr', '.ape', '.au', '.mka', '.pcm', '.ra',
        '.wv', '.opus'
    },
    # Video formats supported by ffmpeg
    'video': {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
        '.mpg', '.mpeg', '.3gp', '.3g2', '.ogv', '.ts', '.mts', '.m2ts'
    }
}

def get_media_type(file_path):
    """
    Determine if the file is audio or video based on its extension.
    
    Args:
        file_path (str or Path): Path to the media file
        
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
        file_path (str or Path): Path to the media file
        
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
    Converts to WAV format with parameters optimized for Whisper.
    
    Args:
        file_path (str or Path): Path to input video file
        output_path (str or Path): Path for output audio file
        
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [
        'ffmpeg', '-i', str(file_path),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # Convert to WAV
        '-ar', '16000',  # 16kHz sample rate (required by Whisper)
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

def get_speech_segments(model, audio_path, merge_threshold_ms=1000, language=None):
    """
    Use Whisper to detect speech segments in the audio, with improved handling
    of overlapping segments and proper chronological ordering.
    
    Args:
        model: Loaded Whisper model
        audio_path (str or Path): Path to audio file
        merge_threshold_ms (int): Maximum gap between words to be considered same segment
        language (str, optional): ISO language code for transcription
        
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
            elif start_time <= current_segment["end"] + merge_threshold_ms:
                # Word starts before or within threshold of current segment
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

def extract_segments(input_path, output_path, segments, padding_ms=250):
    """
    Extract and concatenate segments from any media file using FFmpeg.
    Maintains original quality by using stream copy when possible.
    
    Args:
        input_path (str or Path): Path to input media file
        output_path (str or Path): Path for output media file
        segments (list): List of speech segments with start/end times
        padding_ms (int): Padding in milliseconds around segments
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not segments:
        logging.warning(f"No speech segments found in {input_path}")
        return False

    try:
        # Create temporary directory for segment files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create concat file for FFmpeg
            concat_file = Path(temp_dir) / "concat.txt"
            segment_files = []
            
            # Process each segment
            for i, segment in enumerate(segments):
                # Calculate times with padding
                start = max(0, segment["start"] - padding_ms) / 1000
                end = min(get_media_duration(input_path) * 1000, 
                         segment["end"] + padding_ms) / 1000
                duration = end - start
                
                # Output segment file
                segment_file = Path(temp_dir) / f"segment_{i}{Path(input_path).suffix}"
                segment_files.append(segment_file)
                
                # Extract segment with original quality
                cmd = [
                    'ffmpeg', '-i', str(input_path),
                    '-ss', str(start),
                    '-t', str(duration),
                    '-c', 'copy',  # Copy all codecs to maintain quality
                    '-avoid_negative_ts', '1',
                    '-y', str(segment_file)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            
            # Create concat file
            with open(concat_file, 'w') as f:
                for segment_file in segment_files:
                    f.write(f"file '{segment_file}'\n")
            
            # Concatenate segments
            cmd = [
                'ffmpeg', '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',  # Copy all codecs
                '-y', str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            return True
            
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")
        return False

def process_file(args):
    """
    Process a single media file to remove non-speech segments.
    
    Args:
        args (tuple): Contains (input_path, model, padding_ms, merge_threshold_ms, language)
        
    Returns:
        tuple: (success: bool, result: str)
    """
    input_path, model, padding_ms, merge_threshold_ms, language = args
    try:
        print(f"\nProcessing: {input_path.name}")
        # Create output path with _trimmed suffix
        output_path = input_path.parent / f"{input_path.stem}_trimmed{input_path.suffix}"
        
        # Determine media type
        media_type = get_media_type(input_path)
        if not media_type:
            print(f"✗ Skipped unsupported format: {input_path.name}")
            return False, f"Unsupported format: {input_path}"

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio = Path(temp_dir) / "temp_audio.wav"
            
            # Extract audio if video, or convert audio to WAV
            print(f"⟳ Extracting/loading audio from {input_path.name}")
            if media_type == 'video':
                if not extract_audio(input_path, temp_audio):
                    print(f"✗ Failed to extract audio from {input_path.name}")
                    return False, f"Failed to extract audio from {input_path}"
                audio_path = temp_audio
            else:
                # For audio files, create WAV for Whisper
                try:
                    cmd = [
                        'ffmpeg', '-i', str(input_path),
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',
                        '-ac', '1',
                        '-y', str(temp_audio)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    audio_path = temp_audio
                except subprocess.CalledProcessError as e:
                    print(f"✗ Failed to process audio: {input_path.name}")
                    return False, f"Failed to process audio: {str(e)}"

            # Get speech segments
            print(f"⟳ Detecting speech in {input_path.name}")
            segments = get_speech_segments(model, audio_path, merge_threshold_ms, language)
            
            if not segments:
                print(f"✗ No speech segments found in {input_path.name}")
                return False, f"No speech segments found in {input_path}"

            # Process the segments
            print(f"⟳ Trimming {input_path.name}")
            success = extract_segments(input_path, output_path, segments, padding_ms)

            if success:
                print(f"✓ Completed: {input_path.name}")
            else:
                print(f"✗ Failed: {input_path.name}")

            return success, input_path

    except Exception as e:
        print(f"✗ Error processing {input_path.name}: {str(e)}")
        return False, f"Error processing {input_path}: {str(e)}"

def find_media_files(directory):
    """
    Recursively find all supported media files in directory.
    
    Args:
        directory (str or Path): Directory to search
        
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
    parser.add_argument("input_dir", type=str, 
                        help="Input directory containing media files")
    parser.add_argument("--model", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size to use")
    parser.add_argument("--padding", type=int, default=250,
                        help="Padding in milliseconds around speech segments")
    parser.add_argument("--merge-threshold", type=int, default=1000,
                        help="Maximum gap (ms) between words to be considered same segment")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--language", type=str, default=None,
                        help="ISO 639-1 language code (e.g., 'en' for English)")
    args = parser.parse_args()
    
    # Check for ffmpeg installation
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("FFmpeg is not installed or not found in PATH. Please install FFmpeg first.")
        return

    # Load Whisper model
    logging.info(f"Loading Whisper {args.model} model...")
    try:
        model = whisper.load_model(args.model)
    except Exception as e:
        logging.error(f"Error loading Whisper model: {str(e)}")
        return
    
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
        # Create process arguments for each file
        process_args = [(f, model, args.padding, args.merge_threshold, args.language) 
                       for f in media_files]
        
        # Process files and show progress
        for success, result in tqdm(
            executor.map(process_file, process_args),
            total=len(media_files),
            desc="Processing files"
        ):
            if success:
                success_count += 1
            else:
                logging.error(result)
    
    # Show final summary
    logging.info(f"Processing complete: {success_count}/{len(media_files)} files successfully processed")

if __name__ == "__main__":
    main()