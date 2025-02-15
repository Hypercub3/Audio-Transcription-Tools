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