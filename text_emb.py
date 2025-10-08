import argparse
import math
from pathlib import Path
import numpy as np
import json
import soundfile as sf
from sentence_transformers import SentenceTransformer

DEFAULT_FILENAME = "text_emb_2s.npz"

def resolve_output_path(destination: str) -> Path:
    dest_path = Path(destination)
    if dest_path.is_dir() or destination.endswith(("/", "\\")):
        return dest_path / DEFAULT_FILENAME
    return Path(dest_path)

def extract_and_save_text_features(
    transcript_path: Path,
    output_path: Path,
    chunk_duration=2.0,
    context_window=20.0
):
    """Extract text features (current+context embeddings) and save as .npz with timestamps."""
    
    # Load transcript
    with open(transcript_path) as f:
        transcript = json.load(f)
    
    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    segments = transcript['segments']

    # 🔑 Use wav file in parent folder if available
    wav_path = transcript_path.parent.parent / (transcript_path.stem + ".wav")
    if wav_path.exists():
        f = sf.SoundFile(wav_path)
        duration = len(f) / f.samplerate
    else:
        duration = segments[-1]['end']

    num_chunks = math.floor(duration / chunk_duration)
    
    text_features = []
    timestamps = []
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_duration
        chunk_end = chunk_start + chunk_duration
        timestamps.append((chunk_start, chunk_end))
        
        # Current text (segments with majority overlap)
        current_text = []
        for seg in segments:
            seg_start_chunk = int(seg['start'] // chunk_duration)
            seg_end_chunk = int(seg['end'] // chunk_duration)
            
            if seg_start_chunk == chunk_idx or (
                seg_end_chunk == chunk_idx and 
                seg['end'] - chunk_start > seg['start'] - (seg_start_chunk * chunk_duration)
            ):
                current_text.append(seg['text'])
        
        # Context text (all overlapping withinwindow)
        context_start = max(0, chunk_start - context_window)
        context_end = chunk_end + context_window
        context_text = []
        for seg in segments:
            if seg['start'] < context_end and seg['end'] > context_start:
                context_text.append(seg['text'])
        
        # Embed (handle empty text)
        current_embed = (model.encode(' '.join(current_text)) 
                        if current_text else np.zeros(384, dtype=np.float32))
        context_embed = (model.encode(' '.join(context_text)) 
                        if context_text else np.zeros(384, dtype=np.float32))
        
        # Concatenate: [current (384) + context (384)] = 768 dims
        text_features.append(np.concatenate([current_embed, context_embed]))
    
    # Convert to arrays
    text_features = np.array(text_features, dtype=np.float32)
    timestamps = np.array(timestamps, dtype=np.float32)
    
    # Save as .npz
    np.savez(output_path, embeddings=text_features, timestamps=timestamps)
    print(f"✅ Saved text features to {output_path} "
          f"(emb {text_features.shape}, ts {timestamps.shape})")
    
    return text_features, timestamps

def parse_args():
    parser = argparse.ArgumentParser(description="Extract 2s text embeddings with context (+ timestamps)")
    parser.add_argument("source", help="Path to transcript JSON file")
    parser.add_argument("destination", help="Output file or directory for the embeddings")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {source_path}")

    output_path = resolve_output_path(args.destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extract_and_save_text_features(source_path, output_path)
 
