import os
import subprocess
from pathlib import Path

def run(cmd, cwd=None):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    os.environ["YTDLP_PO_TOKEN_SERVER"] = "http://127.0.0.1:4416"
    with open("scrape.txt") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    for vid in video_ids:
        print(f"\n=== Processing {vid} ===")
        vid_dir = Path(vid)
        vid_dir.mkdir(parents=True, exist_ok=True)

        wav_download = f"{vid}.%(ext)s"
        run([
            "python3.10", "-m", "yt_dlp",
                "-f", "bestaudio",
                "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",  # the "0" must come right after this flag
            "--extractor-args", "youtube:player_client=all;po_token=web",
            "-o", wav_download,
            f"https://www.youtube.com/watch?v={vid}"
        ])

        # Step 3+4: rename and move wav
        wav_file = Path(f"{vid}.wav")
        target_wav = vid_dir / f"{vid}.wav"
        if wav_file.exists():
            wav_file.rename(target_wav)
        else:
            raise FileNotFoundError(f"Download failed: {wav_file}")

        # Step 5: WhisperX diarization
        diar_dir = vid_dir / "diarization"
        diar_dir.mkdir(parents=True, exist_ok=True)
        run([
            "python3.10", "-m", "whisperx",
            str(target_wav),
            "--model", "small",
            "--output_dir", str(diar_dir),
            "--device", "cpu",
            "--compute_type", "int8"
        ])

        # Step 6: acoustic embeddings
        run([
            "python3.10", "acoustic_emb.py",
            str(target_wav),
            str(vid_dir)
        ])

        # Step 7: DVA embeddings
        run([
            "python3.10", "dva_emb.py",
            str(target_wav),
            str(vid_dir)
        ])

        # Step 8: Text embeddings (input.json = diarization/<vidID>.json)
        transcript_json = diar_dir / f"{vid}.json"
        if transcript_json.exists():
            run([
                "python3.10", "text_emb.py",
                str(transcript_json),
                str(vid_dir)
            ])
        else:
            print(f"⚠️ Transcript JSON not found for {vid}, skipping text embeddings.")
        
        # Step 9: Heatmap
        run([
            "python3.10", "replay_graph_v2.py",
            vid
        ])
        print(f"✅ Finished {vid}")

if __name__ == "__main__":
    main()

