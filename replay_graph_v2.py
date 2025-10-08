import json
import os
import subprocess
import sys
import math

def process_video(video_id: str):
    # Step 1: Run the yt-most-replayed script
    repo_path = os.path.abspath("yt-most-replayed")
    cmd = ["uv", "run", "main.py", "--video_id", video_id]
    subprocess.run(cmd, cwd=repo_path, check=True)

    # Step 2: Locate the input JSON file
    input_json = os.path.join(repo_path, "output", "info", f"{video_id}.json")
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Could not find {input_json}")

    with open(input_json, "r") as f:
        data = json.load(f)

    if "heatmap" not in data:
        raise KeyError(f"No 'heatmap' key in {input_json}")

    heatmap = data["heatmap"]

    # Step 3: Expand into 2-second intervals
    expanded = []
    for segment in heatmap:
        start, end, value = segment["start_time"], segment["end_time"], segment["value"]
        t = math.floor(start / 2) * 2  # align to nearest multiple of 2
        while t < end:
            expanded.append({
                "time": round(t, 2),
                "intensity": round(value, 2)
            })
            t += 2

    # Step 4: Save to <videoID>/heatmap.json (outside repo)
    output_dir = os.path.abspath(video_id)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "heatmap.json")

    with open(output_file, "w") as f:
        json.dump(expanded, f, indent=2)

    print(f"✅ Saved processed heatmap to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_heatmap.py <videoID>")
        sys.exit(1)

    video_id = sys.argv[1]
    process_video(video_id)

