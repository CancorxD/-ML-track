import json
from pathlib import Path
from typing import List, Dict, Any
from sd_pipeline import generate_frame




def save_info(shot_list: List[Dict[str, Any]]) -> str:
    description_path = Path("storyboard") / "index.json"
    description_path.parent.mkdir(parents=True, exist_ok=True)


    with description_path.open("w", encoding="utf-8") as f:
        json.dump(shot_list, f, ensure_ascii=False, indent=4)


    for shot in shot_list:
        frame_num = shot.get("frame")
        desc = shot.get("description")
        if frame_num is None or desc is None:
            print("Skipping shot (missing 'frame' or 'description'): ", shot)
            continue


    prompt = (
    f"{desc}, Aldar Köse, middle-aged man, short bushy beard, "
    "cunning smile, wearing colorful Kazakh robe and feathered hat, "
    "consistent style, cinematic lighting, ultra-realistic, detailed face, detailed background"
    )


    seed = 42 + int(frame_num)
    try:
        image = generate_frame(prompt, seed=seed)
        image_path = description_path.parent / f"frame_{frame_num}.png"
        image.save(image_path)
        print("Saved", image_path)
    except Exception as e:
        print(f"Failed to generate/save frame {frame_num}: {e}")


    return f"Descriptions saved to: {description_path}"