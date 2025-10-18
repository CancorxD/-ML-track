"""
Module responsible for talking to OpenAI and parsing the JSON shot list.
"""
import os
import json
from typing import List, Dict, Any
from openai import OpenAI


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the environment. Do NOT hardcode keys in source files.")


_client = OpenAI(api_key=OPENAI_API_KEY)




def generate_shot_list(script: str) -> List[Dict[str, Any]]:
    prompt = f"""
Given the following script: "{script}"
Generate a list of 6–10 detailed scene descriptions for a storyboard. Each scene should include:
- A short description of the action.
- Visual details (Aldar Köse's appearance, setting, camera angle).
- Ensure Aldar Köse is recognizable (e.g., wears traditional Kazakh clothing, has a cunning smile).
Return the result as a JSON list with fields 'frame' and 'description'.
    """


    resp = _client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = resp.choices[0].message.content


    try:
        shots = json.loads(raw_text)
        if not isinstance(shots, list):
            raise ValueError("Model returned JSON but not a list.")
        return shots
    except Exception:
        # print raw output for debugging then raise to let caller handle fallback
        print("LLM output (raw):")
        print(raw_text)
        raise