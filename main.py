"""
Entry point. Orchestrates prompt input, LLM shot-list generation and image creation.
"""
from scripts.input_handler import get_prompt
from scripts.llm import generate_shot_list
from scripts.io_utils import save_info
from scripts.fallback import FALLBACK_SHOT_LIST




def main() -> None:
    script = get_prompt("Введите сценарий (2–4 предложения, описывающие историю с Aldar Köse): ")
    print("Введённый сценарий:", script)


    try:
        shot_list = generate_shot_list(script)
    except Exception as e:
        print("Falling back to predefined shots due to error:", e)
        shot_list = FALLBACK_SHOT_LIST


    result_msg = save_info(shot_list)
    print(result_msg)


if __name__ == "__main__":
    main()

