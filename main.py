import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import openai
import json
import os
from openai import OpenAI
import openai


#Constraints!
reformatted_shot_list = [
    {'frame': 1, 'description': "Aldar Köse approaches the greedy merchant's stall with a cunning smile on his face. Aldar Köse, dressed in traditional Kazakh clothing with a feathered hat, waltzes up to a bustling marketplace. His cunning smile is framed by his bushy beard. The camera angle is from a low position, making Aldar appear larger than life."},
    {'frame': 2, 'description': 'Aldar Köse and the merchant engage in a lively conversation, Aldar pointing to a horse. Aldar gestures excitedly to a sturdy horse tied up nearby. The merchant, a portly man draped in luxurious robes, looks intrigued yet cautious. The camera captures their exchange from a middle-long shot angle.'},
    {'frame': 3, 'description': "Aldar Köse hatches his plan and begins to spin his elaborate scheme to the merchant. The scene is shot from the merchant's perspective. Aldar leans in, speaking in hushed tones. His eyes sparkle with a hint of mischief."},
    {'frame': 4, 'description': "The merchant seems skeptical, but Aldar continues his persuasive tactics. The camera switches to a high angle shot focusing on Aldar, capturing the merchant's hesitant expressions and Aldar's determined, cunning face."},
    {'frame': 5, 'description': "Winning over the merchant, Aldar Köse shakes hands with him to finalize the deal. A close shot of their handshake, Aldar's grin is even wider, but the merchant still appears unsure. The transaction is happening under the bright midday sun, casting vibrant colors over the bustling marketplace."},
    {'frame': 6, 'description': "Aldar takes the reins of the horse and mounts it, waving the merchant farewell. The camera transitions to a long shot as Aldar hops onto the horse. Aldar's vibrant traditional clothes stand out against the dusty marketplace ground."},
    {'frame': 7, 'description': 'Aldar rides away from the merchant, still laughing and waving. Captured from a low angle, Aldar looks triumphant as he rides away on his newly acquired horse. His laughter can be visually perceived through his wide-open mouth and uplifted head.'},
    {'frame': 8, 'description': "The merchant, left in Aldar's dust, realizes his mistake. Shot from Aldar's perspective, the merchant stands alone in the marketplace, a shocked expression on his face as he grasps the emptiness of his mistake. The surrounding crowd seem oblivious, carrying on with their trading."}
]


openai.api_key = ""  #paste your OpenAI API key to Colab Secrets (name: OPENAI_API_KEY)
try:
    client = OpenAI(api_key=openai.api_key)
except Exception as e:
    print("Fuck OpenAI! Reason: ", e)

#Generating the scene descriptions using the prompt in ChatGPT OpenAI
def generate_shot_list(script):
    prompt = f"""
    Given the following script: "{script}"
    Generate a list of 6–10 detailed scene descriptions for a storyboard. Each scene should include:
    - A short description of the action.
    - Visual details (Aldar Köse's appearance, setting, camera angle).
    - Ensure Aldar Köse is recognizable (e.g., wears traditional Kazakh clothing, has a cunning smile).
    Return the result as a JSON list with fields 'frame' and 'description'.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(response.choices[0].message.content)


#Generating frames for.... winning HackNU!
def generate_frame(prompt, seed=42):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt,
        num_inference_steps=50,
        generator=generator,
        guidance_scale=7.5
    ).images[0]
    return image


if __name__ == "__main__":
    script = input("Введите сценарий (2–4 предложения, описывающие историю с Aldar Köse): ")
    print(f"Введённый сценарий: {script}")

    shot_list = generate_shot_list(script)
    print("Описания кадров:")
    print(shot_list)


    os.makedirs("storyboard", exist_ok=True)
    with open("storyboard/index.json", "w") as f:
        json.dump(shot_list, f, indent=4)
    print("Описания сохранены в storyboard/index.json")


    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16  # Используем torch_dtype вместо dtype
    )
    pipe = pipe.to("cuda")  # Переносим на GPU

    


    

    os.makedirs("storyboard", exist_ok=True)
    for shot in reformatted_shot_list:
        frame_num = shot["frame"]
        prompt = f"{shot['description']}, Aldar Köse, middle-aged man, short bushy beard, cunning smile, wearing colorful Kazakh robe and feathered hat, consistent style, cinematic lighting, ultra-realistic, detailed face, detailed background"
        image = generate_frame(prompt, seed=42)
        image.save(f"storyboard/frame_{frame_num}.png")









