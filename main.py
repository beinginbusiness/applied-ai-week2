import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

with open("prompts/v1_zero_shot.txt", "r") as f:
    prompt_template = f.read()

with open("data/transcripts.json", "r") as f:
    transcripts = json.load(f)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for t in transcripts:
    filled_prompt = prompt_template.replace("{transcript}", t["transcript"])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=filled_prompt
    )

    print("=" * 60)
    print(f"Transcript {t['id']}: {t['transcript'][:60]}...")
    print()
    print("Model said:")
    print(response.text)
    print()
    print("Ground truth:")
    print(f"Sentiment: {t['true_sentiment']} | Intent: {t['true_intent']} | Urgency: {t['true_urgency']}")
    print("=" * 60)
    print()