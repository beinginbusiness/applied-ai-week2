import os
import json
import re
import time
from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def parse_response(text):
    result = {}
    for field in ["Sentiment", "Intent", "Urgency", "Injection_detected"]:
        match = re.search(rf"{field}:\s*(\w+)", text, re.IGNORECASE)
        result[field.lower()] = match.group(1).lower() if match else None
    return result


def call_model(provider, filled_prompt):
    if provider == "gemini":
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=filled_prompt
        )
        return response.text
    elif provider == "groq":
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": filled_prompt}]
        )
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_evaluation(prompt_file, provider="gemini", transcripts_file="data/transcripts.json"):
    with open(prompt_file, "r") as f:
        prompt_template = f.read()
    with open(transcripts_file, "r") as f:
        transcripts = json.load(f)

    correct_fields = 0
    total_fields = 0
    results = []

    for t in transcripts:
        filled_prompt = prompt_template.replace("{transcript}", t["transcript"])

        response_text = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response_text = call_model(provider, filled_prompt)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  ⚠️ {t['id']}: API error, retrying in {wait_time}s... ({e})")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ {t['id']}: failed after {max_retries} attempts, skipping")

        if response_text is None:
            continue

        predicted = parse_response(response_text)

        row = {"id": t["id"], "predicted": predicted, "actual": {
            "sentiment": t["true_sentiment"],
            "intent": t["true_intent"],
            "urgency": t["true_urgency"]
        }}

        for field in ["sentiment", "intent", "urgency"]:
            if predicted.get("injection_detected") == "true":
                continue  # don't penalize/score fields we deliberately overrode
            total_fields += 1
            if predicted.get(field) == row["actual"][field]:
                correct_fields += 1
            else:
                row[f"{field}_mismatch"] = True

        results.append(row)

    accuracy = correct_fields / total_fields * 100 if total_fields else 0
    print(f"\n{'=' * 50}")
    print(f"Prompt: {prompt_file} | Provider: {provider}")
    print(f"Accuracy: {correct_fields}/{total_fields} fields correct ({accuracy:.1f}%)")
    print(f"{'=' * 50}\n")

    for row in results:
        mismatches = [k for k in row if k.endswith("_mismatch")]
        flag = " ⚠️ MISMATCH" if mismatches else ""
        print(f"{row['id']}{flag}: predicted={row['predicted']} | actual={row['actual']}")

    return accuracy


if __name__ == "__main__":
    print("\n\n### TESTING v1 (zero-shot) ###")
    run_evaluation("prompts/v1_zero_shot.txt", provider="groq")

    print("\n\n### TESTING v4 (few-shot) ###")
    run_evaluation("prompts/v4_few_shot.txt", provider="groq")

    print("\n\n### TESTING v5 (chain-of-thought) ###")
    run_evaluation("prompts/v5_chain_of_thought.txt", provider="groq")