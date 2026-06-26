import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

prompt = "Write one creative sentence describing a rainy day."

for temp in [0.0, 1.0, 2.0]:
    print(f"\n{'='*50}")
    print(f"TEMPERATURE = {temp}")
    print(f"{'='*50}")
    for run in range(3):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        print(f"Run {run+1}: {response.choices[0].message.content.strip()}")