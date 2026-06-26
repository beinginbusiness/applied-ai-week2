import tiktoken
import json

# tiktoken's encodings are technically OpenAI's, but they're close enough
# to Gemini/Claude's tokenization to demonstrate the concept accurately
encoding = tiktoken.get_encoding("cl100k_base")

with open("data/transcripts.json", "r") as f:
    transcripts = json.load(f)

print(f"{'ID':<6}{'Characters':<12}{'Tokens':<10}")
print("-" * 30)

total_tokens = 0
for t in transcripts:
    text = t["transcript"]
    tokens = encoding.encode(text)
    total_tokens += len(tokens)
    print(f"{t['id']:<6}{len(text):<12}{len(tokens):<10}")

print("-" * 30)
print(f"Total tokens across all 6 transcripts: {total_tokens}")

# Show what tokens actually look like for one example
print("\nBreaking down T002 into its actual tokens:")
sample = transcripts[1]["transcript"]
token_ids = encoding.encode(sample)
for tid in token_ids[:15]:  # just the first 15, full list would be long
    piece = encoding.decode([tid])
    print(f"  Token ID {tid}: '{piece}'")