import pickle
with open("abstract_data/abstract_input_10000.pickle", "rb") as file:
   input_texts =  pickle.load( file)

with open("abstract_data/abstract_response_10000.pickle", "rb") as file:
   response_texts = pickle.load(file)

temp = 1.5

import openai
from openai import OpenAI

client = OpenAI(api_key="")

import time
import random
import concurrent.futures
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def extract_generated_sentences(generated_output):
    pattern = r'<gen(\d)>(.*)'
    results = {}

    for line in generated_output.strip().split('\n'):
        match = re.match(pattern, line)
        if match:
            gen_num = int(match.group(1))
            sentence = match.group(2).strip()
            results[gen_num] = sentence

    if len(results) != 3:
        missing = [i for i in range(1, 4) if i not in results]
        raise ValueError(f"Missing or malformed lines for tags: {', '.join(f'<gen{i}>' for i in missing)}")

    return [results[i] for i in range(1, 4)]

class RateLimitException(Exception):
    pass

@retry(
    retry=retry_if_exception_type(RateLimitException),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(15)
)
def call_openai_api(client, prompt, n=30):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            n=n,
            max_tokens=200,
            temperature=temp,
        )
        return response
    except Exception as e:
        error_message = str(e).lower()
        if "rate limit" in error_message or "429" in error_message:
            wait_time_match = re.search(r'try again in (\d+)ms', error_message)
            if wait_time_match:
                wait_ms = int(wait_time_match.group(1))
                wait_time = (wait_ms / 1000) + random.uniform(0.1, 0.5)
            else:
                wait_time = random.uniform(1, 3)

            print(f"Rate limit hit. Waiting for {wait_time:.2f} seconds before retry...")
            time.sleep(wait_time)
            raise RateLimitException("Rate limit exceeded")
        else:
            raise

def process_item(client, idx, sentences):
    masked_article = f"{sentences[0]} (<Missing Sentence>) {sentences[1]} (<Missing Sentence>) {sentences[2]} (<Missing Sentence>)"
    prompt = (
        f"The following text has missing sentences marked as (<Missing Sentence>) between existing ones:\n\n"
        f"{masked_article}\n\n"
        "Please fill in a coherent sentence at each (*) so that the entire paragraph reads smoothly. "
        "Return exactly 3 generated sentences, each prefixed with <gen1>, <gen2>, <gen3>, one per line."
        "Do not repeat the existing sentences."
    )

    successful_generations = []
    max_attempts = 20

    while len(successful_generations) < 30 and max_attempts > 0:
        try:
            needed = 30 - len(successful_generations)
            response = call_openai_api(client, prompt, n=min(needed + 2, 30))
            response_contents = [choice.message.content for choice in response.choices]

            for content in response_contents:
                try:
                    generated = extract_generated_sentences(content)
                    successful_generations.append(generated)
                    if len(successful_generations) >= 30:
                        break
                except ValueError:
                    continue

            max_attempts -= 1

        except Exception:
            max_attempts -= 1

    if len(successful_generations) == 30:
        return successful_generations
    else:
        print(f"Warning: Could only get {len(successful_generations)} valid generations for item {idx}")
        return successful_generations if successful_generations else None

def process_with_rate_limiting(client, input_texts, max_concurrent=5, batch_size=20):
    all_responses = []

    for batch_start in range(0, len(input_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(input_texts))
        batch = input_texts[batch_start:batch_end]

        print(f"Processing batch {batch_start // batch_size + 1}, items {batch_start} to {batch_end - 1}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [
                executor.submit(process_item, client, idx + batch_start, sentences)
                for idx, sentences in enumerate(batch)
            ]

            # Ensure order of responses matches order of input_texts
            batch_results = [future.result() for future in futures]

        all_responses.extend(batch_results)

        if batch_end < len(input_texts):
            wait_time = random.uniform(1, 3)
            time.sleep(wait_time)

    return all_responses

# Example usage (commented out)
# all_responses = process_with_rate_limiting(client, input_texts, max_concurrent=3, batch_size=10)

all_response = process_with_rate_limiting(client, input_texts[0:10])

all_response = process_with_rate_limiting(client, input_texts)


import pickle

with open("abstract_data/all_response_abstract_9000_new.pickle", "wb") as file:
    pickle.dump(all_response, file)


generations = []

for i in range(9000):
  cur = []
  for j in range(3):
    cur1 = []
    for k in range(20):
      cur1.append(all_response[i][k][j])
    cur.append(cur1)
  generations.append(cur)

from sentence_transformers import SentenceTransformer

# Load a BERT-based model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Compact and fast

from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Safe starting point; increase if GPU memory allows
max_workers = 8

def compute_similarity(i):
    try:
        res = []
        for j in range(3):
          embeddings = model.encode(
              [input_texts[i][j], input_texts[i][min(j + 1, 2)] ] + [response_texts[i][j]] + generations[i][j],
              show_progress_bar=False
          )
          res.append(cosine_similarity(embeddings))
        return i, res
    except Exception as e:
        print(f"[Error @ {i}] {e}")
        return i, None

similarities = [None] * 9000  # Preallocate

start = time.time()
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(compute_similarity, i) for i in range(9000)]
    for future in as_completed(futures):
        i, result = future.result()
        similarities[i] = result
        if i % 100 == 0:
            print(f"Processed: {i}")

print("Finished in", round(time.time() - start, 2), "sec")

import numpy as np

similarities = (np.array(similarities) + 1) / 2

import pickle

with open("abstract_data/similarities_abstract_9000_new.pickle", "wb") as file:
    pickle.dump(similarities, file)
