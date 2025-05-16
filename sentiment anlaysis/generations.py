import time
import random
import concurrent.futures
import re
import os
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Custom exception
class RateLimitException(Exception):
    pass

# Retry logic for API call
@retry(
    retry=retry_if_exception_type(RateLimitException),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(15)
)
def call_openai_api(client, prompt: str, n: int = 1):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            n=10,
            max_tokens=250,
            temperature=1.5,
        )
        return response
    except Exception as e:
        if "rate limit" in str(e).lower() or "429" in str(e).lower():
            print("Rate limit encountered, backing off...")
            time.sleep(random.uniform(1, 3))
            raise RateLimitException()
        else:
            raise

def analyze_sentiment(client, neg_review: str) -> str:
    prompt =  (
        f"The following is a negative customer review of a product:\n\n"
        f"---\n{neg_review}\n---\n\n"
        f"Summarize the *sentiment* of this review in 1–2 sentences. "
        f"Focus on the emotional tone and key reasons for dissatisfaction. "
        f"Do not paraphrase the review. Do not use more words than the original review. "
        f"Make it concise and abstracted (e.g., 'frustrated by poor quality and misleading description')."
    )
    response = call_openai_api(client, prompt)
    return response.choices[0].message.content.strip(), prompt

def generate_diverse_negatives(client, sentiment_summary: str) -> List[str]:
    prompt = (
        f"Based on the following sentiment summary:\n\n"
        f"\"{sentiment_summary}\"\n\n"
        f"Generate 5 **diverse** and **realistic** negative reviews for the same product. "
        f"Each review should reflect a different personal situation, tone, or set of complaints, "
        f"while staying consistent with the overall sentiment above.\n\n"
        f"Do NOT paraphrase or copy from the original input. Be original and emotionally expressive.\n\n"
        f"Return the reviews prefixed with <rev1> to <rev5>:"
    )
    response = call_openai_api(client, prompt)
    content = response.choices[0].message.content

    reviews = []
    for i in range(1, 6):
        match = re.search(rf"<rev{i}>[:\-]?\s*(.+?)(?=\n<rev{i+1}>|\Z)", content, re.DOTALL | re.IGNORECASE)
        if match:
            reviews.append(match.group(1).strip())
        else:
            reviews.append(f"[Missing <rev{i}> or malformed format]")

    return reviews

def process_neg_review(client, idx: int, neg_review: str):
    try:
        summary, prompt_summary = analyze_sentiment(client, neg_review)
        generated_reviews = generate_diverse_negatives(client, summary)
        return {
            "original_review": neg_review,
            "sentiment_summary": summary,
            "prompt_used": prompt_summary,
            "generated_reviews": generated_reviews
        }
    except Exception as e:
        print(f"[Index {idx}] Error: {e}")
        return None

def generate_diverse_negative_reviews(client, neg_reviews: List[str], max_concurrent: int = 5):
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [
            executor.submit(process_neg_review, client, idx, review)
            for idx, review in enumerate(neg_reviews)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                all_results.append(result)
    return all_results
