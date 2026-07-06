"""
ΣΤΑΔΙΟ 2: G-Eval (LLM-as-a-Judge) αξιολόγηση με το ταχύτατο Groq API (ΔΩΡΕΑΝ Tier)
-----------------------------------------------------------------------
Διαβάζει το CSV που παράγει το 01_calculate_consistency.py (στήλες
question, mean_consistency, sample_strategy, sample_indices, 1..10) και
στέλνει τις αντιπροσωπευτικές απαντήσεις κάθε ερώτησης για 3 κριτήρια:
Accuracy, Readability, Completeness.

Αλλαγές σε σχέση με την πρώτη έκδοση:
  1. ΚΡΙΣΙΜΟ: Το "llama3-70b-8192" είναι ΑΠΟΣΥΡΜΕΝΟ (decommissioned) από
     το Groq -> κάθε κλήση θα απέτυχε με 400 error. Αντικαταστάθηκε με
     "openai/gpt-oss-120b", το μοντέλο που προτείνει επίσημα το Groq ως
     διάδοχο για general-purpose/reasoning tasks, με fallback λίστα.
  2. Preflight έλεγχος: μικρή δοκιμαστική κλήση πριν ξεκινήσει όλο το
     loop, ώστε να μάθεις αμέσως αν κάτι δεν δουλεύει (λάθος key, μοντέλο
     αποσυρμένο) αντί να το ανακαλύψεις μετά από αρκετές ερωτήσεις.
  3. Σωστός διαχωρισμός σφαλμάτων: 401 (λάθος API key) και 400 (π.χ.
     αποσυρμένο μοντέλο) ΔΕΝ κάνουν retry — δεν θα διορθωθούν ποτέ έτσι.
     Μόνο πραγματικό 429 (rate limit) κάνει retry με backoff.
  4. Αυτόματο fallback σε επόμενο μοντέλο αν το πρωτεύον αποτύχει στο
     preflight έλεγχο.
"""

import csv
import os
import sys
import re
import time

from openai import OpenAI, APIStatusError, AuthenticationError, RateLimitError

# --- ΡΥΘΜΙΣΕΙΣ ---
INPUT_CSV = "chatbot_results_with_consistency.csv"
OUTPUT_CSV = "geval_results.csv"

# Μοντέλα προς δοκιμή, με τη σειρά (το πρώτο διαθέσιμο θα χρησιμοποιηθεί).
# openai/gpt-oss-120b: επίσημα προτεινόμενο από το Groq ως state-of-the-art
# reasoning μοντέλο (Οκτώβριος 2025+), αντικαθιστά τα παλιά llama3-70b/llama-3.3-70b.
CANDIDATE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]

MAX_RETRIES = 5
REQUEST_DELAY_SECONDS = 0.4  # μικρή καθυστέρηση, σεβασμός στα TPM limits

csv.field_size_limit(100_000_000)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

CRITERIA = {
    "Accuracy": """Evaluate the response based on SCIENTIFIC ACCURACY and DETECTION OF HALLUCINATIONS.
    - Score 5: Entirely accurate based on international medical guidelines (e.g. tinnitus clinical guidelines). No contradictions, no hallucinations.
    - Score 4: Mostly accurate. Minor omissions that do not pose any clinical risk.
    - Score 3: Moderately accurate. Contains ambiguous phrasing or minor technical errors.
    - Score 2: Low accuracy. Contains outdated information or borderline misleading statements.
    - Score 1: Dangerous or completely wrong. Contains clear medical hallucinations or harmful advice.""",

    "Readability": """Evaluate the response based on READABILITY for a lay patient (not a medical professional).
    - Score 5: Very clear, plain language, well-structured, easy for a patient to understand.
    - Score 4: Mostly clear, minor jargon that doesn't hinder understanding.
    - Score 3: Understandable but has notable jargon or awkward phrasing.
    - Score 2: Difficult to follow, heavy unexplained jargon or poor structure.
    - Score 1: Incomprehensible to a lay patient.""",

    "Completeness": """Evaluate the response based on COMPLETENESS relative to what a good clinical answer should cover for this question.
    - Score 5: Fully covers all clinically relevant aspects of the question, no important omission.
    - Score 4: Covers most relevant aspects, minor omission of a non-critical detail.
    - Score 3: Covers the core of the question but omits some relevant aspects.
    - Score 2: Superficial, misses several relevant aspects.
    - Score 1: Fails to actually address the question.""",
}

RATING_REGEX = re.compile(r"Rating:\s*([1-5])")


def build_prompt(question, chatbot_answer, criterion_name, rubric):
    return f"""You are an expert Otolaryngologist (ENT) evaluating a medical chatbot's response about tinnitus.

Criterion: {criterion_name}
{rubric}

Evaluation Steps:
1. Read the user's question carefully.
2. Analyze the chatbot's answer strictly with respect to the "{criterion_name}" criterion above.
3. Provide a brief 1-2 sentence justification (in English).
4. Output the final rating as a single integer between 1 and 5.

User Question: {question}
Chatbot Answer: {chatbot_answer}

Output format (exactly, no extra text, strictly follow this):
Justification: <your text>
Rating: <integer 1-5>"""


def pick_working_model():
    """Δοκιμάζει τα CANDIDATE_MODELS με τη σειρά και επιστρέφει το πρώτο που δουλεύει."""
    for model_name in CANDIDATE_MODELS:
        print(f"[INFO] Δοκιμή μοντέλου: {model_name} ...")
        try:
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Reply with only the word OK."}],
                max_tokens=5,
            )
            print(f"[INFO] Επιτυχία! Χρησιμοποιείται το: {model_name}")
            return model_name
        except AuthenticationError as e:
            print(f"[FATAL] Λάθος GROQ_API_KEY ({e}). Έλεγξε το key σου στο https://console.groq.com/keys")
            return None
        except APIStatusError as e:
            print(f"    [SKIP] '{model_name}' απέτυχε (HTTP {e.status_code}): {e.message}")
        except Exception as e:
            print(f"    [SKIP] '{model_name}' απέτυχε: {e}")
    return None


def call_geval(model_name, question, chatbot_answer, criterion_name, rubric):
    prompt = build_prompt(question, chatbot_answer, criterion_name, rubric)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Μέγιστη σταθερότητα στην κρίση
            )
            content = response.choices[0].message.content.strip()

            match = RATING_REGEX.search(content)
            rating = int(match.group(1)) if match else None

            justification = content
            if "Rating:" in content:
                justification = content.split("Rating:")[0].replace("Justification:", "").strip()

            return justification, rating

        except RateLimitError as e:
            # Πραγματικό rate limit -> αξίζει retry με exponential backoff
            wait = 3 * attempt
            print(f"    [WARN] Rate limit ({e}), retry σε {wait}s (προσπάθεια {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
        except APIStatusError as e:
            # π.χ. 400 (μοντέλο/παράμετροι), 401, 404 -> δεν διορθώνεται με retry
            print(f"    [ERROR] Μη ανακτήσιμο σφάλμα (HTTP {e.status_code}): {e.message}")
            return None, None
        except Exception as e:
            wait = 3 * attempt
            print(f"    [WARN] Σφάλμα σύνδεσης στο Groq ({e}), retry σε {wait}s...")
            time.sleep(wait)

    print("    [ERROR] Απέτυχαν όλες οι προσπάθειες — παραλείπεται.")
    return None, None


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] Δεν βρέθηκε το environment variable GROQ_API_KEY.")
        print("        cmd.exe:      set GROQ_API_KEY=gsk_...")
        print("        PowerShell:   $env:GROQ_API_KEY=\"gsk_...\"")
        return

    print("[INFO] Έλεγχος διαθέσιμων μοντέλων στο Groq...")
    model_name = pick_working_model()
    if model_name is None:
        print("[FATAL] Κανένα μοντέλο δεν είναι διαθέσιμο αυτή τη στιγμή. "
              "Έλεγξε https://console.groq.com/docs/models για ενημερωμένη λίστα.")
        return

    try:
        with open(INPUT_CSV, mode="r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"[ERROR] Δεν βρέθηκε το '{INPUT_CSV}'. Τρέξτε πρώτα το Στάδιο 1.")
        return

    results = []
    total_calls = 0

    print(f"[INFO] Έναρξη αξιολόγησης μέσω Groq ({model_name}) για {len(rows)} ερωτήσεις...")
    for idx, row in enumerate(rows, 1):
        question = row["question"]
        strategy = row["sample_strategy"]
        indices = [i.strip() for i in row["sample_indices"].split(",") if i.strip()]

        print(f"[{idx}/{len(rows)}] '{question[:40]}...' -> στρατηγική: {strategy} ({len(indices)} απάντηση/εις)")

        for ans_idx in indices:
            answer_text = row[ans_idx]
            per_criterion = {}

            for criterion_name, rubric in CRITERIA.items():
                justification, rating = call_geval(model_name, question, answer_text, criterion_name, rubric)
                per_criterion[criterion_name] = (justification, rating)
                total_calls += 1
                time.sleep(REQUEST_DELAY_SECONDS)

            results.append({
                "question": question,
                "answer_index": ans_idx,
                "mean_consistency": row["mean_consistency"],
                "sample_strategy": strategy,
                "judge_model": model_name,
                "accuracy_rating": per_criterion["Accuracy"][1],
                "accuracy_justification": per_criterion["Accuracy"][0],
                "readability_rating": per_criterion["Readability"][1],
                "readability_justification": per_criterion["Readability"][0],
                "completeness_rating": per_criterion["Completeness"][1],
                "completeness_justification": per_criterion["Completeness"][0],
            })

    if not results:
        print("[WARN] Δεν παράχθηκαν αποτελέσματα.")
        return

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("-" * 60)
    print(f"[SUCCESS] Αποθηκεύτηκε με επιτυχία το: {OUTPUT_CSV}")
    print(f"[INFO] Μοντέλο-κριτής: {model_name}")
    print(f"[INFO] Συνολικές κλήσεις API: {total_calls}")


if __name__ == "__main__":
    main()