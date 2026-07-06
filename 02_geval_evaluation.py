"""
ΣΤΑΔΙΟ 2: G-Eval (LLM-as-a-Judge) αξιολόγηση με ΔΩΡΕΑΝ open-source μοντέλα
(μέσω Hugging Face Inference Providers)
-----------------------------------------------------------------------
Διαβάζει το CSV που παράγει το 01_calculate_consistency.py (στήλες
question, mean_consistency, sample_strategy, sample_indices, 1..10) και
στέλνει τις αντιπροσωπευτικές απαντήσεις κάθε ερώτησης σε ένα δωρεάν,
open-source instruct μοντέλο, για 3 κριτήρια: Accuracy, Readability,
Completeness.

Αλλαγές σε σχέση με την πρώτη έκδοση:
  1. ΣΗΜΑΝΤΙΚΟ: Το "meta-llama/Meta-Llama-3-8B-Instruct" είναι "gated"
     μοντέλο — η Meta απαιτεί χειροκίνητη αποδοχή license στη σελίδα του
     μοντέλου στο HF πριν το token σου μπορεί να το καλέσει (αλλιώς 403).
     Εδώ, αν το πρωτεύον μοντέλο αποτύχει, δοκιμάζονται αυτόματα ungated,
     αξιόπιστα διαθέσιμα εναλλακτικά μοντέλα (fallback list).
  2. Preflight έλεγχος: μία δοκιμαστική κλήση πριν ξεκινήσει όλο το loop,
     ώστε να μάθεις αμέσως αν υπάρχει πρόβλημα (λάθος token, gated μοντέλο,
     μη διαθέσιμο provider) αντί να το ανακαλύψεις μετά από 138 κλήσεις.
  3. Σαφής διάγνωση σφαλμάτων: ξεχωρίζει 403 (gated/permissions), 404
     (μοντέλο μη διαθέσιμο σε κανέναν inference provider αυτή τη στιγμή),
     429 (πραγματικό rate limit -> αυτό ΜΟΝΟ κάνει retry).
  4. Συμβατό 100% με τη δομή εξόδου του 01_calculate_consistency.py.
"""

import csv
import os
import re
import sys
import time

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# --- ΡΥΘΜΙΣΕΙΣ ---
INPUT_CSV = "chatbot_results_with_consistency.csv"
OUTPUT_CSV = "geval_results.csv"

# Πρωτεύον μοντέλο + ungated εναλλακτικά (με τη σειρά που θα δοκιμαστούν)
CANDIDATE_MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",             # ungated, δημοφιλές fine-tune του Mistral
    "mistralai/Mistral-7B-Instruct-v0.3",        # ungated fallback #1
    "Qwen/Qwen2.5-7B-Instruct",                   # ungated fallback #2
]

MAX_RETRIES = 4
REQUEST_DELAY_SECONDS = 1.0  # ευγενική καθυστέρηση για το δωρεάν rate limit

csv.field_size_limit(100_000_000)

# ΣΗΜΕΙΩΣΗ: το header x-use-cache:false απενεργοποιεί μόνο το caching
# πανομοιότυπων prompts (χρήσιμο ώστε κάθε ερώτηση να παίρνει "φρέσκια"
# generation) — ΔΕΝ επηρεάζει τη χρέωση/routing μέσω Inference Providers.
# Αν έχεις εξαντλήσει τα μηνιαία δωρεάν credits, θα πάρεις 402 ό,τι header
# κι αν βάλεις — δες τις σημειώσεις στο τέλος του αρχείου για πραγματικές λύσεις.
client = InferenceClient(
    api_key=os.environ.get("HF_TOKEN"),
    headers={"x-use-cache": "false"},
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
    """
    Δοκιμάζει τα CANDIDATE_MODELS με τη σειρά, με μία μικρή δοκιμαστική
    κλήση το καθένα, και επιστρέφει το πρώτο που δουλεύει.
    Αυτό αποφεύγει να ανακαλύψεις μετά από 100+ κλήσεις ότι το μοντέλο
    που διάλεξες είναι gated ή δεν φιλοξενείται πια από κανέναν provider.
    """
    for model_name in CANDIDATE_MODELS:
        print(f"[INFO] Δοκιμή μοντέλου: {model_name} ...")
        try:
            client.chat_completion(
                model=model_name,
                messages=[{"role": "user", "content": "Reply with only the word OK."}],
                max_tokens=5,
            )
            print(f"[INFO] Επιτυχία! Χρησιμοποιείται το: {model_name}")
            return model_name
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 402:
                print(f"    [SKIP] '{model_name}' -> 402 Payment Required: εξαντλήθηκαν τα μηνιαία δωρεάν "
                      f"credits του HF account σου για Inference Providers (ισχύει ανεξαρτήτως μοντέλου).")
            elif status == 403:
                print(f"    [SKIP] '{model_name}' είναι gated ή χωρίς άδεια πρόσβασης (403). "
                      f"Αν θέλεις αυτό το μοντέλο, ζήτησε πρόσβαση στη σελίδα του στο HF.")
            elif status == 404:
                print(f"    [SKIP] '{model_name}' δεν είναι διαθέσιμο σε κανέναν inference provider αυτή τη στιγμή (404).")
            else:
                print(f"    [SKIP] '{model_name}' απέτυχε (HTTP {status}): {e}")
        except Exception as e:
            print(f"    [SKIP] '{model_name}' απέτυχε: {e}")
    return None


def call_geval(model_name, question, chatbot_answer, criterion_name, rubric):
    prompt = build_prompt(question, chatbot_answer, criterion_name, rubric)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat_completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()

            match = RATING_REGEX.search(content)
            rating = int(match.group(1)) if match else None

            justification = content
            if "Rating:" in content:
                justification = content.split("Rating:")[0].replace("Justification:", "").strip()

            return justification, rating

        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 402:
                print(f"    [FATAL] 402 Payment Required: εξαντλήθηκαν τα μηνιαία δωρεάν credits "
                      f"του HF account σου. Retry δεν θα βοηθήσει — δες τις λύσεις στο τέλος του αρχείου.")
                return None, None
            elif status == 429:
                # πραγματικό rate limit -> αξίζει retry
                wait = 5 * attempt
                print(f"    [WARN] Rate limit (429), retry σε {wait}s (προσπάθεια {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                # π.χ. 403/404/500 -> δεν θα διορθωθεί με retry, σταματάμε αμέσως
                print(f"    [ERROR] Μη ανακτήσιμο σφάλμα (HTTP {status}): {e}")
                return None, None
        except Exception as e:
            wait = 5 * attempt
            print(f"    [WARN] Σφάλμα σύνδεσης ({e}), retry σε {wait}s (προσπάθεια {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)

    print("    [ERROR] Απέτυχαν όλες οι προσπάθειες — παραλείπεται.")
    return None, None


def main():
    if not os.environ.get("HF_TOKEN"):
        print("[ERROR] Δεν βρέθηκε το environment variable HF_TOKEN.")
        print("        cmd.exe:      set HF_TOKEN=hf_...")
        print("        PowerShell:   $env:HF_TOKEN=\"hf_...\"")
        return

    print("[INFO] Έλεγχος διαθέσιμων μοντέλων...")
    model_name = pick_working_model()
    if model_name is None:
        print("[FATAL] Κανένα από τα υποψήφια μοντέλα δεν είναι διαθέσιμο αυτή τη στιγμή.")
        print("        Έλεγξε το HF_TOKEN σου ή πρόσθεσε άλλα μοντέλα στη λίστα CANDIDATE_MODELS.")
        return

    try:
        with open(INPUT_CSV, mode="r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"[ERROR] Δεν βρέθηκε το '{INPUT_CSV}'. Τρέξτε πρώτα το 01_calculate_consistency.py.")
        return

    results = []
    total_calls = 0

    for idx, row in enumerate(rows, 1):
        question = row["question"]
        strategy = row["sample_strategy"]
        indices = [i.strip() for i in row["sample_indices"].split(",") if i.strip()]

        print(f"[{idx}/{len(rows)}] '{question[:50]}...' -> στρατηγική: {strategy} ({len(indices)} απάντηση/εις)")

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
    print(f"[SUCCESS] Αποθηκεύτηκε: {OUTPUT_CSV}")
    print(f"[INFO] Μοντέλο-κριτής: {model_name}")
    print(f"[INFO] Συνολικές κλήσεις API: {total_calls}")


if __name__ == "__main__":
    main()