"""
ΣΤΑΔΙΟ 1: Pairwise Semantic Similarity (Consistency) με Sentence-BERT
-----------------------------------------------------------------------
Διαβάζει ένα CSV (question, 1..10) και για κάθε ερώτηση υπολογίζει:
  - mean_consistency : μέσος όρος των 45 pairwise cosine similarities
  - variance         : διακύμανση των 45 similarities
  - min_similarity / max_similarity
  - sample_strategy  : "single" (υψηλό consistency) ή "triple" (χαμηλό)
  - sample_indices    : ποιες απαντήσεις (1-10) πρέπει να σταλούν στο G-Eval
                         (Στάδιο 2), ώστε να μη χρειάζεται να ξαναϋπολογίσεις
                         embeddings εκεί.

Αλλαγές σε σχέση με την πρώτη έκδοση:
  1. Μοντέλο: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
     Το προηγούμενο μοντέλο (symanto/sn-xlm-roberta-...-xnli) είναι μοντέλο
     Natural Language Inference (entailment/contradiction), όχι μοντέλο
     εκπαιδευμένο για Semantic Textual Similarity. Δίνει αναξιόπιστα cosine
     similarity scores. Το paraphrase-multilingual-mpnet-base-v2 είναι
     εκπαιδευμένο ειδικά σε STS/paraphrase δεδομένα και υποστηρίζει και τις
     3 γλώσσες σου (el, de, en) με συγκρίσιμα scores μεταξύ γλωσσών.
  2. Batch encoding: όλα τα κείμενα (380 αντί για 38x10 ξεχωριστές κλήσεις)
     encodάρονται μαζί -> πολύ πιο γρήγορο.
  3. Normalized embeddings -> ακριβές cosine similarity.
  4. Robust CSV reading (BOM, μεγάλα πεδία, κενές απαντήσεις).
  5. Υπολογισμός medoid/outlier answer per ερώτηση, ώστε το script 2 να
     ξέρει ακριβώς ποια/ποιες απαντήσεις να στείλει στον LLM-Judge, σύμφωνα
     με τη στρατηγική δειγματοληψίας του info_for_the_code.txt.
"""

import csv
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# --- ΡΥΘΜΙΣΕΙΣ ---
INPUT_CSV = "chatbot_results_de.csv"
OUTPUT_CSV = "chatbot_results_with_consistency.csv"

# Κατώφλια όπως ορίζονται στο info_for_the_code.txt
HIGH_CONSISTENCY_THRESHOLD = 0.85   # >= -> στέλνουμε 1 απάντηση στο G-Eval
# οτιδήποτε κάτω από αυτό -> στέλνουμε 3 (min, median, max)

csv.field_size_limit(100_000_000)  # οι ιατρικές απαντήσεις μπορεί να είναι μακριές

print("[INFO] Φόρτωση multilingual S-BERT μοντέλου (παρακαλώ περιμένετε)...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


def compute_pairwise_matrix(answers):
    """Επιστρέφει το (10x10) cosine similarity matrix για μία ερώτηση."""
    embeddings = model.encode(
        answers,
        convert_to_numpy=True,
        normalize_embeddings=True,  # -> dot product == cosine similarity
        show_progress_bar=False,
    )
    sim_matrix = embeddings @ embeddings.T
    return sim_matrix


def analyze_question(answers):
    """
    Υπολογίζει consistency metrics + επιλέγει ποιες απαντήσεις πρέπει να
    σταλούν στο Στάδιο 2 (G-Eval), σύμφωνα με τη στρατηγική δειγματοληψίας.
    """
    sim_matrix = compute_pairwise_matrix(answers)

    n = len(answers)
    triu_idx = np.triu_indices(n, k=1)
    pairwise_similarities = sim_matrix[triu_idx]

    mean_consistency = float(np.mean(pairwise_similarities))
    variance = float(np.var(pairwise_similarities))
    min_sim = float(np.min(pairwise_similarities))
    max_sim = float(np.max(pairwise_similarities))

    # "Κεντρικότητα" κάθε απάντησης = μέση ομοιότητα με τις υπόλοιπες 9
    # (χωρίς τη διαγώνιο, που είναι πάντα 1.0)
    row_sums = sim_matrix.sum(axis=1) - 1.0
    avg_similarity_per_answer = row_sums / (n - 1)

    if mean_consistency >= HIGH_CONSISTENCY_THRESHOLD:
        # Μία μόνο, αντιπροσωπευτική απάντηση: η "medoid" -> αυτή με τη
        # μεγαλύτερη μέση ομοιότητα με όλες τις άλλες (πιο "τυπική").
        strategy = "single"
        chosen = [int(np.argmax(avg_similarity_per_answer))]
    else:
        # 3 απαντήσεις: η πιο "τυπική" (max avg similarity), η πιο
        # "ακραία"/outlier (min avg similarity), και η μεσαία (median rank),
        # ώστε ο Judge να δει το εύρος της ποιοτικής διακύμανσης.
        strategy = "triple"
        order = np.argsort(avg_similarity_per_answer)
        low_idx = int(order[0])
        high_idx = int(order[-1])
        mid_idx = int(order[len(order) // 2])
        # αφαίρεση διπλότυπων σε edge cases (π.χ. πολύ λίγες μοναδικές τιμές)
        chosen = list(dict.fromkeys([high_idx, mid_idx, low_idx]))

    # μετατροπή από 0-based index σε 1-based (ταιριάζει με τις στήλες 1..10 του CSV)
    chosen_1_based = [i + 1 for i in chosen]

    return {
        "mean_consistency": round(mean_consistency, 4),
        "variance": round(variance, 4),
        "min_similarity": round(min_sim, 4),
        "max_similarity": round(max_sim, 4),
        "sample_strategy": strategy,
        "sample_indices": ",".join(str(i) for i in chosen_1_based),
    }


def main():
    print("--- ΕΝΑΡΞΗ ΔΙΑΔΙΚΑΣΙΑΣ ΥΠΟΛΟΓΙΣΜΟΥ ΑΠΟ CSV ---")

    try:
        with open(INPUT_CSV, mode="r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = [r for r in reader if any(cell.strip() for cell in r)]
    except FileNotFoundError:
        print(f"[ERROR] Δεν βρέθηκε το αρχείο '{INPUT_CSV}'.")
        return

    n_answer_cols = len(headers) - 1
    print(f"[INFO] {len(rows)} ερωτήσεις, {n_answer_cols} απαντήσεις/ερώτηση.")
    print("-" * 60)

    final_data = []
    for idx, row in enumerate(rows, 1):
        question = row[0]
        answers = [a if a.strip() else " " for a in row[1:1 + n_answer_cols]]  # αποφυγή κενών strings στο SBERT

        if len(answers) != n_answer_cols:
            print(f"[WARN] Γραμμή {idx}: αναμενόμενες {n_answer_cols} απαντήσεις, βρέθηκαν {len(answers)} — παραλείπεται.")
            continue

        print(f"[{idx}/{len(rows)}] '{question[:50]}...'")
        stats = analyze_question(answers)

        new_row = [
            question,
            stats["mean_consistency"],
            stats["variance"],
            stats["min_similarity"],
            stats["max_similarity"],
            stats["sample_strategy"],
            stats["sample_indices"],
        ] + answers
        final_data.append(new_row)

    new_headers = [
        "question", "mean_consistency", "variance", "min_similarity",
        "max_similarity", "sample_strategy", "sample_indices",
    ] + [str(i) for i in range(1, n_answer_cols + 1)]

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(new_headers)
        writer.writerows(final_data)

    print("-" * 60)
    print(f"[SUCCESS] Αποθηκεύτηκε: {OUTPUT_CSV}")
    n_single = sum(1 for r in final_data if r[5] == "single")
    n_triple = len(final_data) - n_single
    total_geval_calls_per_criterion = n_single * 1 + n_triple * 3
    print(f"[INFO] {n_single} ερωτήσεις -> 1 απάντηση, {n_triple} ερωτήσεις -> 3 απαντήσεις.")
    print(f"[INFO] Συνολικές G-Eval κλήσεις ανά κριτήριο: {total_geval_calls_per_criterion} "
          f"(αντί για {len(final_data) * n_answer_cols})")


if __name__ == "__main__":
    main()
