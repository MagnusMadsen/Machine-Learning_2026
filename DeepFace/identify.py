from similarity import compare_to_set

# Husk at skifte MATCH_THRESHOLD afhængigt af den anvendte model!
#| Model      | MATCH |
#| ---------- | ----- |
#| ArcFace    | ~0.68 |
#| FaceNet512 | ~0.75 |


MATCH_THRESHOLD = 0.68

def identify_person(embedding, database):
    best_name = "Skyd"
    best_score = 0.0

    for name, reference_set in database.items():
        score = compare_to_set(embedding, reference_set)

        if score > best_score:
            best_score = score
            best_name = name

    if best_score < MATCH_THRESHOLD:
        return "Skyd", best_score

    return best_name, best_score
