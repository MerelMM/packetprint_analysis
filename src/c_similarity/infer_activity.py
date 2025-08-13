from c_similarity.c_similarity import compute_c_similarity

# During inference
# new_segments = [set of word indices per segment]
psi_min = 0.1

similarity_scores = compute_c_similarity(new_segments, merger, clf, threshold=psi_min)

for i, (psi, is_active) in enumerate(similarity_scores):
    print(f"Segment {i}: Ψ_A = {psi:.3f} → {'ACTIVE' if is_active else 'inactive'}")
