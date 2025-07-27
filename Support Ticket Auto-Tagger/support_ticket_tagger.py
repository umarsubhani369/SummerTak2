from sentence_transformers import SentenceTransformer, util

# Load lightweight model (fast, only ~80MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define your candidate tags
TAGS = [
    "login issue", "account locked", "password reset", "payment problem",
    "technical bug", "app crash", "feature request", "slow performance",
    "email not received", "update issue"
]

def get_ticket_tags(ticket_text, top_k=3):
    # Encode the ticket and tags
    ticket_embedding = model.encode(ticket_text, convert_to_tensor=True)
    tag_embeddings = model.encode(TAGS, convert_to_tensor=True)

    # Calculate cosine similarity between ticket and tags
    cosine_scores = util.cos_sim(ticket_embedding, tag_embeddings)[0]

    # Get top-k tags
    top_indices = cosine_scores.argsort(descending=True)[:top_k]
    top_tags = [TAGS[i] for i in top_indices]

    return top_tags
