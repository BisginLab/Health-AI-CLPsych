# --- Imports
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import datasets as ds  # HF Datasets (pip install datasets)

"""
Generate a bell-curve-style graph (histogram + normal fit)
for per-user word counts built from post-level CSVs.
"""

# --- Adjustable params
max_posts_per_user = 10
step_value = 50          # bin width
max_value = 750          # cap for x-axis (optional)
dataset_dir = "/shared/DATA/reddit/crowd/test"  # <- set this

df_X_path = os.path.join(dataset_dir, "shared_task_posts_test.csv")
df_y_path = os.path.join(dataset_dir, "crowd_test.csv")
separator = "\n\n"

# --- Load data
# Post-level features
df_X = ds.load_dataset("csv", data_files=df_X_path)["train"]
# User-level labels (gives us user_id list to iterate)
df_y = ds.load_dataset("csv", data_files=df_y_path)["train"]

# --- Build per-user concatenated text (only SuicideWatch posts, up to N per user)
def get_matching_posts(user_row):
    uid = user_row["user_id"]
    # filter returns a new Dataset; iterate it to collect posts
    matching = df_X.filter(lambda r: r["user_id"] == uid)
    # keep only SW posts, drop None bodies, cap at max_posts_per_user
    bodies = [
        r["post_body"] for r in matching
        if (r.get("post_body") is not None) and (r.get("subreddit") == "SuicideWatch")
    ][:max_posts_per_user]
    return {"text": separator.join(bodies)}

print("Mapping user posts to their corresponding texts...")
df_with_text = df_y.map(get_matching_posts, batched=False, desc="Mapping user posts")

# --- Compute word counts per user
def safe_word_count(txt):
    if not txt or not isinstance(txt, str):
        return 0
    return len(txt.split())

word_counts = [safe_word_count(row["text"]) for row in df_with_text]

# Optional: drop zeros (users with no matched posts)
word_counts = [c for c in word_counts if c > 0]

# --- Make a “bell graph” (histogram + normal PDF overlay)
if len(word_counts) == 0:
    raise ValueError("No word counts found. Check your paths and filtering conditions.")

arr = np.array(word_counts, dtype=float)

# Fit a normal distribution to the data
mu = np.mean(arr)
sigma = np.std(arr, ddof=1) if len(arr) > 1 else 1.0  # avoid div/0 for tiny samples

# Binning
max_x = max_value if max_value is not None else max(arr) if len(arr) else 0
bins = np.arange(0, max_x + step_value, step_value)

plt.figure(figsize=(8, 5))
# density=True to show probability density; looks more “bell-like”
counts, _, _ = plt.hist(arr, bins=bins, density=True, edgecolor="black")

# Normal PDF overlay
x = np.linspace(0, max_x, 500)
pdf = (1.0 / (sigma * np.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
plt.plot(x, pdf, linewidth=2)

plt.title("Per-User Word Counts (Histogram) with Normal Fit")
plt.xlabel("Words per user (capped)")
plt.ylabel("Density")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Save or show
out_path = os.path.join(os.getcwd(), "user_word_count_bell_curve.png")
plt.savefig(out_path, dpi=150)
print(f"Saved plot to: {out_path}")
# plt.show()
