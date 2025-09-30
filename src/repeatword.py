import pandas as pd

# Ask the user to enter sentences (separated by a period or comma)
user_input = input("Enter a few short sentences (separate with commas):\n")

# Split into a list of sentences
sentences = [s.strip() for s in user_input.split(",")]

# Convert sentences into one big string (lowercased)
text = " ".join(sentences).lower()

# Split into words
words = text.split()

# Create a Pandas Series
word_series = pd.Series(words)

# Define stopwords
stopwords = {"i", "the", "was", "not"}

# Filter out stopwords
filtered_words = word_series[~word_series.isin(stopwords)]

# Count frequencies
word_counts = filtered_words.value_counts()

# Show top 3 words
print("\nTop 3 words:\n", word_counts.head(3))