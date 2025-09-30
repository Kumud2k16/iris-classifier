import pandas as pd
import numpy as np

# Step 1: Generate 10 random scores between 0 and 100
scores = np.random.randint(0, 101, size=10)
df = pd.DataFrame(scores, columns=["Score"])
print("Scores:\n", df)

# Step 2: Set threshold
threshold = 50

# Step 3: Count how many are above threshold
count_above = (df["Score"] > threshold).sum()

# Step 4: Compute the average
average_score = df["Score"].mean()

# Step 5: Print results
print("Hello")
print(f"Number of scores above {threshold}: {count_above}")
print("Average score: {average_score:.2f}")