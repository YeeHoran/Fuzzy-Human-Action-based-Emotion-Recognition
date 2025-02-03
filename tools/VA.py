import numpy as np
import matplotlib.pyplot as plt

# Define emotion categories with their Valence and Arousal values
emotions = {
    "Happy": (0.8, 0.8),
    "Excited": (0.7, 1.0),
    "Surprised": (0.0, 0.8),
    "Neutral": (0.0, 0.0),
    "Sad": (-0.8, -0.8),
    "Angry": (-0.8, 0.8),
    "Relaxed": (0.8, -0.5),
}

# Extract coordinates and labels
valence = np.array([emotions[e][0] for e in emotions])
arousal = np.array([emotions[e][1] for e in emotions])
labels = list(emotions.keys())

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 7))

# Draw axes
ax.axhline(0, color='gray', linewidth=1, linestyle='--')
ax.axvline(0, color='gray', linewidth=1, linestyle='--')

# Scatter plot of emotions
ax.scatter(valence, arousal, color="red", s=100, label="Emotions")

# Annotate points
for i, label in enumerate(labels):
    ax.text(valence[i] + 0.05, arousal[i] + 0.05, label, fontsize=12)

# Labels and limits
ax.set_xlabel("Valence (Pleasure)")
ax.set_ylabel("Arousal (Activation)")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_title("Valence-Arousal 2D Emotion Model")
ax.grid(True, linestyle="--", alpha=0.5)

# Show the plot
plt.show()
