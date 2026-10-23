
import matplotlib.pyplot as plt
import numpy as np

from data_loader import ACTIVITIES, load_dataset

X_train, y_train, _, _ = load_dataset()

fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

for activity_id, ax in enumerate(axes):
    # Select the first training window for this activity
    idx = np.flatnonzero(y_train == activity_id)[0]
    signal = X_train[idx]

    t = np.arange(128) / 50.0

    ax.plot(t, signal[:, 0], label="X")
    ax.plot(t, signal[:, 1], label="Y")
    ax.plot(t, signal[:, 2], label="Z")

    ax.set_title(ACTIVITIES[activity_id].replace("_", " "))
    ax.set_ylabel("Acceleration")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3)

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("UCI HAR: Accelerometer Signals by Activity", fontsize=15)
fig.tight_layout()
fig.savefig("results/activity_signals.png", dpi=160)
plt.show()
