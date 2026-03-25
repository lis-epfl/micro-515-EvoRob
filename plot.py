import numpy as np
from pathlib import Path

ckpt_path = Path("/Users/farahelsousy/Desktop/evolutionary_robotics/micro-515-EvoRob/results/20260318_185115_phase_hybrid_residual_ckpts_best so far ice")

full_f = np.load(ckpt_path / "full_f.npy")

best_so_far = -np.inf

print("\n📈 Fitness evolution:\n")

for gen in range(full_f.shape[0]):
    fitness = full_f[gen]

    gen_mean = np.mean(fitness)
    gen_best = np.max(fitness)

    best_so_far = max(best_so_far, gen_best)

    print(
        f"Gen {gen:03d}: "
        f"Mean={gen_mean:.2f}, "
        f"Best={gen_best:.2f}, "
        f"Best so far={best_so_far:.2f}"
    )
    import matplotlib.pyplot as plt

means = np.mean(full_f, axis=1)
bests = np.max(full_f, axis=1)
best_so_far = np.maximum.accumulate(bests)

plt.plot(means, label="Mean")
plt.plot(bests, label="Best")
plt.plot(best_so_far, label="Best so far")

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.grid()
plt.show()