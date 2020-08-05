import matplotlib.pyplot as plt

accuracies = ys = [39.91, 58.01, 60.86, 68.52]
times      = xs = [0.097, 0.944, 1.767, 2.600]
labels     = ['SBL', '1LC', '2LC', '3LC']

plt.scatter(x=xs, y=ys, c='r', s=40)
plt.plot(xs, ys, c='r')

Δx = max(xs) - min(xs)
Δy = max(ys) - min(ys)

for i, label in enumerate(labels):
    x, y = xs[i], ys[i]
    plt.text(x+0.02*Δx, y-0.1*Δy, label, fontsize='xx-large')

plt.xlim(min(xs) - 0.1 * Δx, max(xs) + 0.2 * Δx)
plt.ylim(min(ys) - 0.2 * Δy, max(ys) + 0.2 * Δy)
plt.xlabel('Time (s)', fontsize='xx-large')
plt.ylabel('Accuracy', fontsize='xx-large')
plt.title('Speed-accuracy tradeoff using light curtains', fontsize='xx-large')
plt.xticks(fontsize='x-large')
plt.yticks(fontsize='x-large')
plt.tight_layout()

plt.savefig("tradeoff.png", format="png", dpi=200)