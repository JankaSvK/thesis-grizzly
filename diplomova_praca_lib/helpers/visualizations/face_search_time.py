import matplotlib.pyplot as plt
import numpy as np

from helpers.visualizations.visualize_request_stats import save_plot

dominik_linear = [89, 21, 4, 600, 65, 69, 7, 67, 69, 341]
jana_linear = [30, 14, 10, 115, 175, 122, 21, 85, 145, 76]

dominik_som = [5, 57, 60, 600, 55, 155, 7, 357, 22, 115]
jana_som = [6, 60, 45, 320, 8, 185, 5, 130, 5, 55]

# plt.boxplot([dominik_linear + jana_linear, dominik_som + jana_som], labels=['a', 'b'])
bplot = plt.boxplot([dominik_linear, dominik_som, jana_linear, jana_som],
                    labels=["Linear Search", "Traversal Search", "Linear Search", "Traversal Search"],
                    patch_artist=True)

colors = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel("Time[s]")
plt.grid(axis='y')
plt.text(1, 570, np.argmax(dominik_linear) + 1)
plt.text(1, 310, np.argsort(dominik_linear)[-2] + 1)
plt.text(2, 570, np.argmax(dominik_som) + 1)
plt.text(2, 330, np.argsort(dominik_som)[-2] + 1)
plt.text(4.05, 295, np.argmax(jana_som) + 1)
save_plot(plt, "face_search_time")
plt.show()

print(np.mean(dominik_linear))
print(np.mean(dominik_som))

print(np.mean(jana_linear))
print(np.mean(jana_som))

print(np.median(dominik_linear))
print(np.median(dominik_som))

print(np.median(jana_linear))
print(np.median(jana_som))

