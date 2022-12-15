import matplotlib.pyplot as plt
import pandas as pd
import sys

if len(sys.argv) < 2:
    input('Nincs megadott fájl.')
    exit()

print(sys.argv[1])
data_frame = pd.read_csv(sys.argv[1])
data_frame.head()

ax = plt.axes(projection='3d')
ax.set_xlabel('upper_letter')
ax.set_ylabel('number')
ax.set_zlabel('distance')
fg = ax.scatter3D(data_frame['upper_letter_ratio'],data_frame['number_ratio'],data_frame['distance_from_last'], c=data_frame['cluster_labels'])

plt.show()