'''
    Script to run KNN algorithm using MPI commands
'''
import pandas as pd
import numpy as np
from mpi4py import MPI

sample_df = pd.read_csv('sampled_data.csv')
test_df = pd.read_csv('test_data.csv')
predicted_df = test_df.copy()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set up our list of neighbors and a dict to keep track of highest genre count in neighbor lst
neighbor_list = np.zeros(size)
knn_dict = {}

# Perform KNN algorithm
for index, row in test_df.iterrows():
    neighbor = abs(row['runtimeMinutes'] - sample_df.iloc[rank]['runtimeMinutes'])**2 +\
        abs(row['averageRating'] - sample_df.iloc[rank]['averageRating'])**2
    comm.Gather(neighbor, neighbor_list, root=0)

    if rank == 0:
        indeces = np.argpartition(neighbor_list, 10)

        # Count the appearance of genres in nearest 10 neighbors
        for idx in indeces[:10]:
            if sample_df.iloc[idx]['genres'] not in knn_dict:
                knn_dict[sample_df.iloc[idx]['genres']] = 1
            else:
                knn_dict[sample_df.iloc[idx]['genres']] += 1

        predicted_df.loc[index, 'genres'] = max(knn_dict, key=knn_dict.get)
        knn_dict = {}

if rank == 0:
    predicted_df.to_csv(path_or_buf='test_data.csv')
