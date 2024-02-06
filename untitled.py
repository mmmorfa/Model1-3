import numpy as np

#matrix = np.array([[1, 2, 3],[4, 5, 6],[7, 5, 0]])
matrix = np.zeros((3,4))

# Value to find
value_to_find = 0

# Find indices where value equals value_to_find
indices = np.where(matrix == value_to_find)

print(50_000_000)

if len(indices[0]) > 0:
    # Print the indices of the value
    print(f"The value {value_to_find} is located at indices:")
    for i in range(len(indices[0])):
        print(f"({indices[0][i]}, {indices[1][i]})")
else:
    print(f"The value {value_to_find} is not found in the matrix.")