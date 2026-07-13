import numpy as np
import tensornetwork as tn

tn.set_default_backend("numpy")

X = tn.Node(np.reshape(np.array([0, 1, 1, 0]), (2, 2)))
X_p = tn.Node(np.reshape(np.array([0, 1, 1, 0]), (2, 2)))
Z = tn.Node(np.reshape(np.array([1, 0, 0, -1]), (2, 2)))

# Tr(X)
edge = X[1] ^ X[0]
identity = tn.contract(edge)
# Should be the trace of X (i.e., 0)
print("Tr(X) =")
print(identity.tensor)

# X^2
edge = X[1] ^ X_p[0]
identity = tn.contract(edge)
# Should be I
print("X**2 =")
print(identity.tensor)

# Equivalent to X @ Z
edge =  X[1] ^ Z[0]
c = tn.contract(edge)
# Should be -iY
print("X @ Z =")
print(c.tensor)


# General matrix multiplication test
matrix_A = np.random.rand(3, 3)
matrix_B = np.random.rand(3, 3)

node_A = tn.Node(matrix_A, name="Node_A")
node_B = tn.Node(matrix_B, name="Node_B")

shared_edge = node_A[1] ^ node_B[0]

node_C = tn.contract(shared_edge)

result_tn = node_C.tensor
result_np = np.dot(matrix_A, matrix_B)

print("\nThe random 3x3 tensors match :", np.allclose(result_tn, result_np))