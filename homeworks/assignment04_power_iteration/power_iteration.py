import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """

    vector = np.random.rand(data.shape[0])
    vector = vector / np.linalg.norm(vector)

    for _ in range(num_steps):
        new_vector = np.dot(data, vector)
        new_vector = new_vector / np.linalg.norm(new_vector)
        
        vector = new_vector

    eigenvalue = float(np.dot(vector.T, np.dot(data, vector)))
    return eigenvalue, vector