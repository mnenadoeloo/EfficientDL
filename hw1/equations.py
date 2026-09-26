import numpy as np

FLOPS_S2_COEF = 17712      
FLOPS_CONST = 313344    
PARAMS_BYTES = 4_161_296  
MEMORY_S2B_COEF = 44        
BYTES_S2B_COEF = 196         


def flops(image_size, batch):
    """FLOPs of one forward pass. -> float / ndarray."""
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    return B * (FLOPS_S2_COEF * S**2 + FLOPS_CONST)


def memory(image_size, batch):
    """Peak GPU memory of one forward pass, in bytes. -> float / ndarray."""
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    return PARAMS_BYTES + MEMORY_S2B_COEF * S**2 * B


def bytes_moved(image_size, batch):
    """Total bytes read+written across all layers (for latency/energy)."""
    S = np.asarray(image_size, dtype=np.float64)
    B = np.asarray(batch, dtype=np.float64)
    return BYTES_S2B_COEF * S**2 * B + PARAMS_BYTES


def latency(image_size, batch, theta):
    """theta = (theta0, theta1, theta2):
    theta0 -- fixed launch/dispatch overhead [s]
    theta1 -- 1 / effective FLOP/s          [s per FLOP]
    theta2 -- 1 / effective bytes/s         [s per byte]
    Linear roofline surrogate (sum instead of max), fit by least squares.
    -> float / ndarray, seconds.
    """
    theta0, theta1, theta2 = theta
    return theta0 + theta1 * flops(image_size, batch) + theta2 * bytes_moved(image_size, batch)


def energy(image_size, batch, theta_energy):
    """theta_energy = (theta0, theta1, theta2, p_idle, theta3, theta4):
    first three calibrate Latency(), then
    p_idle -- static/idle GPU power [W]
    theta3 -- energy per FLOP  [J per FLOP]
    theta4 -- energy per byte [J per byte]
    -> float / ndarray, joules.
    """
    theta0, theta1, theta2, p_idle, theta3, theta4 = theta_energy
    t = latency(image_size, batch, (theta0, theta1, theta2))
    return (p_idle * t
            + theta3 * flops(image_size, batch)
            + theta4 * bytes_moved(image_size, batch))


if __name__ == "__main__":
    S = np.array([32, 224, 512])
    B = np.array([1, 32, 256])
    print("FLOPs:", flops(S, B))
    print("Memory (bytes):", memory(S, B))
    print("Bytes moved:", bytes_moved(S, B))
