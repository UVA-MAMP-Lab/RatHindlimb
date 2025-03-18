import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from opensim import Function

def calcpennation_single(m_lm, lm_opt, alpha_opt):
    """Calculate pennation angle for single value with bounds checking."""
    # Validate inputs
    if not (0 <= alpha_opt < np.pi/2):
        raise ValueError("alpha_opt must be between 0 and pi/2")
    if lm_opt <= 0:
        raise ValueError("lm_opt must be positive")
    # Handle zero/near-zero case
    if np.abs(m_lm) < 1e-6:
        return np.pi/2
    l = np.clip(lm_opt * np.sin(alpha_opt) / m_lm, 0, 1)
    return np.clip(np.arcsin(l), 0, np.pi/2)

def calcpennation(lm_norm, lm_opt, alpha_opt):
    """Vectorized pennation angle calculation."""
    # Validate inputs
    if not np.all(lm_norm > 0):
        raise ValueError("lm_norm must be positive")
    
    # TODO: Vectorize
    m_lm = lm_norm * lm_opt
    alpha = np.zeros_like(m_lm)
    for i in range(len(m_lm)):
        alpha[i] = calcpennation_single(m_lm[i], lm_opt, alpha_opt)
    return alpha

def evaluateCurve(curve, points, inverse=False):
    """
    Evaluate a curve (OpenSim Function or numpy array) at given points.
    
    Args:
        curve: OpenSim Function or numpy array [2 x n]
        points: Points at which to evaluate the curve
        inverse: Whether to evaluate inverse of the curve
    """
    points = np.asarray(points)
    
    if isinstance(curve, Function):
        evalfunc = calcCurve(curve, inverse)
        return np.array([evalfunc(float(x)) for x in points])
    else:
        # Handle numpy array curves
        if inverse:
            f = interp1d(curve[1, :], curve[0, :], kind='cubic', 
                        bounds_error=False, fill_value='extrapolate')
        else:
            f = interp1d(curve[0, :], curve[1, :], kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
        return f(points)

def calcCurve(curve: Function, inverse: bool):
    """Wrap OpenSim Function calcValue method to handle scalar inputs."""
    if inverse:
        return lambda x: calcInverse(curve, float(x))
    else:
        return lambda x: curve.calcValue(float(x))

def calcInverse(curve: Function, value: float, start: float = 0):
    """
    Calculate the inverse of an OpenSim curve at a given value.
    Uses scalar optimization to find the input that produces the target value.
    """
    def obj(x):
        return (curve.calcValue(float(x[0])) - value)**2
    
    result = minimize(obj, x0=[start], method='BFGS')
    return float(result.x[0])
