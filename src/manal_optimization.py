import numpy as np
from scipy.optimize import minimize
from manal_utils import evaluateCurve, calcpennation

class TSLOptimization:
    def __init__(self, lm_opt, alpha_opt, lmt, lm_range, lm0=None, afl=None, pfl=None, tfl=None):
        self.lm_opt = lm_opt
        self.alpha_opt = alpha_opt
        self.lmt = lmt
        self.lm_range = lm_range

        if lm0 is None:
            self.lm0 = np.linspace(lm_range[0], lm_range[1], len(lmt))
        else:
            self.lm0 = lm0

        # TODO: Where exactly does this come from?
        # Active force length curve
        manal_afl = np.array([[-5.00000, 0.000000, 0.401000, 0.402000, 0.403500, 0.527250, 0.628750, 0.718750, 0.861250, 1.045000, 1.217500, 1.438750, 1.618750, 1.620000, 1.621000, 2.200000, 5.000000],
                              [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.226667, 0.636667, 0.856667, 0.950000, 0.993333, 0.770000, 0.246667, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]])
        # Passive force length curve
        manal_pfl = np.array([[-5.00000, 0.998000, 0.999000, 1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000, 1.600000, 1.601000, 1.602000, 5.000000],
                              [0.000000, 0.000000, 0.000000, 0.000000, 0.035000, 0.120000, 0.260000, 0.550000, 1.170000, 2.000000, 2.000000, 2.000000, 2.000000]])
        # Tendon force strain curve
        manal_tfs = np.array([[-10.000, -0.0020, -0.0010, 0.00000, 0.00131, 0.00281, 0.00431, 0.00581, 0.00731, 0.00881, 0.01030, 0.01180, 0.01230, 9.20000, 9.20100, 9.20200, 20.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0108, 0.0257, 0.0435, 0.0652, 0.0915, 0.123, 0.161, 0.208, 0.227, 345.0, 345.0, 345.0, 345.0]])
        # Convert tendon strain to tendon length
        manal_tfl = np.array([manal_tfs[0, :] + 1, manal_tfs[1, :]])

        # Use OpenSim default curves if none provided
        if afl is None:
            afl = np.array(manal_afl)
        if pfl is None:
            pfl = np.array(manal_pfl) 
        if tfl is None:
            tfl = np.array(manal_tfl)[:, 3:-3] # Remove duplicate values

        self.afl = afl
        self.pfl = pfl
        self.tfl = tfl

    def optimize(self, lb=None, ub=None, method='SLSQP', objective='ssdp'):
        if lb is None:
            lb = self.lm_range[0] * np.ones_like(self.lmt)
        if ub is None:
            ub = self.lm_range[1] * np.ones_like(self.lmt)
        result = minimize(lambda x: self.objective(x, method=objective), self.lm0, bounds=list(zip(lb, ub)), method=method)
        return self.calcslacklength(self.lmt, result.x, self.lm_opt)

    def calcslacklength(self, lmt, lm, lm_opt):
        # Ensure numpy arrays
        lmt = np.asarray(lmt)
        lm = np.asarray(lm)
        
        # Input validation
        if np.any(lmt <= 0) or np.any(lm <= 0) or lm_opt <= 0:
            raise ValueError("Length parameters must be positive")
        
        # Calculate the normalized muscle fiber lengths
        lm_norm = lm / lm_opt

        # Calculate the pennation angle
        alpha = calcpennation(lm_norm, lm_opt, self.alpha_opt)
        
        # Evaluate force length curves
        forces = np.zeros((2, len(lm_norm)))
        forces[0, :] = evaluateCurve(self.afl, lm_norm) # Active force length curve 
        forces[1, :] = evaluateCurve(self.pfl, lm_norm) # Passive force length curve

        # Calculate total muscle fiber forces from sum of active and passive forces
        Fm_norm = np.sum(forces, axis=0)
        
        # Calculate normalized tendon forces
        Ft_norm = Fm_norm * np.cos(alpha) # Manal 2004, Eq. 2

        # Calculate normalized tendon lengths from tendon forces
        lt_norm = evaluateCurve(self.tfl, Ft_norm, inverse=True)

        # Calculate slack length
        fiber_proj = lm_opt * lm_norm * np.cos(alpha)
        # Handle zero/near-zero lt_norm values to avoid division by zero
        mask = np.abs(lt_norm) < 1e-6
        lt_s = np.zeros_like(lmt)
        lt_s[~mask] = (lmt[~mask] - fiber_proj[~mask]) / lt_norm[~mask]
        lt_s[mask] = lmt[mask] - fiber_proj[mask]  # When lt_norm ≈ 0, assume lt_s = lmt - fiber_proj
        lt_s = np.clip(lt_s, 0, lmt)  # Maintain physical bounds
        
        return lt_s

    def ssdp(self, lt_s):
        # Calculate error equal to the sum of squared differences between every element of the slack length vector
        diff_matrix = np.tile(lt_s, (len(lt_s), 1))
        difference = diff_matrix - diff_matrix.T
        err = np.sum(difference**2)
        return err
    
    def ssd(self, lt_s):
        # Sum of squared differences between slack lengths and the mean slack length
        return np.sum((lt_s - np.mean(lt_s))**2) 

    def objective(self, lm, method = 'ssdp'):
        # Calcualte slack length 
        lt_s = self.calcslacklength(self.lmt, lm, self.lm_opt)

        if method == 'ssdp':
            err = self.ssdp(lt_s)
        elif method == 'ssd':
            err = self.ssd(lt_s)
        elif method == 'var':
            err = np.var(lt_s)
        elif method == 'std':
            err = np.std(lt_s)
        else:
            raise ValueError(f"Method {method} not recognized")
        return err
