### Schemes

Update schemes for PIPT. All schemes are built using the same structure. 

1. All schemes are built as a class.
2. **__init__.py** initiallizes and sets up the scheme. And the class self-ignites, i.e., the final line in this 
function is always: **self.run_loop()**. This allows us to send the class as a callable method to scipy.minimize

Currently, the following schemes are implemented:
- **EnRML**: Ensemble Randomized Maximum Likelihood
- **ES**: Ensemble smoother
- **ES-MDA**: Ensemble smoother with multiple data assimilation
- **EnKF**: Ensemble Kalman Filter
