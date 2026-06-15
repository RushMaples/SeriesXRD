"""
Workflow:
Iteratively identify, record, and remove from the data each feature in succession
to hone in on less intense data.
1. Background scattering: diamond anvil cell, gasket, etc.
2. Peak/profile fitting (FWHM, see paper) - vecotrization? Potential JAX usage?
3. Compound identification
    a. Deterministic matching using EOS
    b. machine learning - trained on generated simulated data

Finally, create a heatmap for each individual substance. Should receive a multitude of frames;
think about how to keep frame ordering consistent and consider that nearby frames are more likely to share substances
"""