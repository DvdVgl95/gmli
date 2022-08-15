''' Main script to run the GMLI method
    Please provide feature data (X), a treatment indicator (T),
    and the outcome (y).
    The data should be from a randomized control trial.

    The orchestrator Class will run the algorithm N times.
    Then the results of the N different estimates will be aggregated.
    
    For more info check the docstrings of the classes.
    
'''
import model.generic_ml_inference as gmli

# please provide X,y,T
# X, y, T = data[feature_cols], data[outcome], data[treatment_indicator]

gmli_model = gmli.Orchestrator(
    N=100,
    alpha=0.05,
    X=X,
    y=y,
    T=T,
    n_folds=5,
    n_gates=5,
    model='gb'
)
gmli_model.run()
gmli_model.aggregate_results()

# Results are stored as attrubutes
# e.g. the results of the best linear predictor
gmli_model.blp_results
