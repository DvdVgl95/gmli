''' Implementation of Chernozhukov et al (2018):
    "Generic Machine Learning Inference on Heterogeneous Treatment Effects 
    in Randomized Experiments."
    
    Implemented via FirstStageEstimator, BlpEstimator and GatesEstimator.
    Find the paper here: 
    https://www.nber.org/system/files/working_papers/w24678/w24678.pdf
    

Classes:
    BlpEstimator
    FirstStageEstimator
    GatesEstimator
    Orchestrator
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline


class BlpEstimator(BaseEstimator):
    ''' Compute the Best Linear Predictor(BLP)
    
    Y_i = alpha'X_1i + ß_1*D_i + ß_2*D_i*(S_i - E_n,m[S_i])
    alpha'X_1i = alpha_0 + alpha_1 * B(Z_i) + alpha_2 * S(Z_i)
    on the main sample

    Attributes:
        blp_params(array): parameters of the fitted best linear predictor
        blp_se(array): robust standard errors for the blp param estimates
        blp_ci(array): confidence intervals at level self.alpha for blp estimates
        blp_pvalues(array): p-values for the blp estimates 
    '''

    def __init__(self, model=None, alpha=0.05):
        '''
        Arguments:
            alpha(float): alpha for confidence intervals
        '''
        self.alpha = alpha


    def fit(
        self,
        baseline_preds,
        treatment_preds,
        treatment_indicator_main,
        target_main
    ):
        ''' Compute the best linear predictor coefficients

        Arguments:
            baseline_preds(array): predicitons of the untreated model
            treatment_preds(array): predictions of the treated model
            treatment_indicator_main(array): treatment status values for the main sample
            target_main(array): outcome variable of interest
        '''
        demean_treatment_preds = treatment_preds - treatment_preds.mean()

        # propensity score
        prop_score = len(treatment_indicator_main[treatment_indicator_main==1.0])/len(treatment_indicator_main)
        adj_treatment_indicator = treatment_indicator_main - prop_score

        # compute BLP 
        X_BLP = pd.DataFrame(
            (baseline_preds,
            treatment_preds,
            adj_treatment_indicator,
            adj_treatment_indicator*demean_treatment_preds)
        )
        X_BLP = X_BLP.transpose()
        X_BLP.columns = ['a_1', 'a_2', 'ß_1', 'ß_2']
        X_BLP = sm.add_constant(X_BLP)

        blp = sm.OLS(target_main, X_BLP, hasconst=True).fit(use_t=True)

        self.blp_params = blp.params
        summary_df = convert_OLS_summary(blp, alpha=self.alpha/2)
        self.summary = summary_df[summary_df.index.isin(['ß_1', 'ß_2'])]

        return self


class FirstStageEstimator(BaseEstimator):
    ''' Compute the first stage ML proxies
        Use the auxiliary sample of the overall sample splitting.
    '''

    def __init__(
        self,
        model,
        n_folds,
        i,
        epsilon=0.001
    ):
        '''
        Arguments:
            model(str): name of the ML method to use:
                            - "enet": ElasticNet
                            - "rf": RandomForest
                            - "gb": GradientBoostingRegressor
            n_folds(int): number of folds for Kfold parametertuning
            i(int): iteration of the meta loop   
            epsilon(float): tolerance for adding noise to predictions
                            in self.predict() -> check_var_predictions() 
        '''
        self.model = model
        self.n_folds = n_folds
        self.epsilon = epsilon
        self.i = i
    

    def fit(
        self,
        X,
        y,
        check_input=True,
    ):
        ''' Fit first stage models to data seperately

        Arguments:
            X(pd.DataFrame): covariate data, array-like of shape (n_samples, n_features)
            y(pd.Series): outcome variable, array-like of shape (n_samples,)
            check_input(bool): check whether inputs match requirements

        Attributes:
            self._B(): baseline proxy model
            self._S(): treatment proxy model
        '''
        if check_input:
            try:
                assert(X.shape[0] == y.shape[0])
            except:
                raise ValueError(
                    'Input dimensions of X and y do not match.'
                )

        # split on treatment indicator:
        X['T'] = X['T'].astype(int)
        X_t = X[X['T'] == 1]
        X_b = X[X['T'] == 0]

        # drop Treatment indicator
        X_t.drop('T', axis=1, inplace=True)
        X_b.drop('T', axis=1, inplace=True)

        # select corresponding y
        y_t = y[y.index.isin(X_t.index)]
        y_b = y[y.index.isin(X_b.index)]

        model_zoo = {
            'enet': ElasticNetCV,
            'gb': GradientBoostingRegressor,
            'rf': RandomForestRegressor
        }
        assert(self.model in model_zoo.keys())

        if self.model == 'gb' or self.model == 'rf':
            model_B = make_pipeline(
                StandardScaler(with_mean=False),
                model_zoo[self.model](random_state=self.i)
            )
            model_S = make_pipeline(
                StandardScaler(with_mean=False),
                model_zoo[self.model](random_state=self.i)
            )
            model_B.fit(X_b, y_b)
            model_S.fit(X_t, y_t)

        elif self.model == 'enet':
            model_B = make_pipeline(
                StandardScaler(with_mean=False),
                model_zoo[self.model](cv=self.n_folds, random_state=self.i)
            )
            model_S = make_pipeline(
                StandardScaler(with_mean=False),
                model_zoo[self.model](cv=self.n_folds, random_state=self.i)
            )
            model_B.fit(X_b, y_b)
            model_S.fit(X_t, y_t)
        
        self._model_B = model_B
        self._model_S = model_S

        self._fitted = True
        
        return self

    
    def predict(self, X_main, T_main, y_main):
        ''' Perform first stage estimations on main sample.
            Predict y_hat(1) with model_S
            predict y_hat(0) with model_B
            Use for each the real observations of y_0/y_1 and predict
            only the counterfactual, which was not observed.

        Arguments:
            X_main(pd.DataFrame):   frame storing the covariate data
                                    of the main sample
            T_main(pd.Series): treatment indicator for main sample  
            y_main(pd.Series): outcome of interest for main sample
        
        Returns:
            S_z(pd.Series): series storing the true values of y_1 and the predictions y_hat_1 for 
                            observations that had treatment status == 0
            B_z(pd.Series): series storing the true values of y_0 and the predicitons y_hat_0 for 
                            observations that had treatmen status == 1
        '''
        assert(self._fitted == True)

        index_predict_0 = T_main[T_main==1.0].index
        index_predict_1 = T_main[T_main==0.0].index

        X_0 = X_main[X_main.index.isin(index_predict_0)]
        X_1 = X_main[X_main.index.isin(index_predict_1)]

        y_hat_0 = pd.Series(self._model_B.predict(X_0), index=X_0.index)
        y_hat_1 = pd.Series(self._model_S.predict(X_1), index=X_1.index)

        # check variation in predictions
        y_hat_0, y_hat_1 = self.check_var_predictions(
            baseline_preds=y_hat_0,
            treatment_preds=y_hat_1,
            target_main=y_main,
            epsilon=self.epsilon
        )

        B_z = pd.Series(index=X_main.index)
        B_z[B_z.index.isin(index_predict_0)] = y_hat_0
        B_z[B_z.index.isin(index_predict_1)] = y_main[y_main.index.isin(index_predict_1)].values
        y_1 = pd.Series(index=X_main.index)
        y_1[y_1.index.isin(index_predict_1)] = y_hat_1
        y_1[y_1.index.isin(index_predict_0)] = y_main[y_main.index.isin(index_predict_0)].values

        S_z = y_1 - B_z

        return (S_z, B_z)


    @staticmethod
    def check_var_predictions(baseline_preds, treatment_preds, target_main, epsilon):
        ''' Add noise to predictions of first stage if they do not vary.
            As suggested by Chernozhukov et al. (2018, p.29 step 2a)

        Arguments:
            basline_preds(array-like): predictions of the first stage baseline model
            treatment_preds(array-like): predictions of the first stage treatment model
            target_main(array-like): y
            epsiolon(float): criterion constant: if variation in predictions < epsilon, add noise
        '''
        # check variation in proxy estimatess
        var_baseline = np.var(baseline_preds)
        var_treatment = np.var(treatment_preds)
        
        # y sample variance
        sd_y_sample = np.std(target_main)

        # add 1/20th of sample variation of y,
        # as suggested in Chernozhukov et al. 2018
        if var_baseline < epsilon:
            noise= np.random.normal(
                loc=0,
                scale=sd_y_sample/20,
                size=len(baseline_preds)
            )
            indices_b = baseline_preds.index
            baseline_preds = pd.Series(baseline_preds + noise, index=indices_b)
        if var_treatment < epsilon:
            noise= np.random.normal(
                loc=0,
                scale=sd_y_sample/20,
                size=len(treatment_preds)
            )
            indices_t = treatment_preds.index
            treatment_preds = pd.Series(treatment_preds + noise, index=indices_t)

        return baseline_preds, treatment_preds


class GatesEstimator(BaseEstimator):
    ''' Class to estimate the Group Average Treatment Effects(GATEs)
    '''
    def __init__(
        self,
        n_groups,
        alpha,
        model=None,
        store_clan_series=False,
        fit_constant=1
        ):
        '''
        Arguments:
            n_groups(int): number of groups to estimate
            alpha(float): level for confidence intervals of GATES estimates
            model(str): which model was used as first stage
            store_clan_series(boolean): whether to store the unaggregated series
                                        of the least/most affected group as attribute
            fit_constant(int): whether to include constant in GATEs regression, 1=True, 0=False
        
        Attributes:
            self.labels(list): int list as labels for the binning
            self.X_la(float): CLAN average least affected
            self.X_ma(float): CLAN average most affected
        '''
        self.n_groups = n_groups
        self.labels = [n for n in range(1, n_groups+1)]
        self.alpha = alpha
        self.model = model
        self.store_clan_series = store_clan_series
        self.fit_constant = fit_constant


    def fit(
        self,
        baseline_preds,
        treatment_preds,
        treatment_indicator_main,
        target_main,
        X_m
    ):
        ''' Compute the group average treatment effects and CLAN

        Arguments:
            baseline_preds(array): predicitons of the untreated model
            treatment_preds(array): cate estimates
            treatment_indicator_main(array): treatment status values for the main sample
            target_main(array): outcome variable of interest
            X_m(array): covariate data of the main sample
        
        Attributes:
            gates_params(series): parameter estimates of the regression
            gates_ci(ndarray): upper and lower confidence bounds for estimates
            gates_pvalues(series): p-values for OLS estimates
            X_la(pd.Series): mean values of the least affected group covariates
            X_ma(pd.Series): mean values of the most affected group covariates
        '''
        # coerce types
        baseline_preds = pd.Series(baseline_preds)
        treatment_preds = pd.Series(treatment_preds)

        X_GATES = pd.concat(
            [baseline_preds, treatment_preds],
            axis=1
        )
        X_GATES.columns = ['B(Z)', 'S(Z)']

        X_GATES['qtile'] = pd.qcut(
            treatment_preds,
            self.n_groups,
            labels=self.labels,
            duplicates='drop'
        )

        # store indices for most and least affected for CLAN
        self._most_affected = X_GATES[X_GATES['qtile'] == X_GATES['qtile'].max()]
        self._least_affected = X_GATES[X_GATES['qtile'] == X_GATES['qtile'].min()]

        one_hot = OneHotEncoder().fit_transform(
            np.array(X_GATES.qtile).reshape(-1,1)
        )
        one_hot = pd.DataFrame(
            one_hot.toarray(),
            columns=self.labels,
            index=X_GATES.index
        )

        for col in one_hot.columns:
            one_hot[col] = one_hot[col] * treatment_indicator_main
    
        X_GATES = pd.concat(
            [X_GATES, one_hot],
            axis=1
        )
        
        if self.fit_constant == 1:
            X_GATES = sm.add_constant(X_GATES)
        
        X_GATES.drop('qtile', axis=1, inplace=True)

        gates = sm.OLS(target_main, X_GATES).fit(cov_type='HC0')

        self.gates_params = gates.params
        summary_df = convert_OLS_summary(gates, alpha=self.alpha/2)
        select_index = [str(x) for x in self.labels]
        self.summary = summary_df[summary_df.index.isin(select_index)]

        # compute CLAN params
        self.X_la = X_m.loc[X_m.index.isin(self._least_affected.index)].mean()
        self.X_ma = X_m.loc[X_m.index.isin(self._most_affected.index)].mean()

        if self.store_clan_series:
            self.X_la_series = X_m.loc[X_m.index.isin(self._least_affected.index)]
            self.X_ma_series = X_m.loc[X_m.index.isin(self._most_affected.index)]

        return self


class Orchestrator:
    ''' Conduct N runs of the sample splitting and estimation.
        Finally, aggregate results over runs with median.
        as suggested by Chernozhukov et al. 2018

    Attributes:
        i(int): current run number in range [0,N]
        results(dict): dictionary to store results of runs
        blp_results(pd.DataFrame): results of the BLP estimation
        gates_results(pd.DataFrame): results of the GATES estimation
        first_stage_metrics(pd.DataFrame): Chernozhukovs proposed first stage ML metrics
        clan_results(pd.DataFrame): differences between most and least affected groups
    '''

    def __init__(
        self,
        N: int,
        alpha: float,
        X: pd.DataFrame,
        y:pd.Series,
        T:pd.Series,
        n_folds: int,
        n_gates: int,
        model='rf'
        ):
        '''
        Arguments:
            N(int): number of runs to conduct
            alpha(float): confidence level
            X(pd.DataFrame): feature data
            y(pd.Series): outcome data
            T(pd.Series): treatment indicator
            n_folds(int): number of folds for cross validation in first stage
            n_gates(int): number of groups to use for the GATEs
            model(str): which model to use for first stage ml proxies
                        choose from: 
                            - rf (random forest) 
                            - gb (gradient boosting)
                            - enet (elastic-net regression)
        '''
        self.N = int(N)
        self.alpha = float(alpha)
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)
        self.T = pd.Series(T)
        self.n_folds = int(n_folds)
        self.n_gates = int(n_gates)
        try:
            assert(model in ['rf', 'gb', 'enet'])
        except:
            msg = 'The model must be one of ["rf", "gb", "enet"].'
            raise AssertionError(msg)
        self.model = model

        self.i = 0
        self.results = {}
    
    def run(self):
        ''' Run for N times the estimation of BLP and GATES
            also compute the CLANs for least and most affected groups of GATEs
        '''
        while self.i < self.N:
            X_a, X_m, y_a, y_m, T_a, T_m = train_test_split(
                self.X,
                self.y,
                self.T,
                test_size=0.5,
                random_state=self.i,
                stratify=self.T
            )

            # first stage proxy selection
            first_stage_estimator = FirstStageEstimator(
                model=self.model,
                n_folds=self.n_folds,
                i=self.i
            )
            first_stage_estimator.fit(
                X=pd.concat([X_a,T_a], axis=1),
                y=y_a
            )
            S_z, B_z = first_stage_estimator.predict(
                X_main=X_m,
                T_main=T_m,
                y_main=y_m
            )

            # compute BLP
            blp = BlpEstimator(model=self.model)
            blp.fit(
                baseline_preds=B_z,
                treatment_preds=S_z,
                treatment_indicator_main=T_m,
                target_main=y_m
            )

            gates = GatesEstimator(
                n_groups=self.n_gates,
                alpha=self.alpha,
                model=self.model
            )
            gates.fit(
                baseline_preds=B_z,
                treatment_preds=S_z,
                treatment_indicator_main=T_m,
                target_main=y_m,
                X_m=X_m
            )

            # compute performance measures for ML proxies
            measure_1 = blp.blp_params[-1]**2 * np.var(S_z)
            measure_2 = np.mean(gates.gates_params[3:])

            self.results[self.i] = {
                'blp': blp.summary,
                'gates': gates.summary,
                'first_stage_measures': [measure_1, measure_2],
                'clan_la': gates.X_la, 
                'clan_ma': gates.X_ma
            }

            self.i += 1
        
        print('Run finished')
        return self

    def aggregate_results(self):
        ''' Aggregate the results over the N runs
            Use the median to get the median point estimates.   
        '''

        blp_concat_gmli = pd.concat(
            [self.results[i]['blp'] for i in range(self.N)]
        )
        blp = blp_concat_gmli.groupby(level=0).median()
        blp.index = ['ß_1', 'ß_2']
        self.blp_results = blp

        gates_concat_gmli = pd.concat(
            [self.results[i]['gates'] for i in range(self.N)]
        )
        gates = gates_concat_gmli.groupby(level=0).median()
        gates.index = [str(x) for x in range(1,self.n_gates+1)]
        self.gates_results = gates

        # First Stage Performance Metrics Aggregation
        fs = pd.DataFrame(
            [self.results[i]['first_stage_measures'] for i in range(self.N)]
        ).median()
        fs.index = ['Measure 1', 'Measure 2']
        self.first_stage_metrics = fs

        # CLAN aggregation
        clan_la = pd.DataFrame(
            [self.results[i]['clan_la'] for i in range(self.N)]
        )
        clan_ma = pd.DataFrame(
            [self.results[i]['clan_ma'] for i in range(self.N)]
        )
        # test significance of difference in means
        test = sms.CompareMeans(sms.DescrStatsW(clan_la), sms.DescrStatsW(clan_ma))
        summary_diff_test = pd.DataFrame(test.summary())

        index = clan_la.columns
        cols = list(summary_diff_test.iloc[0][1:].astype(str))
        summary_diff_test = summary_diff_test[1:]
        summary_diff_test = summary_diff_test.iloc[:, 1:]
        summary_diff_test.set_index(index, inplace=True)
        summary_diff_test.columns = cols

        clan_la_median = clan_la.median()
        clan_ma_median = clan_ma.median()
        clan = pd.DataFrame(
            [clan_ma_median - clan_la_median, summary_diff_test['P>|t|']],
            index=['Difference', 'P-Value']
        )
        self.clan_results = clan

        return self


# helper function
def convert_OLS_summary(ols_obj, alpha):
    ''' Convert summary table of statsmodels.OLS object
        to pandas DataFrame
    
    Arguments:
        ols_obj(sm.RegressionResultsWrapper):   statsmodels ordinary
                                                least squares result object
    
    Returns:
        df_summary(pd.DataFrame):               frame storing the parameter estimates,
                                                p-values, std. errors, CIs 
    '''
    try:
        assert(isinstance(ols_obj, sm.regression.linear_model.RegressionResultsWrapper))
    except:
        raise TypeError('Input must be an instance of statsmodels OLS Result.')

    df_summary = pd.DataFrame(ols_obj.summary(alpha=alpha).tables[1])
    index = df_summary[0].astype(str).values[1:]
    cols = list(df_summary.iloc[0][1:].astype(str))
    df_summary = df_summary[1:]
    df_summary = df_summary.iloc[:, 1:]
    df_summary.set_index(index, inplace=True)
    df_summary.columns = cols
    df_summary = df_summary.astype(str).astype(float)

    return df_summary
