import os
import random
import time
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import miceforest as mf
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import shap
import streamlit as st
import xgboost as xgb
from catboost import CatBoostRegressor, Pool, cv
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from ngboost import NGBRegressor
from pymoo.optimize import minimize
from optuna.samplers import TPESampler, CmaEsSampler
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost.callback import EarlyStopping
from sklearn.inspection import PartialDependenceDisplay
from mpl_toolkits.mplot3d import Axes3D

st.set_option('deprecation.showPyplotGlobalUse', False)

class ModelApp:
    def __init__(self):
        self.df = None
        self.model = None

    def upload_file(self):
        self.input_filepath = "test_inputSSA.xlsx"
        self.output_filepath = "test_outputSSA.xlsx"

        if self.input_filepath and self.output_filepath:
            self.input_df = pd.read_excel(self.input_filepath)
            self.output_df = pd.read_excel(self.output_filepath)
            self.df_list = [pd.concat([self.input_df, self.output_df[[target]]], axis=1) for target in self.output_df.columns]
            return self.df_list

    def run_model(self, model_type="XGBoost", opt_algo="None"):
        st.session_state['models'] = []
        st.session_state['model_evaluations'] = []

        with st.spinner("Training model..."):
            for df in self.df_list:
                target_feature = df.columns[-1]
                X = df.drop(columns=[target_feature])
                y = df[target_feature]

                X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
                X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

                x_mean, x_std = 0, 1
                y_mean, y_std = 0, 1

                X_train = (X_train - x_mean) / x_std
                X_val = (X_val - x_mean) / x_std
                X_test = (X_test - x_mean) / x_std
                y_train = (y_train - y_mean) / y_std
                y_val = (y_val - y_mean) / y_std
                y_test = (y_test - y_mean) / y_std

                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                st.session_state['X_val'] = X_val
                st.session_state['y_val'] = y_val
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test

                st.session_state['X_train_orig'] = X_train
                st.session_state['y_train_orig'] = y_train
                st.session_state['X_val_orig'] = X_val
                st.session_state['y_val_orig'] = y_val
                st.session_state['X_test_orig'] = X_test
                st.session_state['y_test_orig'] = y_test

                st.session_state['x_mean'] = x_mean
                st.session_state['x_std'] = x_std
                st.session_state['y_mean'] = y_mean
                st.session_state['y_std'] = y_std

                if model_type == "XGBoost":
                    self.run_xgboost(opt_algo)

    def run_xgboost(self, opt_algo='None'):
        param_space = {
            'eta': [0.01, 0.02, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'lambda': [0, 0.01, 0.1, 1, 10]
        }

        def objective(trial):
            params = {
                'eta': trial.suggest_categorical('eta', param_space['eta']),
                'max_depth': trial.suggest_categorical('max_depth', param_space['max_depth']),
                'subsample': trial.suggest_categorical('subsample', param_space['subsample']),
                'colsample_bytree': trial.suggest_categorical('colsample_bytree', param_space['colsample_bytree']),
                'lambda': trial.suggest_categorical('lambda', param_space['lambda']),
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'silent': 1,
                'seed': 42,
                'n_jobs': -1
            }
            model = xgb.XGBRegressor(**params, early_stopping_rounds=20)
            model.fit(st.session_state['X_train'], st.session_state['y_train'],
                      eval_set=[(st.session_state['X_val'], st.session_state['y_val'])],
                      verbose=False)
            preds = model.predict(st.session_state['X_val'])
            rmse = np.sqrt(metrics.mean_squared_error(st.session_state['y_val'], preds))
            return rmse

        if opt_algo == 'TPE':
            study = optuna.create_study(direction='minimize', sampler=TPESampler())
            study.optimize(objective, n_trials=100)
            best_params = study.best_params
        elif opt_algo == 'CMA-ES':
            study = optuna.create_study(direction='minimize', sampler=CmaEsSampler())
            study.optimize(objective, n_trials=100)
            best_params = study.best_params
        else:
            best_params = {
                'eta': 0.02,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'lambda': 1
            }

        params_xgb = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'silent': 1,
            'seed': 42,
            'n_jobs': -1,
            **best_params
        }

        self.model = xgb.XGBRegressor(**params_xgb, early_stopping_rounds=20)
        self.model.fit(st.session_state['X_train'], st.session_state['y_train'],
                       eval_set=[(st.session_state['X_val'], st.session_state['y_val'])],
                       verbose=False)
        st.session_state['models'].append(self.model)

class InversionProblemJMetal(FloatProblem):
    def __init__(self, models, target_values, lower_bound, upper_bound):
        super().__init__()
        self.models = models
        self.target_values = target_values
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.direction_names = ["MINIMIZE", "MINIMIZE"]
        self.history = []
        self.history_objectives = []
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def name(self) -> str:
        return "InversionProblemJMetal"

    def evaluate(self, solution: FloatSolution) -> None:
        inputs = solution.variables
        self.history.append(inputs.copy())
        predictions = [model.predict([inputs])[0] for model in self.models]
        solution.objectives[0] = abs(predictions[0] - self.target_values[0])
        solution.objectives[1] = abs(predictions[1] - self.target_values[1])
        self.history_objectives.append(solution.objectives.copy())

    def get_name(self) -> str:
        return "InversionProblemJMetal"

def nsga3_inversion_jmetal(model_idxs, target_values):
    lower_bound = st.session_state.get('lower_bound', [0.3, 1, 15, 35, 100, 30])
    upper_bound = st.session_state.get('upper_bound', [0.3, 1, 30, 35, 100, 30])

    models = [st.session_state['models'][idx] for idx in model_idxs]
    X_train = st.session_state['X_train']
    num_variables = X_train.shape[1]

    if len(lower_bound) != num_variables or len(upper_bound) != num_variables:
        st.error("Number of parameter bounds does not match number of input features")
        return pd.DataFrame(), None, None

    problem = InversionProblemJMetal(models, target_values, lower_bound, upper_bound)

    algorithm = NSGAIII(
        problem=problem,
        population_size=100,
        reference_directions=UniformReferenceDirectionFactory(2, n_points=92),
        mutation=PolynomialMutation(probability=1.0 / num_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=30),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000),
    )

    with st.spinner("Running optimization..."):
        algorithm.run()

    try:
        solutions = algorithm.get_result()
    except AttributeError:
        solutions = algorithm.solutions

    front = get_non_dominated_solutions(solutions)
    best_solution = front[0]
    solution_df = pd.DataFrame([best_solution.variables], columns=X_train.columns)

    df_history = pd.DataFrame(problem.history, columns=X_train.columns)
    changing_params = [col for col, lb, ub in zip(df_history.columns, problem.lower_bound, problem.upper_bound) if lb != ub]
    df_history_filtered = df_history[changing_params]

    best_values = best_solution.variables

    fig, axes = plt.subplots(len(changing_params), 1, figsize=(10, 2.5 * len(changing_params)), facecolor='white')
    if len(changing_params) == 1:
        axes = [axes]

    param_indices = {param: idx for idx, param in enumerate(X_train.columns) if param in changing_params}

    for ax, param in zip(axes, changing_params):
        ax.plot(df_history_filtered[param], color='#1f77b4', alpha=0.7, label='Optimization Process')
        best_idx = len(df_history_filtered) - 1
        best_val = best_values[param_indices[param]]
        ax.scatter(best_idx, best_val, color='red', marker='*', s=120, edgecolor='black', zorder=10, label='Optimal Value')
        ax.set_title(f'Optimization Process for {param}', fontsize=12)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

        voltage_html = f"""
        <center>
        <div style="margin: 20px 0;">
        <span style="font-size: 24px; font-weight: bold;">Selected Voltage Value:</span>
        <span style="
            font-size: 28px;
            font-weight: 900;
            color: #2c3e50;
            background-color: #ecf0f1;
            border: 2px solid #3498db;
            border-radius: 5px;
            padding: 5px 15px;
            margin-left: 10px;
        ">{best_val:.2f}V</span>
        </div>
        </center>
        """
        st.markdown(voltage_html, unsafe_allow_html=True)

    plt.tight_layout()
    return solution_df, fig, None

if __name__ == "__main__":
    app = ModelApp()
    st.title("Intelligent voltage regulation for direct electric curing of concrete")
    st.info("***Author: Yuting Zhang-School of Civil Engineering, Central South University***")

    uploaded_df_list = app.upload_file()
    if uploaded_df_list is not None:
        app.run_model()

    st.sidebar.header("Parameter Settings")
    param_bounds = []
    param_names = {
        0: "Water-Cement Ratio",
        1: "Cement to Binder Ratio",
        3: "Sand Ratio (%)",
        4: "Specimen Length (mm)",
        5: "Initial Temperature (°C)"
    }
    for i in range(6):
        if i == 2:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                lower = col1.number_input("Voltage Lower Bound (V)", value=15.0, key="param_2_lower")
            with col2:
                upper = col2.number_input("Voltage Upper Bound (V)", value=30.0, key="param_2_upper")
            param_bounds.append((lower, upper))
        else:
            display_name = param_names.get(i, f"Parameter {i}")
            default_values = {0: 0.3, 1: 1.0, 3: 35.0, 4: 100.0, 5: 30.0}
            default_value = default_values.get(i, 0.0)
            value = st.sidebar.number_input(display_name, value=default_value, key=f"param_{i}_value")
            param_bounds.append((value, value))

    st.session_state['lower_bound'] = [pb[0] for pb in param_bounds]
    st.session_state['upper_bound'] = [pb[1] for pb in param_bounds]

    model_idxs = (0, 1)
    model_names = {0: "Target Peak Temperature (°C)", 1: "Target Peak Time (min)"}
    target_values = [
        st.sidebar.number_input(model_names[idx], value=60.0 if idx == 0 else 360.0, key=f"target_{idx}")
        for idx in model_idxs
    ]

    # Add the promotional sentence
    st.markdown("""
            <div style="margin-top: 20px; text-align: center; font-size: 18px;">
            If you encounter similar regression-multi-objective optimization tasks, please visit 
            <a href="https://dec.zhangyuting.cn/" target="_blank">https://dec.zhangyuting.cn/</a>, 
            which includes the full process of data upload, regression prediction, SHAP interpretability analysis, 
            and multi-objective optimization.
            </div>
            """, unsafe_allow_html=True)

    if st.sidebar.button("Run Optimization", key="run_nsga3_inversion_button_unique_1"):
        parameter_result, evolution_plot, _ = nsga3_inversion_jmetal(model_idxs, target_values)
        st.subheader("Optimized Parameters")
        st.dataframe(parameter_result.style.format("{:.4f}"))
        st.pyplot(evolution_plot)

