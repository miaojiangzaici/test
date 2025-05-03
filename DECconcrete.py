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
from jmetal.algorithm.multiobjective.nsgaiii import (
    NSGAIII,
    UniformReferenceDirectionFactory,
)
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
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

            self.df_list = [pd.concat([self.input_df, self.output_df[[target]]], axis=1) for target in
                            self.output_df.columns]
            return self.df_list

    def run_model(self, target_feature, model_type, opt_algo):
        st.session_state['models'] = []
        st.session_state['model_evaluations'] = []

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

            if model_type == "LightGBM":
                self.run_lightgbm(opt_algo)
            elif model_type == "XGBoost":
                self.run_xgboost(opt_algo)
            else:
                st.error("无效的模型选择")

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
        self.evaluate_model()

    def evaluate_model(self):
        actual_y_test = (st.session_state['y_test'] * st.session_state['y_std']) + st.session_state['y_mean']

        if st.session_state['models']:
            st.session_state['trained'] = True
            st.success("模型训练成功。")
        else:
            st.session_state['trained'] = False
            st.error("模型训练失败。")
            return

        model = st.session_state['models'][-1]
        preds = model.predict(st.session_state['X_test'])
        preds = (preds * st.session_state['y_std']) + st.session_state['y_mean']

        mse = metrics.mean_squared_error(actual_y_test, preds)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(actual_y_test, preds)
        r2 = metrics.r2_score(actual_y_test, preds)

        model_eval = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R-squared": r2
        }

        st.session_state['model_evaluations'].append(model_eval)

        idx = len(st.session_state['model_evaluations'])
        st.subheader(f"模型 {idx}")
        st.write(f"均方误差 (MSE): {mse}")
        st.write(f"均方根误差 (RMSE): {rmse}")
        st.write(f"平均绝对误差 (MAE): {mae}")
        st.write(f"决定系数 (R-squared): {r2}")


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

    # 必须实现的抽象方法
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
    # 获取用户设置的参数边界
    lower_bound = st.session_state.get('lower_bound', [0.3, 1, 15, 35, 100, 30])
    upper_bound = st.session_state.get('upper_bound', [0.3, 1, 30, 35, 100, 30])

    models = [st.session_state['models'][idx] for idx in model_idxs]
    X_train = st.session_state['X_train']
    num_variables = X_train.shape[1]

    # 检查参数数量是否匹配
    if len(lower_bound) != num_variables or len(upper_bound) != num_variables:
        st.error("参数边界数量与输入特征数量不匹配")
        return pd.DataFrame(), None, None

    # 初始化问题实例
    problem = InversionProblemJMetal(models, target_values, lower_bound, upper_bound)

    algorithm = NSGAIII(
        problem=problem,
        population_size=100,
        reference_directions=UniformReferenceDirectionFactory(2, n_points=92),
        mutation=PolynomialMutation(probability=1.0 / num_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=30),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000),
    )

    algorithm.run()

    try:
        solutions = algorithm.get_result()
    except AttributeError:
        solutions = algorithm.solutions

    front = get_non_dominated_solutions(solutions)
    best_solution = front[0]
    solution_df = pd.DataFrame([best_solution.variables], columns=X_train.columns)

    df_history = pd.DataFrame(problem.history, columns=X_train.columns)

    changing_params = [
        col for col, lb, ub in zip(df_history.columns, problem.lower_bound, problem.upper_bound)
        if lb != ub
    ]
    df_history_filtered = df_history[changing_params]

    best_values = best_solution.variables

    fig, axes = plt.subplots(
        len(changing_params), 1,
        figsize=(10, 2.5 * len(changing_params)),
        facecolor='white'
    )
    if len(changing_params) == 1:
        axes = [axes]

    param_indices = {param: idx for idx, param in enumerate(X_train.columns)
                     if param in changing_params}

    for ax, param in zip(axes, changing_params):
        line = ax.plot(df_history_filtered[param],
                       color='#1f77b4',
                       alpha=0.7,
                       label='Optimization Process')

        best_idx = len(df_history_filtered) - 1
        best_val = best_values[param_indices[param]]

        ax.scatter(best_idx, best_val,
                   color='red',
                   marker='*',
                   s=120,
                   edgecolor='black',
                   zorder=10,
                   label='Optimal Value')

        ax.set_title(f'智能选取的电压值为：{best_val:.2f}V', fontsize=30)
        ax.set_xlabel('训练步骤', fontsize=8)
        ax.set_ylabel('电压的值', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    df_objectives = pd.DataFrame(problem.history_objectives,
                                 columns=[f"Objective {i + 1}" for i in range(2)])

    fig_obj, ax_obj = plt.subplots(figsize=(10, 4), facecolor='white')
    for i, obj in enumerate(df_objectives.columns):
        ax_obj.plot(df_objectives[obj],
                    alpha=0.7,
                    label=f'Objective {i + 1} ({problem.direction_names[i]})')

    best_idx = len(df_objectives) - 1
    best_objs = df_objectives.iloc[best_idx]
    ax_obj.scatter([best_idx] * len(best_objs), best_objs,
                   color='red', marker='*', s=100,
                   edgecolor='black', zorder=10,
                   label='Optimal Solution')

    ax_obj.set_title('Objective Values Evolution')
    ax_obj.set_xlabel('Evaluation Steps')
    ax_obj.set_ylabel('Objective Value')
    ax_obj.grid(True, linestyle='--', alpha=0.5)
    ax_obj.legend()
    plt.tight_layout()
    return solution_df, fig, fig_obj


if __name__ == "__main__":
    app = ModelApp()
    st.title("直接电养护混凝土升温段电压智能选取模型")

    uploaded_df_list = app.upload_file()
    dataset_placeholder = st.empty()

    if uploaded_df_list is not None:
        with dataset_placeholder.container():
            st.sidebar.title("XGBoost 模型训练")
            st.write("先点击模型训练，输入参数后再进行电压智能选取")

        model_type = st.sidebar.selectbox("选择基准模型", ["XGBoost"])
        opt_algo = st.sidebar.selectbox("选择超参数优化算法", ["None", "TPE", "CMA-ES"])

        if st.sidebar.button("模型训练", key="run_model_button_unique_1"):
            app.run_model(None, model_type, opt_algo)
            dataset_placeholder.empty()

    # 参数边界设置部分
    st.sidebar.header("参数设置")
    param_bounds = []
    param_names = {
        0: "水灰比",
        1: "水泥占胶材比",
        3: "砂率(%)",
        4: "试件长度(mm)",
        5: "初始温度(℃)"
    }
    for i in range(6):
        if i == 2:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                lower = col1.number_input("电压下界(V)", value=15.0, key="param_2_lower")
            with col2:
                upper = col2.number_input("电压上界(V)", value=30.0, key="param_2_upper")
            param_bounds.append((lower, upper))
        else:
            # 获取参数显示名称
            display_name = param_names.get(i, f"参数{i}")
            default_values = {
                0: 0.3,
                1: 1.0,
                3: 35.0,
                4: 100.0,
                5: 30.0
            }
            default_value = default_values.get(i, 0.0)

            value = st.sidebar.number_input(
                f"{display_name}",  # 显示中文名称
                value=default_value,
                key=f"param_{i}_value"
            )
            param_bounds.append((value, value))

    # 保存参数边界到session_state
    st.session_state['lower_bound'] = [pb[0] for pb in param_bounds]
    st.session_state['upper_bound'] = [pb[1] for pb in param_bounds]

    st.sidebar.title("NSGAIII 模型")
    model_idxs = (0, 1)
    model_names = {
        0: "指定的峰值温度(℃)",
        1: "指定的达峰时间(min)"
    }

    target_values = [
        st.sidebar.number_input(
            f"输入{model_names[idx]}",  # 显示中文名称
            value=60.0 if idx == 0 else 360.0,  # 设置合理默认值
            key=f"target_{idx}"
        ) for idx in model_idxs
    ]

    if st.sidebar.button("运行程序，智能选取电压", key="run_nsga3_inversion_button_unique_1"):
        parameter_result, evolution_plot, objective_plot = nsga3_inversion_jmetal(model_idxs, target_values)

        st.subheader("优化后的参数")
        st.dataframe(parameter_result.style.format("{:.4f}"))

        st.pyplot(evolution_plot)


