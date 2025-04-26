
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


class ModelApp:
    def __init__(self):
        self.df = None
        self.model = None

    def upload_file(self):
        """上传输入和输出数据集。"""
        self.input_filepath = st.file_uploader("上传输入数据集 (.xlsx)", type="xlsx", key="input")
        self.output_filepath = st.file_uploader("上传输出数据集 (.xlsx)", type="xlsx", key="output")

        if self.input_filepath and self.output_filepath:
            self.input_df = pd.read_excel(self.input_filepath)
            self.output_df = pd.read_excel(self.output_filepath)

            # 检查上传的文件是否为空
            if self.input_df.empty or self.output_df.empty:
                st.error("一个或两个上传的文件为空。请上传有效的数据集。")
                return None

            # 确保两个DataFrame具有相同的索引
            if not self.input_df.index.equals(self.output_df.index):
                st.error("输入和输出数据集必须具有相同的索引。")
                return None

            # 创建数据集列表，分别组合输入和每个输出列
            self.df_list = [pd.concat([self.input_df, self.output_df[[target]]], axis=1) for target in
                            self.output_df.columns]
            return self.df_list

    def run_model(self, target_feature, model_type, opt_algo):
        """运行选定的模型和优化算法。"""
        st.session_state['models'] = []  # 重置模型列表
        st.session_state['model_evaluations'] = []  # 重置模型评估结果

        for df in self.df_list:
            target_feature = df.columns[-1]  # 最后一列是目标特征
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

            # 更新会话状态
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
            elif model_type == "CatBoost":
                self.run_catboost(opt_algo)
            elif model_type == "NGBoost":
                self.run_ngboost(opt_algo)
            else:
                st.error("无效的模型选择")

    def run_lightgbm(self, opt_algo='None'):
        """运行LightGBM模型和选定的优化算法。"""
        param_space = {
            'learning_rate': [0.01, 0.02, 0.05, 0.1],
            'num_leaves': [31, 63, 127, 255],
            'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
            'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
            'bagging_freq': [0, 1, 2, 4, 8]
        }

        rmse_values = []

        def objective(trial):
            params = {
                'learning_rate': trial.suggest_categorical('learning_rate', param_space['learning_rate']),
                'num_leaves': trial.suggest_categorical('num_leaves', param_space['num_leaves']),
                'feature_fraction': trial.suggest_categorical('feature_fraction', param_space['feature_fraction']),
                'bagging_fraction': trial.suggest_categorical('bagging_fraction', param_space['bagging_fraction']),
                'bagging_freq': trial.suggest_categorical('bagging_freq', param_space['bagging_freq']),
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1,
                'seed': 42,
                'n_jobs': -1
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(st.session_state['X_train'], st.session_state['y_train'],
                      eval_set=[(st.session_state['X_val'], st.session_state['y_val'])],
                      eval_metric='rmse', callbacks=[lgb.early_stopping(stopping_rounds=20)])
            preds = model.predict(st.session_state['X_val'])
            rmse = np.sqrt(metrics.mean_squared_error(st.session_state['y_val'], preds))
            rmse_values.append(rmse)
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
                'learning_rate': 0.02,
                'num_leaves': 127,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.9,
                'bagging_freq': 4
            }

        params_lgb = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'seed': 42,
            'n_jobs': -1,
            **best_params
        }

        self.model = lgb.LGBMRegressor(**params_lgb)
        self.model.fit(st.session_state['X_train'], st.session_state['y_train'],
                       eval_set=[(st.session_state['X_val'], st.session_state['y_val'])],
                       eval_metric='rmse', callbacks=[lgb.early_stopping(stopping_rounds=20)])

        st.session_state['models'].append(self.model)
        self.evaluate_model()

        # 绘制RMSE变化图
        self.plot_rmse_change(rmse_values)

    def plot_rmse_change(self, rmse_values):
        """绘制RMSE随迭代的变化图。"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rmse_values, mode='lines+markers', name='RMSE'))
        fig.update_layout(title='RMSE随迭代的变化图', xaxis_title='迭代次数', yaxis_title='RMSE')
        st.plotly_chart(fig)

    def run_xgboost(self, opt_algo='None'):
        """运行XGBoost模型和选定的优化算法。"""
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

    def run_catboost(self, opt_algo='None'):
        """运行CatBoost模型和选定的优化算法。"""
        param_space = {
            'learning_rate': [0.01, 0.02, 0.05, 0.1],
            'depth': [4, 5, 6, 7],
            'l2_leaf_reg': [3, 5, 7, 9],
            'bagging_temperature': [0.1, 0.2, 0.3, 0.4, 0.5]
        }

        def objective(trial):
            params = {
                'learning_rate': trial.suggest_categorical('learning_rate', param_space['learning_rate']),
                'depth': trial.suggest_categorical('depth', param_space['depth']),
                'l2_leaf_reg': trial.suggest_categorical('l2_leaf_reg', param_space['l2_leaf_reg']),
                'bagging_temperature': trial.suggest_categorical('bagging_temperature',
                                                                 param_space['bagging_temperature']),
                'logging_level': 'Silent',
                'eval_metric': 'RMSE',
                'loss_function': 'RMSE',
                'random_seed': 42
            }
            model = CatBoostRegressor(**params)
            model.fit(st.session_state['X_train'], st.session_state['y_train'],
                      eval_set=(st.session_state['X_val'], st.session_state['y_val']),
                      early_stopping_rounds=20, verbose=False)
            preds = model.predict(st.session_state['X_val'])
            rmse = np.sqrt(metrics.mean_squared_error(st.session_state['y_val'], preds))
            return rmse

        if opt_algo == 'TPE':
            study = optuna.create_study(direction='minimize', sampler=TPESampler())
            study.optimize(objective, n_trials=50)
            best_params = study.best_params
        elif opt_algo == 'CMA-ES':
            study = optuna.create_study(direction='minimize', sampler=CmaEsSampler())
            study.optimize(objective, n_trials=50)
            best_params = study.best_params
        else:
            best_params = {
                'learning_rate': 0.02,
                'depth': 5,
                'l2_leaf_reg': 3,
                'bagging_temperature': 0.4
            }

        params_catboost = {
            'logging_level': 'Silent',
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'random_seed': 42,
            **best_params
        }

        self.model = CatBoostRegressor(**params_catboost)
        self.model.fit(st.session_state['X_train'], st.session_state['y_train'],
                       eval_set=(st.session_state['X_val'], st.session_state['y_val']),
                       early_stopping_rounds=20, verbose=False)

        st.session_state['models'].append(self.model)
        self.evaluate_model()

    def run_ngboost(self, opt_algo='None'):
        """运行NGBoost模型和选定的优化算法。"""
        param_space = {
            'learning_rate': [0.01, 0.02, 0.05, 0.1],
            'n_estimators': [100, 200, 300, 400, 500],
            'minibatch_frac': [0.5, 0.7, 0.9, 1.0]
        }

        def objective(trial):
            params = {
                'learning_rate': trial.suggest_categorical('learning_rate', param_space['learning_rate']),
                'n_estimators': trial.suggest_categorical('n_estimators', param_space['n_estimators']),
                'minibatch_frac': trial.suggest_categorical('minibatch_frac', param_space['minibatch_frac'])
            }
            model = NGBRegressor(**params)
            model.fit(st.session_state['X_train'], st.session_state['y_train'], early_stopping_rounds=20)
            preds = model.predict(st.session_state['X_val'])
            rmse = np.sqrt(metrics.mean_squared_error(st.session_state['y_val'], preds))
            return rmse

        if opt_algo == 'TPE':
            study = optuna.create_study(direction='minimize', sampler=TPESampler())
            study.optimize(objective, n_trials=10)
            best_params = study.best_params
        elif opt_algo == 'CMA-ES':
            study = optuna.create_study(direction='minimize', sampler=CmaEsSampler())
            study.optimize(objective, n_trials=10)
            best_params = study.best_params
        else:
            best_params = {
                'learning_rate': 0.02,
                'n_estimators': 300,
                'minibatch_frac': 0.9
            }

        params_ngboost = {
            **best_params
        }

        self.model = NGBRegressor(**params_ngboost)
        self.model.fit(st.session_state['X_train'], st.session_state['y_train'], early_stopping_rounds=20)

        preds_val = self.model.predict(st.session_state['X_val'])
        val_rmse = np.sqrt(metrics.mean_squared_error(st.session_state['y_val'], preds_val))
        print(f'验证集RMSE: {val_rmse}')

        st.session_state['models'].append(self.model)
        self.evaluate_model()

    def evaluate_model(self):
        """评估训练好的模型并显示结果。"""
        actual_y_test = (st.session_state['y_test'] * st.session_state['y_std']) + st.session_state['y_mean']

        if st.session_state['models']:
            st.session_state['trained'] = True
            st.success("模型训练成功。")
        else:
            st.session_state['trained'] = False
            st.error("模型训练失败。")
            return

        # 获取当前训练好的模型（每次训练后评估）
        model = st.session_state['models'][-1]
        preds = model.predict(st.session_state['X_test'])
        preds = (preds * st.session_state['y_std']) + st.session_state['y_mean']  # 反归一化

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

        # 显示当前模型的评估结果
        idx = len(st.session_state['model_evaluations'])
        st.subheader(f"模型 {idx}")
        st.write(f"均方误差 (MSE): {mse}")
        st.write(f"均方根误差 (RMSE): {rmse}")
        st.write(f"平均绝对误差 (MAE): {mae}")
        st.write(f"决定系数 (R-squared): {r2}")

    def generate_shap_plot(self, dataset, cmap):
        """生成SHAP汇总图。"""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("模型尚未训练。请先训练模型。")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"模型 {idx + 1} SHAP汇总图")

            explainer = shap.TreeExplainer(model)

            if dataset == "训练集":
                shap_values = explainer.shap_values(st.session_state['X_train'])
                data = st.session_state['X_train']
            elif dataset == "验证集":
                shap_values = explainer.shap_values(st.session_state['X_val'])
                data = st.session_state['X_val']
            else:
                shap_values = explainer.shap_values(st.session_state['X_test'])
                data = st.session_state['X_test']

            # 显示SHAP汇总图
            shap.summary_plot(shap_values, data, plot_type="bar", max_display=20, show=False, cmap=cmap)
            st.pyplot(bbox_inches='tight')

            shap.summary_plot(shap_values, data, plot_type="dot", max_display=20, show=False, cmap=cmap)
            st.pyplot(bbox_inches='tight')

            shap.summary_plot(shap_values, data, plot_type="layered_violin", max_display=20, show=False, cmap=cmap)
            st.pyplot(bbox_inches='tight')

            # 保存图表供未来使用
            self.save_plot(f"model_{idx + 1}_shap_summary_bar", shap.summary_plot, shap_values, data, plot_type="bar",
                           max_display=20, show=False, cmap=cmap)
            self.save_plot(f"model_{idx + 1}_shap_summary_dot", shap.summary_plot, shap_values, data, plot_type="dot",
                           max_display=20, show=False, cmap=cmap)
            self.save_plot(f"model_{idx + 1}_shap_summary_violin", shap.summary_plot, shap_values, data, plot_type="layered_violin",
                           max_display=20, show=False, cmap=cmap)

            # 显示SHAP值表
            st.subheader(f"模型 {idx + 1} SHAP值表")

            # 计算平均绝对SHAP值以表示特征重要性
            if isinstance(shap_values, list):
                shap_values_abs_mean = [np.abs(shap_values[class_idx]).mean(axis=0) for class_idx in
                                        range(len(shap_values))]
                importance_df = pd.DataFrame(shap_values_abs_mean, columns=data.columns)
            else:
                shap_values_abs_mean = np.abs(shap_values).mean(axis=0)
                importance_df = pd.DataFrame(shap_values_abs_mean, index=data.columns, columns=["平均SHAP值"])

            # 按重要性排序
            importance_df = importance_df.sort_values(by=["平均SHAP值"], ascending=False)

            # 显示数据框
            st.dataframe(importance_df)

    def generate_dependence_plot(self, feature, interaction_feature):
        """生成SHAP依赖图。"""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("模型尚未训练。请先训练模型。")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"模型 {idx + 1} SHAP依赖图")

            explainer = shap.TreeExplainer(model)

            if dataset == "训练集":
                shap_values = explainer.shap_values(st.session_state['X_train'])
                data = st.session_state['X_train']
            elif dataset == "验证集":
                shap_values = explainer.shap_values(st.session_state['X_val'])
                data = st.session_state['X_val']
            else:
                shap_values = explainer.shap_values(st.session_state['X_test'])
                data = st.session_state['X_test']

            fig, ax = plt.subplots(figsize=(10, 7))
            shap.dependence_plot(
                feature,
                shap_values,
                data,
                interaction_index=interaction_feature if interaction_feature != "None" else None,
                show=False,
                ax=ax,
            )
            st.pyplot(fig)

            features = [feature, interaction_feature]
            fig, ax = plt.subplots(figsize=(10, 7))
            PartialDependenceDisplay.from_estimator(
                model,
                data,
                features=[features],
                kind='average',
                grid_resolution=100,
                contour_kw={'cmap': cmap, 'alpha': 0.8},
                ax=ax
            )
            st.pyplot(fig)
            featureextra = '水泥含量'
            features3d = [feature, interaction_feature, featureextra]
            grid_resolution = 20
            feature_1_vals = np.linspace(data[feature].min(), data[feature].max(), grid_resolution)
            feature_2_vals = np.linspace(data[interaction_feature].min(), data[interaction_feature].max(), grid_resolution)
            feature_3_vals = np.linspace(data[featureextra].min(), data[featureextra].max(), grid_resolution)

            grid_1, grid_2, grid_3 = np.meshgrid(feature_1_vals, feature_2_vals, feature_3_vals)

            grid_points = np.c_[grid_1.ravel(), grid_2.ravel(), grid_3.ravel()]
            X_grid = np.zeros((grid_points.shape[0], data.shape[1]))
            X_grid[:, data.columns.get_loc(feature)] = grid_points[:, 0]
            X_grid[:, data.columns.get_loc(interaction_feature)] = grid_points[:, 1]
            X_grid[:, data.columns.get_loc(featureextra)] = grid_points[:, 2]

            preds = model.predict(X_grid)
            fig = plt.figure(figsize=(10, 8), dpi=1200)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(grid_1.ravel(), grid_2.ravel(), grid_3.ravel(), c=preds, cmap=cmap, alpha=0.8)

            plt.colorbar(sc, ax=ax, label='预测值')
            ax.set_xlabel(feature)
            ax.set_ylabel(interaction_feature)
            ax.set_zlabel(featureextra)
            st.pyplot(fig)
            fig.savefig(f"model_{idx + 1}_3d_pdp_ice_{feature}_{interaction_feature}.svg", format='svg', dpi=600)

            self.save_plot(f"model_{idx + 1}_shap_dependence_{feature}", shap.dependence_plot, feature, shap_values,
                           data, interaction_index=interaction_feature if interaction_feature != "None" else None,
                           show=False)

            self.save_plot(f"model_{idx + 1}_2d_shap_dependence_{feature}_{interaction_feature}", PartialDependenceDisplay.from_estimator, model, data, features=[features],
                           kind='average', grid_resolution=50, contour_kw={'cmap': cmap, 'alpha': 0.8})

    def generate_force_plot(self, sample_index):
        """生成SHAP力图。"""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("模型尚未训练。请先训练模型。")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"模型 {idx + 1} SHAP力图 (样本 {sample_index})")

            explainer = shap.TreeExplainer(model)

            if dataset == "训练集":
                shap_values = explainer.shap_values(st.session_state['X_train'])[sample_index]
                data = st.session_state['X_train'].iloc[sample_index]
            elif dataset == "验证集":
                shap_values = explainer.shap_values(st.session_state['X_val'])[sample_index]
                data = st.session_state['X_val'].iloc[sample_index]
            else:
                shap_values = explainer.shap_values(st.session_state['X_test'])[sample_index]
                data = st.session_state['X_test'].iloc[sample_index]

            expected_value = explainer.expected_value

            shap.initjs()
            shap.force_plot(
                expected_value,
                shap_values,
                data,
                matplotlib=True,
                show=False
            )
            st.pyplot()

            self.save_plot(f"model_{idx + 1}_shap_force_sample_{sample_index}", shap.force_plot, expected_value,
                           shap_values, data, matplotlib=True, show=False)

    def generate_interaction_plot(self):
        """生成SHAP交互汇总图。"""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("模型尚未训练。请先训练模型。")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"模型 {idx + 1} SHAP交互汇总图")

            explainer = shap.TreeExplainer(model)

            if dataset == "训练集":
                shap_values = explainer.shap_interaction_values(st.session_state['X_train'])
                data = st.session_state['X_train']
            elif dataset == "验证集":
                shap_values = explainer.shap_interaction_values(st.session_state['X_val'])
                data = st.session_state['X_val']
            else:
                shap_values = explainer.shap_interaction_values(st.session_state['X_test'])
                data = st.session_state['X_test']

            shap_interaction_values = shap_values
            shap.summary_plot(shap_interaction_values, data, show=False)
            st.pyplot(bbox_inches='tight')

            self.save_plot(f"model_{idx + 1}_shap_interaction", shap.summary_plot, shap_values, data, show=False)

    def generate_heatmap_plot(self, start, end):
        """生成SHAP热图。"""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("模型尚未训练。请先训练模型。")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"模型 {idx + 1} SHAP热图")

            explainer = shap.TreeExplainer(model)

            if dataset == "训练集":
                data = st.session_state['X_train'].iloc[start:end, :]
                feature_names = st.session_state['X_train'].columns
            elif dataset == "验证集":
                data = st.session_state['X_val'].iloc[start:end, :]
                feature_names = st.session_state['X_val'].columns
            else:
                data = st.session_state['X_test'].iloc[start:end, :]
                feature_names = st.session_state['X_test'].columns

            shap_values = explainer.shap_values(data)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            values = shap_values[start:end, :]
            base_values = explainer.expected_value

            shap_explanation = shap.Explanation(values, base_values=base_values, data=data,
                                                feature_names=feature_names)

            shap.plots.heatmap(shap_explanation, show=False)
            st.pyplot()

            self.save_plot(f"model_{idx + 1}_shap_heatmap", shap.plots.heatmap, shap_explanation, show=False)

    def generate_pdp_ice_plots(self, feature):
        """生成PDP/ICE图。"""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("模型尚未训练。请先训练模型。")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"模型 {idx + 1} PDP/ICE图 (特征: {feature})")

            fig, ax = plt.subplots(figsize=(5, 5))
            shap.plots.partial_dependence(feature, model.predict, st.session_state['X_train'], ice=True,
                                          model_expected_value=True, feature_expected_value=True, ax=ax, show=False)
            st.pyplot(fig)

            self.save_plot(f"model_{idx + 1}_pdp_ice_{feature}", shap.plots.partial_dependence, feature, model.predict,
                           st.session_state['X_train'], ice=True, model_expected_value=True, feature_expected_value=True, ax=ax,
                           show=False)
            fig.savefig(f"model_{idx + 1}_pdp_ice_{feature}.svg", format='svg', dpi=600)

    def save_models(self):
        """保存所有训练好的模型到文件。"""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("没有模型可以保存。请先训练模型。")
            return

        models_to_save = {
            'models': st.session_state['models'],
            'model_evaluations': st.session_state['model_evaluations'],
            'X_train': st.session_state['X_train'],
            'y_train': st.session_state['y_train'],
            'X_val': st.session_state['X_val'],
            'y_val': st.session_state['y_val'],
            'X_test': st.session_state['X_test'],
            'y_test': st.session_state['y_test'],
            'X_train_orig': st.session_state['X_train_orig'],
            'y_train_orig': st.session_state['y_train_orig'],
            'X_val_orig': st.session_state['X_val_orig'],
            'y_val_orig': st.session_state['y_val_orig'],
            'X_test_orig': st.session_state['X_test_orig'],
            'y_test_orig': st.session_state['y_test_orig'],
            'x_mean': st.session_state['x_mean'],
            'x_std': st.session_state['x_std'],
            'y_mean': st.session_state['y_mean'],
            'y_std': st.session_state['y_std']
        }
        if not os.path.exists('models'):
            os.makedirs('models')

        file_path = st.text_input("输入文件路径以保存模型:", "models/models.pkl")

        joblib.dump(models_to_save, file_path)
        st.success(f"模型已保存到 {file_path}")

    def load_models(self):
        """从文件加载已保存的模型。"""
        file_path = st.text_input("输入文件路径以加载模型:", "models/models.pkl")

        loaded_models = joblib.load(file_path)
        st.session_state['models'] = loaded_models['models']
        st.session_state['model_evaluations'] = loaded_models['model_evaluations']
        st.session_state['X_train'] = loaded_models['X_train']
        st.session_state['y_train'] = loaded_models['y_train']
        st.session_state['X_val'] = loaded_models['X_val']
        st.session_state['y_val'] = loaded_models['y_val']
        st.session_state['X_test'] = loaded_models['X_test']
        st.session_state['y_test'] = loaded_models['y_test']
        st.session_state['X_train_orig'] = loaded_models['X_train_orig']
        st.session_state['y_train_orig'] = loaded_models['y_train_orig']
        st.session_state['X_val_orig'] = loaded_models['X_val_orig']
        st.session_state['y_val_orig'] = loaded_models['y_val_orig']
        st.session_state['X_test_orig'] = loaded_models['X_test_orig']
        st.session_state['y_test_orig'] = loaded_models['y_test_orig']
        st.session_state['x_mean'] = loaded_models['x_mean']
        st.session_state['x_std'] = loaded_models['x_std']
        st.session_state['y_mean'] = loaded_models['y_mean']
        st.session_state['y_std'] = loaded_models['y_std']
        st.success(f"模型已从 {file_path} 加载")

    def upload_prediction_file(self):
        """上传预测输入数据集。"""
        self.prediction_file = st.file_uploader("上传预测输入数据集 (.csv)", type="csv", key="prediction")
        if self.prediction_file:
            self.prediction_df = pd.read_csv(self.prediction_file)
            st.success("预测输入文件上传成功。")

        def make_predictions_and_plot(self):
            """使用已加载的模型进行预测并绘制结果。"""
            if not hasattr(self, 'prediction_df'):
                st.error("请先上传预测输入文件。")
                return

            if 'models' not in st.session_state or not st.session_state['models']:
                st.error("没有已加载的模型。请先加载模型。")
                return

            all_predictions = []

            for idx, model in enumerate(st.session_state['models']):
                preds = model.predict(self.prediction_df)
                all_predictions.append(preds)
                print(preds)

            # 转置预测值，使每一行是一个样本的预测结果
            all_predictions = np.array(all_predictions).T

            # 限制为前20个模型的预测结果
            max_models_to_plot = 20
            all_predictions = all_predictions[:, :max_models_to_plot]

            # 绘制折线图
            fig_line, ax_line = plt.subplots(figsize=(12, 8))
            for idx, preds in enumerate(all_predictions):
                ax_line.plot(range(1, len(preds) + 1), preds, 'o-', label=f'样本 {idx + 1}')

                # 找到并标注前20个预测中的最大值
                max_pred = max(preds)
                max_pred_idx = np.argmax(preds) + 1
                ax_line.text(max_pred_idx, max_pred, f'{max_pred:.2f}', fontsize=10, verticalalignment='bottom')

            # 设置图表标题和轴标签
            ax_line.set_title('模型预测结果折线图')
            ax_line.set_xlabel('模型编号')
            ax_line.set_ylabel('预测值')
            ax_line.legend()
            st.pyplot(fig_line)

            # 保存图形
            fig_line.savefig("models_predictions.svg", format='svg', dpi=600)

        def save_plot(self, plot_name, plot_function, *args, **kwargs):
            """保存生成的图表。"""
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plot_path = os.path.join('plots', f'{plot_name}.png')
            plot_function(*args, **kwargs, savefig=plot_path)
            st.write(f"图表已保存到 {plot_path}")

        def run_app(self):
            """运行应用程序。"""
            st.title("模型训练和评估工具")

            # 上传数据集
            self.upload_file()

            if self.df_list:
                # 模型选择
                model_type = st.selectbox("选择模型类型", ["LightGBM", "XGBoost", "CatBoost", "NGBoost"])

                # 优化算法选择
                opt_algo = st.selectbox("选择优化算法", ["None", "TPE", "CMA-ES"])

                # 运行模型
                if st.button("训练模型"):
                    self.run_model(None, model_type, opt_algo)

                # 评估模型
                if st.button("评估模型"):
                    self.evaluate_model()

                # 生成SHAP图
                if st.button("生成SHAP图"):
                    dataset = st.selectbox("选择数据集", ["训练集", "验证集", "测试集"])
                    cmap = st.selectbox("选择颜色地图", ["coolwarm", "viridis", "plasma", "magma", "inferno", "cividis"])
                    self.generate_shap_plot(dataset, cmap)

                # 生成SHAP依赖图
                if st.button("生成SHAP依赖图"):
                    feature = st.selectbox("选择特征", st.session_state['X_train'].columns)
                    interaction_feature = st.selectbox("选择交互特征", ["None"] + list(st.session_state['X_train'].columns))
                    self.generate_dependence_plot(feature, interaction_feature)

                # 生成SHAP力图
                if st.button("生成SHAP力图"):
                    sample_index = st.number_input("选择样本索引", min_value=0, max_value=len(st.session_state['X_test']) - 1,
                                                   value=0)
                    self.generate_force_plot(sample_index)

                # 生成SHAP交互汇总图
                if st.button("生成SHAP交互汇总图"):
                    self.generate_interaction_plot()

                # 生成SHAP热图
                if st.button("生成SHAP热图"):
                    start = st.number_input("选择起始样本索引", min_value=0, max_value=len(st.session_state['X_test']) - 10,
                                            value=0)
                    end = st.number_input("选择结束样本索引", min_value=start + 1, max_value=len(st.session_state['X_test']),
                                          value=start + 10)
                    self.generate_heatmap_plot(start, end)

                # 生成PDP/ICE图
                if st.button("生成PDP/ICE图"):
                    feature = st.selectbox("选择特征", st.session_state['X_train'].columns)
                    self.generate_pdp_ice_plots(feature)

                # 保存模型
                if st.button("保存模型"):
                    self.save_models()

                # 加载模型
                if st.button("加载模型"):
                    self.load_models()

                # 上传预测输入文件
                if st.button("上传预测文件"):
                    self.upload_prediction_file()

                # 进行预测并绘制结果
                if st.button("进行预测"):
                    self.make_predictions_and_plot()

if __name__ == "__main__":
    app = ModelApp()

    st.title("直接电养护混凝土升温段电压智能选取模型")

    uploaded_df_list = app.upload_file()

    dataset_placeholder = st.empty()

    if uploaded_df_list is not None:
        with dataset_placeholder.container():
            st.sidebar.title("1. XGBoost 模型训练")
            st.write("输入数据集:")
            st.dataframe(app.input_df)
            # st.write("输出数据集:")
            # st.dataframe(app.output_df.head())

        model_type = st.sidebar.selectbox("选择基线模型", ["LightGBM", "XGBoost", "CatBoost", "NGBoost"])
        opt_algo = st.sidebar.selectbox("选择超参数优化算法", ["None", "TPE", "CMA-ES"])

        if st.sidebar.button("运行模型", key="run_model_button_unique_1"):
            app.run_model(None, model_type, opt_algo)  # 注意：此处不使用 target_feature
            dataset_placeholder.empty()

        st.sidebar.title("2. XGBoost 模型预测")
        # 保存和加载模型
        if st.sidebar.button("保存模型", key="save_models_button_unique"):
            dataset_placeholder.empty()
            app.save_models()
        if st.sidebar.button("加载模型", key="load_models_button_unique"):
            dataset_placeholder.empty()
            app.load_models()

        # 上传预测文件
        app.upload_prediction_file()

        if st.sidebar.button("运行预测", key="run_prediction_button_unique"):
            dataset_placeholder.empty()
            app.make_predictions_and_plot()

        st.sidebar.title("3. XGBoost 模型解释性")
        dataset = st.sidebar.selectbox("选择 SHAP 解释数据集",
                                       ["训练集", "验证集", "测试集"])
        cmap = st.sidebar.selectbox(
            "选择颜色方案",
            ["viridis", "Spectral", "coolwarm", "RdYlGn", "RdYlBu", "RdBu", "RdGy", "PuOr", "BrBG", "PRGn", "PiYG"]
        )

        if st.sidebar.button("生成 SHAP 汇总图", key="generate_shap_summary_plot_button_unique"):
            dataset_placeholder.empty()
            app.generate_shap_plot(dataset, cmap)

        features = app.input_df.columns.tolist()
        features.append("None")
        feature = st.sidebar.selectbox("选择 SHAP 依赖图特征", features, index=0)
        interaction_feature = st.sidebar.selectbox("选择交互特征", features, index=len(features) - 1)

        if st.sidebar.button("生成依赖图", key="generate_dependence_plot_button_unique"):
            dataset_placeholder.empty()
            app.generate_dependence_plot(feature, interaction_feature)

        sample_index = st.sidebar.number_input("选择力图的样本索引", min_value=0,
                                               max_value=len(st.session_state.X_train) - 1)
        if st.sidebar.button("生成 SHAP 力图", key="generate_force_plot_button_unique"):
            dataset_placeholder.empty()
            app.generate_force_plot(sample_index)

        if st.sidebar.button("生成交互图", key="generate_interaction_plot_button_unique"):
            dataset_placeholder.empty()
            app.generate_interaction_plot()

        heatmap_range = st.sidebar.text_input("选择热图数据范围 (起始:结束)", "0:20")
        range_values = heatmap_range.split(":")
        start, end = int(range_values[0]), int(range_values[1])
        if st.sidebar.button("生成 SHAP 热图", key="generate_heatmap_button_unique"):
            dataset_placeholder.empty()
            app.generate_heatmap_plot(start, end)

        pdp_features = st.sidebar.selectbox("选择 PDP/ICE 特征", st.session_state.X_train.columns.tolist())
        if st.sidebar.button("生成 PDP-ICE 图", key="generate_pdp_ice_plot_button_unique"):
            dataset_placeholder.empty()
            app.generate_pdp_ice_plots(pdp_features)

        st.sidebar.title("4. NSGAIII 模型")
        model_idxs = st.sidebar.multiselect("选择模型索引", range(len(st.session_state['models'])),
                                            default=[0])
        target_values = [st.sidebar.number_input(f"输入模型 {idx} 的目标值", value=0.0) for idx in
                         model_idxs]
        if st.sidebar.button("运行 NSGA-III 逆向优化", key="run_nsga3_inversion_button_unique_1"):
            # best_solution_df = nsga3_inversion_jmetal(model_idxs, target_values)
            # st.write("最佳解决方案:")
            # st.dataframe(best_solution_df)

            # 正确解包返回值
            parameter_result, evolution_plot, objective_plot = nsga3_inversion_jmetal(model_idxs, target_values)

            # 显示优化结果表格
            st.subheader("优化后的参数")
            st.dataframe(parameter_result.style.format("{:.4f}"))  # 单独显示DataFrame

            # 显示参数变化曲线
            st.subheader("参数演化过程")
            st.pyplot(evolution_plot)  # 单独显示图像
            # 显示目标值变化曲线
            st.subheader("目标值演化过程")
            st.pyplot(objective_plot)
