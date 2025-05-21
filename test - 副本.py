import os
import random
import time

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import miceforest as mf
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
        """Upload input and output datasets."""
        self.input_filepath = st.file_uploader("Upload Input Dataset (.xlsx)", type="xlsx", key="input")
        self.output_filepath = st.file_uploader("Upload Output Dataset (.xlsx)", type="xlsx", key="output")

        if self.input_filepath and self.output_filepath:
            self.input_df = pd.read_excel(self.input_filepath)
            self.output_df = pd.read_excel(self.output_filepath)

            # Check if the uploaded files are empty
            if self.input_df.empty or self.output_df.empty:
                st.error("One or both of the uploaded files are empty. Please upload valid datasets.")
                return None

            # Ensure both DataFrames have the same index
            if not self.input_df.index.equals(self.output_df.index):
                st.error("Input and Output datasets must have the same index.")
                return None

            # Create dataset list by combining input with each output column separately
            self.df_list = [pd.concat([self.input_df, self.output_df[[target]]], axis=1) for target in
                            self.output_df.columns]
            return self.df_list

    def run_model(self, target_feature, model_type, opt_algo):
        """Run the selected model with the chosen optimization algorithm."""
        st.session_state['models'] = []  # Reset model list
        st.session_state['model_evaluations'] = []  # Reset model evaluation results

        for df in self.df_list:
            target_feature = df.columns[-1]  # The last column is the target feature
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

            # Update session state
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
                st.error("Invalid model selection")

    def run_lightgbm(self, opt_algo='None'):
        """Run LightGBM model with the chosen optimization algorithm."""
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

        # Plot RMSE change
        self.plot_rmse_change(rmse_values)

    def plot_rmse_change(self, rmse_values):
        """Plot the change in RMSE over iterations."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rmse_values, mode='lines+markers', name='RMSE'))
        fig.update_layout(title='RMSE Change Over Iterations', xaxis_title='Iteration', yaxis_title='RMSE')
        st.plotly_chart(fig)

    def run_xgboost(self, opt_algo='None'):
        """Run XGBoost model with the chosen optimization algorithm."""
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
        """Run CatBoost model with the chosen optimization algorithm."""
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
        """Run NGBoost model with the chosen optimization algorithm."""
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
        print(f'Validation RMSE: {val_rmse}')

        st.session_state['models'].append(self.model)
        self.evaluate_model()

    def evaluate_model(self):
        """Evaluate the trained model and display the results."""
        actual_y_test = (st.session_state['y_test'] * st.session_state['y_std']) + st.session_state['y_mean']

        if st.session_state['models']:
            st.session_state['trained'] = True
            st.success("Model has been trained successfully.")
        else:
            st.session_state['trained'] = False
            st.error("Model training failed.")
            return

        # Get the current trained model (evaluated after each training)
        model = st.session_state['models'][-1]
        preds = model.predict(st.session_state['X_test'])
        preds = (preds * st.session_state['y_std']) + st.session_state['y_mean']  # Reverse normalization

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

        # Display the results of the current model
        idx = len(st.session_state['model_evaluations'])
        st.subheader(f"Model {idx}")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"R-squared: {r2}")

    def generate_shap_plot(self, dataset, cmap):
        """Generate SHAP summary plots for the trained models."""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("Model has not been trained. Please train the model first.")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"Model {idx + 1} SHAP Summary Plot")

            explainer = shap.TreeExplainer(model)

            if dataset == "Training Set":
                shap_values = explainer.shap_values(st.session_state.X_train)
                data = st.session_state.X_train
            elif dataset == "Validation Set":
                shap_values = explainer.shap_values(st.session_state.X_val)
                data = st.session_state.X_val
            else:
                shap_values = explainer.shap_values(st.session_state.X_test)
                data = st.session_state.X_test

                # Display SHAP summary plots
            shap.summary_plot(shap_values, data, plot_type="bar", max_display=20, show=False, cmap=cmap)
            st.pyplot(bbox_inches='tight')

            shap.summary_plot(shap_values, data, plot_type="dot", max_display=20, show=False, cmap=cmap)
            st.pyplot(bbox_inches='tight')

            shap.summary_plot(shap_values, data, plot_type="layered_violin", max_display=20, show=False, cmap=cmap)
            st.pyplot(bbox_inches='tight')

            # Save plots for future use
            self.save_plot(f"model_{idx + 1}_shap_summary_bar", shap.summary_plot, shap_values, data, plot_type="bar",
                           max_display=20, show=False, cmap=cmap)
            self.save_plot(f"model_{idx + 1}_shap_summary_dot", shap.summary_plot, shap_values, data, plot_type="dot",
                           max_display=20, show=False, cmap=cmap)
            self.save_plot(f"model_{idx + 1}_shap_summary_violin", shap.summary_plot, shap_values, data, plot_type="layered_violin",
                           max_display=20, show=False, cmap=cmap)

            # Display table of SHAP values
            st.subheader(f"Model {idx + 1} SHAP Values Table")

            # Average absolute SHAP values over all samples to represent feature importance
            if isinstance(shap_values, list):
                # For tree-based models with multi-class outputs
                shap_values_abs_mean = [np.abs(shap_values[class_idx]).mean(axis=0) for class_idx in
                                        range(len(shap_values))]
                importance_df = pd.DataFrame(shap_values_abs_mean, columns=data.columns)
            else:
                shap_values_abs_mean = np.abs(shap_values).mean(axis=0)
                importance_df = pd.DataFrame(shap_values_abs_mean, index=data.columns, columns=["Average SHAP Value"])

                # Sort by importance
            importance_df = importance_df.sort_values(by=["Average SHAP Value"], ascending=False)

            # Display the dataframe
            st.dataframe(importance_df)

    def generate_dependence_plot(self, feature, interaction_feature):
        """Generate SHAP dependence plots for the trained models."""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("Model has not been trained. Please train the model first.")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"Model {idx + 1} SHAP Dependence Plot")

            explainer = shap.TreeExplainer(model)

            if dataset == "Training Set":
                shap_values = explainer.shap_values(st.session_state.X_train)
                data = st.session_state.X_train
            elif dataset == "Validation Set":
                shap_values = explainer.shap_values(st.session_state.X_val)
                data = st.session_state.X_val
            else:
                shap_values = explainer.shap_values(st.session_state.X_test)
                data = st.session_state.X_test

            fig, ax = plt.subplots(figsize=(10, 7))
            shap.dependence_plot(
                feature,
                shap_values,
                st.session_state.X_train,
                interaction_index=interaction_feature if interaction_feature != "None" else None,
                show=False,
                ax=ax,
            )
            st.pyplot(fig)

            features=[feature, interaction_feature]

            fig, ax = plt.subplots(figsize=(10, 7))
            PartialDependenceDisplay.from_estimator(
                model,
                st.session_state.X_train,
                features=[features],
                kind='average',
                grid_resolution=100,
                contour_kw={'cmap': cmap, 'alpha': 0.8},
                ax=ax
            )
            st.pyplot(fig)
            featureextra= 'cement content'
            features3d=[feature, interaction_feature,featureextra]
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

            # for col in data.columns:
            #     if col not in [feature, interaction_feature, featureextra]:
            #         X_grid[:, data.columns.get_loc(col)] = data[col].mean()

            # custom_feature_values = {
            #     'voltage': 30,
            #     'specimen length': 100,
            #     'initial temperature': 30
            # }
            #
            # for col, value in custom_feature_values.items():
            #     if col in data.columns:
            #         X_grid[:, data.columns.get_loc(col)] = value

            preds = model.predict(X_grid)
            fig = plt.figure(figsize=(10, 8), dpi=1200)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(grid_1.ravel(), grid_2.ravel(), grid_3.ravel(), c=preds, cmap=cmap, alpha=0.8)

            plt.colorbar(sc,ax = ax,label = 'Prediction')
            ax.set_xlabel(feature)
            ax.set_ylabel(interaction_feature)
            ax.set_zlabel(featureextra)
            st.pyplot(fig)
            fig.savefig(f"model_{idx + 1}_3d_pdp_ice_{feature}_{interaction_feature}.svg", format='svg', dpi=600)

            self.save_plot(f"model_{idx + 1}_shap_dependence_{feature}", shap.dependence_plot, feature, shap_values,
                           data, interaction_index=interaction_feature if interaction_feature != "None" else None,
                           show=False)

            self.save_plot(f"model_{idx + 1}_2d_shap_dependence_{feature}_{interaction_feature}", PartialDependenceDisplay.from_estimator, model, st.session_state.X_train, features=[features],
                           kind='average', grid_resolution=50, contour_kw={'cmap': cmap, 'alpha': 0.8})

    def generate_force_plot(self, sample_index):
        """Generate SHAP force plots for the trained models."""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("Model has not been trained. Please train the model first.")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"Model {idx + 1} SHAP Force Plot for Sample {sample_index}")

            explainer = shap.TreeExplainer(model)

            if dataset == "Training Set":
                shap_values = explainer.shap_values(st.session_state.X_train)[sample_index]
                data = st.session_state.X_train.iloc[sample_index]
            elif dataset == "Validation Set":
                shap_values = explainer.shap_values(st.session_state.X_val)[sample_index]
                data = st.session_state.X_val.iloc[sample_index]
            else:
                shap_values = explainer.shap_values(st.session_state.X_test)[sample_index]
                data = st.session_state.X_test.iloc[sample_index]

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
        """Generate SHAP interaction summary plots for the trained models."""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("Model has not been trained. Please train the model first.")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"Model {idx + 1} SHAP Interaction Summary Plot")

            explainer = shap.TreeExplainer(model)

            if dataset == "Training Set":
                shap_values = explainer.shap_interaction_values(st.session_state.X_train)
                data = st.session_state.X_train
            elif dataset == "Validation Set":
                shap_values = explainer.shap_interaction_values(st.session_state.X_val)
                data = st.session_state.X_val
            else:
                shap_values = explainer.shap_interaction_values(st.session_state.X_test)
                data = st.session_state.X_test

            shap_interaction_values = shap_values
            shap.summary_plot(shap_interaction_values, data, show=False)
            st.pyplot(bbox_inches='tight')

            self.save_plot(f"model_{idx + 1}_shap_interaction", shap.summary_plot, shap_values, data, show=False)

    def generate_heatmap_plot(self, start, end):
        """Generate SHAP heatmap plots for the trained models."""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("Model has not been trained. Please train the model first.")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"Model {idx + 1} SHAP Heatmap Plot")

            explainer = shap.TreeExplainer(model)

            if dataset == "Training Set":
                data = st.session_state.X_train.iloc[start:end, :]
                feature_names = st.session_state.X_train.columns
            elif dataset == "Validation Set":
                data = st.session_state.X_val.iloc[start:end, :]
                feature_names = st.session_state.X_val.columns
            else:
                data = st.session_state.X_test.iloc[start:end, :]
                feature_names = st.session_state.X_test.columns

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
        """Generate PDP/ICE plots for the trained models."""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("Model has not been trained. Please train the model first.")
            return

        for idx, model in enumerate(st.session_state['models']):
            st.subheader(f"Model {idx + 1} PDP/ICE Plot for Feature: {feature}")

            fig, ax = plt.subplots(figsize=(5, 5))
            shap.plots.partial_dependence(feature, model.predict, st.session_state.X_train, ice=True,
                                          model_expected_value=True, feature_expected_value=True, ax=ax, show=False)
            st.pyplot(fig)

            self.save_plot(f"model_{idx + 1}_pdp_ice_{feature}", shap.plots.partial_dependence, feature, model.predict,
                           st.session_state.X_train, ice=True, model_expected_value=True, feature_expected_value=True, ax=ax,
                           show=False)
            fig.savefig(f"model_{idx + 1}_pdp_ice_{feature}_g.svg", format='svg', dpi=600)


    def save_models(self):
        """Save all trained models to a file."""
        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("No models to save. Please train the models first.")
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

        file_path = st.text_input("Enter the file path to save the models:", "models/models.pkl")

        joblib.dump(models_to_save, file_path)
        st.success(f"Models saved to {file_path}")

    def load_models(self):
        """Load previously saved models from a file."""
        file_path = st.text_input("Enter the file path to load the models:", "models/models.pkl")

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
        st.success(f"Models loaded from {file_path}")

    def upload_prediction_file(self):
        """Upload the prediction input dataset."""
        self.prediction_file = st.file_uploader("Upload Prediction Input Dataset (.csv)", type="csv",
                                                key="prediction")
        if self.prediction_file:
            self.prediction_df = pd.read_csv(self.prediction_file)
            st.success("Prediction input file uploaded successfully.")

    def make_predictions_and_plot(self):
        """Make predictions using the loaded models and plot the results."""
        if not hasattr(self, 'prediction_df'):
            st.error("Please upload the prediction input file first.")
            return

        if 'models' not in st.session_state or not st.session_state['models']:
            st.error("No models loaded. Please load the models first.")
            return

        all_predictions = []

        for idx, model in enumerate(st.session_state['models']):
            preds = model.predict(self.prediction_df)
            all_predictions.append(preds)
            print(preds)
            # Transpose the predictions so that each row is a list of predictions from each model
        all_predictions = np.array(all_predictions).T

        # Limit to the first 20 model predictions
        max_models_to_plot = 20
        all_predictions = all_predictions[:, :max_models_to_plot]

        # Plot line chart
        fig_line, ax_line = plt.subplots()
        for idx, preds in enumerate(all_predictions):
            ax_line.plot(range(1, len(preds) + 1), preds, 'o-', label=f'Sample {idx + 1}')

            # Find and annotate the maximum value among the first 20 predictions
            max_index = np.argmax(preds)
            max_value = preds[max_index]
            ax_line.annotate(f'Max: {max_value}', xy=(max_index + 1, max_value),
                             xytext=(max_index + 1, max_value + 0.05 * max_value),
                             arrowprops=dict(facecolor='black', shrink=0.05))

        ax_line.set_xlabel('Model Number')
        ax_line.set_ylabel('Predicted Value')
        ax_line.set_title("Model Predictions (Line Plot)")
        ax_line.legend()
        st.pyplot(fig_line)

        # Plot curve chart
        fig_curve, ax_curve = plt.subplots()
        for idx, preds in enumerate(all_predictions):
            x_values = range(1, len(preds) + 1)
            z = np.polyfit(x_values, preds, 3)  # Fit a 3rd degree polynomial
            p = np.poly1d(z)
            x_curve = np.linspace(1, len(preds), 100)
            y_curve = p(x_curve)
            ax_curve.plot(x_curve, y_curve, label=f'Sample {idx + 1}')
            ax_curve.plot(x_values, preds, 'o')  # Original points

            # Find and annotate the maximum value among the first 20 predictions
            max_index = np.argmax(preds)
            max_value = preds[max_index]
            ax_curve.annotate(f'Max: {max_value}', xy=(max_index + 1, max_value),
                              xytext=(max_index + 1, max_value + 0.05 * max_value),
                              arrowprops=dict(facecolor='black', shrink=0.05))

        ax_curve.set_xlabel('Model Number')
        ax_curve.set_ylabel('Predicted Value')
        ax_curve.set_title("Model Predictions (Curve Plot)")
        ax_curve.legend()
        st.pyplot(fig_curve)

    def save_plot(self, filename, plot_func, *args, **kwargs):
        """Save plot as SVG with specified DPI."""
        if not os.path.exists('pictures'):
            os.makedirs('pictures')

        plt.figure()
        plot_func(*args, **kwargs)
        plt.savefig(f"pictures/{filename}.svg", format='svg', dpi=600, bbox_inches='tight')
        plt.savefig(f"pictures/{filename}.png", format='png', dpi=600, bbox_inches='tight')
        plt.close()


# NSGAiii
class InversionProblemJMetal(FloatProblem):
    def __init__(self, models, target_values, num_variables):
        super().__init__()
        self.models = models
        self.target_values = target_values
        self.lower_bound = [0.3, 1, 15, 35, 100, 30]
        self.upper_bound = [0.3, 1, 30, 35, 100, 30]
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.direction_names = ["MINIMIZE", "MINIMIZE"]  # 添加实例属性
        # 新增历史记录
        self.history = []
        self.history_objectives = []  # 新增目标值记录

    def evaluate(self, solution: FloatSolution) -> None:
        inputs = solution.variables
        # 记录每次评估的变量
        self.history.append(inputs.copy())
        predictions = [model.predict([inputs])[0] for model in self.models]
        solution.objectives[0] = abs(predictions[0] - self.target_values[0])
        solution.objectives[1] = abs(predictions[1] - self.target_values[1])
        self.history_objectives.append(solution.objectives.copy())  # 记录目标值
    def get_name(self) -> str:
        return "InversionProblemJMetal"

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def name(self) -> str:
        return self.get_name()

# NSGAiii
# class InversionProblemJMetal(FloatProblem):
#     def __init__(self, model, target_value, num_variables):
#         super().__init__()
#         self.model = model
#         self.target_value = target_value
#         self.lower_bound = [0.3, 0.5, 10, 30, 100, 0]
#         self.upper_bound = [0.5, 1, 30, 42, 515, 35]
#
#         self.obj_directions = [self.MINIMIZE]
#
#     def evaluate(self, solution: FloatSolution) -> None:
#         inputs = solution.variables
#         prediction = self.model.predict([inputs])[0]
#         solution.objectives[0] = abs(prediction - self.target_value)
#
#     def create_solution(self) -> FloatSolution:
#         new_solution = FloatSolution(
#             self.lower_bound,
#             self.upper_bound,
#             self.number_of_objectives(),
#             self.number_of_constraints()
#         )
#         new_solution.variables = [
#             random.uniform(lb, ub) for lb, ub in zip(self.lower_bound, self.upper_bound)
#         ]
#         return new_solution
#
#     def get_name(self) -> str:
#         return "InversionProblemJMetal"
#
#     def number_of_objectives(self) -> int:
#         return 1
#
#     def number_of_constraints(self) -> int:
#         return 0
#
#     def number_of_variables(self) -> int:
#         return len(self.lower_bound)
#
#     def name(self) -> str:
#         return self.get_name()
#
#
# def nsga3_inversion_jmetal(model_idx, target_value):
#
#     model = st.session_state['models'][model_idx]
#     X_train = st.session_state['X_train']
#
#     num_variables = X_train.shape[1]
#
#     problem = InversionProblemJMetal(model, target_value, num_variables)
#
#     algorithm = NSGAIII(
#         problem=problem,
#         population_size=92,
#         reference_directions=UniformReferenceDirectionFactory(1, n_points=92),
#         mutation=PolynomialMutation(probability=1.0 / num_variables, distribution_index=20),
#         crossover=SBXCrossover(probability=1.0, distribution_index=30),
#         termination_criterion=StoppingByEvaluations(max_evaluations=25000),
#     )
#
#     algorithm.run()
#
#     try:
#         solutions = algorithm.get_result()
#     except AttributeError:
#         solutions = algorithm.solutions
#
#     front = get_non_dominated_solutions(solutions)
#
#     best_solution = front[0]
#     solution_df = pd.DataFrame([best_solution.variables], columns=X_train.columns)
#
#     return solution_df


def nsga3_inversion_jmetal(model_idxs, target_values):
    models = [st.session_state['models'][idx] for idx in model_idxs]
    X_train = st.session_state['X_train']
    num_variables = X_train.shape[1]

    # 初始化问题实例（包含历史记录）
    problem = InversionProblemJMetal(models, target_values, num_variables)

    # 配置优化算法
    algorithm = NSGAIII(
        problem=problem,
        population_size=100,
        reference_directions=UniformReferenceDirectionFactory(2, n_points=92),
        mutation=PolynomialMutation(probability=1.0 / num_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=30),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000),
    )

    # 执行优化
    algorithm.run()

    # 获取结果
    try:
        solutions = algorithm.get_result()
    except AttributeError:
        solutions = algorithm.solutions

    # 提取最优解
    front = get_non_dominated_solutions(solutions)
    best_solution = front[0]
    solution_df = pd.DataFrame([best_solution.variables], columns=X_train.columns)

    # 生成参数变化可视化图表
    df_history = pd.DataFrame(problem.history, columns=X_train.columns)

    # 过滤固定参数（上下界相同的列）
    changing_params = [
        col for col, lb, ub in zip(df_history.columns, problem.lower_bound, problem.upper_bound)
        if lb != ub
    ]
    df_history_filtered = df_history[changing_params]

    # 提取最佳参数值（来自优化结果）
    best_values = best_solution.variables  # 假设这是包含所有参数的数组

    # 创建多子图可视化（添加最佳值标记）
    fig, axes = plt.subplots(
        len(changing_params), 1,
        figsize=(10, 2.5 * len(changing_params)),
        facecolor='white'
    )
    if len(changing_params) == 1:
        axes = [axes]  # 处理单参数情况

    # 获取参数对应的最佳值索引
    param_indices = {param: idx for idx, param in enumerate(X_train.columns)
                     if param in changing_params}

    for ax, param in zip(axes, changing_params):
        # 绘制参数变化曲线
        line = ax.plot(df_history_filtered[param],
                       color='#1f77b4',
                       alpha=0.7,
                       label='Optimization Process')

        # 标记最佳值位置（最后一代结果）
        best_idx = len(df_history_filtered) - 1
        best_val = best_values[param_indices[param]]

        # 添加最佳值标记
        ax.scatter(best_idx, best_val,
                   color='red',
                   marker='*',
                   s=120,
                   edgecolor='black',
                   zorder=10,
                   label='Optimal Value')

        # 图表装饰
        ax.set_title(f'Parameter: {param}', fontsize=10)
        ax.set_xlabel('Evaluation Steps', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)

        # 显示图例
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    # 生成目标值变化可视化图表
    df_objectives = pd.DataFrame(problem.history_objectives,
                                 columns=[f"Objective {i + 1}" for i in range(problem.number_of_objectives())])

    fig_obj, ax_obj = plt.subplots(figsize=(10, 4), facecolor='white')
    for i, obj in enumerate(df_objectives.columns):
        ax_obj.plot(df_objectives[obj],
                    alpha=0.7,
                    label=f'Objective {i + 1} ({problem.direction_names[i]})')  # 使用实例属性

    # 标记最佳解位置
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

    # 保存优化过程到Excel
    df_variables = pd.DataFrame(problem.history, columns=X_train.columns)
    df_objectives = pd.DataFrame(
        problem.history_objectives,
        columns=[f"Objective {i + 1}" for i in range(problem.number_of_objectives())]
    )
    with pd.ExcelWriter("youhuajieguo.xlsx") as writer:
        df_variables.to_excel(writer, sheet_name="Parameters", index=False)
        df_objectives.to_excel(writer, sheet_name="Objectives", index=False)

    # 保存SVG格式图片
    fig.savefig("parameter_evolution.svg", format="svg", bbox_inches="tight")
    fig_obj.savefig("objective_evolution.svg", format="svg", bbox_inches="tight")

    return solution_df, fig, fig_obj  # 返回三个对象







if __name__ == "__main__":
    app = ModelApp()

    st.title("General GUI for regression-multi-objective optimization tasks applicable to concrete problems")
    st.info("***Author: Yuting Zhang-School of Civil Engineering, Central South University***")

    uploaded_df_list = app.upload_file()

    dataset_placeholder = st.empty()

    if uploaded_df_list is not None:
        with dataset_placeholder.container():
            st.sidebar.title("1.XGBoost model training")
            st.write("Input Dataset:")
            st.dataframe(app.input_df)
            # st.write("Output Dataset:")
            # st.dataframe(app.output_df.head())

        model_type = st.sidebar.selectbox("Select Baseline Model", ["LightGBM", "XGBoost", "CatBoost", "NGBoost"])
        opt_algo = st.sidebar.selectbox("Select Hyperparameter Optimization Algorithm", ["None", "TPE", "CMA-ES"])

        if st.sidebar.button("Run Model", key="run_model_button_unique_1"):
            app.run_model(None, model_type, opt_algo)  # Note: target_feature is not used here
            dataset_placeholder.empty()

        st.sidebar.title("2.XGBoost model predicting")
        # Save and load models
        if st.sidebar.button("Save Models", key="save_models_button_unique"):
            dataset_placeholder.empty()
            app.save_models()
        if st.sidebar.button("Load Models", key="load_models_button_unique"):
            dataset_placeholder.empty()
            app.load_models()

        # Upload prediction file
        app.upload_prediction_file()

        if st.sidebar.button("Run Prediction", key="run_prediction_button_unique"):
            dataset_placeholder.empty()
            app.make_predictions_and_plot()

        st.sidebar.title("3.XGBoost model interpretability")
        dataset = st.sidebar.selectbox("Select SHAP Explanation Dataset",
                                       ["Training Set", "Validation Set", "Test Set"])
        cmap = st.sidebar.selectbox(
            "Select Color Scheme",
            ["viridis", "Spectral", "coolwarm", "RdYlGn", "RdYlBu", "RdBu", "RdGy", "PuOr", "BrBG", "PRGn", "PiYG"]
        )

        if st.sidebar.button("Generate SHAP Plot", key="generate_shap_summary_plot_button_unique"):
            dataset_placeholder.empty()
            app.generate_shap_plot(dataset, cmap)

        features = app.input_df.columns.tolist()
        features.append("None")
        feature = st.sidebar.selectbox("Select SHAP Dependence Plot Feature", features, index=0)
        interaction_feature = st.sidebar.selectbox("Select Interaction Feature", features, index=len(features) - 1)

        if st.sidebar.button("Generate Dependence Plot", key="generate_dependence_plot_button_unique"):
            dataset_placeholder.empty()
            app.generate_dependence_plot(feature, interaction_feature)

        sample_index = st.sidebar.number_input("Select Force Plot Sample Index", min_value=0,
                                               max_value=len(st.session_state.X_train) - 1)
        if st.sidebar.button("Generate SHAP Force Plot", key="generate_force_plot_button_unique"):
            dataset_placeholder.empty()
            app.generate_force_plot(sample_index)

        if st.sidebar.button("Generate Interaction Plot", key="generate_interaction_plot_button_unique"):
            dataset_placeholder.empty()
            app.generate_interaction_plot()

        heatmap_range = st.sidebar.text_input("Select Heatmap Data Range (start:end)", "0:20")
        range_values = heatmap_range.split(":")
        start, end = int(range_values[0]), int(range_values[1])
        if st.sidebar.button("Generate SHAP Heatmap", key="generate_heatmap_button_unique"):
            dataset_placeholder.empty()
            app.generate_heatmap_plot(start, end)

        pdp_features = st.sidebar.selectbox("Select PDP/ICE Feature", st.session_state.X_train.columns.tolist())
        if st.sidebar.button("Generate PDP-ICE Plot", key="generate_pdp_ice_plot_button_unique"):
            dataset_placeholder.empty()
            app.generate_pdp_ice_plots(pdp_features)

        st.sidebar.title("4.NSGAIII model")
        model_idxs = st.sidebar.multiselect("Select Model Indices", range(len(st.session_state['models'])),
                                            default=[0])
        target_values = [st.sidebar.number_input(f"Enter Target Value for Model {idx}", value=0.0) for idx in
                         model_idxs]
        if st.sidebar.button("Run NSGA-III Inversion", key="run_nsga3_inversion_button_unique_1"):
            # best_solution_df = nsga3_inversion_jmetal(model_idxs, target_values)
            # st.write("Best Solution:")
            # st.dataframe(best_solution_df)

            # 正确解包返回值
            parameter_result, evolution_plot, objective_plot = nsga3_inversion_jmetal(model_idxs, target_values)

            # 显示优化结果表格
            st.subheader("Optimized Parameters")
            st.dataframe(parameter_result.style.format("{:.4f}"))  # 单独显示DataFrame

            # 显示参数变化曲线
            st.subheader("Parameter Evolution Process")
            st.pyplot(evolution_plot)  # 单独显示图像
            # 显示目标值变化曲线
            st.subheader("Objective Values Evolution")
            st.pyplot(objective_plot)

        # model_idx = st.sidebar.number_input("Select Model Index", min_value=0,
        #                                     max_value=len(st.session_state['models']) - 1, value=20)
        # target_value = st.sidebar.number_input("Enter Target Value", value=0.0)
        # if st.sidebar.button("Run NSGA-III Inversion", key="run_nsga3_inversion_button_unique_1"):
        #     best_solution_df = nsga3_inversion_jmetal(model_idx, target_value)
        #     st.write("Best Solution:")
        #     st.dataframe(best_solution_df)

