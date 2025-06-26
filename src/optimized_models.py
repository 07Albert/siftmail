import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import joblib
from .libraries import *

class OptimizedModelTrainer:
    """Efficient model training with optimized hyperparameter search"""
    
    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models = {}
        
    def train_knn_optimized(self, X: np.ndarray, y: np.ndarray, 
                           n_iter: int = 50) -> Tuple[Any, Dict[str, Any]]:
        """Optimized KNN training with RandomizedSearchCV"""
        
        # More efficient parameter space
        param_distributions = {
            'n_neighbors': np.arange(1, 31),  # Reduced range
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],  # Removed cosine for speed
            'p': [1, 2]  # For Minkowski distance
        }
        
        # Use RandomizedSearchCV instead of GridSearchCV
        knn = KNeighborsClassifier()
        random_search = RandomizedSearchCV(
            knn, param_distributions, 
            n_iter=n_iter,  # Much faster than exhaustive search
            cv=5,  # Reduced from 10 to 5
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        random_search.fit(X, y)
        
        # Store the best model
        self.models['knn'] = random_search.best_estimator_
        
        return random_search.best_estimator_, random_search.best_params_
    
    def train_decision_tree_optimized(self, X: np.ndarray, y: np.ndarray,
                                    n_iter: int = 30) -> Tuple[Any, Dict[str, Any]]:
        """Optimized Decision Tree training"""
        
        param_distributions = {
            'max_depth': [None] + list(range(3, 21)),
            'min_samples_split': range(2, 11),
            'min_samples_leaf': range(1, 6),
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random']
        }
        
        dt = DecisionTreeClassifier(random_state=self.random_state)
        random_search = RandomizedSearchCV(
            dt, param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        random_search.fit(X, y)
        self.models['decision_tree'] = random_search.best_estimator_
        
        return random_search.best_estimator_, random_search.best_params_
    
    def train_random_forest_optimized(self, X: np.ndarray, y: np.ndarray,
                                    n_iter: int = 20) -> Tuple[Any, Dict[str, Any]]:
        """Optimized Random Forest training"""
        
        param_distributions = {
            'n_estimators': [50, 100, 200, 300],  # Reduced options
            'max_depth': [None] + list(range(3, 16)),
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        random_search = RandomizedSearchCV(
            rf, param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring='f1_weighted',
            n_jobs=1,  # RF already uses parallel processing
            random_state=self.random_state,
            verbose=0
        )
        
        random_search.fit(X, y)
        self.models['random_forest'] = random_search.best_estimator_
        
        return random_search.best_estimator_, random_search.best_params_
    
    def train_random_forest_fast(self, X: np.ndarray, y: np.ndarray,
                                n_iter: int = 10) -> Tuple[Any, Dict[str, Any]]:
        """Fast Random Forest training optimized for speed"""
        
        # Optimized parameter space for speed
        param_distributions = {
            'n_estimators': [50, 100],  # Reduced for speed
            'max_depth': [5, 10, 15],   # Limited depth
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['sqrt', 'log2'],  # Removed None for speed
            'bootstrap': [True, False]
        }
        
        rf = RandomForestClassifier(
            random_state=self.random_state, 
            n_jobs=self.n_jobs,
            warm_start=True  # Speeds up training
        )
        
        random_search = RandomizedSearchCV(
            rf, param_distributions,
            n_iter=n_iter,
            cv=3,  # Reduced CV folds for speed
            scoring='f1_weighted',
            n_jobs=1,  # RF already uses parallel processing
            random_state=self.random_state,
            verbose=0
        )
        
        random_search.fit(X, y)
        self.models['random_forest_fast'] = random_search.best_estimator_
        
        return random_search.best_estimator_, random_search.best_params_

    def train_random_forest_production(self, X: np.ndarray, y: np.ndarray,
                                     n_iter: int = 15) -> Tuple[Any, Dict[str, Any]]:
        """Production-ready Random Forest with balanced speed/performance"""
        
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', None]  # Important for spam detection
        }
        
        rf = RandomForestClassifier(
            random_state=self.random_state, 
            n_jobs=self.n_jobs,
            oob_score=True  # Out-of-bag score for validation
        )
        
        random_search = RandomizedSearchCV(
            rf, param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring='f1_weighted',
            n_jobs=1,
            random_state=self.random_state,
            verbose=0
        )
        
        random_search.fit(X, y)
        self.models['random_forest_production'] = random_search.best_estimator_
        
        return random_search.best_estimator_, random_search.best_params_
    
    def train_naive_bayes_optimized(self, X: np.ndarray, y: np.ndarray,
                                   n_iter: int = 20) -> Tuple[Any, Dict[str, Any]]:
        """Optimized Naive Bayes training with hyperparameter tuning"""
        
        # For GaussianNB, we can tune var_smoothing
        param_distributions = {
            'var_smoothing': np.logspace(-10, -8, 100)  # Log-uniform distribution
        }
        
        nb = GaussianNB()
        random_search = RandomizedSearchCV(
            nb, param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        random_search.fit(X, y)
        self.models['naive_bayes'] = random_search.best_estimator_
        
        return random_search.best_estimator_, random_search.best_params_

    def train_ensemble_optimized(self, X: np.ndarray, y: np.ndarray,
                                n_iter: int = 15) -> Tuple[Any, Dict[str, Any]]:
        """Optimized Ensemble training with voting classifier"""
        
        # Create base models
        knn = KNeighborsClassifier(n_neighbors=5)
        dt = DecisionTreeClassifier(random_state=self.random_state, max_depth=10)
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=self.n_jobs)
        nb = GaussianNB()
        
        # Ensemble parameter space
        param_distributions = {
            'voting': ['hard', 'soft'],
            'weights': [
                [1, 1, 1, 1],  # Equal weights
                [2, 1, 2, 1],  # Give more weight to RF and KNN
                [1, 2, 2, 1],  # Give more weight to tree-based models
                [2, 1, 1, 2]   # Give more weight to KNN and NB
            ]
        }
        
        # Create voting classifier
        from sklearn.ensemble import VotingClassifier
        ensemble = VotingClassifier(
            estimators=[
                ('knn', knn),
                ('dt', dt),
                ('rf', rf),
                ('nb', nb)
            ],
            voting='hard'
        )
        
        # For ensemble, we'll use a simpler approach since VotingClassifier
        # doesn't work directly with RandomizedSearchCV
        # Instead, we'll optimize the base models and use voting
        
        # Optimize base models first
        print("Optimizing base models for ensemble...")
        
        # Optimize KNN for ensemble
        knn_params = {
            'n_neighbors': np.arange(3, 16),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        knn_search = RandomizedSearchCV(
            KNeighborsClassifier(), knn_params,
            n_iter=10, cv=3, scoring='f1_weighted',
            n_jobs=self.n_jobs, random_state=self.random_state, verbose=0
        )
        knn_search.fit(X, y)
        best_knn = knn_search.best_estimator_
        
        # Optimize Decision Tree for ensemble
        dt_params = {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        dt_search = RandomizedSearchCV(
            DecisionTreeClassifier(random_state=self.random_state), dt_params,
            n_iter=10, cv=3, scoring='f1_weighted',
            n_jobs=self.n_jobs, random_state=self.random_state, verbose=0
        )
        dt_search.fit(X, y)
        best_dt = dt_search.best_estimator_
        
        # Optimize Random Forest for ensemble
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5]
        }
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs), rf_params,
            n_iter=10, cv=3, scoring='f1_weighted',
            n_jobs=1, random_state=self.random_state, verbose=0
        )
        rf_search.fit(X, y)
        best_rf = rf_search.best_estimator_
        
        # Create final ensemble
        final_ensemble = VotingClassifier(
            estimators=[
                ('knn', best_knn),
                ('dt', best_dt),
                ('rf', best_rf),
                ('nb', GaussianNB())
            ],
            voting='soft',  # Use soft voting for better performance
            weights=[1, 1, 2, 1]  # Give more weight to Random Forest
        )
        
        # Fit the ensemble
        final_ensemble.fit(X, y)
        self.models['ensemble'] = final_ensemble
        
        best_params = {
            'knn_params': knn_search.best_params_,
            'dt_params': dt_search.best_params_,
            'rf_params': rf_search.best_params_,
            'voting': 'soft',
            'weights': [1, 1, 2, 1]
        }
        
        return final_ensemble, best_params
    
    def create_pipeline(self, model_name: str, preprocessing_steps: list = None) -> Pipeline:
        """Create a scikit-learn pipeline for efficient preprocessing + training"""
        
        steps = []
        
        # Add preprocessing steps
        if preprocessing_steps:
            steps.extend(preprocessing_steps)
        
        # Add the model
        if model_name in self.models:
            steps.append((model_name, self.models[model_name]))
        
        return Pipeline(steps)
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Efficient model evaluation"""
        
        y_pred = model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro')
        }
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model"""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
    
    def load_model(self, model_name: str, filepath: str):
        """Load trained model"""
        self.models[model_name] = joblib.load(filepath) 