#!/usr/bin/env python3
"""
Optimized Machine Learning Pipeline
Efficient spam classification with caching and optimized training
"""

import yaml
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

from src.data_manager import DataManager
from src.optimized_models import OptimizedModelTrainer
from src.metrics import metrics
from src.clear_terminal import clear_terminal
from src.DataAnalysis import DataAnalysis

class OptimizedMLPipeline:
    """Main pipeline class for efficient ML operations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.data_manager = DataManager(
            data_path=self.config['data']['path'],
            features_path=self.config['data']['features_path'],
            cache_dir=self.config['cache']['cache_dir']
        )
        self.trainer = OptimizedModelTrainer(
            n_jobs=self.config['training']['n_jobs'],
            random_state=self.config['training']['random_state']
        )
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_knn_optimized(self):
        """Run optimized KNN training and evaluation"""
        print("Loading data...")
        X, y, dataType, features = self.data_manager.load_data()
        
        # Use default preprocessing or get from user
        preprocessing_config = {
            'transformation': self.config['preprocessing']['default_transformation'],
            'feature_selection': self.config['preprocessing']['default_feature_selection'],
            'balancing': self.config['preprocessing']['default_balancing']
        }
        
        print("Applying preprocessing...")
        X_processed, y_processed = self.data_manager.get_preprocessing_pipeline(preprocessing_config)
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            X_processed, y_processed, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y_processed
        )
        
        print("Training KNN model...")
        start_time = time.time()
        
        model, best_params = self.trainer.train_knn_optimized(
            train_x, train_y,
            n_iter=self.config['training']['knn']['n_iter']
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        results = self.trainer.evaluate_model(model, test_x, test_y)
        
        # Display results
        clear_terminal()
        print("=" * 50)
        print("OPTIMIZED KNN RESULTS")
        print("=" * 50)
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Best parameters: {best_params}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
        
        # Save model if configured
        if self.config['output']['save_models']:
            models_dir = Path(self.config['output']['models_dir'])
            models_dir.mkdir(exist_ok=True)
            self.trainer.save_model('knn', models_dir / 'knn_model.pkl')
            print(f"Model saved to {models_dir / 'knn_model.pkl'}")
    
    def run_decision_tree_optimized(self):
        """Run optimized Decision Tree training and evaluation"""
        print("Loading data...")
        X, y, dataType, features = self.data_manager.load_data()
        
        preprocessing_config = {
            'transformation': self.config['preprocessing']['default_transformation'],
            'feature_selection': self.config['preprocessing']['default_feature_selection'],
            'balancing': self.config['preprocessing']['default_balancing']
        }
        
        print("Applying preprocessing...")
        X_processed, y_processed = self.data_manager.get_preprocessing_pipeline(preprocessing_config)
        
        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            X_processed, y_processed, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y_processed
        )
        
        print("Training Decision Tree model...")
        start_time = time.time()
        
        model, best_params = self.trainer.train_decision_tree_optimized(
            train_x, train_y,
            n_iter=self.config['training']['decision_tree']['n_iter']
        )
        
        training_time = time.time() - start_time
        
        results = self.trainer.evaluate_model(model, test_x, test_y)
        
        clear_terminal()
        print("=" * 50)
        print("OPTIMIZED DECISION TREE RESULTS")
        print("=" * 50)
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Best parameters: {best_params}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    
    def run_random_forest_optimized(self):
        """Run optimized Random Forest training and evaluation"""
        print("Loading data...")
        X, y, dataType, features = self.data_manager.load_data()
        
        preprocessing_config = {
            'transformation': self.config['preprocessing']['default_transformation'],
            'feature_selection': self.config['preprocessing']['default_feature_selection'],
            'balancing': self.config['preprocessing']['default_balancing']
        }
        
        print("Applying preprocessing...")
        X_processed, y_processed = self.data_manager.get_preprocessing_pipeline(preprocessing_config)
        
        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            X_processed, y_processed, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y_processed
        )
        
        print("Training Random Forest model...")
        start_time = time.time()
        
        model, best_params = self.trainer.train_random_forest_optimized(
            train_x, train_y,
            n_iter=self.config['training']['random_forest']['n_iter']
        )
        
        training_time = time.time() - start_time
        
        results = self.trainer.evaluate_model(model, test_x, test_y)
        
        clear_terminal()
        print("=" * 50)
        print("OPTIMIZED RANDOM FOREST RESULTS")
        print("=" * 50)
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Best parameters: {best_params}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    
    def run_naive_bayes_optimized(self):
        """Run optimized Naive Bayes training and evaluation"""
        print("Loading data...")
        X, y, dataType, features = self.data_manager.load_data()
        
        preprocessing_config = {
            'transformation': self.config['preprocessing']['default_transformation'],
            'feature_selection': self.config['preprocessing']['default_feature_selection'],
            'balancing': self.config['preprocessing']['default_balancing']
        }
        
        print("Applying preprocessing...")
        X_processed, y_processed = self.data_manager.get_preprocessing_pipeline(preprocessing_config)
        
        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            X_processed, y_processed, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y_processed
        )
        
        print("Training Naive Bayes model...")
        start_time = time.time()
        
        model, best_params = self.trainer.train_naive_bayes_optimized(
            train_x, train_y,
            n_iter=20
        )
        
        training_time = time.time() - start_time
        
        results = self.trainer.evaluate_model(model, test_x, test_y)
        
        clear_terminal()
        print("=" * 50)
        print("OPTIMIZED NAIVE BAYES RESULTS")
        print("=" * 50)
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Best parameters: {best_params}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    
    def run_ensemble_optimized(self):
        """Run optimized Ensemble training and evaluation"""
        print("Loading data...")
        X, y, dataType, features = self.data_manager.load_data()
        
        preprocessing_config = {
            'transformation': self.config['preprocessing']['default_transformation'],
            'feature_selection': self.config['preprocessing']['default_feature_selection'],
            'balancing': self.config['preprocessing']['default_balancing']
        }
        
        print("Applying preprocessing...")
        X_processed, y_processed = self.data_manager.get_preprocessing_pipeline(preprocessing_config)
        
        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            X_processed, y_processed, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y_processed
        )
        
        print("Training Ensemble model...")
        start_time = time.time()
        
        model, best_params = self.trainer.train_ensemble_optimized(
            train_x, train_y,
            n_iter=15
        )
        
        training_time = time.time() - start_time
        
        results = self.trainer.evaluate_model(model, test_x, test_y)
        
        clear_terminal()
        print("=" * 50)
        print("OPTIMIZED ENSEMBLE RESULTS")
        print("=" * 50)
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Ensemble configuration: {best_params}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
        
        # Show individual model performances
        print("\n" + "=" * 30)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("=" * 30)
        estimators = model.estimators_
        for i, (name, estimator) in enumerate(model.estimators_):
            individual_results = self.trainer.evaluate_model(estimator, test_x, test_y)
            print(f"{name.upper()}: Accuracy={individual_results['accuracy']:.4f}, "
                  f"F1={individual_results['f1_weighted']:.4f}")
    
    def run_data_analysis(self):
        """Run comprehensive data analysis on the dataset"""
        print("Running data analysis...")
        DataAnalysis()
    
    def benchmark_all_models(self):
        """Run all models and compare performance"""
        print("Running comprehensive model benchmark...")
        
        # Load data once
        X, y, dataType, features = self.data_manager.load_data()
        
        preprocessing_config = {
            'transformation': 'standard',  # Use standard scaling for fair comparison
            'feature_selection': 'none',
            'balancing': 'none'
        }
        
        X_processed, y_processed = self.data_manager.get_preprocessing_pipeline(preprocessing_config)
        
        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(
            X_processed, y_processed, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y_processed
        )
        
        models_to_test = [
            ('KNN', self.trainer.train_knn_optimized),
            ('Decision Tree', self.trainer.train_decision_tree_optimized),
            ('Random Forest', self.trainer.train_random_forest_optimized),
            ('Naive Bayes', self.trainer.train_naive_bayes_optimized),
            ('Ensemble', self.trainer.train_ensemble_optimized)
        ]
        
        results = {}
        
        for model_name, train_func in models_to_test:
            print(f"\nTraining {model_name}...")
            start_time = time.time()
            
            model, best_params = train_func(train_x, train_y)
            training_time = time.time() - start_time
            
            eval_results = self.trainer.evaluate_model(model, test_x, test_y)
            
            results[model_name] = {
                'training_time': training_time,
                'accuracy': eval_results['accuracy'],
                'f1_weighted': eval_results['f1_weighted'],
                'best_params': best_params
            }
        
        # Display comparison
        clear_terminal()
        print("=" * 60)
        print("MODEL BENCHMARK RESULTS")
        print("=" * 60)
        print(f"{'Model':<15} {'Time(s)':<10} {'Accuracy':<10} {'F1-Weighted':<12}")
        print("-" * 60)
        
        for model_name, result in results.items():
            print(f"{model_name:<15} {result['training_time']:<10.2f} "
                  f"{result['accuracy']:<10.4f} {result['f1_weighted']:<12.4f}")
        
        print("=" * 60)

def main():
    """Main application entry point"""
    pipeline = OptimizedMLPipeline()
    
    while True:
        clear_terminal()
        print("=" * 50)
        print("OPTIMIZED MACHINE LEARNING PIPELINE")
        print("=" * 50)
        print("1 | Optimized KNN")
        print("2 | Optimized Decision Tree")
        print("3 | Optimized Random Forest")
        print("4 | Optimized Naive Bayes")
        print("5 | Optimized Ensemble")
        print("6 | Benchmark All Models")
        print("7 | Clear Cache")
        print("8 | Data Analysis")
        print("9 | Exit")
        print("=" * 50)
        print("Choose an option: ")
        
        try:
            choice = int(input())
            
            if choice == 1:
                pipeline.run_knn_optimized()
            elif choice == 2:
                pipeline.run_decision_tree_optimized()
            elif choice == 3:
                pipeline.run_random_forest_optimized()
            elif choice == 4:
                pipeline.run_naive_bayes_optimized()
            elif choice == 5:
                pipeline.run_ensemble_optimized()
            elif choice == 6:
                pipeline.benchmark_all_models()
            elif choice == 7:
                pipeline.data_manager.clear_cache()
                print("Cache cleared!")
            elif choice == 8:
                pipeline.run_data_analysis()
            elif choice == 9:
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
                
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 