import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Tuple, Dict, Any
from .libraries import *
from .metrics import metrics
from .clear_terminal import clear_terminal

warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Comprehensive data analysis class for spam dataset"""
    
    def __init__(self, data_path: str = "spambase/spambase.csv", 
                 features_path: str = "spambase/features.txt"):
        self.data_path = data_path
        self.features_path = features_path
        self.df = None
        self.X = None
        self.y = None
        self.features = None
        self.data_types = None
        self._load_data()
    
    def _load_data(self):
        """Load and prepare the dataset"""
        try:
            # Load features
            with open(self.features_path, "r") as f:
                lines = f.readlines()
            
            raw = [x.strip().split(":") for x in lines]
            self.features = [x[0] for x in raw] + ["spam"]
            self.data_types = [x[1].strip()[:-1] for x in raw]
            
            # Load dataset
            self.df = pd.read_csv(self.data_path, names=self.features, index_col=False)
            self.X = self.df.iloc[:, :-1]  # Features
            self.y = self.df.iloc[:, -1]   # Target
            
            print(f"✓ Dataset loaded successfully!")
            print(f"  - Shape: {self.df.shape}")
            print(f"  - Features: {len(self.features)-1}")
            print(f"  - Target classes: {self.y.unique()}")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find data file - {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def basic_statistics(self):
        """Display comprehensive basic statistics"""
        clear_terminal()
        print("=" * 60)
        print("BASIC STATISTICS")
        print("=" * 60)
        
        # Dataset overview
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total samples: {len(self.df):,}")
        print(f"   Features: {len(self.features)-1}")
        print(f"   Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Target distribution
        print(f"\n🎯 TARGET DISTRIBUTION:")
        target_counts = self.y.value_counts()
        target_percentages = self.y.value_counts(normalize=True) * 100
        for class_name, count in target_counts.items():
            percentage = target_percentages[class_name]
            print(f"   Class {class_name}: {count:,} samples ({percentage:.1f}%)")
        
        # Feature statistics
        print(f"\n📈 FEATURE STATISTICS:")
        stats = self.X.describe()
        print(stats.round(3))
        
        # Data types and null values
        print(f"\n🔍 DATA QUALITY:")
        print(f"   Null values: {self.df.isnull().sum().sum()}")
        print(f"   Duplicate rows: {self.df.duplicated().sum()}")
        
        input("\nPress Enter to continue...")
    
    def data_quality_analysis(self):
        """Analyze data quality issues"""
        clear_terminal()
        print("=" * 60)
        print("DATA QUALITY ANALYSIS")
        print("=" * 60)
        
        # Null values analysis
        null_counts = self.df.isnull().sum()
        null_percentages = (null_counts / len(self.df)) * 100
        
        print(f"\n🔍 NULL VALUES ANALYSIS:")
        if null_counts.sum() == 0:
            print("   ✓ No null values found in the dataset!")
        else:
            null_df = pd.DataFrame({
                'Feature': null_counts.index,
                'Null_Count': null_counts.values,
                'Null_Percentage': null_percentages.values
            }).sort_values('Null_Count', ascending=False)
            print(null_df[null_df['Null_Count'] > 0].to_string(index=False))
        
        # Duplicate analysis
        print(f"\n🔄 DUPLICATE ANALYSIS:")
        duplicates = self.df.duplicated()
        print(f"   Total duplicate rows: {duplicates.sum()}")
        print(f"   Duplicate percentage: {(duplicates.sum() / len(self.df)) * 100:.2f}%")
        
        # Data type analysis
        print(f"\n📋 DATA TYPE ANALYSIS:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} features")
        
        input("\nPress Enter to continue...")
    
    def outlier_analysis(self):
        """Comprehensive outlier analysis"""
        clear_terminal()
        print("=" * 60)
        print("OUTLIER ANALYSIS")
        print("=" * 60)
        
        # IQR method
        Q1 = self.X.quantile(0.25)
        Q3 = self.X.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_iqr = ((self.X < lower_bound) | (self.X > upper_bound)).sum()
        
        # Z-score method
        z_scores = np.abs((self.X - self.X.mean()) / self.X.std())
        outliers_zscore = (z_scores > 3).sum()
        
        # Percentile method
        outliers_percentile = ((self.X < self.X.quantile(0.01)) | 
                              (self.X > self.X.quantile(0.99))).sum()
        
        print(f"\n📊 OUTLIER DETECTION METHODS:")
        print(f"   IQR Method (1.5*IQR): {outliers_iqr.sum():,} total outliers")
        print(f"   Z-Score Method (>3σ): {outliers_zscore.sum():,} total outliers")
        print(f"   Percentile Method (1-99%): {outliers_percentile.sum():,} total outliers")
        
        # Top features with most outliers
        print(f"\n🔝 TOP 10 FEATURES WITH MOST OUTLIERS (IQR method):")
        outlier_summary = pd.DataFrame({
            'Feature': outliers_iqr.index,
            'Outlier_Count': outliers_iqr.values,
            'Outlier_Percentage': (outliers_iqr.values / len(self.X)) * 100
        }).sort_values('Outlier_Count', ascending=False)
        
        print(outlier_summary.head(10).to_string(index=False))
        
        input("\nPress Enter to continue...")
    
    def correlation_analysis(self):
        """Analyze feature correlations"""
        clear_terminal()
        print("=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Calculate correlation matrix
        corr_matrix = self.df.corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_corr_pairs.append({
                        'Feature1': corr_matrix.columns[i],
                        'Feature2': corr_matrix.columns[j],
                        'Correlation': corr_value
                    })
        
        print(f"\n🔗 HIGHLY CORRELATED FEATURES (|r| > 0.8):")
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
            print(high_corr_df.to_string(index=False))
        else:
            print("   No highly correlated features found (|r| > 0.8)")
        
        # Target correlations
        print(f"\n🎯 FEATURE-TARGET CORRELATIONS:")
        target_corrs = corr_matrix['spam'].sort_values(key=abs, ascending=False)
        print("   Top 10 features most correlated with target:")
        for feature, corr in target_corrs.head(11).items():  # 11 to include target itself
            if feature != 'spam':
                print(f"   {feature}: {corr:.4f}")
        
        input("\nPress Enter to continue...")
    
    def distribution_analysis(self):
        """Analyze feature distributions"""
        clear_terminal()
        print("=" * 60)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        # Skewness and kurtosis
        skewness = self.X.skew().sort_values(key=abs, ascending=False)
        kurtosis = self.X.kurtosis().sort_values(key=abs, ascending=False)
        
        print(f"\n📊 DISTRIBUTION SHAPE ANALYSIS:")
        print(f"   Most skewed features (top 5):")
        for feature, skew in skewness.head().items():
            print(f"   {feature}: {skew:.3f}")
        
        print(f"\n   Most kurtotic features (top 5):")
        for feature, kurt in kurtosis.head().items():
            print(f"   {feature}: {kurt:.3f}")
        
        # Feature ranges
        print(f"\n📏 FEATURE RANGES:")
        ranges = self.X.max() - self.X.min()
        print(f"   Largest range: {ranges.idxmax()} ({ranges.max():.2f})")
        print(f"   Smallest range: {ranges.idxmin()} ({ranges.min():.2f})")
        
        # Zero variance features
        zero_var_features = self.X.var()[self.X.var() == 0]
        if len(zero_var_features) > 0:
            print(f"\n⚠️  ZERO VARIANCE FEATURES:")
            print(f"   {list(zero_var_features.index)}")
        else:
            print(f"\n✓ No zero variance features found")
        
        input("\nPress Enter to continue...")
    
    def feature_importance_analysis(self):
        """Analyze feature importance using multiple methods"""
        clear_terminal()
        print("=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        # Variance-based importance
        variances = self.X.var().sort_values(ascending=False)
        
        # Mutual information
        try:
            mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
            mi_df = pd.DataFrame({
                'Feature': self.X.columns,
                'Mutual_Info': mi_scores
            }).sort_values('Mutual_Info', ascending=False)
        except:
            mi_df = None
        
        print(f"\n📊 VARIANCE-BASED IMPORTANCE (Top 10):")
        for feature, var in variances.head(10).items():
            print(f"   {feature}: {var:.4f}")
        
        if mi_df is not None:
            print(f"\n🔗 MUTUAL INFORMATION IMPORTANCE (Top 10):")
            for _, row in mi_df.head(10).iterrows():
                print(f"   {row['Feature']}: {row['Mutual_Info']:.4f}")
        
        input("\nPress Enter to continue...")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        clear_terminal()
        print("=" * 60)
        print("COMPREHENSIVE SUMMARY REPORT")
        print("=" * 60)
        
        # Save report to file
        report_path = Path("data_analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("SPAM DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            f.write(f"Dataset Shape: {self.df.shape}\n")
            f.write(f"Features: {len(self.features)-1}\n")
            f.write(f"Target Classes: {list(self.y.unique())}\n\n")
            
            # Target distribution
            target_counts = self.y.value_counts()
            f.write("Target Distribution:\n")
            for class_name, count in target_counts.items():
                percentage = (count / len(self.df)) * 100
                f.write(f"  Class {class_name}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Data quality
            f.write(f"Data Quality:\n")
            f.write(f"  Null values: {self.df.isnull().sum().sum()}\n")
            f.write(f"  Duplicates: {self.df.duplicated().sum()}\n")
            f.write(f"  Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
            
            # Feature statistics
            f.write("Feature Statistics:\n")
            f.write(self.X.describe().to_string())
            f.write("\n\n")
        
        print(f"✓ Summary report saved to: {report_path}")
        input("\nPress Enter to continue...")

def DataAnalysis():
    """Main data analysis function with improved interface"""
    try:
        analyzer = DataAnalyzer()
        
        while True:
            clear_terminal()
            print("=" * 60)
            print("COMPREHENSIVE DATA ANALYSIS")
            print("=" * 60)
            print("1 | Basic Statistics")
            print("2 | Data Quality Analysis")
            print("3 | Outlier Analysis")
            print("4 | Correlation Analysis")
            print("5 | Distribution Analysis")
            print("6 | Feature Importance Analysis")
            print("7 | Generate Summary Report")
            print("8 | Back to Main Menu")
            print("=" * 60)
            
            try:
                choice = int(input("Choose an option: "))
                
                if choice == 1:
                    analyzer.basic_statistics()
                elif choice == 2:
                    analyzer.data_quality_analysis()
                elif choice == 3:
                    analyzer.outlier_analysis()
                elif choice == 4:
                    analyzer.correlation_analysis()
                elif choice == 5:
                    analyzer.distribution_analysis()
                elif choice == 6:
                    analyzer.feature_importance_analysis()
                elif choice == 7:
                    analyzer.generate_summary_report()
                elif choice == 8:
                    break
                else:
                    print("Invalid choice. Please try again.")
                    input("Press Enter to continue...")
                    
            except ValueError:
                print("Please enter a valid number.")
                input("Press Enter to continue...")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
    except Exception as e:
        print(f"❌ Error initializing data analysis: {e}")
        input("Press Enter to continue...")

