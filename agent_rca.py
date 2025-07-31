import pandas as pd
import os
from typing import Optional, Dict, Any


class AgentRCA:
    """
    Root Cause Analysis Agent that analyzes anomalies detected by the anomaly detection agent.
    """
    
    def __init__(self):
        """Initialize the RCA agent."""
        self.base_path = os.path.dirname(__file__)
        self.results_folder = os.path.join(self.base_path, "results")
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load anomaly detection results for root cause analysis.
        
        Args:
            filename: Name of the file containing anomaly detection results
            
        Returns:
            DataFrame containing the anomaly detection results
        """
        # Look for the file in the results folder
        file_path = os.path.join(self.results_folder, filename)
        
        # If not found in results, try the current directory
        if not os.path.exists(file_path):
            file_path = filename
            
        # If still not found, try adding .csv extension
        if not os.path.exists(file_path) and not filename.endswith('.csv'):
            file_path = f"{filename}.csv"
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find anomaly results file: {filename}")
            
        try:
            df = pd.read_csv(file_path)
            print(f"[RCA:LOAD] Loaded {len(df)} rows from {file_path}")
            
            # Validate that this looks like anomaly detection results
            if 'anomaly_label' not in df.columns:
                print(f"[RCA:WARNING] No 'anomaly_label' column found. Available columns: {list(df.columns)}")
            else:
                anomaly_count = (df['anomaly_label'] == 0).sum()
                normal_count = (df['anomaly_label'] == 1).sum()
                print(f"[RCA:DATA] Found {anomaly_count} anomalies and {normal_count} normal records")
                
            return df
            
        except Exception as e:
            raise Exception(f"Error loading anomaly results file {file_path}: {str(e)}")
            
    def analyze_errors(self, anomalies_df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze anomalies to determine root causes.
        
        Args:
            anomalies_df: DataFrame containing detected anomalies
            original_df: Optional original dataset for additional context
            
        Returns:
            Dictionary containing root cause analysis results
        """
        print(f"[RCA:ANALYZE] Starting root cause analysis on {len(anomalies_df)} anomalies")
        
        # Basic analysis for now - can be expanded
        analysis_results = {
            "total_anomalies": len(anomalies_df),
            "anomaly_types": {},
            "patterns": [],
            "recommendations": []
        }
        
        # Analyze each anomaly
        for idx, row in anomalies_df.iterrows():
            print(f"[RCA:ITEM] Anomaly {idx}: {dict(row)}")
            
        print(f"[RCA:COMPLETE] Root cause analysis completed")
        return analysis_results
