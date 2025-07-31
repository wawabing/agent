import pandas as pd
import json
import os
import math
import tiktoken
from openai import OpenAI
from typing import Optional, Dict, Any, List, Callable, TypedDict
import hashlib
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.io import arff
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
import time
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import numpy as np

# our agents
from agent_detector import AgentDetector
from agent_rca import AgentRCA


# Define the state schema for our LangGraph anomaly pipeline
class AnomalyPipelineState(TypedDict):
    df: pd.DataFrame  # The current dataframe
    clean_df: Optional[pd.DataFrame]  # Preprocessed dataframe
    schema_id: Optional[str]  # Schema ID for memory lookups
    validated_df: Optional[pd.DataFrame]  # DataFrame with LLM validations
    anomaly_df: Optional[pd.DataFrame]  # DataFrame with anomaly predictions
    heuristic: Optional[Dict]  # Derived heuristic rule
    agreement: Optional[float]  # Agreement between heuristic and IF
    use_heuristic: Optional[bool]  # Whether to use the heuristic
    rule_applied: Optional[bool]  # Whether the rule has already been applied
    evaluated: Optional[bool]  # Whether results have been evaluated
    filename: str  # Current filename
    skip_llm_validation: Optional[bool]  # Whether to skip LLM validation
    skip_ml_detection: Optional[bool]  # Whether to skip ML detection
    log_structure: Optional[Dict]  # Log structure analysis results
    early_heuristic: Optional[Dict]  # Early derived heuristic

# LangGraph node functions - these ones are for the agent that takes raw data and detects errors
def load_data(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """Load and set up the initial dataframe"""
    print(f"[PHASE:LOAD] Loading dataset: {state['filename']}")
    file_path = f"demo-data/{state['filename']}.csv"
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[ERROR] File {file_path} not found. Please check the path and filename.")
        return state
    
    #print columns


    df.drop(columns=[" Label"], inplace=True, errors="ignore")
    print(f"[PHASE:LOAD] Loaded {len(df)} rows and {len(df.columns)} columns from {file_path} into dataframe")
    print("\n\n\n")    
    return {**state, "df": df}

def preprocess_data(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """Preprocess the data"""
    print(f"[PHASE:PREPERATION] Starting data preprocessing")
    clean_df, schema_id = agent.preprocess(state["df"])
    print("\n\n\n")
    return {**state, "clean_df": clean_df, "schema_id": schema_id}

def analyze_log_structure(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """
    Analyze log structure early to determine the best approach:
    1. If log level column found: Create comprehensive compound heuristic from ALL unique values
    2. If no log level column: Run anomaly detection and derive heuristics from extreme samples
    """
    df = state["clean_df"]
    schema_id = state["schema_id"]

    print(f"[STRUCTURE] Analyzing dataset structure")

    # Identify log structure
    log_structure, existed = agent.identify_log_structure(df, schema_id)

    # Check if we have a log level column
    log_level_column = log_structure.get("log_level_column")

    if log_level_column:
        if not existed:
            print(f"[STRUCTURE] Found log level column '{log_level_column}' - creating comprehensive heuristics")
            
            # Generate comprehensive log level heuristics from ALL unique values
            comprehensive_heuristics = agent.generate_comprehensive_log_level_heuristics(df, log_structure)
            
            if comprehensive_heuristics:
                # Use the comprehensive heuristic (should be just one)
                best_heuristic = comprehensive_heuristics[0]
                
                print(f"[DECISION] Using log-level heuristics - skipping LLM validation and ML detection")
                
                # Store the comprehensive heuristic in memory with high confidence
                agent._add_or_increment_heuristic(schema_id, best_heuristic, confidence_increment=4)  # Very high confidence for comprehensive rules
                
                # Set flags to skip everything else and use this comprehensive heuristic
                return {
                    **state, 
                    "log_structure": log_structure,
                    "early_heuristic": best_heuristic,
                    "skip_llm_validation": True,
                    "skip_ml_detection": True,  # We can skip ML too since we have comprehensive rules
                    "use_heuristic": True
                }
        elif log_structure.get("has_new_log_values", False):
            print(f"[STRUCTURE] New log values detected - updating heuristics")
            
            # Generate new comprehensive heuristics with all values (old + new)
            comprehensive_heuristics = agent.generate_comprehensive_log_level_heuristics(df, log_structure)
            
            if comprehensive_heuristics:
                # Replace the old comprehensive heuristic with the new one
                best_heuristic = comprehensive_heuristics[0]
                
                print(f"[DECISION] Updated log-level heuristics - skipping LLM validation and ML detection")
                
                # For comprehensive rules that may have updated values, we need special handling
                # Remove old comprehensive heuristics first, then add the new one
                existing_heuristics = agent.heuristic_memory.get(schema_id, [])
                non_comprehensive_heuristics = [h for h in existing_heuristics 
                                              if not h.get("rule", {}).get("comprehensive", False)]
                
                # Update memory with non-comprehensive rules only, then add the new comprehensive rule
                agent.heuristic_memory[schema_id] = non_comprehensive_heuristics
                agent._save_memory(agent.heuristic_memory, "heuristic")
                
                # Add the new comprehensive rule
                agent._add_or_increment_heuristic(schema_id, best_heuristic, confidence_increment=4)
                
                # Set flags to skip everything else and use this comprehensive heuristic
                return {
                    **state, 
                    "log_structure": log_structure,
                    "early_heuristic": best_heuristic,
                    "skip_llm_validation": True,
                    "skip_ml_detection": True,  # We can skip ML too since we have comprehensive rules
                    "use_heuristic": True
                }
            else:
                print(f"[ERROR] Could not generate updated heuristics")
        else:
            print(f"[DECISION] Using existing log-level heuristics - skipping LLM validation")
            # Use existing heuristics from memory
            existing_heuristics = agent.heuristic_memory.get(schema_id, [])
            if existing_heuristics:
                # Sort by confidence and get the best rule (not the wrapper)
                sorted_heuristics = sorted(existing_heuristics, key=lambda x: x.get("confidence", 0), reverse=True)
                best_heuristic = sorted_heuristics[0].get("rule", {})  # Extract the actual rule
                
                return {
                    **state, 
                    "log_structure": log_structure, 
                    "early_heuristic": best_heuristic,  # ✅ Now it's a single rule object
                    "skip_llm_validation": True,
                    "skip_ml_detection": True,  # Also skip ML since we have good heuristics
                    "use_heuristic": True,
                }
            else:
                print(f"[ERROR] No existing heuristics found despite cache hit")
                return {**state, "log_structure": log_structure, "skip_llm_validation": False}
    else:
        # No log level column found - check textual content ratio
        textual_cols = df.select_dtypes(include=["object", "category"]).columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        total_cols = len(df.columns)
        textual_ratio = len(textual_cols) / total_cols
        
        print(f"[STRUCTURE] No log level column found")
        print(f"[STRUCTURE] Data composition: {len(textual_cols)} textual, {len(numeric_cols)} numeric ({textual_ratio:.1%} textual)")
        
        # Check if we have sufficient textual content (≥50%) to attempt heuristic derivation
        if textual_ratio >= 0.5:
            print(f"[STRUCTURE] Sufficient textual content ({textual_ratio:.1%}) - attempting heuristic derivation")
            
            # Step 1: Try anomaly-based heuristic extraction (10 most + 10 least anomalous)
            print(f"[HEURISTIC:STEP1] Attempting anomaly-based heuristic extraction")
            anomaly_based_heuristics = agent.derive_heuristics_from_anomaly_scores(df, schema_id)
            
            if anomaly_based_heuristics:
                best_heuristic = anomaly_based_heuristics[0]
                print(f"[DECISION] Derived heuristic from anomaly patterns - skipping LLM validation")
                
                # NOTE: Heuristic already added to memory by derive_heuristics_from_anomaly_scores()
                # No need to manually add it again here
                
                return {
                    **state, 
                    "log_structure": log_structure,
                    "early_heuristic": best_heuristic,
                    "skip_llm_validation": True,  # Skip LLM since we have a heuristic
                    "use_heuristic": True
                }
            
            # Step 2: Anomaly-based failed, try textual-only heuristic as last ditch attempt
            print(f"[HEURISTIC:STEP2] Anomaly-based failed - attempting textual-only heuristic extraction")
            textual_heuristic = agent.get_textual_heuristic(df[textual_cols], schema_id)
            
            if textual_heuristic:
                print(f"[DECISION] Derived textual heuristic - skipping LLM validation")
                return {
                    **state, 
                    "log_structure": log_structure,
                    "early_heuristic": textual_heuristic,
                    "skip_llm_validation": True,  # Skip LLM since we have a heuristic
                    "use_heuristic": True
                }
            
            # Step 3: Both heuristic attempts failed → Fall back to pure ML detection
            print(f"[HEURISTIC:STEP3] Both heuristic extraction methods failed")
            print(f"[DECISION] No reliable patterns found in {textual_ratio:.1%} textual data - falling back to ML-only anomaly detection")
            return {
                **state, 
                "log_structure": log_structure, 
                "skip_llm_validation": True,   # Skip LLM - if it can't derive heuristics, it can't label reliably
                "skip_ml_detection": False,    # Use ML detection instead
                "use_heuristic": False         # No heuristics available
            }
        
        else:
            # <50% textual content - skip LLM validation entirely
            print(f"[STRUCTURE] Insufficient textual content ({textual_ratio:.1%}) for heuristic derivation or LLM validation")
            print(f"[DECISION] Proceeding directly to ML-based anomaly detection")
            
            return {
                **state, 
                "log_structure": log_structure, 
                "skip_llm_validation": True,  # Skip LLM for predominantly numeric data
                "skip_ml_detection": False,   # We need ML for numeric data
                "use_heuristic": False        # No heuristics for predominantly numeric
            }

def run_llm_validation_conditional(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """
    Run LLM validation only if needed (CURRENTLY DISABLED IN 3-ROUTE SYSTEM).
    
    IMPORTANT: This function is kept for future use but is currently skipped in the 3-route system.
    The new logic skips LLM validation because:
    - If LLM can't derive heuristics, it can't reliably label individual records either
    - Better to fall back to ML-only detection instead of "blind guessing"
    
    This function would only be used if we had existing heuristics that needed LLM validation.
    """
    print("\n\n\n")
    # Skip LLM validation if we found good early heuristics OR if dataset is purely numeric
    if state.get("skip_llm_validation", False):
        print(f"[LLM VALIDATION] Skipping validation (3-route system - no blind guessing)")
        print("\n\n\n")
        return state
    
    # Otherwise run the original LLM validation logic
    schema_id = state["schema_id"]
    hyperparams = agent.hyperparam_memory.get(schema_id)
    
    if hyperparams is None:
        print(f"[LLM] Running validation and hyperparameter optimization")
        validated_df = agent.llm_validate_predictions(state["clean_df"])
        
        # Add this section to tune and save hyperparameters immediately
        agent.tune_isolation_forest_by_llm(validated_df, schema_id)
        
        return {**state, "validated_df": validated_df}
    
    print(f"[LLM] Using cached hyperparameters")
    print("\n\n\n")
    return state

def detect_anomalies(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """Detect anomalies using Isolation Forest (or skip if we have comprehensive heuristics)"""
    schema_id = state["schema_id"]
    
    # Check if we can skip ML detection entirely (for comprehensive log-level heuristics)
    if state.get("skip_ml_detection", False):
        print(f"[ROUTE:1] PURE HEURISTIC (Log-Level Rules) - Skipping ML anomaly detection")
        
        # Use validated_df if available, otherwise use clean_df
        if state.get("validated_df") is not None:
            input_df = state["validated_df"]
        else:
            input_df = state["clean_df"]
            
        # Create a dummy anomaly_df with just the input data (no ML predictions)
        anomaly_df = input_df.copy()
        anomaly_df["anomaly_label"] = 1  # Initialize all as normal, heuristic will override
        
        print("\n\n\n")
        return {**state, "anomaly_df": anomaly_df}
    
    # Determine which route we're taking based on available heuristics
    heuristics = agent.heuristic_memory.get(schema_id, [])
    has_heuristic = heuristics and isinstance(heuristics, list) and len(heuristics) > 0
    use_heuristic = state.get("use_heuristic", False)
    
    # Display the correct route
    if has_heuristic and use_heuristic:
        print(f"[ROUTE:3] HYBRID (ML + Heuristic) - Combining ML detection with derived heuristics")
        print(f"[APPROACH:HYBRID] Will combine Isolation Forest with rule-based patterns")
    else:
        print(f"[ROUTE:2] ML-ONLY (Unsupervised) - Pure machine learning anomaly detection")
        print(f"[APPROACH:ML-ONLY] Using Isolation Forest without heuristic guidance")
    
    print(f"[MODEL:INFO] Using Isolation Forest with dynamic threshold adjustment")
    
    # Use validated_df if available, otherwise use clean_df
    if state.get("validated_df") is not None:
        input_df = state["validated_df"]
    else:
        input_df = state["clean_df"]
        
    if input_df is None:
        raise ValueError("No dataframe available for anomaly detection")

    anomaly_df = agent.detect_anomalies_pure_and_hybrid(input_df, schema_id, label_col="y")
    print("\n\n\n")
    return {**state, "anomaly_df": anomaly_df}

def apply_heuristic(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """Apply heuristic rule if available and enabled"""
    
    # Skip if we're not supposed to use heuristics
    if not state.get("use_heuristic", False):
        print(f"[ROUTE:CONFIRM] ML-ONLY route - No heuristic to apply, using ML results only")
        print("\n\n\n")
        return state
    
    # NEW: Check if this is a hybrid route that already applied heuristics in ML stage
    ml_ran = not state.get("skip_ml_detection", False)
    if ml_ran and state.get("use_heuristic", False):
        print(f"[ROUTE:CONFIRM] HYBRID route - Heuristics already integrated with ML in Stage 5")
        print(f"[ROUTE:CONFIRM] Skipping Stage 6 heuristic application to prevent double-application")
        print(f"[ROUTE:CONFIRM] Final results already reflect combined ML + heuristic decision")
        print("\n\n\n")
        return state
    
    print(f"[ROUTE:CONFIRM] PURE HEURISTIC route - Applying rule-based patterns only")
    
    # Get the dataframe to work with
    df = state["anomaly_df"]
    schema_id = state["schema_id"]
    
    if df is None:
        print(f"[HEURISTIC:ERROR] No dataframe available for heuristic application")
        return state
    
    # Check if we have an early heuristic from structure analysis
    heuristic = state.get("early_heuristic")
    
    # If no early heuristic, try to get the best one from memory
    if not heuristic:
        existing_heuristics = agent.heuristic_memory.get(schema_id, [])
        
        if not existing_heuristics:
            print(f"[HEURISTIC] No heuristics found in memory for schema {schema_id}")
            print("\n\n\n")
            return state
        
        # Sort by confidence and get the best one
        sorted_heuristics = sorted(existing_heuristics, key=lambda x: x.get("confidence", 0), reverse=True)
        best_heuristic_entry = sorted_heuristics[0]
        heuristic = best_heuristic_entry.get("rule", {})
        confidence = best_heuristic_entry.get("confidence", 0)
        
        print(f"[HEURISTIC] Using best heuristic from memory (confidence: {confidence})")
    else:
        print(f"[HEURISTIC] Using early heuristic from structure analysis")
    
    # DEFENSIVE CHECK: Handle case where heuristic might be a list instead of dict
    if isinstance(heuristic, list):
        print(f"[HEURISTIC:WARNING] Received list instead of single heuristic, using first item")
        if heuristic and isinstance(heuristic[0], dict) and "rule" in heuristic[0]:
            heuristic = heuristic[0]["rule"]
        elif heuristic and isinstance(heuristic[0], dict):
            heuristic = heuristic[0]
        else:
            print(f"[HEURISTIC:ERROR] Invalid heuristic format in list")
            return state

    if not heuristic or not isinstance(heuristic, dict):
        print(f"[HEURISTIC:ERROR] No valid heuristic found or wrong type: {type(heuristic)}")
        return state
    
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Check if this is a compound rule
    if heuristic.get("type") == "compound" and heuristic.get("logic") == "OR":
        print(f"[HEURISTIC] Applying compound rule with OR logic")
        
        rules = heuristic.get("rules", [])
        normal_label = heuristic.get("normal_label", 1)
        anomaly_label = heuristic.get("anomaly_label", 0)
        
        # Start with all False (no matches)
        matches = pd.Series(False, index=df.index)
        
        # Apply each sub-rule with OR logic
        applied_rules = []
        for rule in rules:
            col = rule.get("column")
            op = rule.get("operator")
            val = rule.get("value")
            
            if col not in df.columns:
                print(f"[HEURISTIC:WARNING] Column '{col}' not found in dataframe")
                continue
            
            # Apply the rule condition
            if op == "==":
                rule_matches = (df[col] == val)
            elif op == "!=":
                rule_matches = (df[col] != val)
            elif op == "contains":
                rule_matches = df[col].astype(str).str.contains(str(val), case=False, na=False)
            else:
                print(f"[HEURISTIC:WARNING] Unsupported operator '{op}' - skipping rule")
                continue
            
            # OR the results together
            matches = matches | rule_matches
            match_count = rule_matches.sum()
            applied_rules.append(f"{col} {op} '{val}' ({match_count} matches)")
            
        # Apply the compound rule results
        # If any rule matches → normal_label, otherwise → anomaly_label
        df["anomaly_label"] = matches.apply(lambda x: normal_label if x else anomaly_label)
        
        # Print summary
        total_matches = matches.sum()
        rule_text = " OR ".join(applied_rules)
        print(f"[HEURISTIC:APPLIED] Compound rule: {rule_text}")
        print(f"[HEURISTIC:RESULT] {total_matches} records matched → label {normal_label}")
        
    else:
        # Handle simple (single) rules
        print(f"[HEURISTIC] Applying simple rule")
        
        col = heuristic.get("column")
        op = heuristic.get("operator", "==")
        val = heuristic.get("value")
        normal_label = heuristic.get("normal_label", 1)
        anomaly_label = heuristic.get("anomaly_label", 0)
        
        if not col or col not in df.columns:
            print(f"[HEURISTIC:ERROR] Column '{col}' not found in dataframe")
            return state
        
        # Apply the single rule
        if op == "==":
            matches = (df[col] == val)
        elif op == "!=":
            matches = (df[col] != val)
        elif op == "contains":
            matches = df[col].astype(str).str.contains(str(val), case=False, na=False)
        else:
            print(f"[HEURISTIC:ERROR] Unsupported operator '{op}'")
            return state
        
        # Apply the rule results
        df["anomaly_label"] = matches.apply(lambda x: normal_label if x else anomaly_label)
        
        # Print summary
        match_count = matches.sum()
        print(f"[HEURISTIC:APPLIED] Rule: {col} {op} '{val}' → {match_count} matches → label {normal_label}")
    
    # Calculate final anomaly distribution
    anomaly_count = (df["anomaly_label"] == 0).sum()
    normal_count = (df["anomaly_label"] == 1).sum()
    total_count = len(df)
    
    print(f"[HEURISTIC:SUMMARY] Final distribution: {anomaly_count} anomalies, {normal_count} normal ({anomaly_count/total_count:.1%} anomalous)")
    
    # Update state with the modified dataframe
    state["anomaly_df"] = df
    state["rule_applied"] = True
    
    print("\n\n\n")
    return state

def evaluate_results(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """Evaluate and save results"""
    # Check if we've already evaluated results
    if state.get("evaluated") is True:
        return state
        
    print(f"[PHASE:EVALUATE] Assessing detection performance")
    
    # Visualize the results if plotting is enabled
    try:
        agent.plot_isolation_forest_results(state["anomaly_df"])
    except Exception as e:
        print(f"[VISUAL:SKIP] Visualization skipped - {str(e)}")
    
    # Evaluate the results
    llm_col = "llm_validated" if "llm_validated" in state["anomaly_df"].columns else None
    agent.evaluate_and_print(
        state["anomaly_df"],
        ground_truth_col=None,
        ml_pred_col="anomaly_label",
        llm_pred_col=llm_col
    )
    
    # Print summary of detected anomalies
    anomalies_df = state["anomaly_df"][state["anomaly_df"]["anomaly_label"] == 0]
    anomaly_count = len(anomalies_df)
    total_count = len(state["anomaly_df"])
    
    # Determine the detection strategy used based on what actually ran
    llm_ran = state.get("validated_df") is not None
    ml_ran = not state.get("skip_ml_detection", False)
    heuristic_used = state.get("use_heuristic", False)
    
    if heuristic_used and not ml_ran and not llm_ran:
        strategy = "Pure Heuristic (Log-Level Rules)"
    elif heuristic_used and ml_ran:
        strategy = "Hybrid (ML + Heuristic)"
    elif llm_ran and ml_ran:
        strategy = "Hybrid (LLM + ML)"
    elif llm_ran and not ml_ran:
        strategy = "LLM Validation Only"
    elif ml_ran and not heuristic_used:
        strategy = "ML-Only (Unsupervised)"
    else:
        strategy = "Unknown Strategy"
    
    print(f"\n[SUMMARY:DETECT] {anomaly_count} anomalies in {total_count} records ({anomaly_count/total_count:.1%})")
    print(f"[SUMMARY:METHOD] Detection strategy: {strategy}")
    
    # Print more information if a heuristic was used
    if state.get("use_heuristic") and state.get("heuristic"):
        rule = state["heuristic"]
        agreement = state.get("agreement", 0)
        
        # Handle different types of rules in the summary
        if rule.get("type") == "compound" and rule.get("operator") == "OR":
            # For compound rules, build a readable description
            rule_descriptions = []
            for sub_rule in rule.get("rules", []):
                col = sub_rule.get("column")
                op = sub_rule.get("operator")
                val = sub_rule.get("value")
                rule_descriptions.append(f"{col} {op} '{val}'")
            
            rule_text = " OR ".join(rule_descriptions)
            print(f"[SUMMARY:RULE] Final rule: If {rule_text} (agreement: {agreement:.2%})")
        else:
            # For simple rules
            col = rule.get("column")
            op = rule.get("operator")
            val = rule.get("value")
            print(f"[SUMMARY:RULE] Final rule: If {col} {op} '{val}' (agreement: {agreement:.2%})")
    
    # If we have a small number of anomalies, print them for quick review
    if anomaly_count > 0 and anomaly_count <= 10:
        print("\n[DETAIL:ANOMALIES] Log entries flagged as anomalous:")
        
        # Look for the most informative column to display
        content_columns = ["Content", "Message", "Description", "Body", "Text", "Log"]
        display_col = next((col for col in content_columns if col in anomalies_df.columns), None)
        
        if display_col:
            for i, (_, row) in enumerate(anomalies_df.iterrows()):
                level = row.get("Level", "N/A")
                content = row[display_col]
                print(f"  {i+1}. [{level}] {content[:100]}..." if len(content) > 100 else f"  {i+1}. [{level}] {content}")
        else:
            # Just show row numbers if no good content column found
            print(f"  [INDICES:ANOMALY] Row numbers: {list(anomalies_df.index)}")
    
    # Save the results (both full results and errors-only file)
    agent.save_results(state["anomaly_df"], filename=f"{state['filename']}_anomalies_validated.csv")
    print("\n\n\n")
    # Mark that evaluation is complete
    return {**state, "evaluated": True}




# Define the state schema for our LangGraph root cause analysis pipeline
class RCAPipelineState(TypedDict):
    anomaly_df: pd.DataFrame  # DataFrame with detected anomalies
    rca_results: Optional[Dict[str, Any]]  # Results of the root cause analysis
    filename: str  # Current filename for results
    evaluated: bool  # Whether results have been evaluated
    rca_summary: Optional[str]  # Summary of the RCA findings



# LangGraph node functions - these ones are for the agent that takes the errors and performs root cause analysis
def load_results(state: RCAPipelineState, agent: AgentRCA) -> RCAPipelineState:
    """Load the anomaly results for root cause analysis"""
    print(f"[RCA:LOAD] Loading anomaly results for RCA from {state['filename']}")
    file_path = f"results/{state['filename']}_anomalies_validated_errors.csv"
    
    try:
        df = agent.load_data(f"{state['filename']}_anomalies_validated_errors.csv")
    except FileNotFoundError:
        print(f"[RCA:ERROR] File {file_path} not found. Please check the path and filename.")
        return state
    
    print(f"[RCA:LOAD] Successfully loaded anomaly results")
    print("\n\n\n")
    return {**state, "anomaly_df": df}

def run_rca_analysis(state: RCAPipelineState, agent: AgentRCA) -> RCAPipelineState:
    """Run root cause analysis on the detected anomalies"""
    anomaly_df = state["anomaly_df"]
    
    if anomaly_df is None or len(anomaly_df) == 0:
        print(f"[RCA:SKIP] No data available for root cause analysis")
        return {**state, "rca_results": {}, "rca_summary": "No data to analyze"}
    
    # Filter to only anomalies (where anomaly_label == 0)
    anomalies_only = anomaly_df[anomaly_df["anomaly_label"] == 0].copy()
    
    if len(anomalies_only) == 0:
        print(f"[RCA:SKIP] No anomalies found in the dataset")
        return {**state, "rca_results": {}, "rca_summary": "No anomalies detected"}
    
    print(f"[RCA:ANALYZE] Running root cause analysis on {len(anomalies_only)} anomalies")
    
    # Run the analysis
    rca_results = agent.analyze_errors(anomalies_only, anomaly_df)
    
    # Create a summary
    summary = f"Analyzed {len(anomalies_only)} anomalies out of {len(anomaly_df)} total records"
    
    print(f"[RCA:COMPLETE] Root cause analysis completed")
    print("\n\n\n")
    
    return {**state, "rca_results": rca_results, "rca_summary": summary, "evaluated": True}







# Create a LangGraph workflow for the anomaly detection pipeline
# This pipeline intelligently routes based on data characteristics and available heuristics
def create_anomaly_pipeline(agent: AgentDetector):
    # Create the graph
    workflow = StateGraph(AnomalyPipelineState)
    
    # Add nodes with clear descriptions - intelligent 3-route pipeline
    print(f"[WORKFLOW:CREATE] Setting up intelligent anomaly detection pipeline with 7 stages")
    print(f"[STRATEGY:ADAPTIVE] Pipeline automatically selects optimal detection approach:")
    print(f"  Route 1: Pure Heuristic (Log-Level Rules) - Log level column found")
    print(f"  Route 2: ML-Only (Unsupervised) - <50% textual OR no patterns found")
    print(f"  Route 3: Hybrid (ML + Heuristic) - Heuristic successfully derived")
    
    workflow.add_node("load_data", lambda state: load_data(state, agent))                           # Stage 1: Data Loading
    workflow.add_node("preprocess_data", lambda state: preprocess_data(state, agent))               # Stage 2: Data Preprocessing  
    workflow.add_node("analyze_log_structure", lambda state: analyze_log_structure(state, agent))  # Stage 3: Structure Analysis & Route Selection
    workflow.add_node("run_llm_validation", lambda state: run_llm_validation_conditional(state, agent))  # Stage 4: LLM Validation (KEPT FOR FUTURE USE)
    workflow.add_node("detect_anomalies", lambda state: detect_anomalies(state, agent))             # Stage 5: ML Detection (Conditional)
    workflow.add_node("apply_heuristic", lambda state: apply_heuristic(state, agent))               # Stage 6: Heuristic Application (Conditional)
    workflow.add_node("evaluate_results", lambda state: evaluate_results(state, agent))             # Stage 7: Results & Strategy Reporting
    
    # Define edges - intelligent routing based on data characteristics
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "preprocess_data")
    workflow.add_edge("preprocess_data", "analyze_log_structure")         # Smart route selection happens here
    workflow.add_edge("analyze_log_structure", "run_llm_validation")      # LLM validation (kept but mostly skipped)
    workflow.add_edge("run_llm_validation", "detect_anomalies")           # ML detection (conditional based on route)
    workflow.add_edge("detect_anomalies", "apply_heuristic")              # Heuristic application (conditional based on route)
    workflow.add_edge("apply_heuristic", "evaluate_results")              # Results evaluation with strategy detection
    workflow.add_edge("evaluate_results", END)
    
    print(f"[WORKFLOW:ROUTING] Decision logic:")
    print(f"  Stage 3 (analyze_log_structure) determines which route to take")
    print(f"  Stages 4-6 execute conditionally based on selected route")
    print(f"  Stage 7 reports which strategy was actually used")
    
    # Compile the graph
    return workflow.compile()

# Create a LangGraph workflow for the root cause analysis pipeline
def create_rca_pipeline(agent: AgentRCA):
    """Create the root cause analysis pipeline"""
    rca_workflow = StateGraph(RCAPipelineState)
    
    # Add nodes for RCA
    rca_workflow.add_node("load_results", lambda state: load_results(state, agent))
    rca_workflow.add_node("run_rca_analysis", lambda state: run_rca_analysis(state, agent))
    
    # Define edges
    rca_workflow.set_entry_point("load_results")
    rca_workflow.add_edge("load_results", "run_rca_analysis")
    rca_workflow.add_edge("run_rca_analysis", END)
    
    return rca_workflow.compile()

def main():
    # Initialize agents
    detector_agent = AgentDetector()
    rca_agent = AgentRCA()
    
    # Create the pipelines
    anomaly_pipeline = create_anomaly_pipeline(detector_agent)
    rca_pipeline = create_rca_pipeline(rca_agent)

    # Define the input state for anomaly detection
    initial_state = AnomalyPipelineState(
        df=None,
        filename="hybrid-route-data",
        clean_df=None,        
        schema_id=None,
        validated_df=None, 
        anomaly_df=None,
        heuristic=None,
        agreement=None,
        use_heuristic=None,
        rule_applied=None,
        evaluated=None,
        skip_llm_validation=None,
        skip_ml_detection=None,
        log_structure=None,
        early_heuristic=None
    )
    
    # Run the anomaly detection pipeline
    print(f"[START:ANOMALY_PIPELINE] Beginning anomaly detection workflow")
    anomaly_result = anomaly_pipeline.invoke(initial_state)
    print(f"[END:ANOMALY_PIPELINE] Anomaly detection workflow completed\n\n\n")


    # Define the RCA pipeline state
    rca_state = RCAPipelineState(
        anomaly_df=None,  # Will be loaded from file
        rca_results=None,
        filename=initial_state["filename"],  # Use same filename
        evaluated=False,
        rca_summary=None
    )

    # Now run the RCA pipeline on the results
    print(f"[START:RCA_PIPELINE] Beginning root cause analysis workflow")
    rca_result = rca_pipeline.invoke(rca_state)
    print(f"[END:RCA_PIPELINE] Root cause analysis workflow completed")
    
    # Print final summary
    if rca_result.get("rca_summary"):
        print(f"[FINAL:SUMMARY] {rca_result['rca_summary']}")


if __name__ == "__main__":
    main()