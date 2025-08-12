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
    print(f"\n\n\n[PHASE:LOAD_DATA] Starting data loading phase")
    print(f"[LOAD_DATA] Loading dataset: {state['filename']}")
    file_path = f"demo-data/{state['filename']}.csv"
    
    try:
        df = pd.read_csv(file_path)
        print(f"[LOAD_DATA] Successfully loaded {len(df)} rows and {len(df.columns)} columns from {file_path}")
    except FileNotFoundError:
        print(f"[LOAD_DATA:ERROR] File {file_path} not found. Please check the path and filename.")
        return state
    
    # Clean up any label columns that might interfere
    df.drop(columns=[" Label"], inplace=True, errors="ignore")
    print(f"[LOAD_DATA] Data preparation complete - ready for preprocessing")
    print(f"[PHASE:LOAD_DATA] ✓ Data loading phase completed")
    print("\n\n\n")    
    return {**state, "df": df}

def preprocess_data(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """Preprocess the data"""
    print(f"[PHASE:PREPROCESS_DATA] Starting data preprocessing phase")
    print(f"[PREPROCESS_DATA] Analyzing column structure and applying ML-optimized preprocessing")
    clean_df, schema_id = agent.preprocess(state["df"])
    print(f"[PREPROCESS_DATA] Preprocessing complete - data optimized for ML algorithms")
    print(f"[PREPROCESS_DATA] Schema ID: {schema_id}")
    print(f"[PHASE:PREPROCESS_DATA] ✓ Data preprocessing phase completed")
    print("\n\n\n")
    return {**state, "clean_df": clean_df, "schema_id": schema_id}

def analyze_log_structure(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """
    Analyze log structure early to determine the best approach:
    1. If log level column found: Create comprehensive compound heuristic from ALL unique values
    2. If no log level column: Run anomaly detection and derive heuristics from extreme samples
    
    IMPORTANT: Uses original dataframe to identify ALL columns including timestamps,
    but subsequent processing uses clean_df for ML operations.
    """
    print(f"[PHASE:ANALYZE_LOG_STRUCTURE] Starting log structure analysis and route selection phase")
    print(f"[ANALYZE_LOG_STRUCTURE] Examining dataset characteristics to determine optimal detection approach")
    
    # CHANGED: Use original dataframe for structure analysis to capture ALL columns including timestamps
    original_df = state["df"]  # This has all original columns including timestamps
    clean_df = state["clean_df"]  # This is preprocessed for ML (timestamps dropped)
    schema_id = state["schema_id"]

    print(f"[ANALYZE_LOG_STRUCTURE] Using original dataframe to capture all column types (including timestamps)")

    # Identify log structure using the ORIGINAL dataframe (includes timestamps)
    log_structure, existed = agent.identify_log_structure(original_df, schema_id)

    # Check if we have a log level column
    log_level_column = log_structure.get("log_level_column")

    if log_level_column:
        # IMPORTANT: For heuristic generation, we need to use the clean_df since that's what ML will process
        # But first, verify the log level column still exists after preprocessing
        if log_level_column not in clean_df.columns:
            print(f"[ANALYZE_LOG_STRUCTURE:WARNING] Log level column '{log_level_column}' was dropped during preprocessing")
            print(f"[ANALYZE_LOG_STRUCTURE:DECISION] Falling back to non-log-level analysis due to preprocessing conflict")
            # Treat as if no log level column was found
            log_level_column = None
        else:
            print(f"[ANALYZE_LOG_STRUCTURE] ✓ Log level column '{log_level_column}' preserved after preprocessing")
    
    if log_level_column:
        if not existed:
            print(f"[ANALYZE_LOG_STRUCTURE:ROUTE_DECISION] ROUTE 1: PURE HEURISTIC selected")
            print(f"[ANALYZE_LOG_STRUCTURE:REASONING] Log level column '{log_level_column}' found - can create comprehensive rule-based detection")
            
            # Generate comprehensive log level heuristics using CLEAN dataframe for consistency with ML
            comprehensive_heuristics = agent.generate_comprehensive_log_level_heuristics(clean_df, log_structure, schema_id)
            
            if comprehensive_heuristics:
                # Use the comprehensive heuristic (should be just one)
                best_heuristic = comprehensive_heuristics[0]
                
                print(f"[ANALYZE_LOG_STRUCTURE:SUCCESS] Comprehensive log-level heuristic created successfully")
                print(f"[ANALYZE_LOG_STRUCTURE:DECISION] Skipping LLM validation and ML detection - rule-based detection sufficient")
                print(f"[PHASE:ANALYZE_LOG_STRUCTURE] ✓ Route selection completed - PURE HEURISTIC pipeline")
                
                # Store the comprehensive heuristic in memory with high confidence
                agent._add_or_increment_heuristic(schema_id, best_heuristic, confidence_increment=4)  # Very high confidence for comprehensive rules
                
                # Update log structure with detection approach
                agent.update_log_structure_with_detection_approach(schema_id, "pure_heuristic")
                
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
            
            # Generate new comprehensive heuristics with all values (old + new) using CLEAN dataframe
            comprehensive_heuristics = agent.generate_comprehensive_log_level_heuristics(clean_df, log_structure, schema_id)
            
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
                
                # Update log structure with detection approach
                agent.update_log_structure_with_detection_approach(schema_id, "pure_heuristic")
                
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
            print(f"[ANALYZE_LOG_STRUCTURE:HEURISTIC_REUSE] Using existing log-level heuristics from memory")
            print(f"[ANALYZE_LOG_STRUCTURE:DECISION] Skipping LLM validation - reliable heuristics cached")
            print(f"[PHASE:ANALYZE_LOG_STRUCTURE] ✓ Route selection completed - PURE HEURISTIC pipeline (from cache)")
            
            # Update log structure with detection approach (for cached case)
            agent.update_log_structure_with_detection_approach(schema_id, "pure_heuristic")
            
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
        # No log level column found - check for message column and textual content ratio using CLEAN dataframe
        message_column = log_structure.get("message_column")
        textual_cols = clean_df.select_dtypes(include=["object", "category"]).columns
        numeric_cols = clean_df.select_dtypes(include=["number"]).columns
        total_cols = len(clean_df.columns)
        textual_ratio = len(textual_cols) / total_cols
        
        print(f"[STRUCTURE] No log level column found")
        print(f"[STRUCTURE] Message column found: {message_column if message_column else 'None'}")
        print(f"[STRUCTURE] Data composition: {len(textual_cols)} textual, {len(numeric_cols)} numeric ({textual_ratio:.1%} textual)")
        
        # NEW LOGIC: LLM is suitable if message column exists OR if ≥50% textual content
        llm_suitable = message_column is not None or textual_ratio >= 0.5
        
        # Check if we should attempt heuristic derivation
        if llm_suitable:
            if message_column:
                print(f"[ANALYZE_LOG_STRUCTURE:LLM_SUITABLE] Message column '{message_column}' found - LLM validation suitable")
                print(f"[ANALYZE_LOG_STRUCTURE:REASONING] Message column enables LLM processing regardless of textual ratio")
            else:
                print(f"[ANALYZE_LOG_STRUCTURE:TEXTUAL_ANALYSIS] Sufficient textual content ({textual_ratio:.1%}) - attempting heuristic derivation")
                print(f"[ANALYZE_LOG_STRUCTURE:REASONING] High textual ratio indicates potential for pattern-based detection")
            
            # Step 1: Try anomaly-based heuristic extraction (10 most + 10 least anomalous)
            print(f"[ANALYZE_LOG_STRUCTURE:STEP1] Attempting anomaly-based heuristic extraction from data patterns")
            anomaly_based_heuristics = agent.derive_heuristics_from_anomaly_scores(clean_df, schema_id)
            
            if anomaly_based_heuristics:
                best_heuristic = anomaly_based_heuristics[0]
                print(f"[ANALYZE_LOG_STRUCTURE:SUCCESS] Anomaly-based heuristic derived successfully")
                print(f"[ANALYZE_LOG_STRUCTURE:DECISION] Skipping LLM validation - pattern-based heuristic sufficient")
                print(f"[PHASE:ANALYZE_LOG_STRUCTURE] ✓ Route selection completed - HYBRID pipeline with anomaly patterns")
                
                # Update log structure with detection approach
                agent.update_log_structure_with_detection_approach(schema_id, "hybrid_ml_heuristic")
                
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
            print(f"[ANALYZE_LOG_STRUCTURE:STEP2] Anomaly-based patterns insufficient - trying textual-only heuristic extraction")
            print(f"[ANALYZE_LOG_STRUCTURE:REASONING] Focusing on text patterns in {len(textual_cols)} textual columns")
            textual_heuristic = agent.get_textual_heuristic(clean_df[textual_cols], schema_id)
            
            if textual_heuristic:
                print(f"[ANALYZE_LOG_STRUCTURE:SUCCESS] Textual pattern heuristic created successfully")
                print(f"[ANALYZE_LOG_STRUCTURE:DECISION] Skipping LLM validation - textual patterns sufficient")
                print(f"[PHASE:ANALYZE_LOG_STRUCTURE] ✓ Route selection completed - HYBRID pipeline with textual patterns")
                
                # Update log structure with detection approach
                agent.update_log_structure_with_detection_approach(schema_id, "hybrid_ml_heuristic")
                
                return {
                    **state, 
                    "log_structure": log_structure,
                    "early_heuristic": textual_heuristic,
                    "skip_llm_validation": True,  # Skip LLM since we have a heuristic
                    "use_heuristic": True
                }
            
            # Step 3: Both heuristic attempts failed → Check if LLM fallback is possible
            print(f"[ANALYZE_LOG_STRUCTURE:STEP3] Both heuristic extraction methods failed to find reliable patterns")
            
            if message_column:
                print(f"[ANALYZE_LOG_STRUCTURE:LLM_FALLBACK] Message column '{message_column}' available - attempting LLM validation")
                print(f"[ANALYZE_LOG_STRUCTURE:REASONING] Message column enables LLM processing even without heuristics")
                print(f"[PHASE:ANALYZE_LOG_STRUCTURE] ✓ Route selection completed - LLM+ML pipeline")
                
                # Update log structure with detection approach
                agent.update_log_structure_with_detection_approach(schema_id, "hybrid_ml_heuristic")
                
                return {
                    **state, 
                    "log_structure": log_structure, 
                    "skip_llm_validation": False,  # Use LLM since message column exists
                    "skip_ml_detection": False,    # Also use ML detection
                    "use_heuristic": False         # No heuristics available
                }
            else:
                print(f"[ANALYZE_LOG_STRUCTURE:DECISION] No pattern-based detection possible - routing to ML-only detection")
                print(f"[ANALYZE_LOG_STRUCTURE:REASONING] {textual_ratio:.1%} textual data insufficient for pattern matching")
                print(f"[PHASE:ANALYZE_LOG_STRUCTURE] ✓ Route selection completed - ML-ONLY pipeline")
                
                # Update log structure with detection approach
                agent.update_log_structure_with_detection_approach(schema_id, "pure_ml")
                
                return {
                    **state, 
                    "log_structure": log_structure, 
                    "skip_llm_validation": True,   # Skip LLM - if it can't derive heuristics, it can't label reliably
                    "skip_ml_detection": False,    # Use ML detection instead
                    "use_heuristic": False         # No heuristics available
                }
        
        else:
            # Neither message column nor sufficient textual content - skip LLM validation entirely
            print(f"[ANALYZE_LOG_STRUCTURE:NUMERIC_DATA] No message column and insufficient textual content ({textual_ratio:.1%}) for LLM processing")
            print(f"[ANALYZE_LOG_STRUCTURE:DECISION] No log level, no message column, <50% textual - routing directly to ML detection")
            print(f"[ANALYZE_LOG_STRUCTURE:REASONING] LLM requires either message column or ≥50% textual content for effective processing")
            print(f"[PHASE:ANALYZE_LOG_STRUCTURE] ✓ Route selection completed - ML-ONLY pipeline")
            
            return {
                **state, 
                "log_structure": log_structure, 
                "skip_llm_validation": True,  # Skip LLM - insufficient textual data and no message column
                "skip_ml_detection": False,   # We need ML for predominantly numeric data
                "use_heuristic": False        # No heuristics for predominantly numeric
            }

def run_llm_validation_conditional(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """
    Run LLM validation only if needed based on data characteristics.
    
    LLM validation is suitable when:
    - Log level column exists (but already handled by pure heuristic route)
    - Message column exists (enables LLM processing regardless of textual %)
    - ≥50% textual content (traditional textual analysis threshold)
    
    LLM validation is skipped when:
    - Pure heuristic route with comprehensive log-level rules
    - No message column AND <50% textual content (insufficient for LLM)
    """
    print(f"\n[PHASE:4] === LLM VALIDATION CONDITIONAL ===")
    
    # Skip LLM validation if we found good early heuristics OR if data is unsuitable for LLM
    if state.get("skip_llm_validation", False):
        print(f"[LLM_VALIDATION:SKIP] LLM validation bypassed - either pure heuristic route or unsuitable data")
        print(f"[LLM_VALIDATION:REASONING] Either comprehensive log-level rules OR insufficient textual data for LLM")
        print(f"[LLM_VALIDATION:DECISION] Proceeding directly to next phase without LLM intervention")
        print(f"[PHASE:LLM_VALIDATION] ✓ LLM validation phase completed (skipped)")
        return state
    
    # Otherwise run the original LLM validation logic
    schema_id = state["schema_id"]
    hyperparams = agent.hyperparam_memory.get(schema_id)
    
    if hyperparams is None:
        print(f"[LLM_VALIDATION:EXECUTE] Running LLM validation and hyperparameter optimization")
        print(f"[LLM_VALIDATION:REASONING] No cached hyperparameters found - need LLM to validate and optimize")
        validated_df = agent.llm_validate_predictions(state["clean_df"])
        
        # Add this section to tune and save hyperparameters immediately
        agent.tune_isolation_forest_by_llm(validated_df, schema_id)
        print(f"[LLM_VALIDATION:SUCCESS] Validation completed and hyperparameters cached")
        print(f"[PHASE:LLM_VALIDATION] ✓ LLM validation phase completed")
        
        return {**state, "validated_df": validated_df}
    
    print(f"[LLM_VALIDATION:CACHE] Using cached hyperparameters from previous runs")
    print(f"[LLM_VALIDATION:REASONING] Hyperparameters already optimized for this data schema")
    print(f"[PHASE:LLM_VALIDATION] ✓ LLM validation phase completed (from cache)")
    return state

def detect_anomalies(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """Detect anomalies using Isolation Forest (or skip if we have comprehensive heuristics)"""
    schema_id = state["schema_id"]
    
    print(f"\n[PHASE:5] === ANOMALY DETECTION ===")
    
    # Check if we can skip ML detection entirely (for comprehensive log-level heuristics)
    if state.get("skip_ml_detection", False):
        print(f"[DETECT_ANOMALIES:ROUTE] PURE HEURISTIC selected - skipping ML anomaly detection")
        print(f"[DETECT_ANOMALIES:REASONING] Comprehensive log-level rules sufficient for detection")
        print(f"[DETECT_ANOMALIES:DECISION] Using rule-based patterns only, no statistical ML needed")
        
        # Use validated_df if available, otherwise use clean_df
        if state.get("validated_df") is not None:
            input_df = state["validated_df"]
        else:
            input_df = state["clean_df"]
            
        # Create a dummy anomaly_df with just the input data (no ML predictions)
        anomaly_df = input_df.copy()
        # FIXED: Initialize all as normal (0), heuristic will set anomalies to 1
        anomaly_df["anomaly_label"] = 0  # Start with all normal, heuristic will override to anomaly (1) where needed
        
        print(f"[PHASE:DETECT_ANOMALIES] ✓ Anomaly detection phase completed (ML skipped)")
        return {**state, "anomaly_df": anomaly_df}
    
    # Determine which route we're taking based on available heuristics
    heuristics = agent.heuristic_memory.get(schema_id, [])
    has_heuristic = heuristics and isinstance(heuristics, list) and len(heuristics) > 0
    use_heuristic = state.get("use_heuristic", False)
    
    # Display the correct route
    if has_heuristic and use_heuristic:
        print(f"[DETECT_ANOMALIES:ROUTE] HYBRID selected - combining ML detection with derived heuristics")
        print(f"[DETECT_ANOMALIES:REASONING] Both statistical patterns (ML) and rule patterns available")
        print(f"[DETECT_ANOMALIES:APPROACH] Isolation Forest + rule-based pattern integration")
    else:
        print(f"[DETECT_ANOMALIES:ROUTE] ML-ONLY selected - pure machine learning anomaly detection")
        print(f"[DETECT_ANOMALIES:REASONING] No reliable heuristic patterns found, using statistical approach")
        print(f"[DETECT_ANOMALIES:APPROACH] Isolation Forest with dynamic threshold adjustment")
    
    print(f"[DETECT_ANOMALIES:MODEL] Isolation Forest algorithm with optimized hyperparameters")
    
    # Use validated_df if available, otherwise use clean_df
    if state.get("validated_df") is not None:
        input_df = state["validated_df"]
        print(f"[DETECT_ANOMALIES:INPUT] Using LLM-validated dataframe for ML processing")
    else:
        input_df = state["clean_df"]
        print(f"[DETECT_ANOMALIES:INPUT] Using clean dataframe for ML processing")
        
    if input_df is None:
        raise ValueError("No dataframe available for anomaly detection")

    print(f"[DETECT_ANOMALIES:EXECUTE] Running Isolation Forest on {len(input_df)} records")
    anomaly_df = agent.detect_anomalies_pure_and_hybrid(input_df, schema_id, label_col="y")
    print(f"[DETECT_ANOMALIES:SUCCESS] Anomaly detection completed with predictions generated")
    print(f"[PHASE:DETECT_ANOMALIES] ✓ Anomaly detection phase completed")
    return {**state, "anomaly_df": anomaly_df}

def apply_heuristic(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """Apply heuristic rule if available and enabled"""
    
    print(f"\n[PHASE:6] === APPLY HEURISTIC ===")
    
    # Skip if we're not supposed to use heuristics
    if not state.get("use_heuristic", False):
        print(f"[APPLY_HEURISTIC:ROUTE] ML-ONLY confirmed - no heuristic patterns to apply")
        print(f"[APPLY_HEURISTIC:DECISION] Using pure ML results without rule-based modifications")
        print(f"[PHASE:APPLY_HEURISTIC] ✓ Heuristic application phase completed (skipped)")
        return state
    
    # NEW: Check if this is a hybrid route that already applied heuristics in ML stage
    ml_ran = not state.get("skip_ml_detection", False)
    if ml_ran and state.get("use_heuristic", False):
        print(f"[APPLY_HEURISTIC:ROUTE] HYBRID confirmed - heuristics already integrated with ML")
        print(f"[APPLY_HEURISTIC:REASONING] Heuristic patterns were combined with ML during detection phase")
        print(f"[APPLY_HEURISTIC:DECISION] Skipping standalone heuristic application to prevent double-application")
        print(f"[PHASE:APPLY_HEURISTIC] ✓ Heuristic application phase completed (already applied)")
        return state
    
    print(f"[APPLY_HEURISTIC:ROUTE] PURE HEURISTIC confirmed - applying rule-based patterns only")
    print(f"[APPLY_HEURISTIC:REASONING] No ML was run, using heuristic patterns as sole detection method")
    
    # Get the dataframe to work with
    df = state["anomaly_df"]
    schema_id = state["schema_id"]
    
    if df is None:
        print(f"[APPLY_HEURISTIC:ERROR] No dataframe available for heuristic application")
        print(f"[PHASE:APPLY_HEURISTIC] ✗ Heuristic application phase failed")
        return state
    
    # Check if we have an early heuristic from structure analysis
    heuristic = state.get("early_heuristic")
    
    # If no early heuristic, try to get the best one from memory
    if not heuristic:
        existing_heuristics = agent.heuristic_memory.get(schema_id, [])
        
        if not existing_heuristics:
            print(f"[APPLY_HEURISTIC:ERROR] No heuristics found in memory for schema {schema_id}")
            print(f"[PHASE:APPLY_HEURISTIC] ✗ Heuristic application phase failed")
            return state
        
        # Sort by confidence and get the best one
        sorted_heuristics = sorted(existing_heuristics, key=lambda x: x.get("confidence", 0), reverse=True)
        best_heuristic_entry = sorted_heuristics[0]
        heuristic = best_heuristic_entry.get("rule", {})
        confidence = best_heuristic_entry.get("confidence", 0)
        
        print(f"[APPLY_HEURISTIC:SOURCE] Using best heuristic from memory (confidence: {confidence})")
    else:
        print(f"[APPLY_HEURISTIC:SOURCE] Using early heuristic from structure analysis phase")
    
    # DEFENSIVE CHECK: Handle case where heuristic might be a list instead of dict
    if isinstance(heuristic, list):
        print(f"[APPLY_HEURISTIC:WARNING] Received list instead of single heuristic, using first item")
        if heuristic and isinstance(heuristic[0], dict) and "rule" in heuristic[0]:
            heuristic = heuristic[0]["rule"]
        elif heuristic and isinstance(heuristic[0], dict):
            heuristic = heuristic[0]
        else:
            print(f"[APPLY_HEURISTIC:ERROR] Invalid heuristic format in list")
            print(f"[PHASE:APPLY_HEURISTIC] ✗ Heuristic application phase failed")
            return state

    if not heuristic or not isinstance(heuristic, dict):
        print(f"[APPLY_HEURISTIC:ERROR] No valid heuristic found or wrong type: {type(heuristic)}")
        print(f"[PHASE:APPLY_HEURISTIC] ✗ Heuristic application phase failed")
        return state
    
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Check if this is a compound rule
    if heuristic.get("type") == "compound" and heuristic.get("logic") == "OR":
        print(f"[APPLY_HEURISTIC:TYPE] Applying compound rule with OR logic")
        print(f"[APPLY_HEURISTIC:REASONING] Multiple patterns combined - if any match, record is anomalous")
        
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
                print(f"[APPLY_HEURISTIC:WARNING] Column '{col}' not found in dataframe")
                continue
            
            # Apply the rule condition
            if op == "==":
                rule_matches = (df[col] == val)
            elif op == "!=":
                rule_matches = (df[col] != val)
            elif op == "contains":
                rule_matches = df[col].astype(str).str.contains(str(val), case=False, na=False)
            else:
                print(f"[APPLY_HEURISTIC:WARNING] Unsupported operator '{op}' - skipping rule")
                continue
            
            # OR the results together
            matches = matches | rule_matches
            match_count = rule_matches.sum()
            applied_rules.append(f"{col} {op} '{val}' ({match_count} matches)")
            
        # FIXED: Apply the compound rule results with correct logic
        # If any rule matches → anomaly (1), otherwise → normal (0)
        df["anomaly_label"] = matches.astype(int)  # True becomes 1 (anomaly), False becomes 0 (normal)
        df["heuristic_anomaly"] = matches.astype(int)  # Add heuristic_anomaly column that matches anomaly_label
        
        # Print summary
        total_matches = matches.sum()
        rule_text = " OR ".join(applied_rules)
        print(f"[APPLY_HEURISTIC:EXECUTION] Compound rule applied: {rule_text}")
        print(f"[APPLY_HEURISTIC:RESULT] {total_matches} records matched compound conditions → labeled as anomalous")
        
    else:
        # Handle simple (single) rules
        print(f"[APPLY_HEURISTIC:TYPE] Applying simple rule pattern")
        print(f"[APPLY_HEURISTIC:REASONING] Single condition match determines anomaly classification")
        
        col = heuristic.get("column")
        op = heuristic.get("operator", "==")
        val = heuristic.get("value")
        normal_label = heuristic.get("normal_label", 1)
        anomaly_label = heuristic.get("anomaly_label", 0)
        
        if not col or col not in df.columns:
            print(f"[APPLY_HEURISTIC:ERROR] Column '{col}' not found in dataframe")
            print(f"[PHASE:APPLY_HEURISTIC] ✗ Heuristic application phase failed")
            return state
        
        # Apply the single rule
        if op == "==":
            matches = (df[col] == val)
        elif op == "!=":
            matches = (df[col] != val)
        elif op == "contains":
            matches = df[col].astype(str).str.contains(str(val), case=False, na=False)
        else:
            print(f"[APPLY_HEURISTIC:ERROR] Unsupported operator '{op}'")
            print(f"[PHASE:APPLY_HEURISTIC] ✗ Heuristic application phase failed")
            return state
        
        # FIXED: Apply the rule results with correct logic
        # If rule matches → anomaly (1), otherwise → normal (0)
        df["anomaly_label"] = matches.astype(int)  # True becomes 1 (anomaly), False becomes 0 (normal)
        df["heuristic_anomaly"] = matches.astype(int)  # Add heuristic_anomaly column that matches anomaly_label
        
        # Print summary
        match_count = matches.sum()
        print(f"[APPLY_HEURISTIC:EXECUTION] Simple rule applied: {col} {op} '{val}'")
        print(f"[APPLY_HEURISTIC:RESULT] {match_count} records matched simple condition → labeled as anomalous")
    
    # Calculate final anomaly distribution - FIXED LOGIC
    anomaly_count = (df["anomaly_label"] == 1).sum()  # Count anomalies (label 1)
    normal_count = (df["anomaly_label"] == 0).sum()   # Count normal (label 0)
    total_count = len(df)
    
    print(f"[APPLY_HEURISTIC:SUMMARY] Final classification: {anomaly_count} anomalies, {normal_count} normal ({anomaly_count/total_count:.1%} anomalous)")
    print(f"[PHASE:APPLY_HEURISTIC] ✓ Heuristic application phase completed")
    
    # Update state with the modified dataframe
    state["anomaly_df"] = df
    state["rule_applied"] = True
    
    return state

def evaluate_results(state: AnomalyPipelineState, agent: AgentDetector) -> AnomalyPipelineState:
    """Evaluate and save results"""
    print(f"\n[PHASE:7] === EVALUATE RESULTS ===")
    
    # Check if we've already evaluated results
    if state.get("evaluated") is True:
        print(f"[EVALUATE_RESULTS:SKIP] Results already evaluated - skipping duplicate evaluation")
        print(f"[PHASE:EVALUATE_RESULTS] ✓ Evaluation phase completed (from cache)")
        return state
        
    print(f"[EVALUATE_RESULTS:START] Assessing detection performance and generating final report")
    
    # Visualize the results only if ML was used (anomaly_score exists)
    ml_was_used = not state.get("skip_ml_detection", False)
    has_anomaly_score = "anomaly_score" in state["anomaly_df"].columns
    
    if ml_was_used and has_anomaly_score:
        print(f"[EVALUATE_RESULTS:VISUALIZATION] Generating ML anomaly score plots")
        try:
            agent.plot_isolation_forest_results(state["anomaly_df"])
            print(f"[EVALUATE_RESULTS:SUCCESS] Visualization plots generated successfully")
        except Exception as e:
            print(f"[EVALUATE_RESULTS:WARNING] Visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        if not ml_was_used:
            print(f"[EVALUATE_RESULTS:VISUALIZATION] Pure heuristic route - no ML scores to visualize")
        else:
            print(f"[EVALUATE_RESULTS:VISUALIZATION] No anomaly scores found - visualization requires ML detection")
    
    # Evaluate the results
    llm_col = "llm_validated" if "llm_validated" in state["anomaly_df"].columns else None
    print(f"[EVALUATE_RESULTS:ANALYSIS] Computing detection performance metrics")
    agent.evaluate_and_print(
        state["anomaly_df"],
        ground_truth_col=None,
        ml_pred_col="anomaly_label",
        llm_pred_col=llm_col
    )
    
    # Print summary of detected anomalies - FIXED LOGIC
    anomalies_df = state["anomaly_df"][state["anomaly_df"]["anomaly_label"] == 1]  # Anomalies have label 1
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
    
    print(f"\n[EVALUATE_RESULTS:SUMMARY] Detection completed: {anomaly_count} anomalies in {total_count} records ({anomaly_count/total_count:.1%})")
    print(f"[EVALUATE_RESULTS:STRATEGY] Detection method used: {strategy}")
    
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
            print(f"[EVALUATE_RESULTS:RULE] Applied pattern: If {rule_text} (agreement: {agreement:.2%})")
        else:
            # For simple rules
            col = rule.get("column")
            op = rule.get("operator")
            val = rule.get("value")
            print(f"[EVALUATE_RESULTS:RULE] Applied pattern: If {col} {op} '{val}' (agreement: {agreement:.2%})")
    
    # If we have a small number of anomalies, print them for quick review
    if anomaly_count > 0 and anomaly_count <= 10:
        print(f"\n[EVALUATE_RESULTS:DETAILS] Sample anomalous log entries detected:")
        
        # Look for the most informative column to display
        content_columns = ["Content", "Message", "Description", "Body", "Text", "Log"]
        display_col = next((col for col in content_columns if col in anomalies_df.columns), None)
        
        if display_col:
            for i, (_, row) in enumerate(anomalies_df.iterrows()):
                level = row.get("Level", "N/A")
                content = row[display_col]
                truncated_content = f"{content[:100]}..." if len(content) > 100 else content
                print(f"  {i+1}. [{level}] {truncated_content}")
        else:
            # Just show row numbers if no good content column found
            print(f"  [EVALUATE_RESULTS:DETAILS] Anomalous row indices: {list(anomalies_df.index)}")
    
    # Merge anomaly results back to original dataframe for complete context
    print(f"[EVALUATE_RESULTS:MERGE] Merging anomaly results back to original dataframe for RCA analysis")
    original_df = state["df"].copy()
    anomaly_df = state["anomaly_df"]
    
    # Validate that both dataframes have the same number of rows
    if len(original_df) != len(anomaly_df):
        print(f"[EVALUATE_RESULTS:WARNING] Row count mismatch: original={len(original_df)}, processed={len(anomaly_df)}")
        print(f"[EVALUATE_RESULTS:WARNING] Using processed dataframe as-is for safety")
        complete_anomaly_df = anomaly_df
    else:
        # Create a complete dataframe that has original columns + anomaly results
        # We'll merge on index to preserve the original structure
        anomaly_results_cols = ["anomaly_label"]
        
        # Add heuristic_anomaly column if it exists
        if "heuristic_anomaly" in anomaly_df.columns:
            anomaly_results_cols.append("heuristic_anomaly")
            
        # Add any ML-specific columns that might exist
        ml_cols = [col for col in anomaly_df.columns if col.startswith(('ml_', 'isolation_', 'anomaly_score', 'llm_'))]
        anomaly_results_cols.extend(ml_cols)
        
        # Only merge columns that actually exist in anomaly_df
        existing_result_cols = [col for col in anomaly_results_cols if col in anomaly_df.columns]
        
        if existing_result_cols:
            # Reset index to ensure we can merge properly
            anomaly_results = anomaly_df[existing_result_cols].reset_index(drop=True)
            original_df = original_df.reset_index(drop=True)
            
            # Add the anomaly results to the original dataframe
            for col in existing_result_cols:
                original_df[col] = anomaly_results[col]
            
            print(f"[EVALUATE_RESULTS:MERGE] Added {len(existing_result_cols)} result columns: {existing_result_cols}")
            print(f"[EVALUATE_RESULTS:MERGE] Complete dataframe: {len(original_df.columns)} columns (original + results)")
            print(f"[EVALUATE_RESULTS:MERGE] Original columns preserved for RCA context")
            
            # Update state with the complete dataframe
            complete_anomaly_df = original_df
        else:
            print(f"[EVALUATE_RESULTS:WARNING] No anomaly result columns found to merge")
            complete_anomaly_df = anomaly_df
    
    # Save the complete results (original dataframe + anomaly results)
    print(f"[EVALUATE_RESULTS:SAVE] Saving complete results with anomaly classifications")
    agent.save_results(complete_anomaly_df, filename=f"{state['filename']}_anomalies_validated.csv")
    print(f"[PHASE:EVALUATE_RESULTS] ✓ Evaluation phase completed successfully")
    
    # Mark that evaluation is complete and update state with complete dataframe
    return {**state, "anomaly_df": complete_anomaly_df, "evaluated": True}




# Define the state schema for our LangGraph root cause analysis pipeline
class RCAPipelineState(TypedDict):
    anomaly_df: pd.DataFrame  # DataFrame with detected anomalies
    original_df: Optional[pd.DataFrame]  # Original DataFrame with all data including normal records
    rca_results: Optional[Dict[str, Any]]  # Results of the root cause analysis
    filename: str  # Current filename for results
    evaluated: bool  # Whether results have been evaluated
    rca_summary: Optional[str]  # Summary of the RCA findings
    schema_id: Optional[str]  # Schema ID from anomaly detection for memory lookup


# LangGraph node functions - these ones are for the agent that takes the errors and performs root cause analysis
def load_results(state: RCAPipelineState, agent: AgentRCA) -> RCAPipelineState:
    """Load the anomaly results for root cause analysis"""
    print(f"[RCA:LOAD] Loading anomaly results for RCA from {state['filename']}")
    file_path = f"results/{state['filename']}_anomalies_validated_errors.csv"
    
    # load error dataset
    try:
        df = agent.load_data(f"{state['filename']}_anomalies_validated_errors.csv")
    except FileNotFoundError:
        print(f"[RCA:ERROR] File {file_path} not found. Please check the path and filename.")
        return state

    # load original dataset
    try:
        original_df = agent.load_data(f"{state['filename']}_anomalies_validated.csv")
    except FileNotFoundError:
        print(f"[RCA:ERROR] Original file {state['filename']}_anomalies_validated.csv not found.")
        return state
    
    print(f"[RCA:LOAD] Successfully loaded anomaly results")
    print("\n\n\n")
    return {**state, "anomaly_df": df, "original_df": original_df}

def run_rca_analysis(state: RCAPipelineState, agent: AgentRCA) -> RCAPipelineState:
    """Run root cause analysis on the detected anomalies"""
    anomaly_df = state["anomaly_df"]
    original_df = state["original_df"]

    # print head of original
    print(f"[RCA:STRUCTURED] Original data head:\n{anomaly_df.head()}")

    schema_id = state.get("schema_id")  # Get schema_id from state
    
    if anomaly_df is None or len(anomaly_df) == 0:
        print(f"[RCA:SKIP] No data available for root cause analysis")
        return {**state, "rca_results": {}, "rca_summary": "No data to analyze"}


    print(f"[RCA:ANALYZE] Running root cause analysis on {len(anomaly_df)} anomalies")
    
    # Run the analysis with schema_id for memory lookup
    rca_results = agent.analyze_errors(anomaly_df, original_df, schema_id)
    
    # Create a summary
    summary = f"Analyzed {len(anomaly_df)} anomalies out of {len(anomaly_df)} total records"
    
    print(f"[RCA:COMPLETE] Root cause analysis completed")
    print("\n\n\n")
    
    return {**state, "rca_results": rca_results, "rca_summary": summary, "evaluated": True}






# Create a LangGraph workflow for the anomaly detection pipeline
# This pipeline intelligently routes based on data characteristics and available heuristics
def create_anomaly_pipeline(agent: AgentDetector):
    # Create the graph
    workflow = StateGraph(AnomalyPipelineState)
    
    # Add nodes with clear descriptions - intelligent 4-route pipeline
    print(f"[WORKFLOW:CREATE] Setting up intelligent anomaly detection pipeline with 7 stages")
    print(f"[STRATEGY:ADAPTIVE] Pipeline automatically selects optimal detection approach:")
    print(f"  Route 1: Pure Heuristic (Log-Level Rules) - Log level column found")
    print(f"  Route 2: LLM+ML (Hybrid) - Message column OR ≥50% textual content available")
    print(f"  Route 3: Hybrid (ML + Derived Heuristic) - Heuristic successfully derived from patterns")
    print(f"  Route 4: ML-Only (Unsupervised) - No log level, no message column, <50% textual")
    
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
    print(f"  Stage 3 (analyze_log_structure) determines which route to take:")
    print(f"    - Log level column → Pure Heuristic route")
    print(f"    - Message column OR ≥50% textual → LLM processing enabled")
    print(f"    - Derive heuristic success → Hybrid route")
    print(f"    - Otherwise → ML-Only route")
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
        filename="parsed_apache_logs-short-1",
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
        original_df=None,
        rca_results=None,
        filename=initial_state["filename"],  # Use same filename
        evaluated=False,
        rca_summary=None,
        schema_id=anomaly_result.get("schema_id")  # Pass schema_id from anomaly detection
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