import pandas as pd
import json
import os
import math
from typing import Optional, Dict, Any, List, Callable, TypedDict
import hashlib
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
import boto3
from botocore.exceptions import ClientError

# Load environment variables from .env file
load_dotenv()

class AgentDetector:
    def __init__(self, model="anthropic.claude-3-sonnet-20240229-v1:0", heuristic_model=None, region="eu-west-2"):
        try:
            base_path = os.path.dirname(__file__)
        except NameError:
            base_path = os.getcwd()
            
        self.preprocess_memory_path = os.path.join(base_path, "memory", "preprocess_memory.json")
        self.hyperparam_memory_path = os.path.join(base_path, "memory", "hyperparam_memory.json")
        self.heuristic_memory_path = os.path.join(base_path, "memory", "heuristic_memory.json")
        self.log_structure_memory_path = os.path.join(base_path, "memory", "log_structure_memory.json")
        
        # AWS Bedrock setup
        self.region = region
        self.model = model
        self.heuristic_model = heuristic_model or model
        
        # Initialize Bedrock client
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            print(f"[agent] AWS Bedrock client initialized with model: {self.model}")
        except Exception as e:
            print(f"[agent] Error initializing Bedrock client: {e}")
            raise
        
        # Load memory (same as before)
        self.preprocess_memory = self._load_memory(self.preprocess_memory_path)
        self.hyperparam_memory = self._load_memory(self.hyperparam_memory_path)
        self.heuristic_memory = self._load_memory(self.heuristic_memory_path)
        self.log_structure_memory = self._load_memory(self.log_structure_memory_path)

    def _call_bedrock_model(self, prompt, system_message=None, max_tokens=4000, temperature=0):
        """
        Helper method to call AWS Bedrock Claude 3 Sonnet with consistent interface
        """
        try:
            # Claude 3 Sonnet format
            messages = []
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
            if system_message:
                body["system"] = system_message
            
            # Make the API call
            response = self.bedrock_client.invoke_model(
                modelId=self.model,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            # Extract text from Claude response
            return response_body['content'][0]['text']
                
        except ClientError as e:
            print(f"[BEDROCK:ERROR] AWS Bedrock API error: {e}")
            raise
        except Exception as e:
            print(f"[BEDROCK:ERROR] Unexpected error calling Bedrock: {e}")
            raise

    def _count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def _load_memory(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}

    def _save_memory(self, memory: dict, memory_type: str):
        if memory_type == "preprocess":
            self.preprocess_memory = memory
            path = self.preprocess_memory_path
        elif memory_type == "hyperparam":
            self.hyperparam_memory = memory
            path = self.hyperparam_memory_path
        elif memory_type == "heuristic":
            self.heuristic_memory = memory
            path = self.heuristic_memory_path
        elif memory_type == "log_structure":
            self.log_structure_memory = memory
            path = self.log_structure_memory_path
        else:
            raise ValueError("Invalid memory type.")

        with open(path, 'w') as f:
            json.dump(memory, f, indent=2)

    def _get_confidence_increment(self, rule):
        """
        Get the confidence increment for a rule based on its type.
        Compound rules get a 1.5x multiplier (rounded up) to encourage their use.
        """
        if rule.get("type") == "compound":
            # Compound rules get 3 points for every 2 that normal rules get
            return 2  # This gives compound rules a 2x boost
        else:
            return 1  # Normal increment for regular rules
    
    def _get_schema_id(self, columns: list[str]) -> str:
        joined = "|".join(columns)
        return hashlib.sha256(joined.encode()).hexdigest()

    def _add_or_increment_heuristic(self, schema_id: str, new_rule: dict, confidence_increment: int = 1):
        """
        Add a new heuristic or increment confidence of existing identical heuristic.
        Returns True if rule was added/updated, False if it was a duplicate.
        """
        existing_heuristics = self.heuristic_memory.get(schema_id, [])
        
        # Check for duplicates by comparing key fields
        for i, entry in enumerate(existing_heuristics):
            existing_rule = entry.get("rule", {})
            
            # Compare the essential fields that make a rule unique
            if self._rules_are_identical(existing_rule, new_rule):
                # Found duplicate - increment confidence
                old_confidence = entry.get("confidence", 1)
                new_confidence = old_confidence + confidence_increment
                existing_heuristics[i]["confidence"] = new_confidence
                
                print(f"[HEURISTIC:DUPLICATE] Found identical rule - incrementing confidence: {old_confidence} → {new_confidence}")
                print(f"[HEURISTIC:DUPLICATE] Rule: {existing_rule.get('column')} {existing_rule.get('operator')} '{existing_rule.get('value')}'")
                
                # Save updated memory
                self.heuristic_memory[schema_id] = existing_heuristics
                self._save_memory(self.heuristic_memory, "heuristic")
                print(f"[HEURISTIC:UPDATED] Updated existing rule with new confidence {new_confidence}")
                return True
        
        # No duplicate found - add as new rule
        new_entry = {
            "rule": new_rule,
            "confidence": confidence_increment
        }
        print(f"[HEURISTIC:NEW] Adding new rule")
        existing_heuristics.append(new_entry)
        self.heuristic_memory[schema_id] = existing_heuristics
        self._save_memory(self.heuristic_memory, "heuristic")
        
        return False

    def _rules_are_identical(self, rule1: dict, rule2: dict) -> bool:
        """
        Check if two heuristic rules are functionally identical.
        Compares the essential fields that define rule behavior.
        """
        # Essential fields that make a rule unique
        essential_fields = ['column', 'operator', 'value', 'normal_label', 'anomaly_label']
        
        # For compound rules, also compare type and logic
        if rule1.get('type') == 'compound' or rule2.get('type') == 'compound':
            if rule1.get('type') != rule2.get('type'):
                return False
            if rule1.get('logic') != rule2.get('logic'):
                return False
            # Compare sub-rules for compound rules
            rules1 = rule1.get('rules', [])
            rules2 = rule2.get('rules', [])
            if len(rules1) != len(rules2):
                return False
            # Check if all sub-rules match (order matters for now)
            for r1, r2 in zip(rules1, rules2):
                if not all(r1.get(field) == r2.get(field) for field in ['column', 'operator', 'value']):
                    return False
            return True
        
        # For simple rules, compare essential fields
        return all(rule1.get(field) == rule2.get(field) for field in essential_fields)

    def _apply_rules(self, df: pd.DataFrame, rules: dict) -> pd.DataFrame:
        df = df.copy()

        if "drop_columns" in rules:
            df.drop(columns=[col for col in rules["drop_columns"] if col in df.columns], inplace=True)

        if "fillna" in rules:
            for col, method in rules["fillna"].items():
                if col not in df.columns:
                    continue
                
                # Handle null/None values properly
                if method is None or method == "null":
                    # For None/null values, skip filling or use appropriate defaults
                    if df[col].dtype == 'object':
                        # For text columns, we can leave NaN as is or use empty string
                        continue  # Skip filling - leave NaN values as they are
                    else:
                        # For numeric columns, use 0 as default
                        df[col] = df[col].fillna(0)
                elif method == "mean":
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        # For non-numeric columns, skip mean calculation
                        continue
                elif method == "ffill":
                    df[col] = df[col].fillna(method="ffill")
                elif method == "bfill":
                    df[col] = df[col].fillna(method="bfill")
                else:
                    # For any other value, try to use it directly
                    try:
                        # Try to convert to appropriate type first
                        if df[col].dtype in ['int64', 'float64']:
                            fill_value = float(method) if method != "" else 0
                        else:
                            fill_value = str(method) if method != "" else ""
                        
                        df[col] = df[col].fillna(fill_value)
                    except (ValueError, TypeError):
                        # If conversion fails, use the value as-is (for strings)
                        df[col] = df[col].fillna(method)

        return df

    @staticmethod
    def safe_json_parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decoding failed:\n{e}")
            print(f"=== Raw response ===\n{text}")
            # Optionally, try to "repair" it (very basic way)
            try:
                repaired = text.strip()
                if repaired.endswith(','):
                    repaired = repaired[:-1]
                if not repaired.endswith('}'):
                    repaired += '}'
                return json.loads(repaired)
            except Exception:
                return None
            
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Check if the dataframe has been processes before, 
        if not then provide LLM with columns and 5 row sample
        to generate preprocess rules like columns to keep or 
        drop, and fillna values.
        """
        schema_id = self._get_schema_id(list(df.columns))

        # print to say the schema id this agent is processing
        print(f"[agent] Processing schema_id: {schema_id}")

        if schema_id in self.preprocess_memory:
            print(f"[agent] Using cached preprocessing rules")
            return self._apply_rules(df, self.preprocess_memory[schema_id]["rules"]), schema_id

        print(f"[agent] No rules for schema_id: {schema_id}. Querying LLM...")

        sample = df.sample(min(len(df), 100)).to_dict(orient="records")
        schema = list(df.columns)

        prompt = (
            f"[[NO PROSE]][[JSON ONLY]]You are a data scientist. Here's a sample of the dataset (comma-separated):\n"
            f"{sample[:5]}\n\n"
            f"Column names: {schema}\n\n"
            f"Return JSON ONLY in this exact format:\n"
            f'{{"{schema_id}": {{"rules": {{"keep_columns": [...], "drop_columns": [...], "fillna": {{"col1": "value", "col2": "value", ...}}}}}}}}\n\n'
            f"- Keep only meaningful data columns.\n"
            f"- Drop index or metadata columns like 'LineId', 'Index', 'RowID', etc.\n"
            f"- Also drop time columns like 'Time' and 'Date\n"
            f"- For each column you KEEP, provide a default fillna value.\n"
            f"- Fill values must be strings, numbers, or nulls, depending on the column type.\n"
            f"- Do NOT add explanations or markdown. Valid JSON only."
        )

        system_msg = (
            "You are a JSON-only assistant. You must respond ONLY with parsable JSON.\n"
            "No text, no markdown, no explanations. No leading/trailing text or commas.\n"
            "All strings must use double quotes. Your JSON must follow this schema exactly:\n"
            '{ "SCHEMA_ID": { "rules": { "keep_columns": [...], "drop_columns": [...], "fillna": { ... } } } }\n'
            "Ensure columns like 'LineId', 'RowID', 'Index', etc., are dropped. Only return relevant data fields."
        )

        raw = self._call_bedrock_model(
            prompt=prompt,
            system_message=system_msg,
            max_tokens=2000,
            temperature=0
        )

        match = re.search(r'\{[\s\S]*?\}', raw)
        if not match:
            raise ValueError("No JSON object found in LLM response.")

        json_str = match.group(0).strip()

        # Auto-fix common LLM bug: unbalanced braces
        if json_str.count('{') > json_str.count('}'):
            json_str += "}" * (json_str.count('{') - json_str.count('}'))

        parsed = self.safe_json_parse(json_str)
        if parsed is None:
            raise ValueError("LLM returned invalid JSON.")

        rules = parsed.get(schema_id, {}).get("rules", {})
        self.preprocess_memory[schema_id] = {"rules": rules}
        self._save_memory(self.preprocess_memory, "preprocess")
        print(f"[agent] Preprocessing rules saved for schema_id: {schema_id}")

        return self._apply_rules(df, rules), schema_id

    def identify_log_structure(self, df, schema_id):
        """
        Ask LLM to analyze the dataset structure and identify key columns.
        Uses memory to avoid re-analyzing the same schema.
        """
        existed = False
        has_new_log_values = False
        
        # Check if we already have log structure for this schema
        if schema_id in self.log_structure_memory:
            existed = True
            cached_structure = self.log_structure_memory[schema_id]
            print(f"[STRUCTURE] Using cached analysis")
            
            # Update log level values if new ones are found
            log_level_column = cached_structure.get("log_level_column")
            if log_level_column and log_level_column in df.columns:
                cached_values = set(cached_structure.get("log_level_values", []))
                current_values = set(df[log_level_column].unique().tolist())
                
                if current_values != cached_values:
                    # Add any new values to the existing list
                    all_values = list(cached_values.union(current_values))
                    cached_structure["log_level_values"] = all_values
                    self.log_structure_memory[schema_id] = cached_structure
                    self._save_memory(self.log_structure_memory, "log_structure")
                    print(f"[STRUCTURE] Found new log level values ({len(all_values)} total)")
                    has_new_log_values = True
                    
            # Add flag to indicate if new log values were found
            cached_structure["has_new_log_values"] = has_new_log_values

            return cached_structure, existed
        
        # Sample a few rows for LLM analysis
        sample_rows = df.head(5).to_dict(orient='records')
        columns = df.columns.tolist()

        print(f"[ANALYZE] Analyzing log structure for {schema_id}")
        
        prompt = f"""Analyze this log data structure with the following columns: {columns}
        Sample log entries:
        {json.dumps(sample_rows, indent=2, default=str)}

        Please identify the following (respond with JSON only):
        1. Which column (if any) contains log levels like INFO, WARN, ERROR, DEBUG?
        2. Which column (if any) contains timestamp information?
        3. Which column (if any) contains line ID / structural information like LineID, ID etc?
        4. Which column (if any) contains the main message or content of the log?
        5. Which columns (if any) contain numerical metrics that might indicate anomalies?
        6. Are there any columns that appear to contain error codes or status information?

        Format your response strictly as JSON with the following structure:
        {{
        "log_level_column": "column_name or null if none exists",
        "timestamp_column": "column_name or null if none exists",
        "lineID_column": "column_name or null if none exists",
        "message_column": "column_name or null if none exists",
        "numerical_metric_columns": ["column1", "column2", ...],
        "error_status_columns": ["column1", "column2", ...]
        }}
        """
        
        system_message = "You are a log analysis expert. Respond with valid JSON only."


        # CHANGED: Query Claude 3 Sonnet via Bedrock instead of OpenAI
        try:
            response = self._call_bedrock_model(
                prompt=prompt,
                system_message=system_message,
                max_tokens=1000,
                temperature=0
            )
                        
            # Clean the response to extract JSON
            response = response.strip()
            
            # Remove any markdown code blocks if present
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse the JSON response
            structure = json.loads(response)
            
            # Add current log level values to the structure
            log_level_column = structure.get("log_level_column")
            if log_level_column and log_level_column in df.columns:
                structure["log_level_values"] = df[log_level_column].unique().tolist()
            
            # Save to memory
            self.log_structure_memory[schema_id] = structure
            self._save_memory(self.log_structure_memory, "log_structure")
            
            print(f"[BEDROCK:SUCCESS] Successfully analyzed log structure using Claude 3 Sonnet")
            return structure, existed
            
        except json.JSONDecodeError as e:
            print(f"[ANALYZE:ERROR] Failed to parse JSON response: {str(e)}")
            print(f"[ANALYZE:ERROR] Raw response: {response}")
            # Return a default structure that won't block processing
            return {
                "log_level_column": None,
                "timestamp_column": None,
                "message_column": None,
                "numerical_metric_columns": [],
                "error_status_columns": []
            }, existed
        except Exception as e:
            print(f"[ANALYZE:ERROR] Failed to analyze log structure: {str(e)}")
            # Return a default structure that won't block processing
            return {
                "log_level_column": None,
                "timestamp_column": None,
                "message_column": None,
                "numerical_metric_columns": [],
                "error_status_columns": []
            }, existed

    def update_log_structure_with_detection_approach(self, schema_id: str, detection_approach: str):
        """
        Update the log structure memory with the detection approach used.
        This ensures RCA agent knows exactly which method was used for detection.
        
        Args:
            schema_id: Schema identifier
            detection_approach: One of 'pure_heuristic', 'hybrid_ml_heuristic', 'pure_ml'
        """
        if schema_id in self.log_structure_memory:
            self.log_structure_memory[schema_id]["detection_approach"] = detection_approach
            self._save_memory(self.log_structure_memory, "log_structure")
            print(f"[LOG_STRUCTURE:APPROACH] Updated detection approach for {schema_id}: {detection_approach}")
        else:
            print(f"[LOG_STRUCTURE:WARNING] No log structure found for schema {schema_id} - cannot update detection approach")

    def generate_comprehensive_log_level_heuristics(self, df, log_structure, schema_id=None):
        """
        Create a comprehensive compound heuristic from ALL unique values in the log level column.
        This is called the first time we see data with a log level column OR when new values are found.
        """
        log_level_column = log_structure.get("log_level_column")
        if not log_level_column or log_level_column not in df.columns:
            print(f"[LOG_LEVEL] No log level column identified")
            return []

        # Get unique values in the log level column
        unique_values = df[log_level_column].unique().tolist()
        stored_values = log_structure.get("log_level_values", [])
        
        print(f"[LOG_LEVEL] Current dataset has {len(unique_values)} unique log level values: {unique_values}")
        print(f"[LOG_LEVEL] Stored values from memory: {stored_values}")
        
        # Use all values from memory (which includes both old and new values)
        all_values = list(set(stored_values))  # Remove duplicates while preserving order
        
        # Check if we have an existing comprehensive heuristic for this schema
        existing_comprehensive = None
        if schema_id and schema_id in self.heuristic_memory:
            existing_heuristics = self.heuristic_memory[schema_id]
            for heuristic_entry in existing_heuristics:
                rule = heuristic_entry.get("rule", {})
                if rule.get("comprehensive", False):
                    existing_comprehensive = rule
                    print(f"[LOG_LEVEL] Found existing comprehensive rule with {len(rule.get('anomalous_values', []))} anomalous values")
                    break

        # If we have new values and an existing comprehensive rule, merge the classifications
        new_values_only = []
        if existing_comprehensive:
            known_anomalous = set(existing_comprehensive.get("anomalous_values", []))
            known_normal = set(existing_comprehensive.get("normal_values", []))
            known_all = known_anomalous.union(known_normal)
            
            new_values_only = [v for v in all_values if v not in known_all]
            if new_values_only:
                print(f"[LOG_LEVEL] Found {len(new_values_only)} NEW values that need classification: {new_values_only}")
            else:
                print(f"[LOG_LEVEL] No new values to classify - using existing comprehensive rule")
                return [existing_comprehensive]

        # Determine which values need LLM classification
        values_to_classify = new_values_only if new_values_only else all_values
        
        print(f"[LOG_LEVEL] Classifying {len(values_to_classify)} values: {values_to_classify}")

        prompt = f"""You are analyzing log data where the column '{log_level_column}' contains log severity levels.
        
        The values that need classification are: {values_to_classify}
        
        Please classify EVERY value as either:
        - ANOMALOUS: Indicates something went wrong (errors, failures, critical issues)
        - NORMAL: Indicates normal operation (info, debug, trace messages, warnings)
        
        Consider common log level conventions:
        - ERROR, FATAL, CRITICAL typically indicate problems or potential issues
        - INFO, DEBUG, TRACE, WARN, WARNING typically indicate normal operation
        
        Important: 
        1. WARN/WARNING should be classified as NORMAL since warnings are not errors
        2. ALL values must be classified (either anomalous or normal)
        3. Only classify the specific values provided above
        
        Respond with JSON only in this format:
        {{
            "anomalous_values": ["value1", "value2", ...],
            "normal_values": ["value1", "value2", ...],
            "confidence": "high|medium|low - your confidence in this classification"
        }}
        """

        system_message = "You are a log analysis expert. Respond with valid JSON only. Classify ALL provided log level values."


        try:
            # CHANGED: Use Bedrock instead of OpenAI
            response = self._call_bedrock_model(
                prompt=prompt,
                system_message=system_message,
                max_tokens=1000,
                temperature=0
            )
            
            # Clean the response to extract JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            classification = json.loads(response)

            # Get new classifications from LLM
            new_anomalous = classification.get("anomalous_values", [])
            new_normal = classification.get("normal_values", [])
            
            print(f"[LOG_LEVEL] LLM classified - Anomalous: {new_anomalous}, Normal: {new_normal}")
            
            # Merge with existing classifications if we have them
            if existing_comprehensive:
                # Combine old and new classifications
                all_anomalous = list(set(existing_comprehensive.get("anomalous_values", []) + new_anomalous))
                all_normal = list(set(existing_comprehensive.get("normal_values", []) + new_normal))
                print(f"[LOG_LEVEL] Merged classifications - Total Anomalous: {all_anomalous}, Total Normal: {all_normal}")
            else:
                # First time - use classifications as-is
                all_anomalous = new_anomalous
                all_normal = new_normal
            
            # Verify all values are classified
            all_classified = set(all_anomalous + all_normal)
            all_unique = set(all_values)
            if all_classified != all_unique:
                missing = all_unique - all_classified
                print(f"[LOG_LEVEL] Warning: unclassified values {missing} - adding to normal")
                # Add missing values to normal by default
                all_normal.extend(list(missing))
            
            # Create comprehensive compound rule that handles all log levels
            if all_anomalous:
                comprehensive_rule = {
                    "type": "compound",
                    "logic": "OR",
                    "rules": [],
                    "normal_label": 0,     # FIXED: Normal records get 0
                    "anomaly_label": 1,    # FIXED: Anomaly records get 1 
                    "source": "comprehensive_log_level_analysis",
                    "confidence_boost": 0.95 if classification.get("confidence") == "high" else 0.85,
                    "comprehensive": True,  # Mark this as a comprehensive rule
                    "all_values": all_values,  # Store all possible values
                    "anomalous_values": all_anomalous,
                    "normal_values": all_normal
                }
                
                for value in all_anomalous:
                    comprehensive_rule["rules"].append({
                        "column": log_level_column,
                        "operator": "==", 
                        "value": value
                    })
                    
                if existing_comprehensive and new_values_only:
                    print(f"[LOG_LEVEL] Updated comprehensive rule: added {len(new_anomalous)} new anomalous values")
                    print(f"[LOG_LEVEL] Total rule now covers {len(all_anomalous)} anomalous values: {all_anomalous}")
                else:
                    print(f"[LOG_LEVEL] Created comprehensive rule with {len(all_anomalous)} anomalous values")
                    
                return [comprehensive_rule]
            else:
                print(f"[LOG_LEVEL] Warning: no anomalous values identified")
                return []
            
        except Exception as e:
            print(f"[LOG_LEVEL] Error generating heuristics: {str(e)}")
            return []
        
    def derive_heuristics_from_anomaly_scores(self, df, schema_id, sample_size=10):
        """
        When no log level column is found, run anomaly detection and use extreme samples
        to derive heuristics via LLM analysis.
        """
        print(f"[HEURISTIC:ANOMALY_BASED] No log level column - deriving heuristics from anomaly patterns")
        
        try:
            # Run quick anomaly detection to get scores
            from sklearn.ensemble import IsolationForest
            
            # Prepare features more carefully
            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            num_cols = df.select_dtypes(include=["number"]).columns
            
            #print(f"[HEURISTIC:DEBUG] Found {len(cat_cols)} categorical and {len(num_cols)} numeric columns")
            
            # Build feature matrix more carefully
            feature_dfs = []
            
            # Add numeric columns if any exist
            if len(num_cols) > 0:
                feature_dfs.append(df[num_cols])
                #print(f"[HEURISTIC:DEBUG] Added {len(num_cols)} numeric columns")
            
            # Add encoded categorical columns if any exist
            if len(cat_cols) > 0:
                df_encoded = pd.get_dummies(df[cat_cols])
                if not df_encoded.empty:
                    feature_dfs.append(df_encoded)
                    #print(f"[HEURISTIC:DEBUG] Added {len(df_encoded.columns)} encoded categorical columns")
            
            # Check if we have any features to work with
            if not feature_dfs:
                #print(f"[HEURISTIC:ERROR] No features available for anomaly detection")
                return []
            
            # Concatenate all feature DataFrames
            X = pd.concat(feature_dfs, axis=1)
            
            if X.empty:
                print(f"[HEURISTIC:ERROR] Feature matrix is empty after concatenation")
                return []
            
            #print(f"[HEURISTIC:DEBUG] Final feature matrix: {X.shape}")
            
            # Quick ML run to get anomaly scores
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X)
            anomaly_scores = model.decision_function(X)
            
            # Add scores to dataframe
            df_with_scores = df.copy()
            df_with_scores['anomaly_score'] = anomaly_scores
            df_with_scores['anomaly_label'] = (model.predict(X) == -1).astype(int)  # 1=anomaly, 0=normal
            
            # Get 10 most anomalous (lowest scores) and 10 least anomalous (highest scores)
            most_anomalous = df_with_scores.nsmallest(sample_size, 'anomaly_score')
            least_anomalous = df_with_scores.nlargest(sample_size, 'anomaly_score')
            
            print(f"[HEURISTIC:SAMPLES] Selected {len(most_anomalous)} most anomalous and {len(least_anomalous)} least anomalous samples")
            
            # Combine samples and prepare for LLM
            sample_df = pd.concat([most_anomalous, least_anomalous])
            
            # Create labels: most anomalous = 0 (anomaly), least anomalous = 1 (normal)
            sample_df.loc[most_anomalous.index, 'label'] = 0
            sample_df.loc[least_anomalous.index, 'label'] = 1
            
            # Clean up sample data for LLM (exclude score/label columns)
            cols_to_exclude = {'anomaly_score', 'anomaly_label'}
            cols_to_include = [col for col in sample_df.columns if col not in cols_to_exclude]
            llm_sample_df = sample_df[cols_to_include]
            
            # Convert to CSV for LLM
            csv_text = llm_sample_df.to_csv(index=False)
            
            # Create LLM prompt - EMPHASIZE ANOMALY DETECTION
            system_msg = (
                "You are an expert log analysis system. Based on extreme anomaly examples, "
                "identify patterns that distinguish anomalous logs from normal logs. "
                "CRITICAL: ONLY derive rules that identify ERRORS/ANOMALIES, never rules that identify normal logs. "
                "Respond with VALID JSON ONLY containing the most reliable anomaly-detecting rule."
            )
            
            prompt = (
                "I have log entries with the most anomalous (label=0) and least anomalous (label=1) examples "
                "based on machine learning anomaly detection.\n\n"
                "TASK: Find the single most reliable pattern that identifies ANOMALIES/ERRORS (label=0).\n\n"
                "CRITICAL REQUIREMENTS:\n"
                "- ONLY create rules that identify anomalies/errors (anomaly_label=1)\n"
                "- NEVER create rules that identify normal logs (normal_label=1)\n"
                "- If you can't find an anomaly-detecting pattern, respond with 'error':'no_pattern_found'\n\n"
                "ANALYSIS PRIORITY (focus on anomaly indicators):\n"
                "1. ERROR KEYWORDS: 'error', 'exception', 'fail', 'failed', 'timeout', 'abort', 'critical'\n"
                "2. ERROR STATUS CODES: 4xx, 5xx, 'FAILED', 'ERROR', 'TIMEOUT'\n"
                "3. NEGATIVE TERMS: 'rejected', 'invalid', 'not found', 'denied', 'blocked', 'unavailable'\n"
                "4. PERFORMANCE ISSUES: 'slow', 'degraded', 'overload', 'capacity', 'limit exceeded'\n"
                "5. SYSTEM PROBLEMS: 'connection lost', 'service down', 'unreachable', 'crashed'\n\n"
                "OUTPUT FORMAT (JSON ONLY) - MUST identify anomalies:\n"
                '{"column": "Message", "operator": "contains", "value": "ERROR", "normal_label": 0, "anomaly_label": 1}\n'
                "Where:\n"
                "- operator: ==, !=, contains\n"
                "- normal_label: 0 (this rule does NOT identify normal logs)\n"
                "- anomaly_label: 1 (this rule DOES identify anomalies)\n\n"
                '{"column": "Status", "operator": "==", "value": "ERROR", "normal_label": 0, "anomaly_label": 1}\n'
                "Where operator is one of: ==, !=, contains\n"
                "If no anomaly-detecting pattern found, respond: no_pattern_found\n\n"
                f"CSV data:\n{csv_text}\n"
            )
            
            # Get response from LLM
            response = self._call_bedrock_model(
                prompt=prompt,
                system_message=system_msg,
                max_tokens=500,
                temperature=0
            )
            
            if response.lower() in {"none", "no obvious heuristic", "no heuristic found", "no rule", "no_pattern_found"}:
                print("[HEURISTIC:ANOMALY_BASED] LLM couldn't identify a pattern from anomaly scores")
                return []
            
            # Clean up the response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]  # Remove ```json
            if response.startswith("```"):
                response = response[3:]   # Remove ```
            if response.endswith("```"):
                response = response[:-3]  # Remove trailing ```
            response = response.strip()
            
            # Handle special responses
            if "no_pattern_found" in response.lower() or response.lower().strip() == "no_pattern_found":
                print("[HEURISTIC:ANOMALY_BASED] LLM couldn't identify a pattern from anomaly scores")
                return []
            
            # Parse the response as JSON
            try:
                rule = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"[HEURISTIC:ERROR] Failed to parse LLM response as JSON: {e}")
                print(f"[HEURISTIC:ERROR] Raw response: {response}")
                return []

            
            # CRITICAL FILTER: Only accept anomaly-detecting heuristics for hybrid route
            if rule.get("anomaly_label") != 1:
                print(f"[HEURISTIC:FILTER] Rejecting normal-detecting rule: {rule}")
                print(f"[HEURISTIC:FILTER] Hybrid route requires anomaly-detecting heuristics only")
                return []
            
            # Fix any missing fields with defaults (ensuring anomaly detection)
            if "normal_label" not in rule:
                rule["normal_label"] = 0  # This rule does NOT identify normal logs
            if "anomaly_label" not in rule:
                rule["anomaly_label"] = 1  # This rule DOES identify anomalies
            if "operator" not in rule or rule["operator"] not in ["==", "!=", "contains"]:
                rule["operator"] = "=="
            
            # Double-check the labels are correct for anomaly detection
            if rule["anomaly_label"] != 1 or rule["normal_label"] != 0:
                print(f"[HEURISTIC:ERROR] Invalid labels - rule must have anomaly_label=1, normal_label=0")
                return []
                
            rule["source"] = "anomaly_score_analysis"
            
            # Add or increment existing heuristic
            self._add_or_increment_heuristic(schema_id, rule, confidence_increment=1)
            
            # Print the derived rule
            col, op, val = rule["column"], rule["operator"], rule["value"]
            print(f"[HEURISTIC:ANOMALY_BASED] Derived ANOMALY-DETECTING rule: {col} {op} '{val}'")
            
            return [rule]
            
        except Exception as e:
            print(f"[HEURISTIC:ERROR] Failed to derive heuristics from anomaly scores: {str(e)}")
            return []

    def llm_validate_predictions(self, df: pd.DataFrame, batch_size=10, max_retries=3, retry_delay=2) -> pd.DataFrame:
        validated_flags = []

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].copy()
            batch = batch.drop(columns=[col for col in batch.columns if col.startswith("llm_")], errors="ignore")
            columns = batch.columns.tolist()
            csv_text = batch.to_csv(index=False, header=True)

            prompt = (
                "You are a log analysis expert. Below is a CSV with columns: "
                + ", ".join(columns)
                + ".\n"
                "Each line represents a single log entry. Your task is to detect any indication of an error, fault, or abnormal event "
                "that a human operator would flag (e.g., exceptions, failed operations, warnings, stack traces, unusual codes). "
                "Do not rely on column names alone—interpret the content as a normal sysadmin would.\n"
                f"Return EXACTLY {len(batch)} tokens separated by commas, each either “0” (this entry indicates an error/abnormality) "
                "or “1” (this entry appears normal). No JSON, no list brackets, no extra text.\n\n"
                f"{csv_text}"
            )

            tokens = self._count_tokens(prompt)

            for attempt in range(max_retries):
                try:
                    res = self._call_bedrock_model(
                        prompt=prompt,
                        max_tokens=1000,
                        temperature=0
                    )

                    bits = [int(x.strip()) for x in res.split(",") if x.strip() in {"0", "1"}]

                    if len(bits) != len(batch):
                        raise ValueError(f"Expected {len(batch)} bits, got {len(bits)}")

                    validated_flags.extend(bits)
                    break  # Success, exit retry loop

                except Exception as e:
                    print(f"[validate_rows_with_llm] Error (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        validated_flags.extend([-1] * len(batch))

        print(f"[agent] LLM validation completed for {len(validated_flags)} rows")
        df["llm_validated"] = validated_flags
        return df

    def tune_isolation_forest_by_llm(self, df, schema_id, llm_pred_col="llm_validated"):
        param_grid = {
            "contamination": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
            "n_estimators": [10, 25, 50, 100, 200, 500],
            "max_samples": ["auto", 0.5, 0.75, 1.0],
            "max_features": [0.5, 0.75, 1.0],
            "random_state": [42]
        }
        best_agreement = -1
        best_params = None

        # Drop LLM column for features
        X = df.drop(columns=[llm_pred_col])
        # Encode categorical columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        num_cols = X.select_dtypes(include=["number"]).columns
        X_encoded = pd.get_dummies(X[cat_cols])
        X_final = pd.concat([X[num_cols], X_encoded], axis=1)

        y_llm = df[llm_pred_col]

        for params in ParameterGrid(param_grid):
            try:
                model = IsolationForest(**params)
                model.fit(X_final)
                y_pred = (model.predict(X_final) == 1).astype(int)
                agreement = (y_pred == y_llm).mean()
                if agreement > best_agreement:
                    best_agreement = agreement
                    best_params = params
            except Exception as e:
                print(f"[tune_isolation_forest_by_llm] Skipping params {params} due to error: {e}")

        # extarct param variables and print them nicely
        best_params = {k: v for k, v in best_params.items() if k != "random_state"}
        best_params_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
        print(f"[AGENT] Best Isolation Forest parameters: {best_params_str} with agreement {best_agreement:.2f}")

        self.hyperparam_memory[schema_id] = best_params
        self._save_memory(self.hyperparam_memory, "hyperparam")
        print(f"[AGENT] Saved best hyperparameters to memory")



        return best_params

    def get_textual_heuristic(self, df, schema_id):
        """
        Ask LLM to derive a textual heuristic based on the dataframe.
        This is used when no log level column is found and we need to derive a heuristic from the data.
        """
        # Filter to only textual columns to reduce noise
        text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if not text_cols:
            print(f"[TEXTUAL_HEURISTIC] No textual columns found - cannot derive textual heuristic")
            return None
        
        print(f"[TEXTUAL_HEURISTIC] Analyzing {len(text_cols)} textual columns: {text_cols}")
        
        # Create a sample with only textual columns
        text_df = df[text_cols]
        sample = text_df.sample(min(len(text_df), 100)).to_dict(orient="records")
        
        prompt = (
            f"You are an expert log analyst. I have a dataset with these textual columns: {text_cols}\n\n"
            f"Sample data (first 10 rows):\n"
            f"{json.dumps(sample[:10], indent=2, default=str)}\n\n"
            f"TASK: Identify patterns in the textual data that SPECIFICALLY indicate anomalies or errors.\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"- ONLY create rules that identify ERRORS/ANOMALIES (anomaly_label=1)\n"
            f"- NEVER create rules that identify normal/success patterns (normal_label=1)\n"
            f"- Focus on error-indicating keywords, not success indicators\n\n"
            f"Look for ANOMALY INDICATORS:\n"
            f"1. ERROR KEYWORDS: 'error', 'exception', 'fail', 'failed', 'timeout', 'abort', 'denied', 'rejected'\n"
            f"2. CRITICAL STATUS: 'critical', 'warning', 'alert', 'fault', 'problem', 'emergency'\n"
            f"3. HTTP ERROR CODES: 4xx, 5xx status codes indicating errors\n"
            f"4. NEGATIVE TERMS: 'invalid', 'not found', 'refused', 'blocked', 'unauthorized', 'forbidden'\n"
            f"5. SYSTEM ISSUES: 'connection lost', 'service down', 'unavailable', 'crashed', 'overload'\n\n"
            f"DO NOT LOOK FOR: 'success', 'ok', 'completed', 'accepted', 'normal' (these are normal patterns)\n\n"
            f"Find the SINGLE most reliable ANOMALY-DETECTING pattern.\n\n"
            f"Respond with JSON in this exact format:\n"
            f'{{"column": "column_name", "operator": "==|!=|contains", "value": "error_pattern", "normal_label": 0, "anomaly_label": 1}}\n\n'
            f"Where:\n"
            f"- operator: '==' for exact match, '!=' for not equal, 'contains' for substring match\n"
            f"- normal_label: 0 = this rule does NOT identify normal data\n"
            f"- anomaly_label: 1 = this rule DOES identify anomalies\n\n"
            f"If no ANOMALY-detecting pattern exists, respond with: {{'error': 'no_pattern_found'}}\n"
        )

        system_msg = (
            "You are a log analysis expert. Analyze textual data to find patterns that indicate anomalies/errors. "
            "CRITICAL: ONLY derive rules that identify anomalies, never rules that identify normal logs. "
            "Respond with valid JSON only. Focus exclusively on error-indicating patterns in text fields."
        )

        try:
            response = self._call_bedrock_model(
                prompt=prompt,
                system_message=system_msg,
                max_tokens=500,
                temperature=0
            )

            print(f"[TEXTUAL_HEURISTIC] LLM response: {response}")

            # Handle explicit "no pattern" responses
            if response.lower() in {"none", "no obvious heuristic", "no heuristic found", "no rule"}:
                print(f"[TEXTUAL_HEURISTIC] LLM couldn't identify a textual pattern")
                return None

            # Parse JSON response with robust extraction
            try:
                # Extract JSON from markdown code blocks if present
                json_content = response
                if '```json' in response:
                    # Extract content between ```json and ```
                    start = response.find('```json') + 7
                    end = response.find('```', start)
                    if end > start:
                        json_content = response[start:end].strip()
                elif '```' in response:
                    # Handle generic code blocks
                    start = response.find('```') + 3
                    end = response.find('```', start)
                    if end > start:
                        json_content = response[start:end].strip()
                
                # Clean up any remaining formatting
                json_content = json_content.strip()
                if not json_content:
                    print(f"[TEXTUAL_HEURISTIC] Empty response after JSON extraction")
                    return None
                
                heuristic = json.loads(json_content)
                
                # Check for error response
                if heuristic.get('error') == 'no_pattern_found':
                    print(f"[TEXTUAL_HEURISTIC] LLM found no reliable textual patterns")
                    return None
                
                # Validate required fields
                required_fields = ['column', 'operator', 'value']
                if not all(field in heuristic for field in required_fields):
                    print(f"[TEXTUAL_HEURISTIC] Invalid heuristic format - missing required fields")
                    return None
                
                # Validate column exists
                if heuristic['column'] not in text_cols:
                    print(f"[TEXTUAL_HEURISTIC] Invalid column '{heuristic['column']}' not in textual columns")
                    return None
                
                # CRITICAL FILTER: Only accept anomaly-detecting heuristics for hybrid route
                if heuristic.get("anomaly_label") != 1:
                    print(f"[TEXTUAL_HEURISTIC:FILTER] Rejecting normal-detecting rule: {heuristic}")
                    print(f"[TEXTUAL_HEURISTIC:FILTER] Hybrid route requires anomaly-detecting heuristics only")
                    return None
                
                # Set default labels if not provided (ensuring anomaly detection)
                if 'normal_label' not in heuristic:
                    heuristic['normal_label'] = 0  # This rule does NOT identify normal logs
                if 'anomaly_label' not in heuristic:
                    heuristic['anomaly_label'] = 1  # This rule DOES identify anomalies
                
                # Double-check the labels are correct for anomaly detection
                if heuristic["anomaly_label"] != 1 or heuristic["normal_label"] != 0:
                    print(f"[TEXTUAL_HEURISTIC:ERROR] Invalid labels - rule must have anomaly_label=1, normal_label=0")
                    return None
                
                # Add metadata
                heuristic['source'] = 'textual_analysis'
                heuristic['confidence_boost'] = 0.7  # Medium confidence for textual patterns
                
                print(f"[TEXTUAL_HEURISTIC] Derived ANOMALY-DETECTING rule: {heuristic['column']} {heuristic['operator']} '{heuristic['value']}'")
                
                # Add or increment existing heuristic using helper method
                self._add_or_increment_heuristic(schema_id, heuristic, confidence_increment=2)  # Medium confidence for textual heuristics
                
                return heuristic
                
            except json.JSONDecodeError as e:
                print(f"[TEXTUAL_HEURISTIC] Failed to parse JSON response: {e}")
                print(f"[TEXTUAL_HEURISTIC] Raw response: {response}")
                print(f"[TEXTUAL_HEURISTIC] Extracted JSON content: {json_content if 'json_content' in locals() else 'N/A'}")
                return None
                
        except Exception as e:
            print(f"[TEXTUAL_HEURISTIC] Error calling LLM: {str(e)}")
            return None

    def save_results(self, df: pd.DataFrame, filename: str = "results.csv"):
        base_path = os.path.dirname(__file__)
        results_folder = os.path.join(base_path, "results")
        os.makedirs(results_folder, exist_ok=True)

        # Save the full results dataframe
        save_path = os.path.join(results_folder, filename)
        df.to_csv(save_path, index=False)
        print(f"[agent] Saved full results to {save_path}")
        
        # FIXED: Extract and save only the anomalies (where anomaly_label == 1)
        if "anomaly_label" in df.columns:
            anomalies_df = df[df["anomaly_label"] == 1].copy()
            
            # Create a more readable filename for the anomalies file
            anomalies_filename = filename.replace(".csv", "_errors.csv")
            anomalies_save_path = os.path.join(results_folder, anomalies_filename)
            
            # Save the errors to a separate file
            if len(anomalies_df) == 0:
                print(f"[agent] No errors/anomalies detected!")
                # Create an empty file to maintain consistency
                pd.DataFrame().to_csv(anomalies_save_path, index=False)
            else:
                anomalies_df.to_csv(anomalies_save_path, index=False)
                print(f"[agent] Saved {len(anomalies_df)} errors/anomalies to {anomalies_save_path}")
        else:
            print(f"[agent] No anomaly_label column found - skipping errors file creation")

    def detect_anomalies_pure_and_hybrid(self, df: pd.DataFrame, schema_id, label_col: Optional[str] = "y") -> pd.DataFrame:
        # Get base hyperparams
        hyperparams = self.hyperparam_memory.get(schema_id, {}).copy()
        heuristics = self.heuristic_memory.get(schema_id, [])
        has_heuristic = heuristics and isinstance(heuristics, list) and len(heuristics) > 0

        # If no hyperparameters exist, run unsupervised tuning
        if not hyperparams:
            print("[TUNE:TRIGGER] No cached hyperparameters found - running unsupervised tuning")
            hyperparams = self.tune_isolation_forest_unsupervised(df, schema_id)
            print("[TUNE:COMPLETE] Unsupervised hyperparameter tuning completed")
        else:
            print("[TUNE:CACHED] Using cached hyperparameters from memory")
        
        
        # 1. Select best heuristic by confidence
        best_heuristic = None
        if has_heuristic:
            print(f"[agent] Found {len(heuristics)} heuristics for schema {schema_id}")
            # Sort heuristics by confidence (just to be sure)
            sorted_heuristics = sorted(heuristics, key=lambda x: x.get("confidence", 0), reverse=True)
            if sorted_heuristics:
                best_heuristic = sorted_heuristics[0].get("rule", {})
                col = best_heuristic.get('column', 'Unknown')
                op = best_heuristic.get('operator', 'Unknown') 
                val = best_heuristic.get('value', 'Unknown')
                best_confidence = sorted_heuristics[0].get("confidence", 0)
                print(f"[agent] Using best heuristic (confidence: {best_confidence}): {col} {op} '{val}'")
        
        # Determine and update detection approach based on heuristic availability
        if has_heuristic and best_heuristic:
            detection_approach = "hybrid_ml_heuristic"
            print(f"[DETECTION:APPROACH] Using hybrid approach (ML + heuristics)")
        else:
            detection_approach = "pure_ml" 
            print(f"[DETECTION:APPROACH] Using pure ML approach (no heuristics)")

        # Update the log structure with the detection approach
        self.update_log_structure_with_detection_approach(schema_id, detection_approach)


        # Prepare features
        df_features = df.drop(columns=[label_col]) if label_col in df.columns else df.copy()
        cat_cols = df_features.select_dtypes(include=["object", "category"]).columns
        num_cols = df_features.select_dtypes(include=["number"]).columns
        
        # Handle case where there are no categorical columns (purely numeric data)
        if len(cat_cols) > 0:
            df_encoded = pd.get_dummies(df_features[cat_cols])
            X = pd.concat([df_features[num_cols], df_encoded], axis=1)
        else:
            # Pure numeric data - no encoding needed
            X = df_features[num_cols].copy()
            print(f"[agent] Pure numeric dataset detected - using {len(num_cols)} numeric features")
        
        # 1. Estimate contamination dynamically if possible
        est_contamination = hyperparams.get("contamination", 0.1)  # Default
        
        if best_heuristic:
            col = best_heuristic.get('column', 'Unknown')
            op = best_heuristic.get('operator', 'Unknown') 
            val = best_heuristic.get('value', 'Unknown')
            try:
                print(f"[agent] Estimating contamination using heuristic: {col} {op} '{val}'")
                # Use heuristic to estimate current batch contamination
                anomaly_label = best_heuristic.get("anomaly_label", 1)  # Get the anomaly label (what the rule considers anomalous)
                normal_label = best_heuristic.get("normal_label", 0)

                # Check if this is a compound rule
                if best_heuristic.get("type") == "compound" and (best_heuristic.get("operator") == "OR" or best_heuristic.get("logic") == "OR"):
                    # For compound rules, we need to handle multiple conditions
                    # Start with all False and OR together all rule matches
                    matching = pd.Series(False, index=df.index)
                    
                    for rule in best_heuristic.get("rules", []):
                        col = rule.get("column")
                        op = rule.get("operator")
                        val = rule.get("value")
                        
                        if col in df.columns:
                            if op == "==":
                                sub_matches = (df[col] == val)
                                matching = matching | sub_matches
                            elif op == "!=":
                                sub_matches = (df[col] != val)
                                matching = matching | sub_matches
                            elif op == "contains":
                                sub_matches = df[col].astype(str).apply(lambda x: val in x)
                                matching = matching | sub_matches
                    
                    # Calculate ratio based on compound matches
                    est_ratio = matching.mean() if anomaly_label == 1 else (1 - matching.mean())
                    
                    # Update contamination based on heuristic estimate
                    # Bound the estimate within reasonable limits
                    est_contamination = max(0.01, min(0.5, est_ratio))
                    
                    print(f"[agent] Compound rule estimated anomaly ratio: {est_ratio:.3%}")
                    print(f"[agent] Adjusted contamination from {hyperparams.get('contamination', 0.1):.3f} to {est_contamination:.3f} based on compound heuristic")
                    
                else:
                    # Original single rule code
                    col = best_heuristic.get("column")
                    op = best_heuristic.get("operator") 
                    val = best_heuristic.get("value")
                    
                    if col in df.columns:
                        # Apply the rule condition
                        if op == "==":
                            matching = (df[col] == val)
                        elif op == "!=":
                            matching = (df[col] != val)
                        elif op == "contains":
                            matching = df[col].astype(str).apply(lambda x: val in x)
                        
                        # CORRECTED LOGIC: 
                        # If anomaly_label=0, the rule identifies NORMAL records (matching=normal)
                        # If anomaly_label=1, the rule identifies ANOMALY records (matching=anomaly)
                        if anomaly_label == 0:
                            # Rule identifies normal records, so anomaly ratio = 1 - normal ratio
                            est_ratio = 1 - matching.mean()
                        else:
                            # Rule identifies anomaly records, so anomaly ratio = anomaly ratio
                            est_ratio = matching.mean()
                        
                        # Update contamination based on heuristic estimate (for both cases)
                        # Bound the estimate within reasonable limits
                        est_contamination = max(0.01, min(0.5, est_ratio))
                        
                        print(f"[agent] Estimated anomaly ratio: {est_ratio:.3%}")
                        print(f"[agent] Adjusted contamination from {hyperparams.get('contamination', 0.1):.3f} to {est_contamination:.3f} based on heuristic")

            except Exception as e:
                print(f"[agent] Error estimating contamination: {e}")

        
        # 2. Run Isolation Forest with adjusted contamination
        model = IsolationForest(
            contamination=est_contamination,
            n_estimators=hyperparams.get("n_estimators", 100),
            max_samples=hyperparams.get("max_samples", "auto"),
            max_features=hyperparams.get("max_features", 1.0),
            random_state=hyperparams.get("random_state", 42),
        )
        model.fit(X)
        
        # 3. Generate scores and predictions
        df = df.copy()
        df["anomaly_score"] = model.decision_function(X)
        df["ml_anomaly_label"] = (model.predict(X) == -1).astype(int)
        
        # 4. Apply heuristic as additional signal if we have one
        if best_heuristic:
            col = best_heuristic.get('column', 'Unknown')
            op = best_heuristic.get('operator', 'Unknown') 
            val = best_heuristic.get('value', 'Unknown')
            # Create a composite score incorporating both ML and heuristic
            try:
                print(f"[agent] Applying heuristic rule: {col} {op} '{val}'")    

                # Check if this is a compound rule
                if best_heuristic.get("type") == "compound" and (best_heuristic.get("operator") == "OR" or best_heuristic.get("logic") == "OR"):
                    # For compound rules, we need to handle multiple conditions
                    print(f"[agent] Processing compound rule with {len(best_heuristic.get('rules', []))} conditions")
                    
                    # Start with all False and OR together all rule matches
                    rule_matches = pd.Series(False, index=df.index)
                    
                    for rule in best_heuristic.get("rules", []):
                        col = rule.get("column")
                        op = rule.get("operator")
                        val = rule.get("value")
                        
                        if col in df.columns:
                            if op == "==":
                                sub_matches = (df[col] == val)
                                rule_matches = rule_matches | sub_matches
                                print(f"[agent] Found {sub_matches.sum()} matches for {col} {op} {val}")
                            elif op == "!=":
                                sub_matches = (df[col] != val)
                                rule_matches = rule_matches | sub_matches
                            elif op == "contains":
                                sub_matches = df[col].astype(str).apply(lambda x: val in x)
                                rule_matches = rule_matches | sub_matches
                    
                    # If anomaly_label is 1, matches indicate anomalies
                    # If anomaly_label is 0, non-matches indicate anomalies
                    anomaly_label = best_heuristic.get("anomaly_label", 0)
                    heuristic_anomaly = rule_matches if anomaly_label == 1 else ~rule_matches
                    df["heuristic_anomaly"] = heuristic_anomaly.astype(int)
                    print(f"[agent] Compound rule found {heuristic_anomaly.sum()} potential anomalies")
                
                else:
                    # FIXED: Single rule code with correct logic
                    col = best_heuristic.get("column")
                    op = best_heuristic.get("operator")
                    val = best_heuristic.get("value")
                    anomaly_label = best_heuristic.get("anomaly_label", 1)  # What the rule detects when it matches
                    
                    if col in df.columns:
                        if op == "==":
                            matches = (df[col] == val)
                        elif op == "!=":
                            matches = (df[col] != val)
                        elif op == "contains":
                            matches = df[col].astype(str).str.contains(str(val), case=False, na=False)
                        else:
                            print(f"[agent] Unsupported operator '{op}' - skipping heuristic")
                            matches = pd.Series(False, index=df.index)
                        
                        # Apply the logic based on what the rule is designed to detect
                        if anomaly_label == 1:
                            # Rule detects anomalies when it matches
                            df["heuristic_anomaly"] = matches.astype(int)
                        else:
                            # Rule detects normal behavior when it matches, so invert
                            df["heuristic_anomaly"] = (~matches).astype(int)
                        
                        print(f"[agent] Single rule found {df['heuristic_anomaly'].sum()} potential anomalies")
                
                # Final decision logic - DIRECTLY use estimated contamination
                # This ensures our contamination estimate directly affects results
                
                # Sort scores to get threshold based on estimated contamination rate
                scores = df["anomaly_score"].sort_values()
                
                # Use dynamic threshold from our estimated contamination
                # Calculate threshold position based on est_contamination
                threshold_idx = int(len(scores) * est_contamination)
                threshold_idx = max(1, min(threshold_idx, len(scores)-1))  # Ensure valid index
                
                # Get the threshold value
                dynamic_threshold = scores.iloc[threshold_idx]
                #print(f"[agent] Using dynamic threshold {dynamic_threshold:.3f} (at {est_contamination:.1%} contamination)")
                
                # Primary decision based on adjusted contamination threshold
                ml_decision = (df["anomaly_score"] <= dynamic_threshold).astype(int)

                # Incorporate heuristic as a weighted factor based on confidence
                best_confidence = next((h["confidence"] for h in heuristics if h.get("rule") == best_heuristic), 1)
                heuristic_weight = min(0.9, best_confidence / 10.0)  # Scale with confidence, max 0.9

                # Decision transparency header
                print(f"\n[DECISION:HYBRID] ═══ HYBRID DECISION ANALYSIS ═══")
                print(f"[DECISION:INPUTS] ML contamination threshold: {dynamic_threshold:.3f} (at {est_contamination:.1%})")
                print(f"[DECISION:INPUTS] Heuristic confidence level: {best_confidence}")
                print(f"[DECISION:INPUTS] Calculated heuristic weight: {heuristic_weight:.2f}")
                
                # If confidence is high (>= 3), give more weight to the heuristic
                if best_confidence >= 3:
                    print(f"[DECISION:STRATEGY] HIGH CONFIDENCE heuristic (≥3) - Heuristic-Primary Strategy")
                    print(f"[DECISION:LOGIC] Primary: Heuristic rule detections")
                    print(f"[DECISION:LOGIC] Supplement: ML detections with very low scores (< -0.05)")
                    print(f"[DECISION:LOGIC] Formula: heuristic_matches OR (ml_matches AND anomaly_score < -0.05)")
                    
                    # Use the heuristic decision we computed above
                    heuristic_decision = (df["heuristic_anomaly"] == 1)
                    
                    # Combined decision - trust heuristic more, but use ML for edge cases
                    ml_supplement = ml_decision & (df["anomaly_score"] < -0.05)
                    combined_decision = heuristic_decision | ml_supplement
                    
                    # Detailed breakdown of the decision process
                    print(f"[DECISION:BREAKDOWN] ───────────────────────────────")
                    print(f"[DECISION:BREAKDOWN] Heuristic detections:     {heuristic_decision.sum():4d} anomalies")
                    print(f"[DECISION:BREAKDOWN] ML total detections:      {ml_decision.sum():4d} anomalies")
                    print(f"[DECISION:BREAKDOWN] ML supplement (score<-0.05): {ml_supplement.sum():4d} anomalies")
                    print(f"[DECISION:BREAKDOWN] Combined unique total:    {combined_decision.sum():4d} anomalies")
                    
                    # Show overlap analysis
                    overlap = heuristic_decision & ml_decision
                    heuristic_only = heuristic_decision & ~ml_decision
                    ml_only = ml_supplement & ~heuristic_decision
                    print(f"[DECISION:OVERLAP] ────────────────────────────────")
                    print(f"[DECISION:OVERLAP] Both methods agree:     {overlap.sum():4d} anomalies")
                    print(f"[DECISION:OVERLAP] Heuristic only:         {heuristic_only.sum():4d} anomalies")
                    print(f"[DECISION:OVERLAP] ML supplement only:     {ml_only.sum():4d} anomalies")
                    print(f"[DECISION:FINAL] ══════════════════════════════════")
                    print(f"[DECISION:FINAL] FINAL HYBRID RESULT: {combined_decision.sum()} anomalies")
                    print(f"[DECISION:FINAL] Decision confidence: HIGH (heuristic-driven)")
                    
                else:
                    print(f"[DECISION:STRATEGY] MEDIUM CONFIDENCE heuristic (<3) - ML-Primary Strategy")
                    print(f"[DECISION:LOGIC] Primary: ML model detections")
                    print(f"[DECISION:LOGIC] Supplement: Heuristic detections with negative scores")
                    print(f"[DECISION:LOGIC] Formula: ml_matches OR (heuristic_matches AND anomaly_score < 0)")
                    
                    # Lower confidence - rely more on ML but incorporate heuristic
                    heuristic_supplement = (df["heuristic_anomaly"] == 1) & (df["anomaly_score"] < 0)
                    combined_decision = ml_decision | heuristic_supplement
                    
                    heuristic_decision = (df["heuristic_anomaly"] == 1)
                    
                    # Detailed breakdown for medium confidence
                    print(f"[DECISION:BREAKDOWN] ───────────────────────────────")
                    print(f"[DECISION:BREAKDOWN] ML primary detections:    {ml_decision.sum():4d} anomalies")
                    print(f"[DECISION:BREAKDOWN] Heuristic total:          {heuristic_decision.sum():4d} anomalies")
                    print(f"[DECISION:BREAKDOWN] Heuristic supplement:     {heuristic_supplement.sum():4d} anomalies")
                    print(f"[DECISION:BREAKDOWN] Combined unique total:    {combined_decision.sum():4d} anomalies")
                    
                    # Show overlap analysis
                    overlap = ml_decision & heuristic_supplement
                    ml_only = ml_decision & ~heuristic_supplement
                    heuristic_only = heuristic_supplement & ~ml_decision
                    print(f"[DECISION:OVERLAP] ────────────────────────────────")
                    print(f"[DECISION:OVERLAP] Both methods agree:     {overlap.sum():4d} anomalies")
                    print(f"[DECISION:OVERLAP] ML only:                {ml_only.sum():4d} anomalies")
                    print(f"[DECISION:OVERLAP] Heuristic supplement:   {heuristic_only.sum():4d} anomalies")
                    print(f"[DECISION:FINAL] ══════════════════════════════════")
                    print(f"[DECISION:FINAL] FINAL HYBRID RESULT: {combined_decision.sum()} anomalies")
                    print(f"[DECISION:FINAL] Decision confidence: MEDIUM (ML-driven)")

                # Final label (1=normal, 0=anomaly)
                df["anomaly_label"] = combined_decision.astype(int)
                print(f"[DECISION:OUTPUT] Anomaly labels set: 0={combined_decision.sum()}, 1={len(df) - combined_decision.sum()}")
                print(f"[DECISION:HYBRID] ═══ END DECISION ANALYSIS ═══\n")
                return df
            except Exception as e:
                print(f"[agent] Error applying hybrid detection: {e}")
        
        # 5. Fallback to ML-only result but still use our estimated contamination
        scores = df["anomaly_score"].sort_values()
        threshold_idx = int(len(scores) * est_contamination)
        threshold_idx = max(1, min(threshold_idx, len(scores)-1))
        dynamic_threshold = scores.iloc[threshold_idx]
        print(f"[agent] Using ML-only with dynamic threshold {dynamic_threshold:.3f} (at {est_contamination:.1%} contamination)")
        # FIXED: Remove the inversion - lower scores (≤ threshold) should be anomalies (1)
        df["anomaly_label"] = (df["anomaly_score"] <= dynamic_threshold).astype(int)
        return df

    def tune_isolation_forest_unsupervised(self, df: pd.DataFrame, schema_id: str) -> dict:
        """
        Tune Isolation Forest hyperparameters using unsupervised approach.
        Uses isolation scores to create pseudo-labels and optimize parameters.
        """
        print(f"[TUNE:UNSUPERVISED] Starting unsupervised hyperparameter tuning for schema {schema_id}")
        
        # Prepare feature matrix
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        num_cols = df.select_dtypes(include=["number"]).columns
        
        # Handle case where there are no categorical columns (purely numeric data)
        if len(cat_cols) > 0:
            df_encoded = pd.get_dummies(df[cat_cols])
            X = pd.concat([df[num_cols], df_encoded], axis=1)
        else:
            X = df[num_cols].copy()
        
        print(f"[TUNE:UNSUPERVISED] Feature matrix: {X.shape}")
        
        # Parameter grid for tuning
        param_grid = {
            "contamination": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            "n_estimators": [50, 100, 200, 300],
            "max_samples": ["auto", 0.5, 0.75, 1.0],
            "max_features": [0.5, 0.75, 1.0],
            "random_state": [42]
        }
        
        best_score = -float('inf')
        best_params = None
        
        # Run initial model to get baseline scores for pseudo-labeling
        baseline_model = IsolationForest(contamination=0.1, random_state=42)
        baseline_model.fit(X)
        baseline_scores = baseline_model.decision_function(X)
        
        # Create pseudo-labels based on baseline scores
        # Use bottom 10% as anomalies (label=0), top 10% as normal (label=1)
        score_threshold_low = np.percentile(baseline_scores, 10)
        score_threshold_high = np.percentile(baseline_scores, 90)
        
        # Create conservative pseudo-labels (only very confident examples)
        pseudo_labels = []
        for score in baseline_scores:
            if score <= score_threshold_low:
                pseudo_labels.append(0)  # Very likely anomaly
            elif score >= score_threshold_high:
                pseudo_labels.append(1)  # Very likely normal
            else:
                pseudo_labels.append(-1)  # Uncertain, exclude from evaluation
        
        pseudo_labels = np.array(pseudo_labels)
        confident_mask = pseudo_labels != -1
        
        print(f"[TUNE:UNSUPERVISED] Created {np.sum(pseudo_labels == 0)} anomaly and {np.sum(pseudo_labels == 1)} normal pseudo-labels")
        print(f"[TUNE:UNSUPERVISED] Confident labels: {np.sum(confident_mask)}/{len(pseudo_labels)} ({np.sum(confident_mask)/len(pseudo_labels):.1%})")
        
        if np.sum(confident_mask) < 20:  # Need at least 20 confident examples
            print(f"[TUNE:UNSUPERVISED] Too few confident pseudo-labels, using default parameters")
            default_params = {
                "contamination": 0.1,
                "n_estimators": 100,
                "max_samples": "auto",
                "max_features": 1.0
            }
            self.hyperparam_memory[schema_id] = default_params
            self._save_memory(self.hyperparam_memory, "hyperparam")
            return default_params
        
        # Grid search with pseudo-labels
        from sklearn.model_selection import ParameterGrid
        from sklearn.metrics import f1_score
        
        for params in ParameterGrid(param_grid):
            try:
                model = IsolationForest(**params)
                model.fit(X)
                predictions = (model.predict(X) == -1).astype(int)  # 1=anomaly, 0=normal
                
                # Evaluate only on confident pseudo-labels
                if np.sum(confident_mask) > 0:
                    # Calculate F1 score on confident examples
                    y_true_confident = pseudo_labels[confident_mask]
                    y_pred_confident = predictions[confident_mask]
                    
                    # Also consider silhouette-like score based on isolation scores
                    scores = model.decision_function(X)
                    
                    # Score based on separation between anomaly and normal scores
                    anomaly_scores = scores[predictions == 1]
                    normal_scores = scores[predictions == 0]
                    
                    if len(anomaly_scores) > 0 and len(normal_scores) > 0:
                        separation_score = np.mean(normal_scores) - np.mean(anomaly_scores)
                    else:
                        separation_score = 0
                    
                    # Combined score: F1 on pseudo-labels + separation score
                    f1 = f1_score(y_true_confident, y_pred_confident, average='weighted', zero_division=0)
                    combined_score = 0.7 * f1 + 0.3 * separation_score
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = params.copy()
                        
            except Exception as e:
                print(f"[TUNE:UNSUPERVISED] Skipping params {params} due to error: {e}")
                continue
        
        if best_params is None:
            print(f"[TUNE:UNSUPERVISED] No valid parameters found, using defaults")
            best_params = {
                "contamination": 0.1,
                "n_estimators": 100,
                "max_samples": "auto",
                "max_features": 1.0
            }
        else:
            # Remove random_state for storage
            best_params = {k: v for k, v in best_params.items() if k != "random_state"}
            best_params_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
            print(f"[TUNE:UNSUPERVISED] Best parameters: {best_params_str} (score: {best_score:.3f})")
        
        # Save to memory
        self.hyperparam_memory[schema_id] = best_params
        self._save_memory(self.hyperparam_memory, "hyperparam")
        print(f"[TUNE:UNSUPERVISED] Saved hyperparameters to memory")
        
        return best_params

    def plot_isolation_forest_results(self,df, score_col="anomaly_score", label_col="anomaly_label", ground_truth_col="Level", title="Isolation Forest Anomaly Scores"
        ):
            """
            Plots anomaly scores with anomalies highlighted.
            If ground truth is available: correctly predicted anomalies are green, incorrectly predicted are red crosses.
            If no ground truth: simply highlight detected anomalies in red.
            """
            try:
                # Check if required columns exist
                if score_col not in df.columns:
                    print(f"Warning: {score_col} column not found in dataframe. Cannot create plot.")
                    return
                if label_col not in df.columns:
                    print(f"Warning: {label_col} column not found in dataframe. Cannot create plot.")
                    return
                
                plt.figure(figsize=(14, 5))
                plt.plot(df[score_col], label="Anomaly Score", color="blue", alpha=0.7)
                
                # Check if ground_truth_col exists for supervised evaluation
                if ground_truth_col in df.columns:
                    print(f"[PLOT] Ground truth column '{ground_truth_col}' found - creating supervised visualization")
                    
                    # Compute ground truth: 1=normal, 0=anomaly
                    y_true = df[ground_truth_col].apply(lambda lvl: 1 if lvl in {"INFO", "DEBUG"} else 0).astype(int)
                    y_pred = df[label_col].astype(int)
                    
                    # Correctly predicted anomalies (true anomaly and predicted anomaly)
                    correct_anom = (y_true == 0) & (y_pred == 0)
                    # Incorrectly predicted anomalies (predicted anomaly but not true anomaly)
                    incorrect_anom = (y_true == 1) & (y_pred == 0)

                    # Plot correctly predicted anomalies (green)
                    plt.scatter(df.index[correct_anom], df[score_col][correct_anom], color="green", label="Correct Anomaly", marker="o")
                    # Plot incorrectly predicted anomalies (red X)
                    plt.scatter(df.index[incorrect_anom], df[score_col][incorrect_anom], color="red", label="Incorrect Anomaly", marker="x")
                else:
                    print(f"[PLOT] No ground truth column - creating unsupervised visualization")
                    
                    # Without ground truth, just highlight detected anomalies
                    y_pred = df[label_col].astype(int)
                    detected_anomalies = (y_pred == 1)  # anomaly_label == 1 means anomaly
                    
                    # Plot detected anomalies in red
                    plt.scatter(df.index[detected_anomalies], df[score_col][detected_anomalies], 
                              color="red", label="Detected Anomaly", marker="o", s=50)

                plt.title(title)
                plt.xlabel("Row Index")
                plt.ylabel("Anomaly Score")
                plt.legend()
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Warning: Could not create plot - {str(e)}")
                import traceback
                print(f"Full error: {traceback.format_exc()}")
                return

    def evaluate_and_print(self, df, ground_truth_col, ml_pred_col, llm_pred_col):

        if ground_truth_col is not None:
            # 1) Build y_true = 1 for normal, 0 for error
            y_true = df[ground_truth_col].apply(
                lambda lvl: 1 if lvl in {"INFO", "DEBUG"} else 0
            ).astype(int)

            # 2) Predictions (already 1=normal, 0=anomaly)
            y_ml  = df[ml_pred_col].fillna(1).astype(int)

            def print_metrics(name, y_pred, y_true_ref):
                total = len(y_true_ref)
                acc_norm = accuracy_score(y_true_ref, y_pred)
                prec_norm = precision_score(y_true_ref, y_pred, pos_label=1)
                rec_norm  = recall_score(y_true_ref, y_pred, pos_label=1)
                f1_norm   = f1_score(y_true_ref, y_pred, pos_label=1)
                cm = confusion_matrix(y_true_ref, y_pred, labels=[1, 0])
                prec_anom = precision_score(y_true_ref, y_pred, pos_label=0)
                rec_anom  = recall_score(y_true_ref, y_pred, pos_label=0)

                f1_anom   = f1_score(y_true_ref, y_pred, pos_label=0)
                total_anom = sum(y_true_ref == 0)
                found_anom = sum((y_true_ref == 0) & (y_pred == 0))

                print(f"\n{name} Metrics (class 1 = normal):")
                print(f"  Accuracy:  {acc_norm:.4f}")
                print(f"  Precision: {prec_norm:.4f} (of predicted normal, how many truly normal)")
                print(f"  Recall:    {rec_norm:.4f} (of true normal, how many predicted normal)")
                print(f"  F1-score:  {f1_norm:.4f}")
                print(f"  Confusion Matrix [[TP, FN],[FP, TN]] =\n{cm}")

                print(f"\n{name} Metrics (class 0 = anomaly):")
                print(f"  Precision (anom):  {prec_anom:.4f}")
                print(f"  Recall    (anom):  {rec_anom:.4f}")
                print(f"  F1-score  (anom):  {f1_anom:.4f}")
                print(f"  Anomalies in data: {total_anom}, Anomalies found: {found_anom}")

            print(f"Total rows: {len(df)}")
            print_metrics("ML",  y_ml, y_true)

            # Only print LLM metrics if column exists and has valid predictions
            if llm_pred_col in df.columns:
                y_llm = df[llm_pred_col].fillna(-1).astype(int)
                valid_mask = y_llm.isin([0, 1])
                if valid_mask.any():
                    print_metrics("LLM", y_llm[valid_mask], y_true[valid_mask])
                else:
                    print("\nNo valid LLM predictions to evaluate.")
            else:
                print("\nNo LLM predictions column found.")
        else:
            #output num of anomlies found
            print(f"[agent] No ground truth column provided for evaluation.")
            print(f"Total rows: {len(df)}")
            print(f"Anomalies found by ML: {df['anomaly_label'].value_counts().get(0, 0)}")
            if llm_pred_col in df.columns:
                print(f"Anomalies found by LLM: {df[llm_pred_col].value_counts().get(0, 0)}")   
            else:
                print("No LLM predictions column found.")



