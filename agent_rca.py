import pandas as pd
import os
import json
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import re
from collections import Counter, defaultdict
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import time
from scipy import stats
import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
from datetime import datetime
import random


# Load environment variables from .env file
load_dotenv()

URL = "http://localhost:5000/event"

class AgentRCA:
    """
    Root Cause Analysis Agent that analyzes anomalies detected by the anomaly detection agent.
    """
    
    def __init__(self, model="anthropic.claude-3-sonnet-20240229-v1:0", region="eu-west-2"):
        """Initialize the RCA agent."""
        self.base_path = os.path.dirname(__file__)
        self.results_folder = os.path.join(self.base_path, "results")
        self.memory_folder = os.path.join(self.base_path, "memory")
        
        # Load memory files
        self.log_structure_memory = self._load_memory("log_structure_memory.json")
        self.heuristic_memory = self._load_memory("heuristic_memory.json")
        self.hyperparam_memory = self._load_memory("hyperparam_memory.json")
        self.preprocess_memory = self._load_memory("preprocess_memory.json")
        
        # NEW: Load RCA pattern memory
        self.rca_memory = self._load_memory("rca_template_memory.json")

        # Load RCA forecasting memory
        self.rca_forecast_memory = self._load_memory("rca_forecast_memory.json")

        # AWS Bedrock setup
        self.region = region
        self.model = model
        
        # Initialize Bedrock client
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            self.llm_enabled = True
            print(f"[RCA:LLM] AWS Bedrock client initialized with model: {self.model}")
        except Exception as e:
            print(f"[RCA:LLM] Error initializing Bedrock client: {e}")
            self.llm_enabled = False
    
    def _load_memory(self, filename: str) -> Dict:
        """Load memory from JSON file"""
        try:
            file_path = os.path.join(self.memory_folder, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                if filename == "log_structure_memory.json" and len(data) > 0:
                    print(f"[RCA:MEMORY] Loaded {len(data)} schema configurations")
                return data
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            print(f"[RCA:MEMORY] Error reading {filename}: Invalid JSON format")
            return {}
        except Exception as e:
            print(f"[RCA:MEMORY] Error loading {filename}: {e}")
            return {}
    
    def _save_rca_memory(self):
        """Save RCA memory to file"""
        try:
            file_path = os.path.join(self.memory_folder, "rca_template_memory.json")
            with open(file_path, 'w') as f:
                json.dump(self.rca_memory, f, indent=2)
        except Exception as e:
            print(f"[RCA:MEMORY] Error saving RCA memory: {e}")
    
    def _get_rca_from_memory(self, schema_id: str, pattern: str) -> Optional[Dict]:
        """Get RCA analysis from memory for a specific schema and pattern"""
        if schema_id in self.rca_memory and pattern in self.rca_memory[schema_id]:
            return self.rca_memory[schema_id][pattern]
        return None
    
    def _get_next_template_id(self, schema_id: str) -> int:
        """Get the next incremental template ID for a schema"""
        if schema_id not in self.rca_memory:
            return 0
        
        # Find the highest existing template_id
        max_id = -1
        for pattern_data in self.rca_memory[schema_id].values():
            if isinstance(pattern_data, dict) and 'template_id' in pattern_data:
                max_id = max(max_id, pattern_data['template_id'])
        
        return max_id + 1
    
    def _save_rca_to_memory(self, schema_id: str, pattern: str, analysis: Dict):
        """Save RCA analysis to memory with incremental template ID"""
        if schema_id not in self.rca_memory:
            self.rca_memory[schema_id] = {}
        
        # Check if pattern already exists (to preserve existing template_id)
        existing_analysis = self.rca_memory[schema_id].get(pattern)
        if existing_analysis and 'template_id' in existing_analysis:
            template_id = existing_analysis['template_id']
        else:
            # Assign new incremental template ID
            template_id = self._get_next_template_id(schema_id)
        
        # Add metadata to analysis
        analysis_with_metadata = analysis.copy()
        analysis_with_metadata['template_id'] = template_id
        analysis_with_metadata['pattern_template'] = pattern
        analysis_with_metadata['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        analysis_with_metadata['schema_id'] = schema_id
        
        self.rca_memory[schema_id][pattern] = analysis_with_metadata
        self._save_rca_memory()
        
        print(f"[RCA:MEMORY] Saved template ID {template_id} for pattern: {pattern[:50]}...")
        return template_id

    def _get_forecast_rules_from_memory(self, schema_id: str) -> Optional[List[Dict]]:
        """Get forecasting rules from memory for a specific schema."""
        return self.rca_forecast_memory.get(schema_id, [])

    def _save_forecast_rules_to_memory(self):
        """Save forecasting rules memory to file."""
        try:
            file_path = os.path.join(self.memory_folder, "rca_forecast_memory.json")
            with open(file_path, 'w') as f:
                json.dump(self.rca_forecast_memory, f, indent=2)
        except Exception as e:
            print(f"[RCA:MEMORY] Error saving RCA forecast memory: {e}")

    def _add_forecast_rule_to_memory(self, schema_id: str, rule: Dict):
        """Add a new forecasting rule to memory and save, avoiding duplicates."""
        if schema_id not in self.rca_forecast_memory:
            self.rca_forecast_memory[schema_id] = []
        # Check for duplicate rule
        for existing_rule in self.rca_forecast_memory[schema_id]:
            if (
                existing_rule.get("feature") == rule.get("feature") and
                existing_rule.get("pattern") == rule.get("pattern") and
                abs(existing_rule.get("slope_threshold", 0) - rule.get("slope_threshold", 0)) < 1e-6
            ):
                # Duplicate found, do not add
                return
        self.rca_forecast_memory[schema_id].append(rule)
        self._save_forecast_rules_to_memory()

    def _analyze_pattern_with_claude(self, pattern: str, example: str) -> Optional[Dict]:
        """Analyze an error pattern using Claude to get root cause, severity, and mitigation"""
        if not self.llm_enabled:
            return None
            
        system_message = """You are an expert system administrator and log analyst. Given an error pattern and example, provide:
            1. Root Cause: Brief explanation of what causes this error
            2. Severity: Low, Medium, or High
            3. Mitigation: Specific actionable steps to resolve or prevent this error

            Respond in JSON format with exactly these keys: root_cause, severity, mitigation"""

        prompt = f"""Analyze this error pattern:

            Pattern Template: {pattern}
            Example Message: {example}

            Provide analysis in JSON format with root_cause, severity (Low/Medium/High), and mitigation."""

        try:
            response = self._call_bedrock_model(prompt, system_message, max_tokens=500, temperature=0.1)
            if response:
                # Try to parse JSON response with multiple fallback strategies
                response_clean = response.strip()
                
                # Remove code block markers
                if response_clean.startswith('```json'):
                    response_clean = response_clean[7:-3].strip()
                elif response_clean.startswith('```'):
                    response_clean = response_clean[3:-3].strip()
                
                # Fix common escape character issues
                response_clean = response_clean.replace('\\\\', '\\')  # Fix double backslashes
                response_clean = response_clean.replace('\\"', '"')   # Fix escaped quotes
                
                try:
                    analysis = json.loads(response_clean)
                except json.JSONDecodeError:
                    # Fallback: Try to extract from partial JSON if it looks structured
                    import re
                    root_cause_match = re.search(r'"root_cause"\s*:\s*"([^"]*)"', response_clean)
                    severity_match = re.search(r'"severity"\s*:\s*"([^"]*)"', response_clean)
                    mitigation_match = re.search(r'"mitigation"\s*:\s*"([^"]*)"', response_clean)
                    
                    if root_cause_match and severity_match and mitigation_match:
                        analysis = {
                            'root_cause': root_cause_match.group(1),
                            'severity': severity_match.group(1),
                            'mitigation': mitigation_match.group(1)
                        }
                    else:
                        raise json.JSONDecodeError("Could not extract structured data", response_clean, 0)
                
                # Validate required keys
                if all(key in analysis for key in ['root_cause', 'severity', 'mitigation']):
                    # Normalize severity to standard values
                    severity = analysis['severity'].strip().title()
                    if severity not in ['Low', 'Medium', 'High']:
                        severity = 'Medium'  # Default fallback
                    analysis['severity'] = severity
                    return analysis
                else:
                    print(f"[RCA:LLM] Invalid response format: missing required keys")
                    return None
            
        except json.JSONDecodeError as e:
            print(f"[RCA:LLM] Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            print(f"[RCA:LLM] Error analyzing pattern: {e}")
            return None
            
        return None

    def _normalize_message_to_pattern(self, message: str) -> str:
        """Convert an error message to a normalized pattern template"""
        import re
        
        # Replace common dynamic elements with placeholders
        pattern = message
        
        # Replace timestamps
        pattern = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?', '[TIMESTAMP]', pattern)
        pattern = re.sub(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', '[TIMESTAMP]', pattern)
        
        # Replace IDs and hashes
        pattern = re.sub(r'\b[a-f0-9]{8,}\b', '[ID]', pattern)
        pattern = re.sub(r'\b\d{6,}\b', '[ID]', pattern)
        
        # Replace IP addresses
        pattern = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', pattern)
        
        # Replace file paths
        pattern = re.sub(r'[/\\][^\s]+[/\\][^\s]+', '[PATH]', pattern)
        
        # Replace URLs
        pattern = re.sub(r'https?://[^\s]+', '[URL]', pattern)
        
        # Replace numeric values (but keep small numbers that might be meaningful)
        pattern = re.sub(r'\b\d{4,}\b', '[NUMBER]', pattern)
        
        # Replace UUIDs
        pattern = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '[UUID]', pattern)
        
        return pattern.strip()

    def _call_bedrock_model(self, prompt, system_message=None, max_tokens=1000, temperature=0.3):
        """
        Helper method to call AWS Bedrock Claude 3 Sonnet
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
            print(f"[RCA:BEDROCK] AWS Bedrock API error: {e}")
            return None
        except Exception as e:
            print(f"[RCA:BEDROCK] Unexpected error calling Bedrock: {e}")
            return None
        
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
                anomaly_count = (df['anomaly_label'] == 1).sum()
                normal_count = (df['anomaly_label'] == 0).sum()
                print(f"[RCA:DATA] Found {anomaly_count} anomalies and {normal_count} normal records")
                
            return df
            
        except Exception as e:
            raise Exception(f"Error loading anomaly results file {file_path}: {str(e)}")
    
    def save_rca_results_to_csv(self, errors_df: pd.DataFrame, schema_id: str, filename_suffix: str = "rca_results") -> str:
        """
        Save RCA results to CSV with template IDs for grouping.
        
        Args:
            errors_df: DataFrame with error messages and their assigned template IDs
            schema_id: Schema ID for the analysis
            filename_suffix: Suffix for the output filename
            
        Returns:
            Path to the saved CSV file
        """
        try:
            # Create results DataFrame with template information
            results_data = []
            
            for idx, row in errors_df.iterrows():
                message = str(row.get('message', ''))
                if pd.isna(message) or message.strip() == '':
                    continue
                
                # Get the normalized pattern
                pattern = self._normalize_message_to_pattern(message)
                
                # Get RCA analysis from memory
                rca_analysis = self._get_rca_from_memory(schema_id, pattern)
                
                if rca_analysis:
                    result_row = {
                        'original_message': message,
                        'pattern_template': pattern,
                        'template_id': rca_analysis.get('template_id', -1),
                        'root_cause': rca_analysis.get('root_cause', 'Unknown'),
                        'severity': rca_analysis.get('severity', 'Unknown'),
                        'mitigation': rca_analysis.get('mitigation', 'No mitigation available'),
                        'timestamp': rca_analysis.get('timestamp', ''),
                        'schema_id': schema_id
                    }
                    
                    # Add any additional columns from the original dataframe
                    for col in row.index:
                        if col not in result_row and col != 'message':
                            result_row[f'original_{col}'] = row[col]
                            
                    results_data.append(result_row)
            
            # Create DataFrame
            results_df = pd.DataFrame(results_data)
            
            if len(results_df) == 0:
                print(f"[RCA:CSV] No results to save for schema {schema_id}")
                return ""
            
            # Generate filename with timestamp
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{schema_id[:8]}_{filename_suffix}_{timestamp}.csv"
            filepath = os.path.join(self.results_folder, filename)
            
            # Ensure results folder exists
            os.makedirs(self.results_folder, exist_ok=True)
            
            # Save to CSV
            results_df.to_csv(filepath, index=False)
            
            # Print summary
            template_counts = results_df['template_id'].value_counts().sort_index()
            unique_templates = len(template_counts)
            
            print(f"[RCA:CSV] Saved {len(results_df)} records to {filepath}")
            print(f"[RCA:CSV] {unique_templates} unique templates found:")
            for template_id, count in template_counts.head(5).items():
                severity = results_df[results_df['template_id'] == template_id]['severity'].iloc[0]
                print(f"[RCA:CSV]   Template {template_id}: {count} occurrences (Severity: {severity})")
            
            if len(template_counts) > 5:
                print(f"[RCA:CSV]   ... and {len(template_counts) - 5} more templates")
            
            return filepath
            
        except Exception as e:
            print(f"[RCA:CSV] Error saving RCA results to CSV: {e}")
            return ""
            
    def get_template_summary(self, schema_id: str) -> pd.DataFrame:
        """
        Get a summary of all templates for a given schema.
        
        Args:
            schema_id: Schema ID to get template summary for
            
        Returns:
            DataFrame with template summary information
        """
        if schema_id not in self.rca_memory:
            return pd.DataFrame()
        
        summary_data = []
        for pattern, analysis in self.rca_memory[schema_id].items():
            summary_data.append({
                'template_id': analysis.get('template_id', -1),
                'pattern_template': pattern,
                'root_cause': analysis.get('root_cause', 'Unknown'),
                'severity': analysis.get('severity', 'Unknown'),
                'mitigation': analysis.get('mitigation', 'No mitigation available'),
                'timestamp': analysis.get('timestamp', ''),
                'schema_id': schema_id
            })
        
        summary_df = pd.DataFrame(summary_data)
        if len(summary_df) > 0:
            summary_df = summary_df.sort_values('template_id')
        
        return summary_df

    def analyze_errors(self, anomalies_df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None, schema_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze anomalies to determine root causes using adaptive strategy.
        
        Args:
            anomalies_df: DataFrame containing detected anomalies
            original_df: Optional original dataset for additional context
            schema_id: Schema ID to look up log structure and other memory
            
        Returns:
            Dictionary containing root cause analysis results
        """
        print(f"[RCA:ANALYZE] Starting root cause analysis on {len(anomalies_df)} anomalies")
        
        # Get log structure from memory if schema_id provided
        log_structure = None
        if schema_id and schema_id in self.log_structure_memory:
            log_structure = self.log_structure_memory[schema_id]
            print(f"[RCA:MEMORY] Using cached schema configuration")
        else:
            if not schema_id:
                print(f"[RCA:WARNING] No schema_id provided")
            else:
                # Try to reload memory in case it was lost during initialization
                self.log_structure_memory = self._load_memory("log_structure_memory.json")
                if schema_id in self.log_structure_memory:
                    log_structure = self.log_structure_memory[schema_id]
                    print(f"[RCA:MEMORY] Schema configuration loaded successfully")
                else:
                    print(f"[RCA:WARNING] Schema configuration not found")
        
        # Initialize results structure
        analysis_results = {
            "total_anomalies": len(anomalies_df),
            "rca_approach": "",
            "analysis_results": {},
            "recommendations": [],
            "schema_id": schema_id,
            "log_structure": log_structure
        }
        
        # Check if we have a message column for pattern analysis
        message_column = None
        template_summary = None
        
        if log_structure:
            message_column = log_structure["message_column"]
            if message_column in anomalies_df.columns:
                print(f"[RCA:PATTERN] Using message-based pattern analysis - ROUTE 1")
                # Create a temporary DataFrame with standardized column name for clustering
                temp_df = anomalies_df.copy()
                temp_df['message'] = temp_df[message_column]
                template_summary = self.message_context_present_route(temp_df, schema_id)
                analysis_results["rca_approach"] = "Pattern-Based Error Analysis"
                analysis_results["template_summary"] = template_summary.to_dict('records') if template_summary is not None else []
            else:
                print(f"[RCA:STRUCTURED] Message column not available, using structured data analysis - ROUTE 2")
                absent_results = self.message_context_absent_route(anomalies_df, original_df, schema_id, log_structure)
                #FOR FRONTEND STREAMING
                # events = self.prepare_stream_events(absent_results, original_df, absent_results["trend_analysis"]["forecasting_rules"])
                # self.stream_events(events, "http://localhost:5000/event", delay=0.2)

                analysis_results["rca_approach"] = "Structured Data Analysis"
                analysis_results["absent_route_results"] = absent_results
        else:
            print(f"[RCA:STRUCTURED] no log structure found this is a fallabck path (should not occur)")
            
        # Print results summary
        if analysis_results["rca_approach"] == "Pattern-Based Error Analysis" and template_summary is not None:
            print(f"[RCA:COMPLETE] Generated {len(template_summary)} error patterns")
        elif analysis_results["rca_approach"] == "Structured Data Analysis":
            print(f"[RCA:COMPLETE] Structured data analysis completed successfully")
        
        return analysis_results
    
    def message_context_present_route(self, errors_df, schema_id):
        
        """
        Cluster error messages by pattern and generate insights using Claude LLM.
        Uses memory system to cache analyses and avoid repeated LLM calls.
        
        Args:
            errors_df: DataFrame with error messages
            schema_id: Identifier for the dataset/schema for memory organization
        """
        if 'message' not in errors_df.columns:
            print("[PATTERN_DETECTION:ERROR] No 'message' column found in error data")
            return
        
        # Extract and normalize error patterns
        error_patterns = {}
        total_errors = len(errors_df)
        
        print(f"[PATTERN_DETECTION] Analyzing {total_errors} error messages")
        
        for idx, row in errors_df.iterrows():
            message = str(row['message'])
            if pd.isna(message) or message.strip() == '':
                continue
                
            # Normalize the message to create a pattern template
            pattern = self._normalize_message_to_pattern(message)
            
            if pattern not in error_patterns:
                error_patterns[pattern] = {
                    'count': 0,
                    'examples': [],
                    'pattern': pattern
                }
            
            error_patterns[pattern]['count'] += 1
            if len(error_patterns[pattern]['examples']) < 3:  # Keep up to 3 examples
                error_patterns[pattern]['examples'].append(message)
        
        # Sort patterns by frequency
        sorted_patterns = sorted(error_patterns.items(), 
                               key=lambda x: x[1]['count'], reverse=True)
        
        # Count unique patterns (ones that appear only once)
        unique_patterns = sum(1 for _, data in sorted_patterns if data['count'] == 1)
        
        # Analyze ALL patterns with LLM (including unique ones)
        analyzed_count = 0
        cached_count = 0
        failed_count = 0
        
        # Process all patterns for memory caching
        for pattern, data in sorted_patterns:
            # Check memory first
            rca_analysis = self._get_rca_from_memory(schema_id, pattern)
            
            if rca_analysis:
                cached_count += 1
            else:
                # Generate new analysis for ANY pattern (even unique ones)
                analysis = self._analyze_pattern_with_claude(pattern, data['examples'][0])
                if analysis:
                    # Save to memory
                    self._save_rca_to_memory(schema_id, pattern, analysis)
                    analyzed_count += 1
                else:
                    failed_count += 1
        
        # Print concise summary
        print(f"[PATTERN_DETECTION:SUMMARY] {len(sorted_patterns)} patterns found ({unique_patterns} unique) | New: {analyzed_count} | Cached: {cached_count}")
        
        # Show top 3 patterns with severity and template IDs
        top_patterns = sorted_patterns[:3]
        for i, (pattern, data) in enumerate(top_patterns, 1):
            percentage = (data['count'] / total_errors) * 100
            rca_analysis = self._get_rca_from_memory(schema_id, pattern)
            severity = rca_analysis['severity'] if rca_analysis else 'Unknown'
            template_id = rca_analysis['template_id'] if rca_analysis else 'N/A'
            print(f"[PATTERN_DETECTION:TOP{i}] Template {template_id} | {data['count']} occurrences ({percentage:.1f}%) | Severity: {severity} | {pattern[:50]}{'...' if len(pattern) > 50 else ''}")
        
        # Save results to CSV for analysis and grouping
        csv_path = self.save_rca_results_to_csv(errors_df, schema_id, "pattern_analysis")
        if csv_path:
            print(f"[PATTERN_DETECTION:CSV] Results saved to: {csv_path}")
        
        # Return template summary for further analysis
        return self.get_template_summary(schema_id)
    
    def _normalize_message_to_pattern(self, message: str) -> str:
        
        """
        Normalizes a log or message string by replacing variable components with standardized patterns.
        This function applies a series of regular expression substitutions to the input message,
        replacing elements such as IP addresses, timestamps, file paths, UUIDs, hexadecimal values,
        URLs, email addresses, MAC addresses, long alphanumeric strings (e.g., session IDs, tokens, hashes),
        numbers, and quoted strings with corresponding placeholders.
        
        Enhanced to handle URL encoding and normalize directory traversal attacks.
        
        Args:
            message (str): The input message string to normalize.
        Returns:
            str: The normalized message with variable components replaced by standardized patterns.
        """

        pattern = message
        
        # FIRST: Decode URL encoding to normalize different encoding variants of the same attack
        try:
            import urllib.parse
            # Decode URL encoding once to normalize variants like %c0%af, %5c, %c1%9c etc.
            decoded = urllib.parse.unquote(pattern)
            # If decoding changed something, use the decoded version
            if decoded != pattern:
                pattern = decoded
        except:
            pass  # Keep original if decoding fails
        
        # SECOND: Normalize directory traversal patterns (regardless of original encoding)
        pattern = re.sub(r'\.\.[\\/]', '[TRAVERSAL]', pattern)  # ../
        pattern = re.sub(r'[\\/]\.\.', '[TRAVERSAL]', pattern)  # /..
        pattern = re.sub(r'\.\.[^/\\]*', '[TRAVERSAL]', pattern)  # ..anything
        
        # THIRD: Normalize other URL-encoded attack patterns
        pattern = re.sub(r'%[0-9a-fA-F]{2}', '[ENCODED]', pattern)  # Any remaining URL encoding
        pattern = re.sub(r'\\x[0-9a-fA-F]{2}', '[ENCODED]', pattern)  # Hex encoding
        pattern = re.sub(r'\\[xuU][0-9a-fA-F]+', '[ENCODED]', pattern)  # Unicode escapes
        
        # Replace IP addresses (IPv4)
        pattern = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', pattern)
        
        # Replace IPv6 addresses
        pattern = re.sub(r'\b[0-9a-fA-F:]{10,}\b', '[IPv6]', pattern)
        
        # Replace timestamps (various common formats)
        pattern = re.sub(r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?', '[TIMESTAMP]', pattern)  # ISO format
        pattern = re.sub(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', '[TIMESTAMP]', pattern)  # US format
        pattern = re.sub(r'\w{3} \w{3} \d{1,2} \d{2}:\d{2}:\d{2}', '[TIMESTAMP]', pattern)  # Syslog format
        pattern = re.sub(r'\d{10,13}', '[EPOCH]', pattern)  # Unix epoch timestamps
        
        # Replace file paths (Unix and Windows)
        pattern = re.sub(r'/[a-zA-Z0-9_./\-~]+', '[PATH]', pattern)
        pattern = re.sub(r'[A-Z]:\\[a-zA-Z0-9_\\.\\-]+', '[PATH]', pattern)
        
        # Replace UUIDs (before general numbers to be more specific)
        pattern = re.sub(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '[UUID]', pattern)
        
        # Replace hexadecimal values
        pattern = re.sub(r'\b0x[0-9a-fA-F]+\b', '[HEX]', pattern)
        
        # Replace URLs
        pattern = re.sub(r'https?://[^\s]+', '[URL]', pattern)
        
        # Replace email addresses
        pattern = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '[EMAIL]', pattern)
        
        # Replace MAC addresses
        pattern = re.sub(r'\b[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\b', '[MAC]', pattern)
        
        # Replace long alphanumeric strings (session IDs, tokens, hashes)
        pattern = re.sub(r'\b[a-zA-Z0-9]{16,}\b', '[ID]', pattern)
        
        # Replace all remaining numbers (any sequence of digits)
        pattern = re.sub(r'\b\d+\b', '[NUM]', pattern)
        
        # Replace quoted strings (preserving the structure but not the content)
        pattern = re.sub(r'"[^"]*"', '[QUOTED]', pattern)
        pattern = re.sub(r"'[^']*'", '[QUOTED]', pattern)
        
        return pattern.strip()

    def message_context_absent_route(self, errors_df, original_df, schema_id, log_structure):

        rules = self._get_forecast_rules_from_memory(schema_id)

        if rules:
            print("[RCA:FORECAST] We have seen this dataset before and have forecasting rules")
            for rule in rules:
                print(f"[RCA:FORECAST] - {rule['forecast_rule']} (Confidence: {rule['confidence']})")

        print(f"[RCA:STRUCTURED] Analyzing {len(errors_df)} anomalies in structured data")

        if log_structure.get("log_level_column"):
            #drop column from original
            original_df = original_df.drop(columns=[log_structure["log_level_column"], "heuristic_anomaly"], errors='ignore')

        # --- Process error dataset (test group) ---
        error_df = self._drop_structural_columns(log_structure, errors_df)
        error_df = self._drop_result_columns(log_structure, error_df)

        tokenised_error_df = self._tokenize_dataframe_dynamic(error_df)
        #tokenised_error_df.to_csv(f"demo-data/{schema_id}_tokenized_errors.csv", index=False)
        
        # --- Process control dataset (normal events from original_df) ---
        control_df = original_df[original_df['anomaly_label'] == 0].copy()

        control_df = self._drop_structural_columns(log_structure, control_df)
        control_df = self._drop_result_columns(log_structure, control_df)

        tokenised_control_df = self._tokenize_dataframe_dynamic(control_df)
        #tokenised_control_df.to_csv(f"demo-data/{schema_id}_tokenized_control.csv", index=False)

        print(f"[RCA:STRUCTURED] ✓ Data preprocessing completed successfully")


        # --- Extract sliding windows of tokens around each error ---
        error_windows = self._extract_error_windows(original_df, log_structure)
        # print length descriptively
        print(f"[RCA:STRUCTURED] ✓ Found {len(error_windows)} error windows")

        # --- Extract sliding windows of normal events (11 contiguous normal without error) ---
        normal_windows = self._extract_normal_windows(original_df, log_structure)
        print(f"[RCA:STRUCTURED] ✓ Found {len(normal_windows)} normal windows")

        print(f"[RCA:STRUCTURED] ✓ Sliding Window extraction completed successfully")

        # NEW: Trend Analysis - Extract raw numerical windows for slope calculation
        print(f"\n[RCA:TRENDS] Starting trend analysis on temporal patterns leading to errors...")
        raw_error_windows = self._extract_raw_error_windows(original_df, log_structure, window_size=11)
        print(f"[RCA:TRENDS] ✓ Extracted {len(raw_error_windows)} raw error windows for trend analysis")
        
        # Calculate slopes and trends for each error window
        slopes_df = self._calculate_feature_slopes(raw_error_windows, log_structure)
        print(f"[RCA:TRENDS] ✓ Calculated slope trends for {len(slopes_df.columns)} feature slopes")
        
        # Classify trends and find common patterns
        trend_patterns = self._classify_trends(slopes_df, threshold=0.05)
        common_trend_patterns = self._mine_trend_patterns(trend_patterns)
        
        # Print trend insights
        print(f"\n[RCA:TRENDS] === Temporal Pattern Analysis Results ===")
        print(f"[RCA:TRENDS] Top trend patterns occurring before errors:")
        for i, (pattern, count) in enumerate(common_trend_patterns[:5], 1):
            percentage = (count / len(raw_error_windows)) * 100
            print(f"[RCA:TRENDS] #{i}: {pattern}")
            print(f"[RCA:TRENDS]     → Occurs in {count} of {len(raw_error_windows)} error windows ({percentage:.1f}%)")
        
        # Generate forecasting rules
        forecasting_rules = self._generate_forecasting_rules(slopes_df)
        print(f"\n[RCA:TRENDS] === Predictive Forecasting Rules Generated ===")
        print(f"[RCA:TRENDS] Generated {len(forecasting_rules)} forecasting rules")

        for rule in forecasting_rules:
            self._add_forecast_rule_to_memory(schema_id, rule)
        
        if len(forecasting_rules) == 0:
            print(f"[RCA:TRENDS] No significant trend patterns found for forecasting rules")
            print(f"[RCA:TRENDS] This may indicate stable features or insufficient trend variance")
        else:
            for i, rule in enumerate(forecasting_rules[:5], 1):
                print(f"[RCA:TRENDS] Rule #{i}: {rule['forecast_rule']}")
                print(f"[RCA:TRENDS]     → Confidence: {rule['confidence']} | Feature: {rule['feature']}")

        # Vectorize the extracted windows
        X, y, vectorizer = self._vectorize_windows(error_windows, normal_windows)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Labels shape: {y.shape}")


        feature_names = self._plot_feature_importances(X.toarray() if hasattr(X, 'toarray') else X, y, vectorizer.get_feature_names_out())

        #show time series of activations - cool to show importnant bins activating at the same time
        #self._plot_binary_activation_sequence(X, y, feature_names, top_n=10)

        summary_df = self._analyze_feature_bins(X, y, feature_names, top_n=10, cooccur_top_k=3)


        results = {
            "preprocessed_error_data": error_df,
            "tokenised_error_data": tokenised_error_df,
            "preprocessed_control_data": control_df,
            "tokenised_control_data": tokenised_control_df,
            "error_windows": error_windows,
            "trend_analysis": {
                "slopes_df": slopes_df,
                "trend_patterns": trend_patterns,
                "common_patterns": common_trend_patterns,
                "forecasting_rules": forecasting_rules,
                "raw_error_windows": raw_error_windows
            }
        }

        print(f"\n[RCA:COMPLETE] Structured data analysis completed successfully")
        print(f"[RCA:COMPLETE] Analysis included: {len(error_windows)} tokenized windows + {len(raw_error_windows)} trend windows")
        return results
    
    def _drop_structural_columns(self, log_structure, errors_df):
        """
        Removes specified structural columns (e.g., timestamp and lineID) from the provided DataFrame based on the log structure schema.
        Args:
            log_structure (dict): Dictionary containing schema information, including keys 'timestamp_column' and 'lineID_column' specifying column names to drop.
            errors_df (pd.DataFrame): Input DataFrame containing error log data.
        Returns:
            pd.DataFrame: A copy of the input DataFrame with the specified structural columns removed.
        Side Effects:
            Prints informative messages about the presence and removal of structural columns, as well as a summary of remaining numerical and categorical columns.
        """
        # Extract timestamp and lineID column names from log structure
        timestamp_col = log_structure.get("timestamp_column")
        lineID_col = log_structure.get("lineID_column")

        # Create working copy of the dataframe
        working_df = errors_df.copy()

        # Sort DataFrame to preserve event order

        #change order to srot on TIMESTAMP

        if lineID_col and lineID_col in working_df.columns:
            working_df = working_df.sort_values(by=lineID_col)
        elif timestamp_col and timestamp_col in working_df.columns:
            working_df = working_df.sort_values(by=timestamp_col)
        else:
            print(f"[RCA:STRUCTURED] ⚠ No timestamp or lineID column found for sorting, preserving existing order")

        original_columns = list(working_df.columns)
        columns_to_drop = []

        # Check and prepare timestamp column for removal
        if timestamp_col:
            if timestamp_col in working_df.columns:
                columns_to_drop.append(timestamp_col)
        else:
            print(f"[RCA:STRUCTURED]  No timestamp column specified in schema")

        # Check and prepare lineID column for removal
        if lineID_col:
            if lineID_col in working_df.columns:
                columns_to_drop.append(lineID_col)
        else:
            print(f"[RCA:STRUCTURED]  No lineID column specified in schema")

        # Drop the structural columns
        if columns_to_drop:
            working_df = working_df.drop(columns=columns_to_drop)
        else:
            print(f"[RCA:STRUCTURED]  No structural columns to drop")

        # Show before/after column summary
        remaining_columns = list(working_df.columns)

        # Analyze remaining column types for tokenization
        numerical_cols = []
        categorical_cols = []

        for col in remaining_columns:
            if col == 'anomaly_label':  # Skip the target column
                continue

            if working_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)


        return working_df

    def _drop_result_columns(self, log_structure, errors_df):
        """
        Removes specified result columns (e.g., anomaly_label) from the provided DataFrame based on the log structure schema.

        Args:
            log_structure (dict): Dictionary containing schema information, including keys 'result_columns' specifying column names to drop.
            errors_df (pd.DataFrame): Input DataFrame containing error log data.

        Returns:
            pd.DataFrame: A copy of the input DataFrame with the specified result columns removed.

        Side Effects:
            Prints informative messages about the presence and removal of result columns.
        """
        detect_approach = log_structure.get("detection_approach", "")

        if detect_approach == "pure_heuristic":
            columns = ["anomaly_label", "heuristic_anomaly"]
        elif detect_approach == "hybrid_ml_heuristic":
            columns = ["anomaly_label", "ml_anomaly_label", "heuristic_anomaly", "anomaly_score"]
        elif detect_approach == "pure_ml":
            columns = ["anomaly_label", "ml_anomaly_label", "anomaly_score"]
        else:
            print(f"[RCA:STRUCTURED] ⚠ Unknown detection approach '{detect_approach}', no result columns to drop")

        # Drop the result columns
        working_df = errors_df.drop(columns=columns, errors='ignore')
        return working_df

    def _dynamic_bin_count(self, series, min_bins=3, max_bins=20):
        """
        Calculates the optimal number of bins for histogramming a pandas Series using the Freedman–Diaconis rule.

        Parameters:
            series (pd.Series): The data series to be binned.
            min_bins (int, optional): Minimum number of bins to use. Default is 3.
            max_bins (int, optional): Maximum number of bins to use. Default is 20.

        Returns:
            int: The computed number of bins, constrained between min_bins and max_bins.

        Notes:
            - If the number of unique values in the series is less than or equal to min_bins, returns the number of unique values.
            - Uses the Freedman–Diaconis rule to determine bin width, which is robust to outliers.
            - If the interquartile range (IQR) is zero, returns min_bins.
        """
        # Clean the series and check for edge cases
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return min_bins
        
        unique_vals = clean_series.nunique()
        
        # If we have very few unique values, just use them directly
        if unique_vals <= min_bins:
            return max(1, unique_vals)  # Ensure at least 1 bin
        
        # If all values are the same, return 1 bin
        if clean_series.min() == clean_series.max():
            return 1
        
        try:
            # Freedman–Diaconis rule
            q75, q25 = np.percentile(clean_series, [75, 25])
            iqr = q75 - q25
            
            # If IQR is 0 (all values between Q1 and Q3 are the same), fall back to range-based binning
            if iqr <= 0:
                data_range = clean_series.max() - clean_series.min()
                if data_range == 0:
                    return 1
                # Simple rule: use sqrt of unique values as bin count
                num_bins = min(max_bins, max(min_bins, int(np.sqrt(unique_vals))))
            else:
                bin_width = 2 * iqr * (len(clean_series) ** (-1/3))
                if bin_width <= 0:
                    return min_bins
                
                data_range = clean_series.max() - clean_series.min()
                num_bins = int(np.ceil(data_range / bin_width))
            
            # Ensure the result is within bounds and positive
            final_bins = max(1, min(max_bins, max(min_bins, num_bins)))
            return final_bins
            
        except Exception as e:
            print(f"[RCA:STRUCTURED] ⚠ Error calculating bins for column, using default: {e}")
            return min_bins

    def _tokenize_dataframe_dynamic(self, df):
        """
        Tokenizes a DataFrame by converting numerical columns to binned ranges and categorical columns to prefixed values.
        
        Args:
            df (pd.DataFrame): Input DataFrame to tokenize
            
        Returns:
            pd.Series: Series where each row is a list of tokens representing that row's values
        """
        tokenized_df = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Handle numerical columns with binning
                    clean_col = df[col].dropna()
                    
                    if len(clean_col) == 0:
                        # All NaN values
                        tokenized_df[col] = df[col].apply(lambda x: f"{col}_NaN")
                    elif clean_col.nunique() == 1:
                        # All same value
                        val = clean_col.iloc[0]
                        tokenized_df[col] = df[col].apply(lambda x: f"{col}_{val}" if pd.notna(x) else f"{col}_NaN")
                    else:
                        # Multiple values - apply binning
                        bins = self._dynamic_bin_count(clean_col)
                        
                        try:
                            binned = pd.cut(df[col], bins=bins, duplicates='drop')
                            
                            def format_interval(interval_str):
                                if pd.isna(interval_str) or str(interval_str) == 'nan':
                                    return "NaN"
                                # Example input: "(616.333, 717.667]"
                                # Strip brackets and parentheses
                                stripped = str(interval_str).strip("()[]")
                                # Split on comma and strip spaces
                                try:
                                    left_str, right_str = map(str.strip, stripped.split(","))
                                    # Convert to floats and round to 2 decimals
                                    left = round(float(left_str), 2)
                                    right = round(float(right_str), 2)
                                    # Format as "left-right"
                                    return f"{left}-{right}"
                                except (ValueError, IndexError):
                                    return str(interval_str)
                            
                            tokenized_df[col] = binned.astype(str).apply(
                                lambda x: f"{col}_{format_interval(x)}"
                            )
                            
                        except ValueError as e:
                            print(f"[RCA:STRUCTURED] ⚠ Binning failed for column '{col}': {e}")
                            # FIXED: Ensure fallback includes column prefix
                            tokenized_df[col] = df[col].astype(str).apply(lambda x: f"{col}_{x}" if pd.notna(x) else f"{col}_NaN")
                            
                else:
                    # Handle categorical columns - FIXED: Ensure proper prefixing
                    tokenized_df[col] = df[col].astype(str).apply(lambda x: f"{col}_{x}" if pd.notna(x) else f"{col}_NaN")
                    
            except Exception as e:
                print(f"[RCA:STRUCTURED] ⚠ Error tokenizing column '{col}': {e}")
                # FIXED: Ensure fallback always includes column prefix
                tokenized_df[col] = df[col].astype(str).apply(lambda x: f"{col}_{x}" if pd.notna(x) else f"{col}_NaN")
        
        # Convert each row to a list of tokens
        try:
            result = tokenized_df.apply(list, axis=1)
            return result
        except Exception as e:
            print(f"[RCA:STRUCTURED] ⚠ Error converting to token lists: {e}")
            # Return empty series as fallback
            return pd.Series([[] for _ in range(len(df))], index=df.index)

    def _extract_error_windows(self, original_df, log_structure, window_size_before=11):
        """
        Extract tokenized windows that contain `window_size_before` events immediately
        BEFORE each error (the error row itself is NOT included).

        - Sorts original_df by lineID (preferred) or timestamp.
        - Uses only full windows (skips errors that don't have enough history).
        - Drops structural/result columns once, tokenizes once for efficiency.
        - Returns: list of pd.Series where each Series is tokenized rows for that window.
        """
        # Step 0: validate argument
        if window_size_before <= 0:
            raise ValueError("window_size_before must be > 0")

        # Choose sort column
        sort_col = log_structure.get("lineID_column") or log_structure.get("timestamp_column")

        # Step 1: Sort original_df by timestamp or lineID and reset index so slicing is stable
        if sort_col and sort_col in original_df.columns:
            df_sorted = original_df.sort_values(by=sort_col).reset_index(drop=True)
        else:
            # keep order if no sort column available
            df_sorted = original_df.reset_index(drop=True)

        # Step 2: find error indices in the sorted DF
        error_indices = df_sorted.index[df_sorted['anomaly_label'] == 1].tolist()

        # Step 3: Drop structural & result columns once from the entire sorted df
        df_cleaned = self._drop_structural_columns(log_structure, df_sorted.copy())
        df_cleaned = self._drop_result_columns(log_structure, df_cleaned)

        # Step 4: Tokenize the cleaned df once
        tokenized_df = self._tokenize_dataframe_dynamic(df_cleaned)

        windows = []
        skipped = 0
        for idx in error_indices:
            start = idx - window_size_before
            end = idx  # exclude the error row itself (pre-error only)

            # require a full pre-error window
            if start < 0 or (end - start) != window_size_before:
                skipped += 1
                continue

            # slice tokenized series (preserves original temporal order)
            window = tokenized_df.iloc[start:end]
            windows.append(window)

        if skipped:
            print(f"[RCA:STRUCTURED] ⚠ Skipped {skipped} error(s) without enough pre-history for a full {window_size_before}-row window")

        print(f"[RCA:STRUCTURED] ✓ Extracted {len(windows)} pre-error windows (each {window_size_before} rows)")

        return windows

    def _extract_raw_error_windows(self, original_df, log_structure, window_size=11):
        """
        Extract raw numerical values in temporal windows before errors for trend analysis.
        Returns windows with original numerical values for slope calculation.
        """
        half_window = window_size // 2
        window_size_before = window_size  # Full window before error

        # Sort by lineID (preferred) or timestamp
        sort_col = log_structure.get("lineID_column") or log_structure.get("timestamp_column")
        df_sorted = original_df.sort_values(by=sort_col).reset_index(drop=True)

        # Identify error indices
        error_indices = df_sorted.index[df_sorted['anomaly_label'] == 1].tolist()

        # Get numerical columns only (for trend analysis)
        numerical_cols = []
        for col in df_sorted.columns:
            if pd.api.types.is_numeric_dtype(df_sorted[col]) and col not in ['anomaly_label', 'LineId']:
                if log_structure.get("timestamp_column") != col and log_structure.get("lineID_column") != col:
                    numerical_cols.append(col)

        raw_windows = []
        for idx in error_indices:
            start = idx - window_size_before
            end = idx  # exclude the error row itself

            if start < 0:
                continue  # Skip if not enough history

            # Extract raw numerical values for this window
            window_data = df_sorted.iloc[start:end][numerical_cols].copy()
            raw_windows.append(window_data)

        return raw_windows

    def _calculate_feature_slopes(self, raw_windows, log_structure):
        """
        For each error window, calculate the slope (trend) of each numerical feature.
        Fits a linear trend line to understand if features are rising, falling, or stable before errors.
        """
        slopes_data = []
        
        for window_idx, window in enumerate(raw_windows):
            window_slopes = {}
            
            # For each numerical column in the window
            for col in window.columns:
                values = window[col].values
                time_points = np.arange(len(values))  # 0, 1, 2, ..., window_size-1
                
                # Fit linear regression: values = slope * time + intercept
                if len(values) > 1 and not np.all(np.isnan(values)):
                    try:
                        slope, intercept = np.polyfit(time_points, values, 1)
                        window_slopes[f'{col}_slope'] = slope
                        window_slopes[f'{col}_final_value'] = values[-1]  # Last value in window
                        window_slopes[f'{col}_change_magnitude'] = abs(values[-1] - values[0]) if len(values) > 0 else 0
                        window_slopes[f'{col}_mean_value'] = np.mean(values)
                    except (np.linalg.LinAlgError, ValueError):
                        # Handle edge cases where polyfit fails
                        window_slopes[f'{col}_slope'] = 0
                        window_slopes[f'{col}_final_value'] = values[-1] if len(values) > 0 else np.nan
                        window_slopes[f'{col}_change_magnitude'] = 0
                        window_slopes[f'{col}_mean_value'] = np.mean(values) if len(values) > 0 else np.nan
                else:
                    window_slopes[f'{col}_slope'] = 0
                    window_slopes[f'{col}_final_value'] = np.nan
                    window_slopes[f'{col}_change_magnitude'] = 0
                    window_slopes[f'{col}_mean_value'] = np.nan
            
            slopes_data.append(window_slopes)
        
        return pd.DataFrame(slopes_data)

    def _classify_trends(self, slopes_df, threshold=0.05):
        """
        Classify feature trends as Rising, Falling, or Stable based on slope values.
        """
        trend_patterns = {}
        
        for col in slopes_df.columns:
            if col.endswith('_slope'):
                feature_name = col.replace('_slope', '')
                slopes = slopes_df[col].dropna()  # Remove NaN slopes
                
                # Classify each slope
                trends = []
                for slope in slopes:
                    if slope > threshold:
                        trends.append(f'{feature_name}_RISING')
                    elif slope < -threshold:
                        trends.append(f'{feature_name}_FALLING') 
                    else:
                        trends.append(f'{feature_name}_STABLE')
                
                trend_patterns[feature_name] = trends
        
        return trend_patterns

    def _mine_trend_patterns(self, trend_patterns):
        """
        Find frequent combinations of trends that appear before errors.
        """
        from collections import Counter
        
        if not trend_patterns:
            return []
            
        pattern_counter = Counter()
        
        # Get the number of windows (should be same for all features)
        num_windows = len(next(iter(trend_patterns.values())))
        
        for window_idx in range(num_windows):
            # Get all trends for this window
            window_trends = []
            for feature, trends in trend_patterns.items():
                if window_idx < len(trends):  # Safety check
                    window_trends.append(trends[window_idx])
            
            # Create pattern signature (sorted for consistency)
            if window_trends:
                pattern = ' + '.join(sorted(window_trends))
                pattern_counter[pattern] += 1
        
        return pattern_counter.most_common(10)

    def _generate_forecasting_rules(self, slopes_df):
        """
        Generate actionable forecasting rules based on trend analysis.
        """
        rules = []
        
        for col in slopes_df.columns:
            if col.endswith('_slope'):
                feature_name = col.replace('_slope', '')
                slopes = slopes_df[col].dropna()  # Remove NaN values
                
                if len(slopes) == 0:
                    continue
                    
                mean_slope = slopes.mean()
                std_slope = slopes.std()
                abs_mean_slope = abs(mean_slope)
                
                # Only create rules for significant slopes (more lenient thresholds for real-world data)
                # Use magnitude check OR ratio check (not both) to capture significant trends
                magnitude_check = abs_mean_slope > 0.5  # Absolute magnitude threshold
                ratio_check = std_slope == 0 or abs_mean_slope > 0.5 * std_slope  # Ratio check with division by zero protection
                threshold_check = magnitude_check and ratio_check
                
                if threshold_check:
                    direction = "increases" if mean_slope > 0 else "decreases"
                    confidence = 'High' if std_slope < 0.3 * abs_mean_slope else 'Medium'
                    
                    rule = {
                        'feature': feature_name,
                        'pattern': 'RISING_TREND' if mean_slope > 0 else 'FALLING_TREND',
                        'slope_threshold': abs_mean_slope,
                        'forecast_rule': f"Alert when {feature_name} {direction} by >{abs_mean_slope:.3f} per time unit over 11 time steps",
                        'confidence': confidence,
                        'mean_slope': mean_slope
                    }
                    rules.append(rule)
        
        # Sort by absolute slope magnitude (most significant trends first)
        rules.sort(key=lambda x: abs(x['mean_slope']), reverse=True)
        return rules

    def _extract_normal_windows(self, original_df, log_structure, window_size=11):
        half_window = window_size // 2

        # Sort by lineID (preferred) or timestamp
        sort_col = log_structure.get("lineID_column") or log_structure.get("timestamp_column")
        df_sorted = original_df.sort_values(by=sort_col).reset_index(drop=True)

        # Clean entire df once (drop structural/result columns and tokenize)
        df_cleaned = self._drop_structural_columns(log_structure, df_sorted)
        df_cleaned = self._drop_result_columns(log_structure, df_cleaned)
        tokenized_df = self._tokenize_dataframe_dynamic(df_cleaned)

        normal_windows = []
        n = len(df_sorted)

        # We'll slide a window of size window_size across the dataset,
        # but only keep windows where all events have anomaly_label == 0
        for start_idx in range(n - window_size + 1):
            end_idx = start_idx + window_size
            window_slice = df_sorted.iloc[start_idx:end_idx]

            # Check if window contains any error events
            if window_slice['anomaly_label'].any():
                continue  # skip windows with errors

            # If clean, extract corresponding tokens window
            window_tokens = tokenized_df.iloc[start_idx:end_idx]
            normal_windows.append(window_tokens)

        return normal_windows

    def _vectorize_windows(self, error_windows, normal_windows):
        """
        Converts lists of error and normal windows into vectorized feature representations and labels for machine learning.
        Each window is flattened into a space-joined string of tokens, then vectorized using a CountVectorizer with simple space-based tokenization.
        The resulting feature matrix represents the presence or absence of tokens in each window.
        Args:
            error_windows (list): List of error windows, where each window is a list of token rows.
            normal_windows (list): List of normal windows, where each window is a list of token rows.
        Returns:
            X_dense (np.ndarray): Dense feature matrix of shape (num_windows, num_tokens).
            y (np.ndarray): Array of labels (1 for error windows, 0 for normal windows).
            vectorizer (CountVectorizer): Fitted CountVectorizer instance used for tokenization and vectorization.
        """
        # Flatten tokens in each window to a space-joined string (for CountVectorizer)
        def window_to_string(window):
            return ' '.join(token for row in window for token in row)

        all_windows = error_windows + normal_windows
        window_strings = [window_to_string(w) for w in all_windows]

        # FIXED: Use simple split-based tokenization to preserve full tokens
        # This prevents any regex-based splitting that breaks our carefully crafted tokens
        vectorizer = CountVectorizer(
            binary=True, 
            analyzer='word',
            token_pattern=None,  # Disable regex pattern matching
            tokenizer=lambda x: x.split()  # Simple space-based splitting
        )
        X = vectorizer.fit_transform(window_strings)  # sparse matrix shape: (num_windows, num_tokens)
        
        # Create labels: 1 for error windows, 0 for normal windows
        y = np.array([1]*len(error_windows) + [0]*len(normal_windows))

        # Optionally convert sparse to dense (if memory allows)
        X_dense = X.toarray()

        return X_dense, y, vectorizer

    def _run_ran_for(self, X, y):
        # 1. Split data into train and test sets (stratify to keep class distribution)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 2. Initialize and train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 3. Predict on test set
        y_pred = model.predict(X_test)

        # 4. Print classification metrics
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def _plot_feature_importances(self, X, y, feature_names=None):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        importances = clf.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feat_{i}' for i in range(X.shape[1])]
        
        indices = np.argsort(importances)[::-1][:10]  # top 10 features

        # Print top 20 features with importance
        print("\nTop 10 Feature Importances:")
        for idx in indices:
            print(f"{feature_names[idx]}: {importances[idx]:.6f}")
        
        plt.figure(figsize=(10,6))
        plt.title("Top 10 Feature Importances")
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

        return feature_names
        
    def _plot_binary_activation_sequence(self, X, y, feature_names, top_n=10):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in top_idx]
        
        X_top = pd.DataFrame(X[:, top_idx], columns=top_features)
        
        # Find last time step with any activity
        last_active_col = np.max(np.where(X_top.any(axis=1))[0])
        X_top_trimmed = X_top.iloc[:last_active_col+1, :]
        y_trimmed = y[:last_active_col+1]
        
        plt.figure(figsize=(14, 6))
        sns.heatmap(
            X_top_trimmed.T, 
            cmap=sns.color_palette(["#2166ac", "#b2182b"]),
            cbar=False,
            linewidths=0.5,
            linecolor="lightgray"
        )
        
        plt.yticks(rotation=0)
        plt.xticks(
            ticks=np.arange(0, X_top_trimmed.shape[0], max(1, X_top_trimmed.shape[0] // 15)),
            labels=np.arange(0, X_top_trimmed.shape[0], max(1, X_top_trimmed.shape[0] // 15)),
            rotation=45
        )
        plt.xlabel("Time window index")
        plt.ylabel("Top features")
        plt.title("Binary Activation Sequence of Top Features (Trimmed)")
        
        anomaly_idx = np.where(y_trimmed == 1)[0]
        for idx in anomaly_idx:
            plt.axvline(idx, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _analyze_feature_bins(self, X, y, feature_names, top_n=10, cooccur_top_k=3):
        """
        Analyze feature bins for their correlation with errors and print human-readable insights.

        Args:
            X (np.ndarray or pd.DataFrame): Feature matrix (windows x features), binary 0/1.
            y (np.ndarray): Labels array (1=error window, 0=normal).
            feature_names (list): List of feature bin names, length = number of columns in X.
            top_n (int): Number of top features by feature importance to analyze.
            cooccur_top_k (int): Number of top features to analyze co-occurrence for.

        Returns:
            pd.DataFrame: Summary table with frequency and likelihood ratio info for top_n features.
        """

        # Convert to DataFrame for convenience if needed
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()

        df['error'] = y

        # Calculate feature importances with Random Forest to get top features
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        importances = clf.feature_importances_

        # Sort features by importance descending
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        top_features = importance_df.head(top_n)['feature'].tolist()

        # Calculate frequencies and likelihood ratios for each top feature bin
        summary = []
        for feat in top_features:
            freq_in_error = df[df['error'] == 1][feat].mean()
            freq_in_normal = df[df['error'] == 0][feat].mean()
            freq_in_normal = max(freq_in_normal, 1e-6)  # avoid div zero
            likelihood_ratio = freq_in_error / freq_in_normal

            summary.append({
                'feature': feat,
                'importance': importance_df[importance_df['feature'] == feat]['importance'].values[0],
                'freq_in_error': freq_in_error,
                'freq_in_normal': freq_in_normal,
                'likelihood_ratio': likelihood_ratio
            })

        summary_df = pd.DataFrame(summary).sort_values(by='likelihood_ratio', ascending=False)

        # Print single-feature rules
        # print("\n-- Single Feature Analysis --")
        # for _, row in summary_df.iterrows():
        #     print(f"* Feature '{row.feature}':")
        #     print(f"  - Occurs in {row.freq_in_error:.2%} of error windows")
        #     print(f"  - Occurs in {row.freq_in_normal:.2%} of normal windows")
        #     print(f"  - Likelihood ratio (error/normal): {row.likelihood_ratio:.2f}")
        #     if row.likelihood_ratio > 3:
        #         print("  ==> Strong indicator of errors.")
        #     elif row.likelihood_ratio > 1.5:
        #         print("  ==> Moderate indicator of errors.")
        #     else:
        #         print("  ==> Weak indicator or common in normal windows.")
        #     print()

        # Check co-occurrence of top K features
        print(f"\n-- Co-occurrence Analysis for Top {cooccur_top_k} Features --")
        if cooccur_top_k > len(top_features):
            cooccur_top_k = len(top_features)
        top_k_feats = top_features[:cooccur_top_k]

        cooccur_in_error = df[(df['error'] == 1) & (df[top_k_feats].all(axis=1))].shape[0] / max(df[df['error'] == 1].shape[0], 1)
        cooccur_in_normal = df[(df['error'] == 0) & (df[top_k_feats].all(axis=1))].shape[0] / max(df[df['error'] == 0].shape[0], 1)
        cooccur_in_normal = max(cooccur_in_normal, 1e-6)
        cooccur_lr = cooccur_in_error / cooccur_in_normal

        print(f"When all bins {top_k_feats} are active together:")
        print(f"  - Occurs in {cooccur_in_error:.2%} of error windows")
        print(f"  - Occurs in {cooccur_in_normal:.2%} of normal windows")
        print(f"  - Likelihood ratio: {cooccur_lr:.2f}")

        if cooccur_lr > 3:
            print("  ==> Strong combined predictor of errors.")
        elif cooccur_lr > 1.5:
            print("  ==> Moderate combined predictor of errors.")
        else:
            print("  ==> Weak combined predictor.")

        print("\nSummary table returned for further analysis or reporting.")

        return summary_df
















    #FOR FRONTEND STREAMING TO DEMO APP
    def prepare_stream_events(self, results, original_df, forecast_rules):
        """
        Convert RCA results + original data into a chronological list of events
        with predictions and RCA messages.
        """

        print(forecast_rules)

        events = []
        # Sort by timestamp if you have one
        if "timestamp" in original_df.columns:
            df = original_df.sort_values("timestamp").reset_index(drop=True)
        else:
            df = original_df.reset_index(drop=True)

        for idx, row in df.iterrows():
            # Extract feature values (replace with your feature names)
            features = {
                "response_time": row.get("response_time", None),
                "memory_usage": row.get("memory_usage", None),
                "thread_count": row.get("thread_count", None)
            }

            # Check if any forecasting rule triggers
            prediction = None
            for rule in forecast_rules:
                if self._matches_rule(row, rule):
                    prediction = "alarm"
                    break

            # Check if it's an actual error
            actual_error = row.get("anomaly_label", 0) == 1

            # Build RCA message if prediction triggered
            rca_message = None
            if prediction == "alarm":
                top_feature = rule["feature"]
                rca_message = f"High slope on {top_feature} ({rule['forecast_rule']}) indicates likely error"

            events.append({
                "timestamp": row.get("timestamp", idx),
                "memory_usage": row.get("memory_usage_mb", None),
                "thread_count": row.get("thread_count", None),
                "response_time_ms": row.get("response_time_ms", None),
                "alarm_triggered": prediction == "alarm",
                "rca_message": rca_message,
                "forecast_rule": rule["forecast_rule"] if prediction == "alarm" else None,
                "alarm_feature": rule["feature"] if prediction == "alarm" else None
            })

        return events
    
    def _matches_rule(self, row, rule):
        """Simple placeholder check for matching a forecasting rule."""
        # Example: if slope or bin number > threshold
        feature_val = row.get(rule["feature"], None)
        print(f"Checking {rule['feature']} value {row.get(rule['feature'])} against threshold {rule.get('threshold')}")
        if feature_val is None:
            return False
        # Very naive match logic — replace with your actual bin/slope logic
        return feature_val > rule.get("threshold", float("inf"))

    def stream_events(self, events, endpoint_url, delay=0.5):
        """
        Sends events one-by-one to a backend endpoint at a fixed delay.
        """
        for event in events:
            try:
                requests.post(endpoint_url, json=event)
            except Exception as e:
                print(f"Failed to send event: {e}")
            time.sleep(delay)




# next thing to do is the tokenizing logic with the dataset 
# in absent route ( lineID and timestamp) column dropping 
# logic has been implmented