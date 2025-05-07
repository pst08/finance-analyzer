import pandas as pd
import numpy as np
import chardet
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules
import logging
import warnings
import traceback
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Add a custom JSON encoder to handle NaN values
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

# Function to recursively replace NaN with None
def replace_nan_with_none(obj):
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(item) for item in obj]
    elif isinstance(obj, (float, np.float64, np.float32)) and np.isnan(obj):
        return None
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    else:
        return obj

def clean_data(df):
    """
    Function to clean data by handling missing values.
    Numeric columns are filled with the mean, categorical columns are filled with mode.
    """
    try:
        logger.info(f"Cleaning data with {len(df)} rows")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Check if dataframe is empty
        if len(df) == 0:
            logger.warning("Empty dataframe provided to clean_data")
            return df
            
        # Convert all columns to appropriate types
        # Attempt to convert string numbers to float/int
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass  # Keep as object if conversion fails
        
        # Handle missing numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            if df[col].isna().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)  # Use median instead of mean for robustness
                logger.info(f"Filled {df[col].isna().sum()} NaN values in column {col} with median {median_value}")

        # Handle missing categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isna().sum() > 0:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled {df[col].isna().sum()} NaN values in column {col} with mode {mode_value}")

        # Log NaN counts for each column to help diagnose issues
        nan_counts = df.isna().sum()
        columns_with_nans = nan_counts[nan_counts > 0]
        if not columns_with_nans.empty:
            logger.warning(f"Columns with remaining NaNs: {columns_with_nans.to_dict()}")
        
        # MODIFIED: Only drop rows if we still have NaNs after filling
        # First check if there are any NaNs left
        if df.isna().any().any():
            rows_before = len(df)
            # Only drop rows with a high percentage of NaN values (more than 50%)
            # This ensures we don't lose all data due to a few missing values
            threshold = len(df.columns) * 0.5
            df = df.dropna(thresh=threshold)
            rows_after = len(df)
            if rows_before > rows_after:
                logger.warning(f"Dropped {rows_before - rows_after} rows with more than 50% NaN values")
                # If we've dropped more than 90% of our data, something is wrong
                if rows_after < rows_before * 0.1:
                    logger.error(f"WARNING: Dropped more than 90% of data rows. Check data quality and delimiter settings!")
        
        return df
    except Exception as e:
        logger.error(f"Error in clean_data: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def compute_basic_insights(df):
    """
    Compute basic descriptive statistics and insights from the dataset
    """
    try:
        logger.info("Computing basic insights")
        insights = {}
        
        # Check relevant columns exist
        required_cols = ['Income', 'Groceries', 'Entertainment', 'Eating_Out']
        for col in required_cols:
            if col not in df.columns:
                return {'error': f"Required column '{col}' not found in dataset"}
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Basic statistics
        insights['total_records'] = len(df)
        
        # Descriptive statistics for Income
        if 'Income' in df.columns:
            insights['avg_income'] = float(df['Income'].mean())
            insights['median_income'] = float(df['Income'].median())
            insights['min_income'] = float(df['Income'].min())
            insights['max_income'] = float(df['Income'].max())
            
            # Calculate total income and spending
            total_income = float(df['Income'].sum())
            spending_cols = [col for col in numeric_cols if col not in ['Income', 'Desired_Savings', 'Disposable_Income']]
            total_spent = float(df[spending_cols].sum().sum())
            
            insights.update({
                'total_income': total_income,
                'total_spent': total_spent,
                'potential_savings': total_income - total_spent,
            })
        
        # Most and least spent categories
        expense_cols = [col for col in numeric_cols if col not in ['Income', 'Age', 'Dependents', 'Desired_Savings_Percentage', 'Desired_Savings', 'Disposable_Income']]
        
        if expense_cols:
            expense_totals = df[expense_cols].sum().sort_values(ascending=False)
            insights['expense_breakdown'] = {col: float(expense_totals[col]) for col in expense_totals.index[:5]}
            insights['most_spent_category'] = expense_totals.index[0]
            insights['least_spent_category'] = expense_totals.index[-1]
        
        # Average spending by age group if Age column exists
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '56+'])
            age_spending = df.groupby('Age_Group')['Income'].mean().to_dict()
            insights['avg_income_by_age'] = {str(k): float(v) for k, v in age_spending.items()}
        
        # Replace NaN with None
        insights = replace_nan_with_none(insights)
        return insights
    except Exception as e:
        logger.error(f"Error in compute_basic_insights: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': str(e)}

def cluster_users(df, n_clusters=3):
    """
    Clustering users into spending clusters using KMeans.
    Uses sampling for large datasets to speed up processing.
    """
    try:
        logger.info(f"Clustering {len(df)} users with KMeans (k={n_clusters})")
        
        # Select relevant features for clustering
        features = ['Income', 'Rent', 'Groceries', 'Entertainment', 'Eating_Out', 'Healthcare']
        
        # Check if all features exist
        missing_features = [col for col in features if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing clustering features: {missing_features}")
            # Use available features
            features = [col for col in features if col in df.columns]
            
        if not features:
            return df, {"error": "No valid features for clustering"}
        
        # Get the data for clustering
        X = df[features].copy()
        
        # Handle any remaining NaNs
        X.fillna(X.median(), inplace=True)
        
        # Normalize data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # For large datasets, use sample to speed up clustering
        sample_size = min(10000, len(df))  # Max 10k samples
        if len(df) > sample_size:
            logger.info(f"Using {sample_size} samples for clustering")
            indices = np.random.choice(len(df), sample_size, replace=False)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X_scaled[indices])
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
        
        # Predict clusters for all data
        df['Spending_Cluster'] = kmeans.predict(X_scaled)
        
        # Get stats about each cluster
        cluster_stats = {}
        for cluster in range(n_clusters):
            cluster_df = df[df['Spending_Cluster'] == cluster]
            cluster_stats[f'Cluster_{cluster}'] = {
                'count': int(len(cluster_df)),
                'pct': float(len(cluster_df) / len(df) * 100),
                'avg_income': float(cluster_df['Income'].mean()) if 'Income' in df.columns else 0,
                'avg_spending': float(cluster_df[features[1:]].sum(axis=1).mean()) if len(features) > 1 else 0
            }
        
        # Replace NaN with None
        cluster_stats = replace_nan_with_none(cluster_stats)
        return df, cluster_stats
    except Exception as e:
        logger.error(f"Error in cluster_users: {str(e)}")
        logger.error(traceback.format_exc())
        return df, {"error": str(e)}

def detect_anomalies(df, contamination=0.02):
    """
    Anomaly detection using Isolation Forest.
    Uses sampling for large datasets to speed up processing.
    """
    try:
        logger.info(f"Detecting anomalies with contamination rate {contamination}")
        
        # Select relevant features for anomaly detection
        features = ['Groceries', 'Eating_Out', 'Entertainment', 'Miscellaneous']
        
        # Check if all required features exist
        missing_features = [col for col in features if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing anomaly detection features: {missing_features}")
            # Use available features
            features = [col for col in features if col in df.columns]
            
        if not features:
            return df, {"error": "No valid features for anomaly detection"}
            
        # Get data for anomaly detection
        X = df[features].copy()
        
        # Handle any remaining NaNs
        X.fillna(X.median(), inplace=True)
        
        # For large datasets, use sample to fit the model
        sample_size = min(10000, len(df))  # Max 10k samples
        
        # Adjust contamination for sample size
        adjusted_contamination = contamination
        if len(df) > sample_size:
            # Scale contamination to sample size
            adjusted_contamination = contamination * (len(df) / sample_size)
            adjusted_contamination = min(0.5, adjusted_contamination)  # Cap at 0.5
            logger.info(f"Using adjusted contamination of {adjusted_contamination} for sample")
            
            # Use sample for fitting
            indices = np.random.choice(len(df), sample_size, replace=False)
            iso = IsolationForest(contamination=adjusted_contamination, random_state=42, n_jobs=-1)
            iso.fit(X.iloc[indices])
        else:
            iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            iso.fit(X)
        
        # Predict for all data
        anomalies = iso.predict(X)
        df['Anomaly'] = anomalies  # 1 for normal, -1 for anomaly
        
        # Get summary of anomalies
        anomaly_count = df['Anomaly'].value_counts().to_dict()
        
        # Convert to more readable format
        anomaly_summary = {
            'normal_transactions': int(anomaly_count.get(1, 0)),
            'anomalous_transactions': int(anomaly_count.get(-1, 0)),
            'anomaly_percentage': float(anomaly_count.get(-1, 0) / len(df) * 100)
        }
        
        # Get average values for anomalous vs normal transactions
        if -1 in df['Anomaly'].values:
            normal_df = df[df['Anomaly'] == 1]
            anomaly_df = df[df['Anomaly'] == -1]
            
            anomaly_summary['avg_values'] = {
                'normal': {feat: float(normal_df[feat].mean()) for feat in features},
                'anomalous': {feat: float(anomaly_df[feat].mean()) for feat in features}
            }
        
        # Replace NaN with None
        anomaly_summary = replace_nan_with_none(anomaly_summary)
        return df, anomaly_summary
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {str(e)}")
        logger.error(traceback.format_exc())
        return df, {"error": str(e)}

def generate_recommendations(df):
    """
    Generate savings recommendations using linear regression and association rule mining.
    Uses sampling for association rule mining to handle large datasets.
    """
    try:
        logger.info("Generating recommendations")
        recommendations = {}
        
        # Check if required columns exist
        required_cols = ['Income', 'Desired_Savings']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Missing columns for recommendations: {missing_cols}"}
        
        # Savings regression prediction
        try:
            reg_features = ['Income', 'Rent', 'Loan_Repayment', 'Utilities']
            
            # Use available features
            available_features = [col for col in reg_features if col in df.columns]
            if len(available_features) > 0:
                X = df[available_features].fillna(0)
                y = df['Desired_Savings'].fillna(0)
                
                # Fit regression model
                reg = LinearRegression()
                reg.fit(X, y)
                
                # Make predictions
                df['Predicted_Savings'] = reg.predict(X)
                
                # Calculate potential savings
                recommendations['avg_predicted_savings'] = float(df['Predicted_Savings'].mean())
                
                # Feature importance (coefficients)
                coef_dict = dict(zip(available_features, reg.coef_))
                recommendations['savings_factors'] = {k: float(v) for k, v in coef_dict.items()}
            else:
                recommendations['regression_error'] = "Insufficient features for regression"
        except Exception as reg_error:
            logger.error(f"Error in regression: {str(reg_error)}")
            recommendations['regression_error'] = str(reg_error)
        
        # Association rule mining on a sample
        try:
            # For large datasets, sample to avoid memory issues
            sample_size = min(5000, len(df))
            if len(df) > sample_size:
                logger.info(f"Using {sample_size} samples for association rules")
                rule_df = df.sample(sample_size, random_state=42)
            else:
                rule_df = df
                
            # Create binary features
            savings_mean = rule_df['Desired_Savings_Percentage'].mean() if 'Desired_Savings_Percentage' in rule_df.columns else 0
            rule_df['High_Saver'] = (rule_df.get('Desired_Savings_Percentage', 0) > savings_mean).astype(int)
            
            # Select spending columns to binarize
            spending_cols = [col for col in ['Eating_Out', 'Entertainment', 'Healthcare'] if col in rule_df.columns]
            
            # Create binary dataset
            binarized = pd.DataFrame(index=rule_df.index)
            binarized['High_Saver'] = rule_df['High_Saver']
            
            # Add binary spending features
            for col in spending_cols:
                col_mean = rule_df[col].mean()
                binarized[f'High_{col}'] = (rule_df[col] > col_mean).astype(int)
            
            # Apply Apriori with lower support for large datasets
            min_support = max(0.1, 10 / len(rule_df))  # Adaptive support
            frequent_items = apriori(binarized, min_support=min_support, use_colnames=True, max_len=3)
            
            # Generate rules if we have frequent itemsets
            if not frequent_items.empty:
                rules = association_rules(frequent_items, metric='lift', min_threshold=1.1)
                
                if not rules.empty:
                    # Get top 5 rules
                    top_rules = rules.sort_values('lift', ascending=False).head(5)
                    
                    readable_rules = []
                    for _, row in top_rules.iterrows():
                        readable_rules.append({
                            'antecedents': [str(x) for x in list(row['antecedents'])],
                            'consequents': [str(x) for x in list(row['consequents'])],
                            'support': float(row['support']),
                            'confidence': float(row['confidence']),
                            'lift': float(row['lift'])
                        })
                    
                    recommendations['association_rules'] = readable_rules
                else:
                    recommendations['association_rules'] = []
            else:
                recommendations['association_rules'] = []
                
        except Exception as rule_error:
            logger.error(f"Error in association rules: {str(rule_error)}")
            recommendations['rule_error'] = str(rule_error)
            
        # General spending reduction recommendations
        expense_cols = [col for col in df.columns if col.startswith('Potential_Savings_')]
        if expense_cols:
            potential_savings = df[expense_cols].mean().sort_values(ascending=False)
            recommendations['savings_opportunities'] = {
                col.replace('Potential_Savings_', ''): float(potential_savings[col]) 
                for col in potential_savings.index[:3]
            }
        
        # Replace NaN with None
        recommendations = replace_nan_with_none(recommendations)
        return recommendations
    except Exception as e:
        logger.error(f"Error in generate_recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def analyze_csv(filepath):
    """
    Main function to analyze the CSV data with robust error handling and parsing
    """
    try:
        logger.info(f"Starting analysis of {filepath}")
        
        # First, read the file directly since we already know it's a proper CSV with comma delimiter
        try:
            logger.info(f"Reading CSV with comma delimiter")
            df = pd.read_csv(
                filepath, 
                delimiter=',',  # Your sample data shows comma as the correct delimiter
                low_memory=False,
                on_bad_lines='warn'
            )
            
            # Make sure we actually read something
            if len(df) == 0:
                logger.warning("File appears to be empty or couldn't be parsed correctly")
                raise ValueError("Empty dataframe after reading CSV")
                
            logger.info(f"Successfully read {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Quick check for common issues
            if len(df.columns) == 1:
                logger.warning("Only one column detected. This might indicate a delimiter issue.")
        except Exception as e:
            logger.error(f"Error reading CSV with direct approach: {str(e)}")
            logger.info("Falling back to advanced detection...")
            
            # Try to detect the delimiter and encoding
            try:
                # Read a sample of the file to detect format
                with open(filepath, 'rb') as f:
                    sample = f.read(4096)
                    
                # Try to determine encoding
                detected = chardet.detect(sample)
                encoding = detected['encoding']
                confidence = detected['confidence']
                
                logger.info(f"Detected encoding: {encoding} with confidence: {confidence}")
                
                # If confidence is too low, default to utf-8
                if confidence < 0.7:
                    logger.warning(f"Low confidence in encoding detection ({confidence}), defaulting to utf-8")
                    encoding = 'utf-8'
                
                # Try comma first since our sample uses commas
                delimiter = ','
                logger.info(f"Trying known delimiter: {delimiter}")
                
                # Try with comma first, since that's what your data uses
                df = pd.read_csv(
                    filepath, 
                    delimiter=delimiter,
                    encoding=encoding,
                    low_memory=False,
                    on_bad_lines='warn'
                )
                
                # If successful, log it
                if len(df) > 0:
                    logger.info(f"Successfully read {len(df)} rows with delimiter={delimiter}, encoding={encoding}")
                else:
                    raise ValueError("Empty dataframe after reading with comma delimiter")
                    
            except Exception as e:
                logger.error(f"Error during advanced detection: {str(e)}")
                # Try again with more fallbacks
                
                # Expanding the list of encodings and delimiters to try
                encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252', 'ascii']
                delimiters = [',', ';', '\t', '|']
                
                success = False
                
                for enc in encodings:
                    for delim in delimiters:
                        try:
                            logger.info(f"Trying fallback: delimiter={delim}, encoding={enc}")
                            
                            df = pd.read_csv(
                                filepath, 
                                delimiter=delim,
                                encoding=enc,
                                low_memory=False,
                                on_bad_lines='warn'
                            )
                            
                            # If we have data and columns, mark as success
                            if len(df) > 0 and len(df.columns) > 1:  
                                logger.info(f"Successfully parsed with delimiter={delim}, encoding={enc}")
                                success = True
                                break
                        except Exception as specific_e:
                            logger.debug(f"Failed with delimiter={delim}, encoding={enc}: {str(specific_e)}")
                            continue
                    
                    if success:
                        break
                
                if not success:
                    raise ValueError("Unable to parse CSV file with any combination of delimiter and encoding.")
        
        # Clean data
        df = clean_data(df)
        
        # Ensure we have data left after cleaning
        if len(df) == 0:
            return {
                'status': 'error',
                'error': 'No data left after cleaning. Check data quality and cleaning parameters.',
                'original_row_count': len(pd.read_csv(filepath))
            }
        
        # Compute basic insights
        insights = compute_basic_insights(df)
        
        # Cluster users
        df, cluster_info = cluster_users(df)
        
        # Detect anomalies
        df, anomaly_info = detect_anomalies(df)
        
        # Generate recommendations
        recommendations = generate_recommendations(df)
        
        # Convert sample data to dict while fixing NaN values
        safe_sample_data = []
        for record in df.head(5).to_dict(orient='records'):
            safe_sample_data.append(replace_nan_with_none(record))
        
        # Prepare the result dictionary
        result = {
            'status': 'success',
            'row_count': len(df),
            'columns': df.columns.tolist(),
            'sample_data': safe_sample_data,
            'insights': insights,
            'clustering': cluster_info,
            'anomalies': anomaly_info,
            'recommendations': recommendations
        }
        
        # For debugging, log the types of values
        try:
            json_str = json.dumps(result, cls=NpEncoder)
            logger.info("Successfully serialized result to JSON")
            return json.loads(json_str)
        except Exception as json_error:
            logger.error(f"JSON serialization error: {str(json_error)}")
            # Last resort - try to fix it brute force
            return replace_nan_with_none(result)
        
    except Exception as e:
        logger.error(f"Error analyzing CSV: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'error': str(e),
            'trace': traceback.format_exc()
        }

def forecast_spending():
    """
    Forecast spending using a time series model (placeholder).
    """
    return {
        'status': 'info',
        'message': 'Forecasting module not yet implemented. Placeholder response.'
    }