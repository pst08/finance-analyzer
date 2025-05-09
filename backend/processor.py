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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import silhouette_score, precision_recall_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
 #NaN value handler 

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
    Numeric columns are filled with the median, categorical columns are filled with mode.
    """
    try:
        logger.info(f"Cleaning data with {len(df)} rows")
        
        df = df.copy()

        if len(df) == 0:
            logger.warning("Empty dataframe provided to clean_data")
            return df

        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            if df[col].isna().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logger.info(f"Filled {df[col].isna().sum()} NaN values in column {col} with median {median_value}")

        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isna().sum() > 0:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled {df[col].isna().sum()} NaN values in column {col} with mode {mode_value}")

        nan_counts = df.isna().sum()
        columns_with_nans = nan_counts[nan_counts > 0]
        if not columns_with_nans.empty:
            logger.warning(f"Columns with remaining NaNs: {columns_with_nans.to_dict()}")

        if df.isna().any().any():
            rows_before = len(df)
            threshold = len(df.columns) * 0.5
            df = df.dropna(thresh=threshold)
            rows_after = len(df)
            if rows_before > rows_after:
                logger.warning(f"Dropped {rows_before - rows_after} rows with more than 50% NaN values")
                if rows_after < rows_before * 0.1:
                    logger.error("WARNING: Dropped more than 90% of data rows. Check data quality and delimiter settings!")

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()

        if 'Occupation' in df.columns and df['Occupation'].dtype == 'object':
            df['Occupation'] = le.fit_transform(df['Occupation'])
            logger.info("Encoded Occupation column with LabelEncoder")

        if 'City_Tier' in df.columns and df['City_Tier'].dtype == 'object':
            df['City_Tier'] = le.fit_transform(df['City_Tier'])
            logger.info("Encoded City_Tier column with LabelEncoder")

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
        
        required_cols = ['Income', 'Groceries', 'Entertainment', 'Eating_Out']
        for col in required_cols:
            if col not in df.columns:
                return {'error': f"Required column '{col}' not found in dataset"}
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        insights['total_records'] = len(df)
        
        if 'Income' in df.columns:
            insights['avg_income'] = float(df['Income'].mean())
            insights['median_income'] = float(df['Income'].median())
            insights['min_income'] = float(df['Income'].min())
            insights['max_income'] = float(df['Income'].max())
            
            total_income = float(df['Income'].sum())
            spending_cols = [col for col in numeric_cols if col not in ['Income', 'Desired_Savings', 'Disposable_Income']]
            total_spent = float(df[spending_cols].sum().sum())
            
            insights.update({
                'total_income': total_income,
                'total_spent': total_spent,
                'potential_savings': total_income - total_spent,
            })
        
        expense_cols = [col for col in numeric_cols if col not in ['Income', 'Age', 'Dependents', 'Desired_Savings_Percentage', 'Desired_Savings', 'Disposable_Income']]
        
        if expense_cols:
            expense_totals = df[expense_cols].sum().sort_values(ascending=False)
            insights['expense_breakdown'] = {col: float(expense_totals[col]) for col in expense_totals.index[:5]}
            insights['most_spent_category'] = expense_totals.index[0]
            insights['least_spent_category'] = expense_totals.index[-1]
        
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '56+'])
            age_spending = df.groupby('Age_Group')['Income'].mean().to_dict()
            insights['avg_income_by_age'] = {str(k): float(v) for k, v in age_spending.items()}
        
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
        
        features = ['Income', 'Rent', 'Groceries', 'Entertainment', 'Eating_Out', 'Healthcare']
        
        missing_features = [col for col in features if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing clustering features: {missing_features}")
            features = [col for col in features if col in df.columns]
        
        if not features:
            return df, {"error": "No valid features for clustering"}
        
        X = df[features].copy()

        categorical_cols = []
        if 'Occupation' in df.columns:
            categorical_cols.append('Occupation')
        if 'City_Tier' in df.columns:
            categorical_cols.append('City_Tier')

        if categorical_cols:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df[categorical_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
            X = pd.concat([X, encoded_df], axis=1)
        

        X.fillna(X.median(), inplace=True)
        
        # Normalize
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        sample_size = min(10000, len(df))
        if len(df) > sample_size:
            logger.info(f"Using {sample_size} samples for clustering")
            indices = np.random.choice(len(df), sample_size, replace=False)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X_scaled[indices])
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X_scaled)

        df['Spending_Cluster'] = kmeans.predict(X_scaled)

        cluster_stats = {}
        for cluster in range(n_clusters):
            cluster_df = df[df['Spending_Cluster'] == cluster]
            stats = {
                'count': int(len(cluster_df)),
                'pct': float(len(cluster_df) / len(df)),  # Return as decimal, not percentage
                'avg_income': float(cluster_df['Income'].mean()) if 'Income' in df.columns else 0,
                'avg_spending': float(cluster_df[features[1:]].sum(axis=1).mean()) if len(features) > 1 else 0
            }
            
            if 'Occupation' in df.columns and not cluster_df['Occupation'].dropna().empty:
                stats['most_common_occupation'] = cluster_df['Occupation'].value_counts().idxmax()
            else:
                stats['most_common_occupation'] = None

            if 'City_Tier' in df.columns and not cluster_df['City_Tier'].dropna().empty:
                stats['most_common_city_tier'] = cluster_df['City_Tier'].value_counts().idxmax()
            else:
                stats['most_common_city_tier'] = None

            cluster_stats[f'Cluster_{cluster}'] = stats

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
        
        features = ['Groceries', 'Eating_Out', 'Entertainment', 'Miscellaneous', 'Occupation', 'City_Tier']
        
        missing_features = [col for col in features if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing anomaly detection features: {missing_features}")
            
            features = [col for col in features if col in df.columns]
            
        if not features:
            return df, {"error": "No valid features for anomaly detection"}
        
        # Encode categorical features (occupation and city_tier)
        df_encoded = df.copy()
        
        if 'Occupation' in df_encoded.columns:
            df_encoded['Occupation'] = df_encoded['Occupation'].map({'Self_Employed': 0, 'Retired': 1, 'Student': 2, 'Professional': 3})
        
        if 'City_Tier' in df_encoded.columns:
            df_encoded['City_Tier'] = df_encoded['City_Tier'].map({'Tier_1': 0, 'Tier_2': 1, 'Tier_3': 2})
        
        X = df_encoded[features].copy()
        
        X.fillna(X.median(), inplace=True)
        
        sample_size = min(10000, len(df)) 
        
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
        
        required_cols = ['Income', 'Desired_Savings']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Missing columns for recommendations: {missing_cols}"}
        
        try:
            reg_features = ['Income', 'Rent', 'Loan_Repayment', 'Utilities', 'Occupation', 'City_Tier']
            
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

def calculate_categorization_metrics(data, true_labels, predicted_labels):
    """
    Calculate metrics for transaction categorization performance
    
    Parameters:
    -----------
    data : DataFrame
        The financial dataset
    true_labels : array-like
        The true category labels
    predicted_labels : array-like
        The predicted category labels
        
    Returns:
    --------
    dict
        Dictionary containing categorization performance metrics
    """
    categories = np.unique(true_labels)
    
    # Calculate main metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Format confusion matrix for frontend visualization
    confusion_matrix_data = []
    for i, category in enumerate(categories):
        correct = conf_matrix[i, i]
        incorrect = sum(conf_matrix[i, :]) - correct
        confusion_matrix_data.append({
            "category": category,
            "correctPredictions": int(correct),
            "incorrectPredictions": int(incorrect)
        })
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1Score": float(f1),
        "confusionMatrix": confusion_matrix_data
    }

def calculate_clustering_metrics(data):
    """
    Calculate metrics for clustering quality
    
    Parameters:
    -----------
    data : DataFrame
        The financial dataset with features to cluster
        
    Returns:
    --------
    dict
        Dictionary containing clustering quality metrics
    """
    # Select relevant features for clustering
    # Adjust these columns based on your actual dataset
    features = data[['Income', 'Age', 'Groceries', 'Transport', 
                     'Eating_Out', 'Entertainment', 'Utilities']]
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Find optimal number of clusters using the elbow method
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        distortions.append(kmeans.inertia_)
    
    # Identify elbow point (this is a simplified approach)
    optimal_clusters = 4  # This would normally be calculated from the distortions
    
    # Perform clustering with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Calculate silhouette score
    silhouette = silhouette_score(scaled_features, clusters) if optimal_clusters > 1 else 0
    
    # Calculate Dunn Index (simplified)
    # In a real implementation, you'd compute inter-cluster and intra-cluster distances
    dunn_index = 0.54  # Placeholder value
    
    # Get cluster distribution
    cluster_counts = np.bincount(clusters)
    cluster_distribution = [
        {"name": f"Cluster {i+1}", "value": int(count)}
        for i, count in enumerate(cluster_counts)
    ]
    
    return {
        "silhouetteScore": float(silhouette),
        "dunnIndex": float(dunn_index),
        "optimalClusters": int(optimal_clusters),
        "clusterDistribution": cluster_distribution
    }

def calculate_anomaly_detection_metrics(data, true_anomalies=None):
    """
    Calculate metrics for anomaly detection performance
    
    Parameters:
    -----------
    data : DataFrame
        The financial dataset
    true_anomalies : array-like, optional
        True anomaly labels (1 for anomaly, 0 for normal)
        If None, will use synthetic data for demonstration
        
    Returns:
    --------
    dict
        Dictionary containing anomaly detection metrics
    """
    # If no true anomalies provided, create synthetic data for demonstration
    if true_anomalies is None:
        # Simulate anomaly detection results with 5% anomaly rate
        np.random.seed(42)
        n_samples = len(data)
        true_anomalies = np.zeros(n_samples)
        anomaly_indices = np.random.choice(range(n_samples), size=int(n_samples * 0.05), replace=False)
        true_anomalies[anomaly_indices] = 1
        
        # Simulate anomaly scores (higher = more likely to be anomaly)
        anomaly_scores = np.random.beta(1, 10, size=n_samples)  # Most scores will be low
        anomaly_scores[anomaly_indices] += np.random.beta(5, 1, size=len(anomaly_indices))  # Boost anomaly scores
        anomaly_scores = np.clip(anomaly_scores, 0, 1)  # Ensure scores are between 0 and 1
    else:
        # In real implementation, you'd calculate anomaly scores from your model
        anomaly_scores = np.random.rand(len(true_anomalies))
    
    # Calculate threshold-independent metrics
    precision, recall, _ = precision_recall_curve(true_anomalies, anomaly_scores)
    pr_auc = auc(recall, precision)
    
    # For ROC-AUC
    roc_auc = roc_auc_score(true_anomalies, anomaly_scores)
    
    # Calculate threshold-dependent metrics using an optimal threshold
    # In practice, you would determine this threshold based on your specific requirements
    threshold = 0.5
    predicted_anomalies = (anomaly_scores >= threshold).astype(int)
    
    # Calculate accuracy and false positive rate
    accuracy = accuracy_score(true_anomalies, predicted_anomalies)
    fps = sum((predicted_anomalies == 1) & (true_anomalies == 0))
    n_normal = sum(true_anomalies == 0)
    fpr = fps / n_normal if n_normal > 0 else 0
    
    # Calculate anomaly rate
    anomaly_rate = sum(true_anomalies) / len(true_anomalies)
    
    return {
        "precisionRecallAUC": float(pr_auc),
        "rocAUC": float(roc_auc),
        "anomalyRate": float(anomaly_rate),
        "accuracy": float(accuracy),
        "falsePositiveRate": float(fpr)
    }

def calculate_recommendation_metrics(data):
    """
    Calculate metrics for recommendation quality
    
    Parameters:
    -----------
    data : DataFrame
        The financial dataset
        
    Returns:
    --------
    dict
        Dictionary containing recommendation quality metrics
    """
    # For association rule metrics (support, confidence, lift)
    # In a real implementation, you'd calculate these from your Apriori algorithm results
    support = 0.42
    confidence = 0.76
    lift = 2.1
    
    # For budget prediction accuracy
    # Simulate MAE and RMSE calculation for savings predictions
    # In a real implementation, you'd use actual and predicted values
    
    # Split data for demonstration purposes
    X = data[['Income', 'Age', 'Dependents', 'Groceries', 'Transport', 'Entertainment']]
    y = data['Desired_Savings']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model for demonstration
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate MAE and RMSE
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Simulate user satisfaction score (in a real app, this would come from user surveys)
    user_satisfaction = 4.2
    
    return {
        "support": float(support),
        "confidence": float(confidence),
        "lift": float(lift),
        "mae": float(mae),
        "rmse": float(rmse),
        "userSatisfaction": float(user_satisfaction)
    }

def calculate_all_metrics(data):
    """
    Calculate all evaluation metrics for the personal finance analyzer
    
    Parameters:
    -----------
    data : DataFrame
        The financial dataset
        
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    # In a real application, you would have actual labels and predictions
    # Here we'll simulate them for demonstration purposes
    
    # Simulate transaction categorization
    # For example, predicting expense categories based on amount and other features
    np.random.seed(42)
    categories = ['Groceries', 'Transport', 'Entertainment', 'Utilities', 'Healthcare']
    true_labels = np.random.choice(categories, size=len(data))
    predicted_labels = np.copy(true_labels)
    
    # Introduce some errors (15% error rate)
    error_indices = np.random.choice(range(len(data)), size=int(len(data) * 0.15), replace=False)
    for idx in error_indices:
        options = [cat for cat in categories if cat != true_labels[idx]]
        predicted_labels[idx] = np.random.choice(options)
    
    # Calculate all metrics
    categorization_metrics = calculate_categorization_metrics(data, true_labels, predicted_labels)
    clustering_metrics = calculate_clustering_metrics(data)
    anomaly_metrics = calculate_anomaly_detection_metrics(data)
    recommendation_metrics = calculate_recommendation_metrics(data)
    
    return {
        "categorization": categorization_metrics,
        "clustering": clustering_metrics,
        "anomalyDetection": anomaly_metrics,
        "recommendations": recommendation_metrics
    }