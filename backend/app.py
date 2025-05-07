from flask import Flask, request, jsonify
from processor import analyze_csv, forecast_spending
import os
from flask_cors import CORS
import traceback
import pandas as pd
import io
import re
import logging
import chardet

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# Folder to save uploaded datasets
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limit file size to 50MB

def fix_malformed_csv(filepath):
    """
    Attempts to fix a malformed CSV file where all columns are concatenated.
    Returns path to the fixed file.
    """
    try:
        # First, detect encoding to reduce encoding issues
        with open(filepath, 'rb') as f:
            content = f.read()
            result = chardet.detect(content)
            encoding = result['encoding'] or 'utf-8'
            
        logger.info(f"Detected encoding for fix_malformed_csv: {encoding}")
        
        # Read with detected encoding
        with open(filepath, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        # Split into lines
        lines = content.split('\n')
        if not lines:
            return filepath
            
        # Try to identify column names (from project description)
        expected_columns = [
            'Income', 'Age', 'Dependents', 'Occupation', 'City_Tier', 
            'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
            'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 
            'Education', 'Miscellaneous', 'Desired_Savings_Percentage',
            'Desired_Savings', 'Disposable_Income'
        ]
        
        # Add potential savings columns
        potential_columns = [f'Potential_Savings_{cat}' for cat in [
            'Groceries', 'Transport', 'Eating_Out', 'Entertainment',
            'Utilities', 'Healthcare', 'Education', 'Miscellaneous'
        ]]
        expected_columns.extend(potential_columns)
        
        # First try to find natural delimiter
        header_line = lines[0] if lines else ""
        natural_delimiters = [',', ';', '\t', '|']
        delimiter = None
        
        for d in natural_delimiters:
            if d in header_line and len(header_line.split(d)) > 1:
                delimiter = d
                break
                
        if delimiter:
            # Already has a natural delimiter, no need to fix
            logger.info(f"File already has natural delimiter: {delimiter}")
            return filepath
            
        logger.info("No natural delimiter found, attempting to fix file")
        
        # No natural delimiter found, need to fix the file
        
        # Function to split malformed line
        def split_malformed_line(line):
            # Look for patterns like column name followed by a number or value
            parts = []
            remaining = line
            
            for col in expected_columns:
                pattern = f"({col})([0-9.]+|[A-Za-z_]+)"
                match = re.search(pattern, remaining)
                if match:
                    # Add the column to parts
                    if parts:  # Don't add a comma before the first part
                        parts.append(",")
                    parts.append(match.group(1))
                    
                    # Update remaining to start right after the column name
                    remaining = remaining[remaining.find(col) + len(col):]
                    
                    # Look for a value after the column
                    value_pattern = "^([0-9.]+|[A-Za-z_]+)"
                    value_match = re.search(value_pattern, remaining)
                    if value_match:
                        parts.append(",")
                        parts.append(value_match.group(1))
                        # Update remaining to exclude the value we just found
                        remaining = remaining[len(value_match.group(1)):]
            
            return "".join(parts) if parts else line
        
        # Try to find column names in header
        header_parts = []
        for col in expected_columns:
            if col in header_line:
                header_parts.append(col)
                
        if header_parts:
            # Create new header with commas
            new_header = ",".join(header_parts)
            
            # Create new fixed file with proper CSV structure
            fixed_filepath = filepath + ".fixed.csv"
            with open(fixed_filepath, 'w', encoding='utf-8') as f:
                f.write(new_header + '\n')
                
                # Process data rows
                for i in range(1, len(lines)):
                    if lines[i].strip():  # Skip empty lines
                        fixed_line = split_malformed_line(lines[i])
                        f.write(fixed_line + '\n')
                        
            logger.info(f"Fixed malformed CSV saved to {fixed_filepath}")
            return fixed_filepath
        
        # If we can't fix it, return original file
        logger.warning("Could not fix malformed CSV, returning original file")
        return filepath
    except Exception as e:
        logger.error(f"Error fixing malformed CSV: {str(e)}")
        logger.error(traceback.format_exc())
        return filepath

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint to analyze the uploaded CSV file containing financial user data.
    Returns insights, clustering labels, anomaly flags, and recommendations.
    """
    temp_files = []  # Track temporary files to clean up
    
    try:
        # Check if file exists in request
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return jsonify({'error': 'No file part in the request.'}), 400
            
        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected.'}), 400

        # Check file extension
        if not file.filename.lower().endswith('.csv'):
            logger.error("Invalid file type, not a CSV")
            return jsonify({'error': 'Please upload a valid CSV file.'}), 400

        # Generate a safe filename
        safe_filename = os.path.join(app.config['UPLOAD_FOLDER'], 
                                     f"upload_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        
        # Save the file
        logger.info(f"Saving uploaded file to {safe_filename}")
        file.save(safe_filename)
        temp_files.append(safe_filename)
        
        # Check file size before processing
        file_size = os.path.getsize(safe_filename)
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            logger.error(f"File too large: {file_size} bytes")
            return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 400
            
        # If file is empty, return error
        if file_size == 0:
            logger.error("Uploaded file is empty")
            return jsonify({'error': 'Uploaded file is empty.'}), 400
        
        # First, try to determine if CSV is properly formatted
        try:
            # Read first few bytes to check BOM and encoding
            with open(safe_filename, 'rb') as f:
                raw_sample = f.read(4096)
                
            # Check encoding
            result = chardet.detect(raw_sample)
            encoding = result['encoding'] or 'utf-8'
            confidence = result['confidence']
            
            logger.info(f"Detected encoding: {encoding} with confidence: {confidence}")
            
            # Check for BOM mark (might cause issues)
            has_bom = False
            if raw_sample.startswith(b'\xef\xbb\xbf'):
                has_bom = True
                logger.info("File has UTF-8 BOM mark")
                
            # Try to sniff out the CSV dialect
            try:
                import csv
                # Convert to string with detected encoding
                if has_bom:
                    sample_text = raw_sample[3:].decode(encoding, errors='replace')
                else:
                    sample_text = raw_sample.decode(encoding, errors='replace')
                
                dialect = csv.Sniffer().sniff(sample_text, delimiters=',;\t|')
                logger.info(f"Detected CSV dialect with delimiter: {dialect.delimiter}")

                # Try parsing with detected delimiter
                test_df = pd.read_csv(safe_filename, 
                                    encoding=encoding,
                                    delimiter=dialect.delimiter,
                                    nrows=5)

                
                if len(test_df.columns) <= 1:
                    logger.warning("CSV parsing detected only one column, might need fixing")
                    fixed_filepath = fix_malformed_csv(safe_filename)
                    if fixed_filepath != safe_filename:
                        temp_files.append(fixed_filepath)
                        safe_filename = fixed_filepath
                
            except Exception as sniff_error:
                logger.warning(f"CSV dialect detection failed: {str(sniff_error)}")
                # Try basic parsing
                try:
                    test_df = pd.read_csv(safe_filename, 
                                         encoding=encoding,
                                         nrows=5)
                    
                    if len(test_df.columns) <= 1:
                        logger.warning("CSV parsing detected only one column, might need fixing")
                        fixed_filepath = fix_malformed_csv(safe_filename)
                        if fixed_filepath != safe_filename:
                            temp_files.append(fixed_filepath)
                            safe_filename = fixed_filepath
                            
                except Exception as basic_error:
                    logger.warning(f"Basic CSV parsing failed: {str(basic_error)}")
                    # Try to fix the file
                    fixed_filepath = fix_malformed_csv(safe_filename)
                    if fixed_filepath != safe_filename:
                        temp_files.append(fixed_filepath)
                        safe_filename = fixed_filepath
            
        except Exception as e:
            logger.error(f"Error in pre-processing CSV: {str(e)}")
            # We'll continue and let the processor handle it
        
        # Now call the analyzer function
        logger.info(f"Calling analyze_csv with file: {safe_filename}")
        result = analyze_csv(safe_filename)
        
        # If result indicates error, return appropriate response
        if result.get('status') == 'error':
            logger.error(f"Analysis failed: {result.get('error')}")
            return jsonify(result), 400
        
        # Clean up the uploaded files if successful
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up file {temp_file}: {str(cleanup_error)}")
            
        return jsonify(result)
    except Exception as e:
        # Clean up any temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
                
        # Return the error and traceback for debugging purposes
        error_message = str(e)
        trace = traceback.format_exc()
        logger.error(f"Unexpected error: {error_message}")
        logger.error(trace)
        return jsonify({
            'status': 'error',
            'error': error_message, 
            'trace': trace
        }), 500

@app.route('/forecast', methods=['GET'])
def forecast():
    """
    (Optional) Endpoint for savings or expense forecasting.
    Can be implemented using regression or time series later.
    """
    try:
        result = forecast_spending()
        return jsonify(result)
    except Exception as e:
        # Return the error and traceback for debugging purposes
        logger.error(f"Forecasting error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e), 
            'trace': traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple endpoint to check if the API is running
    """
    return jsonify({'status': 'ok'})

@app.route('/', methods=['GET'])
def home():
    return (
        "<h2>Finance Analyzer API</h2>"
        "<ul>"
        "<li><b>POST /analyze</b> - Analyze a CSV file (multipart/form-data, key: <code>file</code>)</li>"
        "<li><b>GET /forecast</b> - Get savings/expense forecast</li>"
        "<li><b>GET /health</b> - Health check</li>"
        "</ul>"
    )


if __name__ == '__main__':
    # Make sure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)

