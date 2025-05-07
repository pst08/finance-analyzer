import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import 'D:/finance-analyzer-backend/frontend/react-front/src/App.css';  // Make sure to import the CSS file where you will add the @import


function Home() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [previewData, setPreviewData] = useState(null);
  const navigate = useNavigate();

  const validateCSV = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => {
        try {
          const text = event.target.result;
          
          // Check if file is empty
          if (!text || text.trim() === '') {
            reject("The CSV file is empty");
            return;
          }
          
          // Split the file into lines
          const lines = text.split(/\r\n|\n/);
          
          // Check if there's at least a header and one data row
          if (lines.length < 2) {
            reject("The CSV file must contain at least a header row and one data row");
            return;
          }
          
          // Try to detect the delimiter
          const possibleDelimiters = [',', ';', '\t', '|'];
          let delimiter = ','; // Default
          let maxColumns = 0;
          
          // Find the delimiter that results in the most columns
          possibleDelimiters.forEach(delim => {
            const headerCount = lines[0].split(delim).length;
            if (headerCount > maxColumns) {
              maxColumns = headerCount;
              delimiter = delim;
            }
          });
          
          // Parse with the detected delimiter
          const headers = lines[0].split(delimiter);
          
          // Check if we have reasonable headers (more than one column)
          if (headers.length <= 1) {
            // If only one column, it might be a malformed CSV - check for common headers
            const requiredColumns = ['Income', 'Age', 'Dependents', 'Occupation'];
            const headerLine = headers[0];
            
            // Check if all required column names are in the single header
            const hasAllRequiredColumns = requiredColumns.every(col => 
              headerLine.includes(col)
            );
            
            if (hasAllRequiredColumns) {
              // If it looks like all headers are in one column, we'll manually separate them
              const preview = {
                headers: requiredColumns,
                rows: [],
                message: "CSV appears to be malformed. We'll attempt to fix it during upload."
              };
              resolve(preview);
              return;
            } else {
              reject("CSV format appears invalid. Expected multiple columns including Income, Age, etc.");
              return;
            }
          }
          
          // Parse a few rows for preview
          const previewRows = [];
          for (let i = 1; i < Math.min(4, lines.length); i++) {
            if (lines[i].trim() !== '') {
              const values = lines[i].split(delimiter);
              // Make sure we have the same number of values as headers
              if (values.length === headers.length) {
                previewRows.push(
                  Object.fromEntries(headers.map((header, index) => [header, values[index]]))
                );
              }
            }
          }
          
          // Validate required columns
          const requiredColumns = ['Income', 'Age'];
          const missingColumns = requiredColumns.filter(col => 
            !headers.some(header => header.trim() === col)
          );
          
          if (missingColumns.length > 0) {
            reject(`Missing required columns: ${missingColumns.join(', ')}`);
            return;
          }
          
          resolve({
            headers,
            rows: previewRows,
            delimiter
          });
          
        } catch (error) {
          reject(`Error validating CSV: ${error.message}`);
        }
      };
      
      reader.onerror = () => {
        reject("Failed to read the file");
      };
      
      reader.readAsText(file);
    });
  };

  const onFileChange = async (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;
    
    if (!selectedFile.name.endsWith('.csv')) {
      setError("Please select a CSV file");
      setFile(null);
      setPreviewData(null);
      return;
    }
    
    try {
      setError(null);
      setLoading(true);
      
      const preview = await validateCSV(selectedFile);
      setPreviewData(preview);
      setFile(selectedFile);
      setLoading(false);
    } catch (err) {
      setError(err);
      setFile(null);
      setPreviewData(null);
      setLoading(false);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please select a file to upload");
      return;
    }
  
    // Show loading state
    setLoading(true);
    setError(null);
  
    const formData = new FormData();
    formData.append("file", file);
    
    // If we detected a malformed CSV, add a hint for the backend
    if (previewData && previewData.message) {
      formData.append("csv_format", "malformed");
    } else if (previewData && previewData.delimiter) {
      formData.append("delimiter", previewData.delimiter);
    }
  
    try {
      console.log("Submitting file to backend...");
      
      // Make the API call with additional headers and credentials
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        body: formData,
        // Add these settings to help with CORS issues
        credentials: 'include',
        headers: {
          'Accept': 'application/json',
          // Don't set Content-Type with FormData as the browser will set it with the boundary
        },
      });
      
      console.log("Response status:", response.status);
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      // Parse the JSON response
      const result = await response.json();
      console.log("Response from backend:", result);
      
      if (result.status === 'success') {
        console.log("Analysis successful! Navigating to insights...");
        
        // Navigate to insights page with the data
        navigate("/insights", { 
          state: { 
            data: result 
          }
        });
      } else {
        // Show error message
        console.error("Analysis failed:", result);
        setError(`Analysis failed: ${result.error || 'Unknown error'}`);
      }
    
    } catch (error) {
      // Handle network errors with more detail
      console.error('Error uploading file:', error);
      if (error.message && error.message.includes('Failed to fetch')) {
        setError('Connection failed. Make sure the backend server is running at http://localhost:5000');
      } else {
        setError(`Network error: ${error.message}`);
      }
    
    } finally {
      // Hide loading state (runs whether success or error)
      setLoading(false);
    }
    
  };

  return (
    <div className="container" style={{ 
      padding: "40px", 
      maxWidth: "800px", 
      margin: "0 auto", 
      fontFamily: "'Cal Sans', sans-serif" 
    }}>
      <h1 style={{ color: "#333", marginBottom: "30px", textAlign: "center" }}>
        Personal Finance Pattern Analyzer
      </h1>
      
      <div style={{ 
        backgroundColor: "#f5f5f5", 
        padding: "30px", 
        borderRadius: "10px",
        boxShadow: "0 4px 6px rgba(0,0,0,0.1)" 
      }}>
        <h2 style={{ color: "#444", marginBottom: "20px" }}>Upload Your Financial Data</h2>
        
        <p style={{ marginBottom: "20px", color: "#666" }}>
          Upload a CSV file containing your financial transactions to analyze spending patterns, 
          detect anomalies, and receive personalized recommendations.
        </p>
        
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: "20px" }}>
            <input 
              type="file" 
              onChange={onFileChange} 
              style={{ 
                padding: "10px",
                border: "1px solid #ddd",
                borderRadius: "4px",
                width: "100%"
              }}
              accept=".csv"
            />
            {error && (
              <p style={{ color: "red", marginTop: "10px" }}>{error}</p>
            )}
          </div>
          
          {previewData && (
            <div style={{ marginBottom: "20px" }}>
              <h3 style={{ color: "#555", marginBottom: "15px" }}>Data Preview</h3>
              
              {previewData.message && (
                <div style={{ padding: "10px", backgroundColor: "#fff3cd", color: "#856404", borderRadius: "4px", marginBottom: "15px" }}>
                  ⚠️ {previewData.message}
                </div>
              )}
              
              {previewData.rows && previewData.rows.length > 0 && (
                <div style={{ overflowX: "auto", marginBottom: "15px" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr>
                        {previewData.headers.slice(0, 5).map((header, idx) => (
                          <th key={idx} style={{ padding: "8px", textAlign: "left", borderBottom: "2px solid #ddd" }}>
                            {header}
                          </th>
                        ))}
                        {previewData.headers.length > 5 && (
                          <th style={{ padding: "8px", textAlign: "left", borderBottom: "2px solid #ddd" }}>...</th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {previewData.rows.map((row, rowIdx) => (
                        <tr key={rowIdx}>
                          {previewData.headers.slice(0, 5).map((header, colIdx) => (
                            <td key={colIdx} style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>
                              {row[header]}
                            </td>
                          ))}
                          {previewData.headers.length > 5 && (
                            <td style={{ padding: "8px", borderBottom: "1px solid #ddd" }}>...</td>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
          
          <button 
            type="submit" 
            disabled={loading || !file} 
            style={{
              backgroundColor: loading ? "#cccccc" : "#4CAF50",
              color: "white",
              padding: "12px 24px",
              border: "none",
              borderRadius: "4px",
              cursor: loading ? "not-allowed" : "pointer",
              width: "100%",
              fontSize: "16px"
            }}
          >
            {loading ? "Processing..." : "Upload and Analyze"}
          </button>
          
          {loading && (
            <div style={{ marginTop: "20px" }}>
              <div style={{ 
                backgroundColor: "#ddd", 
                height: "20px", 
                borderRadius: "10px", 
                overflow: "hidden" 
              }}>
                <div 
                  style={{ 
                    width: `${progress}%`,
                    backgroundColor: "#4CAF50",
                    height: "100%",
                    transition: "width 0.3s ease"
                  }}
                />
              </div>
              <p style={{ textAlign: "center", marginTop: "10px" }}>
                {progress < 100 ? `Uploading: ${progress}%` : "Processing data..."}
              </p>
            </div>
          )}
        </form>
        
        <div style={{ marginTop: "30px" }}>
          <h3 style={{ color: "#555", marginBottom: "15px" }}>Expected CSV Format</h3>
          <p style={{ color: "#666", marginBottom: "10px" }}>
            Your CSV should include columns for Income, Expenses (Rent, Groceries, etc.), 
            and other financial data. Example columns:
          </p>
          <div style={{ 
            backgroundColor: "#eee", 
            padding: "15px", 
            borderRadius: "5px", 
            overflowX: "auto" 
          }}>
            <code>Income, Age, Dependents, Occupation, City_Tier, Rent, Groceries, Transport, Entertainment, Utilities, ...</code>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;