import React, { useState, useEffect } from "react";
import { 
  Container, Typography, Box, Paper, Alert, AlertTitle, Button 
} from '@mui/material';
import EvaluationMetrics from './Evaluate';

const Metrics = () => {
  const [evaluationData, setEvaluationData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showMetrics, setShowMetrics] = useState(false); // State to toggle visibility of metrics

  useEffect(() => {
    // Check sessionStorage for data
    const storedData = sessionStorage.getItem('financeData');
    if (storedData) {
      try {
        const parsedData = JSON.parse(storedData);
        console.log("Data loaded from sessionStorage:", parsedData);
        setEvaluationData(parsedData);
        setLoading(false);
      } catch (err) {
        console.error("Error parsing stored data:", err);
        setError("Failed to load analysis data. Please ensure data is available.");
        setLoading(false);
      }
    } else {
      // No data available
      setError("No analysis data available. Please upload CSV and run analysis first.");
      setLoading(false);
    }
  }, []);

  const toggleMetrics = () => {
    // Just toggle visibility of the metrics section
    setShowMetrics(!showMetrics);
  };

  const handleRunAnalysis = () => {
    // Placeholder for running analysis directly from this page
    // In a real implementation, you would call your analysis function here
    alert("This would trigger data analysis directly from this page");
    // After analysis is complete, you could set showMetrics to true
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Performance Metrics
        </Typography>
        <Typography variant="body1" paragraph>
          Comprehensive evaluation of the data mining models applied to your financial data.
        </Typography>

        {error ? (
          <Paper sx={{ p: 3, mb: 3 }}>
            <Alert severity="warning">
              <AlertTitle>Data Required</AlertTitle>
              {error}
            </Alert>
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleRunAnalysis}
              >
                Run Analysis
              </Button>
            </Box>
          </Paper>
        ) : loading ? (
          <Paper sx={{ p: 3, mb: 3 }}>
            <Alert severity="info">
              <AlertTitle>Loading Data</AlertTitle>
              Your performance metrics are being loaded. Please wait.
            </Alert>
          </Paper>
        ) : (
          <>
            <Button
              variant="contained"
              color="primary"
              onClick={toggleMetrics}
              style={{ marginBottom: "20px" }}
            >
              {showMetrics ? "Hide Detailed Metrics" : "View Detailed Metrics"}
            </Button>
            
            {/* Always render the component but control visibility with CSS */}
            <div style={{ display: showMetrics ? 'block' : 'none' }}>
              <EvaluationMetrics data={evaluationData} loading={false} />
            </div>
          </>
        )}
      </Box>
    </Container>
  );
};

export default Metrics;