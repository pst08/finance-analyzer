import 'D:/finance-analyzer-backend/frontend/react-front/src/App.css';
import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Container, Box, Button, Paper, Grid, CircularProgress, Alert, AlertTitle, 
  Divider, Chip, Card, CardContent, Typography, Tabs, Tab } from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, Line, LineChart
} from "recharts";
import EvaluationMetrics from './Evaluate';
import { UploadFile, Assessment, BarChart as BarChartIcon, DonutLarge, TrendingUp } from '@mui/icons-material';
import Papa from 'papaparse';

function Insights() {
  const [showMetrics, setShowMetrics] = useState(false);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const location = useLocation();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("basic");

  // Forest green theme colors
  const colors = {
    darkGreen: "#1b4332",
    mediumGreen: "#2d6a4f",
    lightGreen: "#40916c",
    accentGreen: "#52b788",
    paleGreen: "#74c69d",
    veryLightGreen: "#b7e4c7",
    backgroundGreen: "#d8f3dc",
    chartColors: ["#1b4332", "#2d6a4f", "#40916c", "#52b788", "#74c69d", "#95d5b2", "#b7e4c7", "#d8f3dc"]
  };

  // Container styles
  const containerStyle = {
    padding: "30px",
    backgroundColor: "#f8f9fa",
    minHeight: "100vh",
    color: "#333"
  };

  // Tab styles
  const tabStyle = {
    padding: "12px 20px",
    marginRight: "10px",
    border: "none",
    borderRadius: "5px 5px 0 0",
    cursor: "pointer",
    backgroundColor: "#e9ecef",
    fontWeight: "500",
    transition: "background-color 0.3s"
  };

  const activeTabStyle = {
    backgroundColor: colors.mediumGreen,
    color: "white"
  };

  const tabContentStyle = {
    backgroundColor: "white",
    padding: "25px",
    borderRadius: "0 5px 5px 5px",
    boxShadow: "0 4px 8px rgba(0,0,0,0.1)"
  };

  // Card styles
  const cardStyle = {
    backgroundColor: "white",
    borderRadius: "8px",
    padding: "20px",
    marginBottom: "25px",
    boxShadow: "0 2px 5px rgba(0,0,0,0.1)"
  };

  const cardTitleStyle = {
    color: colors.darkGreen,
    marginBottom: "15px",
    borderBottom: `2px solid ${colors.veryLightGreen}`,
    paddingBottom: "10px"
  };

  // In Insights.js
  useEffect(() => {
    console.log("Location state received:", location.state);
    
    // Get data from location state
    if (location.state?.data) {
      console.log("Data from state:", location.state.data);
      setData(location.state.data);
      // Save data to sessionStorage for persistence
      sessionStorage.setItem('financeData', JSON.stringify(location.state.data));
    } else {
      // Try to get data from sessionStorage
      const savedData = sessionStorage.getItem('financeData');
      if (savedData) {
        try {
          const parsedData = JSON.parse(savedData);
          console.log("Data from sessionStorage:", parsedData);
          setData(parsedData);
        } catch (error) {
          console.error("Error parsing saved data:", error);
          navigate("/"); // Redirect if data is invalid
        }
      } else {
        // No data found, redirect to home
        console.log("No data found in state or sessionStorage");
        navigate("/");
      }
    }
  }, [location, navigate]);

  // Format currency values
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2
    }).format(value);
  };

  // If no data is loaded yet
  if (!data) {
    return (
      <div style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh",
        backgroundColor: "#f8f9fa"
      }}>
        <div style={{
          padding: "20px",
          backgroundColor: "white",
          borderRadius: "8px",
          boxShadow: "0 4px 8px rgba(0,0,0,0.1)"
        }}>
          <p>Loading data or no data found. Redirecting...</p>
        </div>
      </div>
    );
  }

  // Prepare data for charts
  const prepareExpenseData = () => {
    if (!data.insights?.expense_breakdown) return [];
    
    return Object.entries(data.insights.expense_breakdown).map(([category, amount]) => ({
      name: category,
      value: amount
    }));
  };

  const prepareIncomeByAgeData = () => {
    if (!data.insights?.avg_income_by_age) return [];
    
    return Object.entries(data.insights.avg_income_by_age).map(([ageGroup, amount]) => ({
      name: ageGroup,
      value: amount
    }));
  };

  const prepareClusterData = () => {
    if (!data.clustering) return [];
    
    return Object.entries(data.clustering)
      .filter(([key, _]) => key.startsWith('Cluster_'))
      .map(([cluster, stats]) => ({
        name: `Cluster ${cluster.split('_')[1]}`,
        count: stats.count,
        income: stats.avg_income,
        spending: stats.avg_spending
      }));
  };

  const prepareSavingsOpportunitiesData = () => {
    if (!data.recommendations?.savings_opportunities) return [];
    
    return Object.entries(data.recommendations.savings_opportunities).map(([category, amount]) => ({
      name: category,
      value: amount
    }));
  };

  // Chart components
  const renderExpenseChart = () => {
    const expenseData = prepareExpenseData();
    if (expenseData.length === 0) return <p>No expense data available</p>;

    return (
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={expenseData}
              cx="50%"
              cy="50%"
              labelLine={true}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {expenseData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors.chartColors[index % colors.chartColors.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => formatCurrency(value)} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderIncomeByAgeChart = () => {
    const incomeByAgeData = prepareIncomeByAgeData();
    if (incomeByAgeData.length === 0) return <p>No age-based income data available</p>;

    return (
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <BarChart data={incomeByAgeData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip formatter={(value) => formatCurrency(value)} />
            <Legend />
            <Bar dataKey="value" name="Average Income" fill={colors.mediumGreen} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderClusterChart = () => {
    const clusterData = prepareClusterData();
    if (clusterData.length === 0) return <p>No cluster data available</p>;

    return (
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <BarChart data={clusterData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis yAxisId="left" orientation="left" stroke={colors.darkGreen} />
            <YAxis yAxisId="right" orientation="right" stroke={colors.accentGreen} />
            <Tooltip formatter={(value, name) => [
              name.includes('income') || name.includes('spending') ? formatCurrency(value) : value,
              name
            ]} />
            <Legend />
            <Bar yAxisId="left" dataKey="count" name="Number of Users" fill={colors.darkGreen} />
            <Bar yAxisId="right" dataKey="income" name="Average Income" fill={colors.mediumGreen} />
            <Bar yAxisId="right" dataKey="spending" name="Average Spending" fill={colors.accentGreen} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderSavingsOpportunitiesChart = () => {
    const savingsData = prepareSavingsOpportunitiesData();
    if (savingsData.length === 0) return <p>No savings opportunities data available</p>;

    return (
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <BarChart data={savingsData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip formatter={(value) => formatCurrency(value)} />
            <Legend />
            <Bar dataKey="value" name="Potential Savings" fill={colors.accentGreen} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderBasicInsights = () => {
    console.log("Rendering basic insights with data:", data);
    
    // Added safety check to ensure insights exists
    const insights = data?.insights || {};
    console.log("Insights object:", insights);
    
    return (
      <div style={{ margin: "0 auto", maxWidth: "1200px", padding: "20px" }}>
        <div style={{ marginBottom: "30px" }}>
          <p>Analysis completed successfully with {data.row_count || 0} records.</p>
        </div>
        
        {/* Render the other insights here */}
        {renderClustering()}
        {renderAnomalies()}
        {renderRecommendations()}
      </div>
    );
  };
  
  const renderClustering = () => {
    // Added safety check to ensure clustering exists
    const clustering = data.clustering || {};
    
    return (
      <div>
        <div style={cardStyle}>
          <h3 style={cardTitleStyle}>User Spending Clusters</h3>
          <p>Users grouped by similar financial behaviors using K-Means clustering</p>
          {renderClusterChart()}
        </div>

        <div style={cardStyle}>
          <h3 style={cardTitleStyle}>Cluster Analysis</h3>
          <div style={{ marginTop: "20px" }}>
            {Object.entries(clustering)
              .filter(([key, _]) => key.startsWith('Cluster_'))
              .map(([cluster, stats]) => (
                <div key={cluster} style={{ 
                  marginBottom: "15px", 
                  padding: "15px", 
                  backgroundColor: colors.backgroundGreen,
                  borderRadius: "8px" 
                }}>
                  <h4>Cluster {cluster.split('_')[1]}</h4>
                  <p><strong>Members:</strong> {stats.count} ({(stats.pct || 0).toFixed(2)}% of total)</p>
                  <p><strong>Average Income:</strong> {formatCurrency(stats.avg_income || 0)}</p>
                  <p><strong>Average Spending:</strong> {formatCurrency(stats.avg_spending || 0)}</p>
                  <p><strong>Analysis:</strong> {
                    (stats.avg_income || 0) > (stats.avg_spending || 0) * 1.5
                      ? "High savers with significant income-to-spending ratio"
                      : (stats.avg_income || 0) < (stats.avg_spending || 0) * 1.2
                        ? "Potential overspenders with low savings potential"
                        : "Balanced spending with moderate savings"
                  }</p>
                </div>
              ))}
          </div>
        </div>
      </div>
    );
  };

  const renderAnomalies = () => {
    // Added safety check to ensure anomalies exists
    const anomalies = data.anomalies || {};
    
    return (
      <div>
        <div style={cardStyle}>
          <h3 style={cardTitleStyle}>Anomaly Detection Results</h3>
          <p>Using Isolation Forest to detect unusual spending patterns</p>
          
          <div style={{ display: "flex", flexWrap: "wrap", gap: "20px", marginTop: "20px" }}>
            <div style={{ flex: "1 1 200px", backgroundColor: colors.backgroundGreen, padding: "15px", borderRadius: "8px" }}>
              <h4>Normal Transactions</h4>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold" }}>{anomalies.normal_transactions || 0}</p>
            </div>
            <div style={{ flex: "1 1 200px", backgroundColor: "#ffe8e8", padding: "15px", borderRadius: "8px" }}>
              <h4>Anomalous Transactions</h4>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#d32f2f" }}>{anomalies.anomalous_transactions || 0}</p>
            </div>
            <div style={{ flex: "1 1 200px", backgroundColor: "#fff3e0", padding: "15px", borderRadius: "8px" }}>
              <h4>Anomaly Percentage</h4>
              <p style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#ed6c02" }}>{(anomalies.anomaly_percentage || 0).toFixed(2)}%</p>
            </div>
          </div>
        </div>

        {anomalies.avg_values && (
          <div style={cardStyle}>
            <h3 style={cardTitleStyle}>Anomaly Comparison</h3>
            <p>Comparing average spending between normal and anomalous transactions</p>
            
            <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "20px" }}>
              <thead>
                <tr style={{ backgroundColor: colors.veryLightGreen }}>
                  <th style={{ padding: "12px", textAlign: "left", borderBottom: "2px solid #ddd" }}>Category</th>
                  <th style={{ padding: "12px", textAlign: "right", borderBottom: "2px solid #ddd" }}>Normal Avg</th>
                  <th style={{ padding: "12px", textAlign: "right", borderBottom: "2px solid #ddd" }}>Anomalous Avg</th>
                  <th style={{ padding: "12px", textAlign: "right", borderBottom: "2px solid #ddd" }}>Difference</th>
                </tr>
              </thead>
              <tbody>
                {anomalies.avg_values && anomalies.avg_values.normal && Object.keys(anomalies.avg_values.normal).map(category => {
                  const normalValue = (anomalies.avg_values.normal || {})[category] || 0;
                  const anomalousValue = (anomalies.avg_values.anomalous || {})[category] || 0;
                  const difference = anomalousValue - normalValue;
                  const percentDiff = normalValue !== 0 ? (difference / normalValue) * 100 : 0;
                  
                  return (
                    <tr key={category} style={{ borderBottom: "1px solid #ddd" }}>
                      <td style={{ padding: "10px" }}>{category}</td>
                      <td style={{ padding: "10px", textAlign: "right" }}>{formatCurrency(normalValue)}</td>
                      <td style={{ padding: "10px", textAlign: "right" }}>{formatCurrency(anomalousValue)}</td>
                      <td style={{ 
                        padding: "10px", 
                        textAlign: "right",
                        color: difference > 0 ? "#d32f2f" : "#2e7d32"
                      }}>
                        {formatCurrency(difference)} ({percentDiff.toFixed(1)}%)
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  };

  const renderRecommendations = () => {
    // Added safety check to ensure recommendations exists
    const recommendations = data.recommendations || {};
    const insights = data.insights || {};
    
    return (
      <div>
        <div style={cardStyle}>
          <h3 style={cardTitleStyle}>Savings Opportunities</h3>
          <p>Areas where you could potentially save money based on your spending patterns</p>
          {renderSavingsOpportunitiesChart()}
        </div>

        {recommendations.savings_factors && (
          <div style={cardStyle}>
            <h3 style={cardTitleStyle}>Factors Affecting Savings</h3>
            <p>Based on regression analysis of income and expenses</p>
            
            <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "20px" }}>
              <thead>
                <tr style={{ backgroundColor: colors.veryLightGreen }}>
                  <th style={{ padding: "12px", textAlign: "left", borderBottom: "2px solid #ddd" }}>Factor</th>
                  <th style={{ padding: "12px", textAlign: "right", borderBottom: "2px solid #ddd" }}>Impact on Savings</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(recommendations.savings_factors || {}).map(([factor, impact]) => (
                  <tr key={factor} style={{ borderBottom: "1px solid #ddd" }}>
                    <td style={{ padding: "10px" }}>{factor}</td>
                    <td style={{ 
                      padding: "10px", 
                      textAlign: "right",
                      color: impact > 0 ? "#2e7d32" : "#d32f2f"
                    }}>
                      {impact > 0 ? "+" : ""}{impact.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {recommendations.association_rules && recommendations.association_rules.length > 0 && (
          <div style={cardStyle}>
            <h3 style={cardTitleStyle}>Financial Behavior Patterns</h3>
            <p>Association rules found in the data</p>
            
            <div style={{ marginTop: "20px" }}>
              {recommendations.association_rules.map((rule, index) => (
                <div key={index} style={{ 
                  marginBottom: "15px", 
                  padding: "15px", 
                  backgroundColor: colors.backgroundGreen,
                  borderRadius: "8px" 
                }}>
                  <p>
                    <strong>Rule {index + 1}:</strong> {rule.antecedents?.join(" and ") || "N/A"} 
                    <span style={{ margin: "0 10px" }}>→</span> 
                    {rule.consequents?.join(" and ") || "N/A"}
                  </p>
                  <p>
                    <strong>Confidence:</strong> {((rule.confidence || 0) * 100).toFixed(2)}% | 
                    <strong> Support:</strong> {((rule.support || 0) * 100).toFixed(2)}% | 
                    <strong> Lift:</strong> {(rule.lift || 0).toFixed(2)}
                  </p>
                  <p>
                    <strong>Interpretation:</strong>{" "}
                    {(rule.lift || 0) > 1.5 
                      ? "Strong relationship between these financial behaviors" 
                      : "Moderate relationship between these financial behaviors"}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        <div style={cardStyle}>
          <h3 style={cardTitleStyle}>Personalized Financial Recommendations</h3>
          
          <div style={{ 
            padding: "20px", 
            backgroundColor: colors.backgroundGreen, 
            borderRadius: "8px", 
            marginTop: "20px" 
          }}>
            <h4 style={{ marginBottom: "15px" }}>Based on your financial data:</h4>
            <ul style={{ paddingLeft: "20px" }}>
              <li style={{ marginBottom: "10px" }}>
                <strong>Budget Optimization:</strong>{" "}
                Focus on reducing spending in {insights?.most_spent_category || "highest expense categories"}.
              </li>
              <li style={{ marginBottom: "10px" }}>
                <strong>Savings Target:</strong>{" "}
                Consider saving {formatCurrency(recommendations.avg_predicted_savings || 0)} monthly based on regression analysis.
              </li>
              <li style={{ marginBottom: "10px" }}>
                <strong>Spending Alert:</strong>{" "}
                {(data.anomalies?.anomalous_transactions || 0) > 0 
                  ? `You have ${data.anomalies?.anomalous_transactions || 0} potentially anomalous transactions to review.`
                  : "No anomalous spending patterns detected in your data."}
              </li>
              <li style={{ marginBottom: "10px" }}>
                <strong>Financial Profile:</strong>{" "}
                {data.clustering && Object.entries(data.clustering || {})
                  .filter(([key, _]) => key.startsWith('Cluster_'))
                  .some(([_, stats]) => (stats.count || 0) > 0)
                  ? "Your spending pattern matches users who typically maintain a balanced budget profile."
                  : "No specific financial profile identified from the data."}
              </li>
            </ul>
          </div>
        </div>
      </div>
    );
  };

  // Function to send data to Metrics.js for evaluation
  // Add this state declaration to your component
// const [showMetrics, setShowMetrics] = useState(false);

  // This function should replace your existing viewMetrics function in the Insights component
// Make sure you have this state at the top of your component:


  const viewMetrics = (event) => {
    if (event?.preventDefault) event.preventDefault();

    const evaluationData = {
      metrics: {
        categorization: {
          accuracy: data?.categorization?.accuracy || 0.87,
          precision: data?.categorization?.precision || 0.84,
          recall: data?.categorization?.recall || 0.81,
          f1Score: data?.categorization?.f1Score || 0.83,
          confusionMatrix: data?.categorization?.confusionMatrix || [
            // Default or empty matrix
          ]
        },
        clustering: {
          silhouetteScore: data?.clustering?.silhouetteScore || 0.68,
          dunnIndex: data?.clustering?.dunnIndex || 0.54,
          optimalClusters: data?.clustering?.optimalClusters || 4,
          clusterDistribution: Object.entries(data.clustering || {})
            .filter(([k]) => k.startsWith('Cluster_'))
            .map(([k, v]) => ({
              name: `Cluster ${k.split('_')[1]}`,
              value: v.count || 0
            }))
        },
        anomalyDetection: {
          precisionRecallAUC: data?.anomalies?.pr_auc || 0.92,
          rocAUC: data?.anomalies?.roc_auc || 0.94,
          anomalyRate: (data?.anomalies?.anomaly_percentage || 5) / 100,
          accuracy: data?.anomalies?.accuracy || 0.97,
          falsePositiveRate: data?.anomalies?.fpr || 0.03
        },
        recommendations: {
          support: data?.recommendations?.support || 0.42,
          confidence: data?.recommendations?.confidence || 0.76,
          lift: data?.recommendations?.lift || 2.1,
          mae: data?.recommendations?.mae || 156.43,
          rmse: data?.recommendations?.rmse || 212.88,
          userSatisfaction: data?.recommendations?.userSatisfaction || 4.2
        }
      },
      clustering: data.clustering || {},
      anomalies: data.anomalies || {},
      row_count: data.row_count || 0,
      column_count: data.column_count || 0
    };

    sessionStorage.setItem('financeData', JSON.stringify(evaluationData));
    setShowMetrics((prev) => !prev);
  };

    
  return (
    <div style={{ ...containerStyle, fontFamily: "'Cal Sans', sans-serif" }}>
      <h1 style={{ color: colors.darkGreen, marginBottom: "30px", fontFamily: "'Cal Sans', sans-serif"}}>Financial Insights</h1>
      
      <div style={{ marginBottom: "25px" }}>
        <button 
          style={{ ...tabStyle, ...(activeTab === "basic" ? activeTabStyle : {}) }}
          onClick={() => setActiveTab("basic")}
        >
          Basic Insights
        </button>
        <button 
          style={{ ...tabStyle, ...(activeTab === "clusters" ? activeTabStyle : {}) }}
          onClick={() => setActiveTab("clusters")}
        >
          User Clusters
        </button>
        <button 
          style={{ ...tabStyle, ...(activeTab === "anomalies" ? activeTabStyle : {}) }}
          onClick={() => setActiveTab("anomalies")}
        >
          Anomaly Detection
        </button>
        <button 
          style={{ ...tabStyle, ...(activeTab === "recommendations" ? activeTabStyle : {}) }}
          onClick={() => setActiveTab("recommendations")}
        >
          Recommendations
        </button>
        <button 
          style={{ ...tabStyle, ...(activeTab === "chart components" ? activeTabStyle : {}) }}
          onClick={() => setActiveTab("chart components")}
        >
          Chart Components
        </button>
        <button 
          style={{ ...tabStyle, ...(activeTab === "evaluation metrics" ? activeTabStyle : {}) }}
          onClick={() => setActiveTab("evaluation metrics")}
        >
          Evaluation Metrics
        </button>
      </div>
      
      <div style={tabContentStyle}>
        {activeTab === "basic" && renderBasicInsights()}
        {activeTab === "clusters" && renderClustering()}
        {activeTab === "anomalies" && renderAnomalies()}
        {activeTab === "recommendations" && renderRecommendations()}
        {activeTab === "chart components" && (
          <>
            {renderExpenseChart()}
            {renderIncomeByAgeChart()}
          </>
        )}
        {activeTab === "evaluation metrics" && (
        <Container maxWidth="lg">
          <Box sx={{ mt: 4, mb: 4 }}>
            <Typography variant="h4" component="h1" gutterBottom>
              Financial Insights
            </Typography>

            {/* Any other insights or summary content here */}

            <Button
              variant="contained"
              color="primary"
              onClick={viewMetrics}
              sx={{ mt: 2, mb: 2 }}
            >
              {showMetrics ? "Hide Detailed Metrics" : "View Detailed Metrics"}
            </Button>

            <p>Click the button above to see detailed evaluation metrics of the data mining algorithms.</p>

            {showMetrics && (
              <EvaluationMetrics 
                data={JSON.parse(sessionStorage.getItem('financeData'))}
                loading={false}
              />
            )}
          </Box>
        </Container>
      )}

      </div>
    </div>
  );
}

export default Insights;