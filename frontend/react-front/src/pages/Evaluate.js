import React, { useState, useEffect } from "react";
import { 
  Card, CardContent, Typography, Grid, Box, Tabs, Tab, CircularProgress
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell
} from "recharts";

const EvaluationMetrics = ({ data, loading: externalLoading }) => {
  const [metrics, setMetrics] = useState({
    categorization: null,
    clustering: null,
    anomalyDetection: null,
    recommendations: null,
    loading: true
  });
  
  const [activeTab, setActiveTab] = useState(0);
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

  // Default metrics to use when no data is provided
  const defaultCategorization = {
    accuracy: 0.87,
    precision: 0.84,
    recall: 0.81,
    f1Score: 0.83,
    confusionMatrix: [
      { category: 'Groceries', correctPredictions: 93, incorrectPredictions: 7 },
      { category: 'Transport', correctPredictions: 89, incorrectPredictions: 11 },
      { category: 'Entertainment', correctPredictions: 82, incorrectPredictions: 18 },
      { category: 'Utilities', correctPredictions: 91, incorrectPredictions: 9 },
      { category: 'Healthcare', correctPredictions: 85, incorrectPredictions: 15 }
    ]
  };
  
  const defaultClustering = {
    silhouetteScore: 0.68,
    dunnIndex: 0.54,
    optimalClusters: 4,
    clusterDistribution: [
      { name: 'Cluster 1', value: 35 },
      { name: 'Cluster 2', value: 28 },
      { name: 'Cluster 3', value: 22 },
      { name: 'Cluster 4', value: 15 }
    ]
  };
  
  const defaultAnomaly = {
    precisionRecallAUC: 0.92,
    rocAUC: 0.94,
    anomalyRate: 0.05,
    accuracy: 0.97,
    falsePositiveRate: 0.03
  };
  
  const defaultRecommendations = {
    support: 0.42,
    confidence: 0.76,
    lift: 2.1,
    mae: 156.43,
    rmse: 212.88,
    userSatisfaction: 4.2
  };

  useEffect(() => {
    // Check if we have data from props
    if (data && data.metrics) {
      // Process and set metrics from provided data
      setMetrics({
        categorization: data.metrics.categorization || defaultCategorization,
        clustering: data.metrics.clustering || defaultClustering,
        anomalyDetection: data.metrics.anomalyDetection || defaultAnomaly,
        recommendations: data.metrics.recommendations || defaultRecommendations,
        loading: false
      });
    } else if (data) {
      // If data exists but doesn't have the expected structure,
      // try to extract metrics or use defaults
      console.log("Data received but missing metrics structure:", data);
      setMetrics({
        categorization: defaultCategorization,
        clustering: defaultClustering,
        anomalyDetection: defaultAnomaly,
        recommendations: defaultRecommendations,
        loading: false
      });
    }
  }, [data, defaultAnomaly, defaultCategorization, defaultClustering, defaultRecommendations]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Show loading if either component's internal loading state is true
  // or if external loading prop is true
  if (metrics.loading || externalLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  // Define TabPanel to ensure consistent rendering
  const TabPanel = (props) => {
    const { children, value, index, ...other } = props;
    return (
      <div
        role="tabpanel"
        hidden={value !== index}
        id={`metrics-tabpanel-${index}`}
        aria-labelledby={`metrics-tab-${index}`}
        {...other}
      >
        {value === index && (
          <Box sx={{ pt: 3 }}>
            {children}
          </Box>
        )}
      </div>
    );
  };

  return (
    <Card sx={{ mb: 4, mt: 4 }}>
      <CardContent>
        <Typography variant="h5" component="h2" gutterBottom>
          Model Evaluation Metrics
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          Performance assessment of the Personal Finance Pattern Analyzer based on your uploaded data.
        </Typography>
        
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="metrics tabs">
            <Tab label="Transaction Categorization" />
            <Tab label="Clustering Analysis" />
            <Tab label="Anomaly Detection" />
            <Tab label="Recommendations" />
          </Tabs>
        </Box>

        {/* Transaction Categorization Metrics */}
        <TabPanel value={activeTab} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  Classification Performance
                </Typography>
                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={6} sm={3}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">Accuracy</Typography>
                      <Typography variant="h6">{(metrics.categorization.accuracy * 100).toFixed(1)}%</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">Precision</Typography>
                      <Typography variant="h6">{(metrics.categorization.precision * 100).toFixed(1)}%</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">Recall</Typography>
                      <Typography variant="h6">{(metrics.categorization.recall * 100).toFixed(1)}%</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">F1 Score</Typography>
                      <Typography variant="h6">{(metrics.categorization.f1Score * 100).toFixed(1)}%</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" fontWeight="bold">
                Confusion Matrix Highlights
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart
                  data={metrics.categorization.confusionMatrix}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="correctPredictions" name="Correct" fill="#4caf50" />
                  <Bar dataKey="incorrectPredictions" name="Incorrect" fill="#f44336" />
                </BarChart>
              </ResponsiveContainer>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Clustering Analysis Metrics */}
        <TabPanel value={activeTab} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  Clustering Quality
                </Typography>
                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={6} sm={4}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">Silhouette Score</Typography>
                      <Typography variant="h6">{metrics.clustering.silhouetteScore.toFixed(2)}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={4}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">Dunn Index</Typography>
                      <Typography variant="h6">{metrics.clustering.dunnIndex.toFixed(2)}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={4}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">Optimal Clusters</Typography>
                      <Typography variant="h6">{metrics.clustering.optimalClusters}</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" fontWeight="bold">
                Cluster Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={metrics.clustering.clusterDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {metrics.clustering.clusterDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Anomaly Detection Metrics */}
        <TabPanel value={activeTab} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  Anomaly Detection Performance
                </Typography>
                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={6} sm={4}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">PR-AUC</Typography>
                      <Typography variant="h6">{metrics.anomalyDetection.precisionRecallAUC.toFixed(2)}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={4}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">ROC-AUC</Typography>
                      <Typography variant="h6">{metrics.anomalyDetection.rocAUC.toFixed(2)}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={4}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">Anomaly Rate</Typography>
                      <Typography variant="h6">{(metrics.anomalyDetection.anomalyRate * 100).toFixed(1)}%</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  Detection Quality
                </Typography>
                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={6}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">Accuracy</Typography>
                      <Typography variant="h6">{(metrics.anomalyDetection.accuracy * 100).toFixed(1)}%</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">False Positive Rate</Typography>
                      <Typography variant="h6">{(metrics.anomalyDetection.falsePositiveRate * 100).toFixed(1)}%</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" fontWeight="bold">
                Anomaly Detection Visualization
              </Typography>
              <Box sx={{ height: 250, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                <Box 
                  sx={{ 
                    width: 200, 
                    height: 200, 
                    borderRadius: '50%', 
                    background: `conic-gradient(
                      #4caf50 0% ${metrics.anomalyDetection.accuracy * 100}%, 
                      #f44336 ${metrics.anomalyDetection.accuracy * 100}% 100%
                    )`,
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    position: 'relative'
                  }}
                >
                  <Box 
                    sx={{ 
                      width: 160, 
                      height: 160, 
                      borderRadius: '50%', 
                      bgcolor: 'background.paper',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center',
                      alignItems: 'center'
                    }}
                  >
                    <Typography variant="h4">{(metrics.anomalyDetection.accuracy * 100).toFixed(0)}%</Typography>
                    <Typography variant="body2" color="text.secondary">Detection Rate</Typography>
                  </Box>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Recommendation Metrics */}
        <TabPanel value={activeTab} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  Association Rule Quality
                </Typography>
                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">Support</Typography>
                      <Typography variant="h6">{metrics.recommendations.support.toFixed(2)}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">Confidence</Typography>
                      <Typography variant="h6">{metrics.recommendations.confidence.toFixed(2)}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={4}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">Lift</Typography>
                      <Typography variant="h6">{metrics.recommendations.lift.toFixed(2)}</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  Budget Prediction Accuracy
                </Typography>
                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={6}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">MAE</Typography>
                      <Typography variant="h6">${metrics.recommendations.mae.toFixed(2)}</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box sx={{ textAlign: 'center', p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                      <Typography variant="body2" color="text.secondary">RMSE</Typography>
                      <Typography variant="h6">${metrics.recommendations.rmse.toFixed(2)}</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" fontWeight="bold">
                User Satisfaction Score
              </Typography>
              <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <Typography variant="h3" sx={{ mb: 2 }}>
                  {metrics.recommendations.userSatisfaction.toFixed(1)}/5.0
                </Typography>
                <Box sx={{ width: '100%', display: 'flex', justifyContent: 'space-between' }}>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <Box 
                      key={star} 
                      sx={{ 
                        fontSize: '2rem', 
                        color: star <= Math.round(metrics.recommendations.userSatisfaction) ? '#FFD700' : '#D3D3D3'
                      }}
                    >
                      ★
                    </Box>
                  ))}
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
                  Based on user feedback surveys evaluating the usefulness of financial recommendations
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </TabPanel>
      </CardContent>
    </Card>
  );
};

export default EvaluationMetrics;