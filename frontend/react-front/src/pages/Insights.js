import 'D:/finance-analyzer-backend/frontend/react-front/src/App.css';
import React, { useState, useEffect } from "react";

import { useLocation, useNavigate } from "react-router-dom";
import { Box, Button} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, Line, ComposedChart, ReferenceLine, Scatter, ScatterChart
} from "recharts";
import { UploadFile} from '@mui/icons-material';

function Insights() {
  const [data, setData] = useState(null);
  const location = useLocation();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("basic");


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

  const containerStyle = {
    padding: "30px",
    backgroundColor: "#f8f9fa",
    minHeight: "100vh",
    color: "#333"
  };

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

  useEffect(() => {
    console.log("Location state received:", location.state);
  
    // Get data from location state
    if (location.state?.data) {
      const incomingData = location.state.data;
  
      // Check if clustering exists, if not, redirect
      if (!incomingData.clustering) {
        console.error("Clustering data missing in location state:", incomingData);
        navigate("/"); // or show an error message
        return;
      }
  
      console.log("Data from state (with clustering):", incomingData);
      setData(incomingData);
  
      // Save to sessionStorage
      sessionStorage.setItem('financeData', JSON.stringify(incomingData));
    } else {
      // Try to get data from sessionStorage
      const savedData = sessionStorage.getItem('financeData');
      if (savedData) {
        try {
          const parsedData = JSON.parse(savedData);
  
          if (!parsedData.clustering) {
            console.error("Clustering data missing in saved sessionStorage:", parsedData);
            navigate("/");
            return;
          }
  
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
  
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2
    }).format(value);
  };

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
    console.log("prepareClusterData called with data:", data); // 👈 ADD THIS

    if (!data || !data.clustering) return [];
    
    return Object.entries(data.clustering)
      .filter(([key, _]) => key.startsWith('Cluster_'))
      .map(([cluster, stats]) => ({
        name: `Cluster ${cluster.split('_')[1]}`,
        count: stats.count || 0,
        income: stats.avg_income || 0,
        spending: stats.avg_spending || 0,
        most_common_occupation: stats.most_common_occupation,
        most_common_city_tier: stats.most_common_city_tier
      }));
  };

  const prepareSavingsOpportunitiesData = () => {
    if (!data.recommendations?.savings_opportunities) return [];
    
    return Object.entries(data.recommendations.savings_opportunities).map(([category, amount]) => ({
      name: category,
      value: amount
    }));
  };

  
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
    // Check if prepareClusterData function exists
    if (typeof prepareClusterData !== 'function') {
      return <p>Chart function is not available</p>;
    }
    
    const clusterData = prepareClusterData();
    if (!clusterData || clusterData.length === 0) {
      return <p>No cluster data available</p>;
    }

    const chartColors = colors || {
      darkGreen: '#2e7d32',
      mediumGreen: '#4caf50',
      accentGreen: '#66bb6a',
      veryLightGreen: '#e8f5e9'
    };

    return (
      <div style={{ width: '100%', height: 400 }}>
        <ResponsiveContainer>
          <ComposedChart data={clusterData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis yAxisId="left" orientation="left" stroke={chartColors.darkGreen} />
            <YAxis yAxisId="right" orientation="right" stroke={chartColors.accentGreen} />
            <Tooltip
              formatter={(value, name, props) => {
                const cluster = props.payload;
                const extraInfo = [];

                if (cluster.most_common_occupation) {
                  extraInfo.push(`Occupation: ${cluster.most_common_occupation}`);
                }

                if (cluster.most_common_city_tier) {
                  extraInfo.push(`City Tier: ${cluster.most_common_city_tier}`);
                }

                return [
                  name.includes('income') || name.includes('spending')
                    ? (typeof formatCurrency === 'function' ? formatCurrency(value) : `$${value}`)
                    : value,
                  `${name}${extraInfo.length ? ` (${extraInfo.join(', ')})` : ''}`
                ];
              }}
              contentStyle={{
                backgroundColor: chartColors.veryLightGreen,
                borderColor: chartColors.mediumGreen
              }}
            />

            <Legend wrapperStyle={{ paddingTop: 10 }} />
            <Bar 
              yAxisId="left" 
              dataKey="count" 
              name="Number of Users" 
              fill={chartColors.darkGreen} 
              radius={[5, 5, 0, 0]} 
            />
            <Line 
              yAxisId="right" 
              type="monotone" 
              dataKey="income" 
              name="Average Income" 
              stroke={chartColors.mediumGreen} 
              strokeWidth={2} 
              dot={{ r: 5 }} 
              activeDot={{ r: 7 }} 
            />
            <Line 
              yAxisId="right" 
              type="monotone" 
              dataKey="spending" 
              name="Average Spending" 
              stroke={chartColors.accentGreen} 
              strokeWidth={2} 
              dot={{ r: 5 }} 
              activeDot={{ r: 7 }} 
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderClustering = () => {
    if (!data || !data.clustering) {
      return <div>Loading clustering data...</div>;
    }
  
    const clustering = data.clustering;
    
    return (
      <div>
        <div style={cardStyle}>
          <h3 style={cardTitleStyle}>User Spending Clusters</h3>
          <p>Users grouped by similar financial behaviors using K-Means clustering</p>
          {renderClusterChart && renderClusterChart()}
        </div>
        <div style={cardStyle}>
          <h3 style={cardTitleStyle}>Cluster Visualization</h3>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart>
              <CartesianGrid />
              <XAxis type="number" dataKey="income" name="Income" unit=" Rs" />
              <YAxis type="number" dataKey="spending" name="Spending" unit=" Rs" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              <Scatter name="Clusters" data={prepareClusterData()} fill="#8884d8" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
  
        <div style={cardStyle}>
          <h3 style={cardTitleStyle}>Cluster Analysis</h3>
          <div style={{ marginTop: "20px", display: "flex", flexWrap: "wrap", gap: "15px" }}>
            {Object.entries(clustering)
              .filter(([key, _]) => key.startsWith('Cluster_'))
              .map(([cluster, stats]) => {
                const savingsRatio = (stats.avg_income || 0) / (stats.avg_spending || 1);
                const clusterColor = savingsRatio > 1.5 
                  ? colors.darkGreen 
                  : savingsRatio < 1.2 
                    ? "#d32f2f" 
                    : colors.mediumGreen;
                
                return (
                  <div key={cluster} style={{ 
                    flex: "1 1 300px",
                    padding: "20px", 
                    backgroundColor: colors.backgroundGreen,
                    borderRadius: "8px",
                    borderLeft: `5px solid ${clusterColor}`
                  }}>
                    <h4>Cluster {cluster.split('_')[1]}</h4>
                    
                    <div style={{ display: "flex", alignItems: "center", marginBottom: "10px" }}>
                      <div style={{ flex: "0 0 70px" }}><strong>Members:</strong></div>
                      <div style={{ 
                        flex: "1", 
                        height: "20px", 
                        backgroundColor: colors.veryLightGreen,
                        borderRadius: "10px"
                      }}>
                        <div style={{ 
                          width: `${Math.min((stats.pct || 0) * 100, 100)}%`, 
                          height: "100%", 
                          backgroundColor: clusterColor,
                          borderRadius: "10px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          color: "#fff",
                          fontSize: "12px",
                          fontWeight: "bold"
                        }}>
                          {stats.count || 0} ({((stats.pct || 0) * 100).toFixed(2)}%)
                        </div>
                      </div>
                    </div>
  
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "15px" }}>
                      <div>
                        <p><strong>Income:</strong> {formatCurrency(stats.avg_income || 0)}</p>
                        <p><strong>Spending:</strong> {formatCurrency(stats.avg_spending || 0)}</p>
                      </div>
                      <div style={{ 
                        width: "100px", 
                        height: "100px", 
                        borderRadius: "50%", 
                        backgroundColor: clusterColor,
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "center",
                        alignItems: "center",
                        color: "#fff"
                      }}>
                        <div style={{ fontSize: "12px" }}>Savings</div>
                        <div style={{ fontSize: "16px", fontWeight: "bold" }}>
                          {savingsRatio.toFixed(2)}x
                        </div>
                      </div>
                    </div>
  
                    <p><strong>Analysis:</strong> {
                      savingsRatio > 1.5
                        ? "High savers with significant income-to-spending ratio"
                        : savingsRatio < 1.2
                          ? "Potential overspenders with low savings potential"
                          : "Balanced spending with moderate savings"
                    }</p>
                    {stats.most_common_occupation && (
                      <p><strong>Most Common Occupation:</strong> {stats.most_common_occupation}</p>
                    )}
                    
                    {stats.most_common_city_tier && (
                      <p><strong>Most Common City Tier:</strong> {stats.most_common_city_tier}</p>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      </div>
    );
  };
  

  const RenderAnomalies = ({ data, colors, formatCurrency, cardStyle, cardTitleStyle }) => {
    const anomalies = data.anomalies || {};
    const [selectedFilter, setSelectedFilter] = useState({
      occupation: 'all',
      city_tier: 'all'
    });
    const [filteredData, setFilteredData] = useState(null);

    const normalPct = anomalies.normal_transactions 
      ? (anomalies.normal_transactions / (anomalies.normal_transactions + (anomalies.anomalous_transactions || 0))) * 100 
      : 0;
    
    const filterAnomaliesByDemographic = (demographicData, occupation, cityTier) => {
      if (!demographicData) return null;
      
      if (occupation === 'all' && cityTier === 'all') {
        return {
          normal: anomalies.normal_transactions || 0,
          anomalous: anomalies.anomalous_transactions || 0,
          avg_values: anomalies.avg_values || {}
        };
      }
      
      let filtered = { ...demographicData };
      
      // Filter by occupation if not 'all'
      if (occupation !== 'all') {
        filtered = filtered[occupation] || { normal: 0, anomalous: 0, avg_values: {} };
      }
      
      // Further filter by city_tier if not 'all'
      if (cityTier !== 'all' && filtered.by_city_tier) {
        filtered = filtered.by_city_tier[cityTier] || { normal: 0, anomalous: 0, avg_values: {} };
      }
      
      return filtered;
    };
    
    useEffect(() => {
      if (!anomalies.demographic_breakdown) return;
      
      const filtered = filterAnomaliesByDemographic(
        anomalies.demographic_breakdown,
        selectedFilter.occupation,
        selectedFilter.city_tier
      );
      
      setFilteredData(filtered);
    }, [anomalies, selectedFilter, filterAnomaliesByDemographic]);
    
    const getUniqueValues = () => {
      const occupations = new Set(['all']);
      const cityTiers = new Set(['all']);
      
      if (anomalies.demographic_breakdown) {
        
        Object.keys(anomalies.demographic_breakdown).forEach(occ => {
          if (occ !== 'by_city_tier') occupations.add(occ);
        });
        
        if (anomalies.demographic_breakdown.by_city_tier) {
          Object.keys(anomalies.demographic_breakdown.by_city_tier).forEach(tier => {
            cityTiers.add(tier);
          });
        }
      }
      
      return {
        occupations: Array.from(occupations),
        cityTiers: Array.from(cityTiers)
      };
    };
    
    const { occupations, cityTiers } = getUniqueValues();
    
    const displayData = filteredData || {
      normal: anomalies.normal_transactions || 0,
      anomalous: anomalies.anomalous_transactions || 0,
      avg_values: anomalies.avg_values || {}
    };
    
    const getAnomalyScoreDistribution = () => {
      if (!anomalies.score_distribution) return [];
      
      return Object.entries(anomalies.score_distribution).map(([score, count]) => ({
        score: parseFloat(score),
        count,
        isAnomaly: parseFloat(score) > anomalies.threshold
      }));
    };
    
    const scoreDistribution = getAnomalyScoreDistribution();
    
    const getDemographicInsights = () => {
      if (!anomalies.demographic_breakdown) return null;
      
      const insights = [];
      const overall = anomalies.anomaly_percentage || 0;
      
      Object.entries(anomalies.demographic_breakdown).forEach(([occ, data]) => {
        if (occ !== 'by_city_tier' && data.total > 0) {
          const rate = (data.anomalous / data.total) * 100;
          const difference = rate - overall;
          
          if (Math.abs(difference) > 5) { 
            insights.push({
              type: 'occupation',
              name: occ,
              rate,
              difference,
              impact: Math.abs(difference) * data.total / anomalies.total_transactions
            });
          }
        }
      });
      
      if (anomalies.demographic_breakdown.by_city_tier) {
        Object.entries(anomalies.demographic_breakdown.by_city_tier).forEach(([tier, data]) => {
          if (data.total > 0) {
            const rate = (data.anomalous / data.total) * 100;
            const difference = rate - overall;
            
            if (Math.abs(difference) > 5) { // Only show significant differences
              insights.push({
                type: 'city_tier',
                name: `Tier ${tier}`,
                rate,
                difference,
                impact: Math.abs(difference) * data.total / anomalies.total_transactions
              });
            }
          }
        });
      }
      
      return insights.sort((a, b) => b.impact - a.impact);
    };
    
    const demographicInsights = getDemographicInsights();
    
    return (
      <div>
        <div style={cardStyle}>
          <h3 style={cardTitleStyle}>Anomaly Detection Results</h3>
          <p>Using Isolation Forest to detect unusual spending patterns</p>
          
          {/* Filter controls */}
          <div style={{ 
            display: "flex", 
            gap: "15px", 
            marginTop: "15px",
            padding: "15px",
            backgroundColor: colors.veryLightGreen,
            borderRadius: "8px"
          }}>
            <div>
              <label htmlFor="occupation-filter" style={{ marginRight: "8px", fontWeight: "bold" }}>
                Occupation:
              </label>
              <select 
                id="occupation-filter"
                value={selectedFilter.occupation}
                onChange={(e) => setSelectedFilter({...selectedFilter, occupation: e.target.value})}
                style={{ padding: "6px", borderRadius: "4px", border: "1px solid #ccc" }}
              >
                {occupations.map(occ => (
                  <option key={occ} value={occ}>{occ === 'all' ? 'All Occupations' : occ}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label htmlFor="city-tier-filter" style={{ marginRight: "8px", fontWeight: "bold" }}>
                City Tier:
              </label>
              <select 
                id="city-tier-filter"
                value={selectedFilter.city_tier}
                onChange={(e) => setSelectedFilter({...selectedFilter, city_tier: e.target.value})}
                style={{ padding: "6px", borderRadius: "4px", border: "1px solid #ccc" }}
              >
                {cityTiers.map(tier => (
                  <option key={tier} value={tier}>
                    {tier === 'all' ? 'All City Tiers' : `Tier ${tier}`}
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <div style={{ display: "flex", flexWrap: "wrap", gap: "20px", marginTop: "20px" }}>
            <div style={{ flex: "1 1 300px", minHeight: "250px" }}>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={[
                      { name: "Normal Transactions", value: displayData.normal || 0, fill: colors.darkGreen },
                      { name: "Anomalous Transactions", value: displayData.anomalous || 0, fill: "#d32f2f" }
                    ]}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                    labelLine={true}
                  >
                  </Pie>
                  <Tooltip formatter={(value) => value} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            <div style={{ flex: "1 1 300px", display: "flex", flexDirection: "column", gap: "10px" }}>
              <div style={{ 
                flex: "1", 
                backgroundColor: colors.backgroundGreen, 
                padding: "15px", 
                borderRadius: "8px",
                display: "flex",
                alignItems: "center",
                gap: "15px"
              }}>
                <div style={{ 
                  width: "50px", 
                  height: "50px", 
                  borderRadius: "50%", 
                  backgroundColor: colors.darkGreen,
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  color: "#fff",
                  fontSize: "24px",
                  fontWeight: "bold"
                }}>
                  ✓
                </div>
                <div>
                  <h4>Normal Transactions</h4>
                  <p style={{ fontSize: "1.5rem", fontWeight: "bold" }}>{displayData.normal || 0}</p>
                </div>
              </div>
              
              <div style={{ 
                flex: "1", 
                backgroundColor: "#ffe8e8", 
                padding: "15px", 
                borderRadius: "8px",
                display: "flex",
                alignItems: "center",
                gap: "15px"
              }}>
                <div style={{ 
                  width: "50px", 
                  height: "50px", 
                  borderRadius: "50%", 
                  backgroundColor: "#d32f2f",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  color: "#fff",
                  fontSize: "24px",
                  fontWeight: "bold"
                }}>
                  !
                </div>
                <div>
                  <h4>Anomalous Transactions</h4>
                  <p style={{ fontSize: "1.5rem", fontWeight: "bold", color: "#d32f2f" }}>{displayData.anomalous || 0}</p>
                  <p style={{ fontSize: "1rem", color: "#d32f2f" }}>
                    {displayData.total > 0 
                      ? ((displayData.anomalous / displayData.total) * 100).toFixed(2) 
                      : (displayData.anomalous / (displayData.normal + displayData.anomalous) * 100).toFixed(2)}% of total
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Demographic Analysis */}
        <div style={cardStyle}>
          <h3 style={cardTitleStyle}>Demographic Analysis</h3>
          <p>Anomaly rates across different demographic segments</p>
          
          <div style={{ display: "flex", flexWrap: "wrap", gap: "20px", marginTop: "20px" }}>
            {/* Occupation-based anomaly rates */}
            <div style={{ flex: "1 1 400px", minHeight: "300px" }}>
              <h4 style={{ marginBottom: "10px" }}>Anomaly Rates by Occupation</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={
                    Object.entries(anomalies.demographic_breakdown || {})
                      .filter(([occ]) => occ !== 'by_city_tier')
                      .map(([occ, data]) => ({
                        name: occ,
                        rate: data.total > 0 ? (data.anomalous / data.total) * 100 : 0,
                        transactions: data.total || 0
                      }))
                      .sort((a, b) => b.rate - a.rate)
                  }
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                  <YAxis type="category" dataKey="name" width={100} />
                  <Tooltip 
                    formatter={(value) => `${value.toFixed(2)}%`}
                    labelFormatter={(value) => `Occupation: ${value}`}
                    contentStyle={{ backgroundColor: colors.veryLightGreen }}
                  />
                  <Bar 
                    dataKey="rate" 
                    name="Anomaly Rate" 
                    fill={colors.darkGreen}
                    background={{ fill: '#eee' }}
                  >
                    {/* Color bars based on comparison to overall rate */}
                    {
                      Object.entries(anomalies.demographic_breakdown || {})
                        .filter(([occ]) => occ !== 'by_city_tier')
                        .map(([occ, data]) => {
                          const rate = data.total > 0 ? (data.anomalous / data.total) * 100 : 0;
                          const overall = anomalies.anomaly_percentage || 0;
                          return (
                            <Cell 
                              key={`cell-${occ}`} 
                              fill={rate > overall * 1.5 ? '#d32f2f' : rate < overall * 0.5 ? colors.darkGreen : '#FF9800'}
                            />
                          );
                        })
                    }
                  </Bar>
                  <ReferenceLine 
                    x={anomalies.anomaly_percentage || 0} 
                    stroke="#000" 
                    strokeDasharray="3 3" 
                    label={{ value: 'Overall', position: 'insideTopRight' }} 
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {/* City Tier-based anomaly rates */}
            <div style={{ flex: "1 1 400px", minHeight: "300px" }}>
              <h4 style={{ marginBottom: "10px" }}>Anomaly Rates by City Tier</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={
                    Object.entries(anomalies.demographic_breakdown?.by_city_tier || {})
                      .map(([tier, data]) => ({
                        name: `Tier ${tier}`,
                        rate: data.total > 0 ? (data.anomalous / data.total) * 100 : 0,
                        transactions: data.total || 0
                      }))
                      .sort((a, b) => b.rate - a.rate)
                  }
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis tickFormatter={(value) => `${value}%`} />
                  <Tooltip 
                    formatter={(value) => `${value.toFixed(2)}%`}
                    labelFormatter={(value) => `${value}`}
                    contentStyle={{ backgroundColor: colors.veryLightGreen }}
                  />
                  <Bar 
                    dataKey="rate" 
                    name="Anomaly Rate" 
                    fill={colors.darkGreen}
                  >
                    {/* Color bars based on comparison to overall rate */}
                    {
                      Object.entries(anomalies.demographic_breakdown?.by_city_tier || {})
                        .map(([tier, data]) => {
                          const rate = data.total > 0 ? (data.anomalous / data.total) * 100 : 0;
                          const overall = anomalies.anomaly_percentage || 0;
                          return (
                            <Cell 
                              key={`cell-${tier}`} 
                              fill={rate > overall * 1.5 ? '#d32f2f' : rate < overall * 0.5 ? colors.darkGreen : '#FF9800'}
                            />
                          );
                        })
                    }
                  </Bar>
                  <ReferenceLine 
                    y={anomalies.anomaly_percentage || 0} 
                    stroke="#000" 
                    strokeDasharray="3 3" 
                    label={{ value: 'Overall', angle: 90, position: 'insideLeft' }} 
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Key Demographic Insights */}
          {demographicInsights && demographicInsights.length > 0 && (
            <div style={{ marginTop: "30px" }}>
              <h4>Key Demographic Insights</h4>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "15px", marginTop: "15px" }}>
                {demographicInsights.slice(0, 3).map((insight, idx) => (
                  <div key={idx} style={{
                    flex: "1 1 300px",
                    backgroundColor: insight.difference > 0 ? "#ffe8e8" : "#e8f5e9",
                    padding: "15px",
                    borderRadius: "8px",
                    border: `1px solid ${insight.difference > 0 ? "#ffcdd2" : "#c8e6c9"}`
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "10px" }}>
                      <div style={{
                        width: "40px",
                        height: "40px",
                        borderRadius: "50%",
                        backgroundColor: insight.difference > 0 ? "#d32f2f" : colors.darkGreen,
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        color: "#fff",
                        fontSize: "20px"
                      }}>
                        {insight.difference > 0 ? "!" : "✓"}
                      </div>
                      <div>
                        <h5 style={{ margin: 0 }}>{insight.name}</h5>
                        <p style={{ margin: 0, fontSize: "0.8rem" }}>{insight.type === 'occupation' ? 'Occupation' : 'City Tier'}</p>
                      </div>
                    </div>
                    
                    <p style={{ 
                      fontWeight: "bold", 
                      color: insight.difference > 0 ? "#d32f2f" : "#2e7d32",
                      fontSize: "1.1rem",
                      margin: "10px 0"
                    }}>
                      {insight.rate.toFixed(2)}% anomaly rate
                      <span style={{ 
                        fontSize: "0.9rem", 
                        backgroundColor: insight.difference > 0 ? "#ffebee" : "#e8f5e9",
                        padding: "2px 6px",
                        borderRadius: "4px",
                        marginLeft: "8px"
                      }}>
                        {insight.difference > 0 ? "+" : ""}{insight.difference.toFixed(2)}% vs. overall
                      </span>
                    </p>
                    
                    <p style={{ fontSize: "0.9rem", margin: 0 }}>
                      {insight.difference > 0 
                        ? `${insight.name} shows a significantly higher anomaly rate than the overall population.`
                        : `${insight.name} shows a significantly lower anomaly rate than the overall population.`
                      }
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Anomaly Score Distribution */}
        {scoreDistribution.length > 0 && (
          <div style={cardStyle}>
            <h3 style={cardTitleStyle}>Anomaly Score Distribution</h3>
            <p>Distribution of isolation forest anomaly scores across all transactions</p>
            
            <div style={{ marginTop: "20px" }}>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={scoreDistribution}
                  margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="score" 
                    type="number"
                    domain={[0, 1]}
                    ticks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
                    label={{ value: 'Anomaly Score', position: 'insideBottom', offset: -20 }}
                  />
                  <YAxis 
                    label={{ value: 'Number of Transactions', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    formatter={(value, name) => [value, 'Transactions']}
                    labelFormatter={(value) => `Score: ${value.toFixed(2)}`}
                  />
                  <Bar dataKey="count" name="Transactions">
                    {
                      scoreDistribution.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={entry.isAnomaly ? '#d32f2f' : colors.darkGreen} 
                        />
                      ))
                    }
                  </Bar>
                  {anomalies.threshold && (
                    <ReferenceLine 
                      x={anomalies.threshold} 
                      stroke="#000" 
                      strokeWidth={2}
                      label={{ value: 'Anomaly Threshold', position: 'top' }} 
                    />
                  )}
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            <div style={{
              display: "flex",
              justifyContent: "center",
              gap: "20px",
              marginTop: "10px"
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                <div style={{ width: "15px", height: "15px", backgroundColor: colors.darkGreen, borderRadius: "2px" }}></div>
                <span>Normal</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                <div style={{ width: "15px", height: "15px", backgroundColor: "#d32f2f", borderRadius: "2px" }}></div>
                <span>Anomalous</span>
              </div>
            </div>
          </div>
        )}

        {displayData.avg_values && (
          <div style={cardStyle}>
            <h3 style={cardTitleStyle}>Anomaly Comparison</h3>
            <p>Comparing average spending between normal and anomalous transactions</p>
            
            <div style={{ marginTop: "20px" }}>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart 
                  data={
                    Object.keys(displayData.avg_values.normal || {}).map(category => {
                      const normalValue = (displayData.avg_values.normal || {})[category] || 0;
                      const anomalousValue = (displayData.avg_values.anomalous || {})[category] || 0;
                      return {
                        category,
                        normal: normalValue,
                        anomalous: anomalousValue,
                        difference: anomalousValue - normalValue
                      };
                    })
                  }
                  layout="vertical"
                  margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="category" width={100} />
                  <Tooltip 
                    formatter={(value) => formatCurrency(value)}
                    contentStyle={{ backgroundColor: colors.veryLightGreen }}
                  />
                  <Legend />
                  <Bar dataKey="normal" name="Normal Avg" fill={colors.darkGreen} />
                  <Bar dataKey="anomalous" name="Anomalous Avg" fill="#d32f2f" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
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
                {displayData.avg_values && displayData.avg_values.normal && Object.keys(displayData.avg_values.normal).map(category => {
                  const normalValue = (displayData.avg_values.normal || {})[category] || 0;
                  const anomalousValue = (displayData.avg_values.anomalous || {})[category] || 0;
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
                        color: difference > 0 ? "#d32f2f" : "#2e7d32",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "flex-end",
                        gap: "5px"
                      }}>
                        {difference > 0 ? "+" : ""}{formatCurrency(difference)} 
                        <span style={{
                          backgroundColor: difference > 0 ? "#ffebee" : "#e8f5e9",
                          padding: "2px 6px",
                          borderRadius: "4px",
                          fontSize: "0.8rem"
                        }}>
                          {difference > 0 ? "+" : ""}{percentDiff.toFixed(1)}%
                        </span>
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

  const renderAnomalies = () => {
    return <RenderAnomalies 
      data={data} 
      colors={colors} 
      formatCurrency={formatCurrency} 
      cardStyle={cardStyle} 
      cardTitleStyle={cardTitleStyle}
    />;
  };


  const renderRecommendations = () => {
    const recommendations = data.recommendations || {};
    const insights = data.insights || {};
  
    const renderSavingsOpportunitiesImproved = () => {
      const savingsData = prepareSavingsOpportunitiesData();
      if (savingsData.length === 0) return <p>No savings opportunities data available</p>;
  
      return (
        <div style={{ width: '100%', height: 350 }}>
          <ResponsiveContainer>
            <BarChart 
              data={savingsData.sort((a, b) => b.value - a.value)} 
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={100} />
              <Tooltip 
                formatter={(value) => formatCurrency(value)} 
                contentStyle={{ backgroundColor: colors.veryLightGreen }}
              />
              <Legend />
              <Bar 
                dataKey="value" 
                name="Potential Savings" 
                fill={colors.accentGreen}
                radius={[0, 4, 4, 0]}
                label={{ position: 'right', formatter: (value) => formatCurrency(value) }}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      );
    };
  
    return (
      <div>
        <div style={cardStyle}>
          <h3 style={cardTitleStyle}>Savings Opportunities</h3>
          <p>Areas where you could potentially save money based on your spending patterns</p>
          {renderSavingsOpportunitiesImproved()}
        </div>
  
        {recommendations.savings_factors && (
          <div style={cardStyle}>
            <h3 style={cardTitleStyle}>Factors Affecting Savings</h3>
            <p>Based on regression analysis of income and expenses, including Occupation and City Tier</p>
  
            <div style={{ marginTop: "20px" }}>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart 
                  data={Object.entries(recommendations.savings_factors).map(([factor, impact]) => ({
                    factor,
                    impact
                  }))}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="factor" width={100} />
                  <Tooltip />
                  <Bar 
                    dataKey="impact" 
                    name="Impact on Savings"
                    fill={(data) => data.impact > 0 ? colors.darkGreen : "#d32f2f"}
                    radius={[0, 4, 4, 0]}
                  />
                  <ReferenceLine x={0} stroke="#000" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "20px" }}>
              <thead>
                <tr style={{ backgroundColor: colors.veryLightGreen }}>
                  <th style={{ padding: "12px", textAlign: "left", borderBottom: "2px solid #ddd" }}>Factor</th>
                  <th style={{ padding: "12px", textAlign: "right", borderBottom: "2px solid #ddd" }}>Impact on Savings</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(recommendations.savings_factors || {})
                  .sort((a, b) => b[1] - a[1])
                  .map(([factor, impact]) => (
                    <tr key={factor} style={{ borderBottom: "1px solid #ddd" }}>
                      <td style={{ padding: "10px" }}>{factor}</td>
                      <td style={{ padding: "10px", textAlign: "right" }}>
                        <span style={{
                          backgroundColor: impact > 0 ? colors.backgroundGreen : "#ffebee",
                          color: impact > 0 ? "#2e7d32" : "#d32f2f",
                          padding: "5px 10px",
                          borderRadius: "4px",
                          fontWeight: "bold"
                        }}>
                          {impact > 0 ? "+" : ""}{impact.toFixed(2)}
                        </span>
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
            <p>Association rules discovered in your financial data</p>
  
            <div style={{ marginTop: "20px", display: "flex", flexWrap: "wrap", gap: "15px" }}>
              {recommendations.association_rules.map((rule, index) => {
                const liftStrength = (rule.lift || 0);
                const getStrengthColor = () => {
                  if (liftStrength > 2) return colors.darkGreen;
                  if (liftStrength > 1.5) return colors.mediumGreen;
                  return colors.accentGreen;
                };
  
                return (
                  <div key={index} style={{ 
                    flex: "1 1 350px",
                    padding: "20px", 
                    backgroundColor: colors.backgroundGreen,
                    borderRadius: "8px",
                    borderLeft: `5px solid ${getStrengthColor()}`
                  }}>
                    <div style={{ 
                      display: "flex", 
                      alignItems: "center", 
                      justifyContent: "space-between",
                      gap: "20px",
                      marginBottom: "15px"
                    }}>
                      <div style={{ flex: "1" }}>
                        <div style={{ padding: "10px", backgroundColor: colors.veryLightGreen, borderRadius: "6px" }}>
                          {rule.antecedents?.join(" and ") || "N/A"}
                        </div>
                      </div>
                      <div style={{ 
                        width: "30px", 
                        height: "30px", 
                        display: "flex", 
                        justifyContent: "center", 
                        alignItems: "center",
                        fontSize: "18px",
                        fontWeight: "bold"
                      }}>
                        →
                      </div>
                      <div style={{ flex: "1" }}>
                        <div style={{ padding: "10px", backgroundColor: colors.veryLightGreen, borderRadius: "6px" }}>
                          {rule.consequents?.join(" and ") || "N/A"}
                        </div>
                      </div>
                    </div>
  
                    <div style={{ display: "flex", justifyContent: "space-between", gap: "10px" }}>
                      <div style={{
                        flex: "1",
                        padding: "10px",
                        textAlign: "center",
                        backgroundColor: colors.veryLightGreen,
                        borderRadius: "6px"
                      }}>
                        <div style={{ fontSize: "0.8rem", color: "#666" }}>Confidence</div>
                        <div style={{ fontWeight: "bold" }}>{((rule.confidence || 0) * 100).toFixed(2)}%</div>
                      </div>
                      <div style={{
                        flex: "1",
                        padding: "10px",
                        textAlign: "center",
                        backgroundColor: colors.veryLightGreen,
                        borderRadius: "6px"
                      }}>
                        <div style={{ fontSize: "0.8rem", color: "#666" }}>Support</div>
                        <div style={{ fontWeight: "bold" }}>{((rule.support || 0) * 100).toFixed(2)}%</div>
                      </div>
                      <div style={{
                        flex: "1",
                        padding: "10px",
                        textAlign: "center",
                        backgroundColor: colors.veryLightGreen,
                        borderRadius: "6px"
                      }}>
                        <div style={{ fontSize: "0.8rem", color: "#666" }}>Lift</div>
                        <div style={{ fontWeight: "bold" }}>{(rule.lift || 0).toFixed(2)}x</div>
                      </div>
                    </div>
  
                    <p style={{ marginTop: "15px" }}>
                      <strong>Interpretation:</strong>{" "}
                      {(rule.lift || 0) > 2 
                        ? "Very strong relationship between these financial behaviors" 
                        : (rule.lift || 0) > 1.5 
                          ? "Strong relationship between these financial behaviors" 
                          : "Moderate relationship between these financial behaviors"}
                    </p>
                  </div>
                );
              })}
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
  
            <div style={{ display: "flex", flexWrap: "wrap", gap: "15px" }}>
              <div style={{
                flex: "1 1 250px",
                padding: "15px",
                backgroundColor: colors.veryLightGreen,
                borderRadius: "8px",
                borderLeft: `5px solid ${colors.darkGreen}`
              }}>
                <h5 style={{ marginBottom: "10px", display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ 
                    width: "24px", 
                    height: "24px", 
                    borderRadius: "50%", 
                    backgroundColor: colors.darkGreen,
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    color: "#fff",
                    fontSize: "14px"
                  }}>1</span>
                  Budget Optimization
                </h5>
                <p>Focus on reducing spending in {insights?.most_spent_category || "highest expense categories"}.</p>
              </div>
  
              <div style={{
                flex: "1 1 250px",
                padding: "15px",
                backgroundColor: colors.veryLightGreen,
                borderRadius: "8px",
                borderLeft: `5px solid ${colors.mediumGreen}`
              }}>
                <h5 style={{ marginBottom: "10px", display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ 
                    width: "24px", 
                    height: "24px", 
                    borderRadius: "50%", 
                    backgroundColor: colors.mediumGreen,
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    color: "#fff",
                    fontSize: "14px"
                  }}>2</span>
                  Savings Target
                </h5>
                <p>Consider saving {formatCurrency(recommendations.avg_predicted_savings || 0)} monthly based on regression analysis.</p>
              </div>
  
              <div style={{
                flex: "1 1 250px",
                padding: "15px",
                backgroundColor: colors.veryLightGreen,
                borderRadius: "8px",
                borderLeft: `5px solid ${(data.anomalies?.anomalous_transactions || 0) > 0 ? "#d32f2f" : colors.accentGreen}`
              }}>
                <h5 style={{ marginBottom: "10px", display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ 
                    width: "24px", 
                    height: "24px", 
                    borderRadius: "50%", 
                    backgroundColor: (data.anomalies?.anomalous_transactions || 0) > 0 ? "#d32f2f" : colors.accentGreen,
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    color: "#fff",
                    fontSize: "14px"
                  }}>3</span>
                  Spending Alert
                </h5>
                <p>
                  {(data.anomalies?.anomalous_transactions || 0) > 0 
                    ? `You have ${data.anomalies?.anomalous_transactions || 0} potentially anomalous transactions to review.`
                    : "No anomalous spending patterns detected in your data."}
                </p>
              </div>
  
              <div style={{
                flex: "1 1 250px",
                padding: "15px",
                backgroundColor: colors.veryLightGreen,
                borderRadius: "8px",
                borderLeft: `5px solid ${colors.accentGreen}`
              }}>
                <h5 style={{ marginBottom: "10px", display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ 
                    width: "24px", 
                    height: "24px", 
                    borderRadius: "50%", 
                    backgroundColor: colors.accentGreen,
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    color: "#fff",
                    fontSize: "14px"
                  }}>4</span>
                  Investment Strategy
                </h5>
                <p>Consider exploring investment options based on your long-term financial goals.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };
  

  const renderEvaluationMetrics = () => {

    const anomalies = data?.anomalies || {};
    const recommendations = data?.recommendations || {};
    const clustering = prepareClusterData();

    const metricsData = {
      clustering: {
        silhouetteScore: clustering?.silhouette_score,
        inertia: clustering?.inertia,
        n_clusters: clustering?.n_clusters
      },
      anomalyDetection: {
        accuracy: anomalies?.accuracy,
        precision: anomalies?.precision,
        recall: anomalies?.recall,
        f1Score: anomalies?.f1_score,
        roc_auc: anomalies?.roc_auc
      },
      recommendations: {
        r2_score: recommendations?.r2_score,
        mse: recommendations?.mse,
        avg_support: recommendations?.avg_rule_support,
        avg_confidence: recommendations?.avg_rule_confidence,
        feature_importance: recommendations?.feature_importance
      }
    };

    return (
      <renderEvaluationMetrics data={{ metrics: metricsData }} loading={false} />
    );
  };
  if (!data) return null;

  return (
    <div style={{ ...containerStyle, fontFamily: "'Cal Sans', sans-serif" }}>
      <h1 style={{ color: colors.darkGreen, marginBottom: "30px", fontFamily: "'Cal Sans', sans-serif"}}>Financial Insights</h1>
      
      <div style={{ marginBottom: "25px" }}>
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
        {activeTab === "clusters" && renderClustering()}
        {activeTab === "anomalies" && renderAnomalies()}
        {activeTab === "recommendations" && renderRecommendations()}
        {activeTab === "chart components" && (
          <>
            {renderExpenseChart()}
            {renderIncomeByAgeChart()}
          </>
        )}
        {activeTab === "evaluation" && renderEvaluationMetrics()}

        <Box sx={{ display: "flex", justifyContent: "space-between", mt: 4 }}>
        <Button variant="outlined" color="primary" onClick={() => navigate("/")} style={{ backgroundColor: colors.backgroundGreen, color: colors.darkGreen, borderColor: colors.darkGreen }}>
          <UploadFile sx={{ mr: 1 }} /> Upload New Data
        </Button>
        <Button variant="contained" color="primary" onClick={() => {
          const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
          const downloadAnchorNode = document.createElement('a');
          downloadAnchorNode.setAttribute("href", dataStr);
          downloadAnchorNode.setAttribute("download", "finance_analysis_results.json");
          document.body.appendChild(downloadAnchorNode);
          downloadAnchorNode.click();
          downloadAnchorNode.remove();
        }} style={{ backgroundColor: colors.mediumGreen }}>
          Export Analysis Results
        </Button>
      </Box>
      </div>
    </div>
  );
}

export default Insights;