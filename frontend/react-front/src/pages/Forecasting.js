import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell
} from "recharts";

function Forecasting() {
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedForecastType, setSelectedForecastType] = useState("savings");
  const [months, setMonths] = useState(6);
  const [savingsGoal, setSavingsGoal] = useState(10000);
  
  // Forest green theme colors
  const colors = {
    darkGreen: "#1b4332",
    mediumGreen: "#2d6a4f",
    lightGreen: "#40916c",
    accentGreen: "#52b788",
    paleGreen: "#74c69d",
    veryLightGreen: "#b7e4c7",
    backgroundGreen: "#d8f3dc",
    chartColors: ["#1b4332", "#2d6a4f", "#40916c", "#52b788", "#74c69d"]
  };

  // Container styles
  const containerStyle = {
    padding: "30px",
    backgroundColor: "#f8f9fa",
    minHeight: "100vh",
    color: "#333"
  };

  // Card styles
  const cardStyle = {
    backgroundColor: "white",
    borderRadius: "8px",
    padding: "20px",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
    marginBottom: "20px"
  };

  // Button styles
  const buttonStyle = {
    backgroundColor: colors.mediumGreen,
    color: "white",
    border: "none",
    padding: "10px 15px",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: "16px",
    marginRight: "10px"
  };

  // Input styles
  const inputStyle = {
    padding: "8px 12px",
    border: `1px solid ${colors.paleGreen}`,
    borderRadius: "4px",
    marginRight: "10px",
    fontSize: "16px"
  };

  // Select styles
  const selectStyle = {
    padding: "8px 12px",
    border: `1px solid ${colors.paleGreen}`,
    borderRadius: "4px",
    marginRight: "10px",
    fontSize: "16px",
    backgroundColor: "white"
  };

  // Form group styles
  const formGroupStyle = {
    marginBottom: "15px",
    display: "flex",
    alignItems: "center"
  };

  const labelStyle = {
    marginRight: "10px",
    minWidth: "120px",
    fontWeight: "500"
  };

  // Header styles
  const headerStyle = {
    color: colors.darkGreen,
    marginBottom: "20px"
  };

  // Load financial data from API
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await axios.get("/api/financial-data");
        setData(response.data);
        setLoading(false);
      } catch (err) {
        setError("Failed to fetch financial data. Please try again later.");
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Generate forecast data based on type and parameters
  const generateForecast = () => {
    setLoading(true);
    
    // In a real app, this would call an API endpoint
    // For this example, we'll generate mock forecast data
    
    setTimeout(() => {
      try {
        let forecast = [];
        const currentDate = new Date();
        
        if (selectedForecastType === "savings") {
          // Mock savings forecast data
          const monthlySavings = data ? calculateAverageMonthlySavings(data) : 1500;
          let cumulativeSavings = data ? getCurrentSavings(data) : 5000;
          
          for (let i = 0; i < months; i++) {
            const forecastDate = new Date(currentDate);
            forecastDate.setMonth(currentDate.getMonth() + i);
            
            cumulativeSavings += monthlySavings;
            
            forecast.push({
              month: forecastDate.toLocaleString('default', { month: 'short', year: 'numeric' }),
              savings: cumulativeSavings,
              goal: savingsGoal,
              projected: cumulativeSavings
            });
          }
        } else if (selectedForecastType === "expenses") {
          // Mock expense forecast data
          const monthlyExpenses = data ? calculateAverageMonthlyExpenses(data) : 2500;
          
          for (let i = 0; i < months; i++) {
            const forecastDate = new Date(currentDate);
            forecastDate.setMonth(currentDate.getMonth() + i);
            
            // Add some randomness to make data more realistic
            const randomFactor = 0.9 + Math.random() * 0.3; // 0.9 to 1.2
            
            forecast.push({
              month: forecastDate.toLocaleString('default', { month: 'short', year: 'numeric' }),
              expenses: Math.round(monthlyExpenses * randomFactor),
              average: monthlyExpenses
            });
          }
        } else if (selectedForecastType === "income") {
          // Mock income forecast data
          const monthlyIncome = data ? calculateAverageMonthlyIncome(data) : 4000;
          
          for (let i = 0; i < months; i++) {
            const forecastDate = new Date(currentDate);
            forecastDate.setMonth(currentDate.getMonth() + i);
            
            // Add some growth over time (1% monthly growth)
            const growthFactor = 1 + (0.01 * i);
            
            forecast.push({
              month: forecastDate.toLocaleString('default', { month: 'short', year: 'numeric' }),
              income: Math.round(monthlyIncome * growthFactor),
              baseline: monthlyIncome
            });
          }
        }
        
        setForecastData(forecast);
        setLoading(false);
      } catch (err) {
        setError("Failed to generate forecast. Please try again.");
        setLoading(false);
      }
    }, 1000); // Simulate API call delay
  };

  // Helper functions for calculations
  const calculateAverageMonthlySavings = (data) => {
    // In a real app, calculate this from actual data
    return data.monthlySavings || 1500;
  };

  const calculateAverageMonthlyExpenses = (data) => {
    // In a real app, calculate this from actual data
    return data.monthlyExpenses || 2500;
  };

  const calculateAverageMonthlyIncome = (data) => {
    // In a real app, calculate this from actual data
    return data.monthlyIncome || 4000;
  };

  const getCurrentSavings = (data) => {
    // In a real app, get this from actual data
    return data.currentSavings || 5000;
  };

  // Calculate savings forecast progress percentage
  const calculateSavingsProgress = () => {
    if (!forecastData || forecastData.length === 0) return 0;
    const lastForecast = forecastData[forecastData.length - 1];
    return Math.min(Math.round((lastForecast.projected / savingsGoal) * 100), 100);
  };

  // Render appropriate chart based on forecast type
  const renderChart = () => {
    if (!forecastData || forecastData.length === 0) return null;

    if (selectedForecastType === "savings") {
      return (
        <div style={{ height: "400px", width: "100%" }}>
          <h3 style={headerStyle}>Savings Forecast</h3>
          <ResponsiveContainer>
            <LineChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" stroke={colors.veryLightGreen} />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="projected" 
                stroke={colors.mediumGreen} 
                strokeWidth={2} 
                dot={{ fill: colors.mediumGreen }} 
                name="Projected Savings"
              />
              <Line 
                type="monotone" 
                dataKey="goal" 
                stroke={colors.accentGreen} 
                strokeWidth={2} 
                strokeDasharray="5 5" 
                dot={false} 
                name="Savings Goal"
              />
            </LineChart>
          </ResponsiveContainer>
          
          <div style={{ marginTop: "20px", textAlign: "center" }}>
            <h4>Progress Towards Goal: {calculateSavingsProgress()}%</h4>
            <div style={{ 
              backgroundColor: colors.backgroundGreen, 
              height: "20px", 
              borderRadius: "10px",
              overflow: "hidden"
            }}>
              <div style={{ 
                backgroundColor: colors.accentGreen, 
                height: "100%", 
                width: `${calculateSavingsProgress()}%`,
                borderRadius: "10px"
              }} />
            </div>
          </div>
        </div>
      );
    } else if (selectedForecastType === "expenses") {
      return (
        <div style={{ height: "400px", width: "100%" }}>
          <h3 style={headerStyle}>Expense Forecast</h3>
          <ResponsiveContainer>
            <BarChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" stroke={colors.veryLightGreen} />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="expenses" fill={colors.lightGreen} name="Projected Expenses" />
              <Line 
                type="monotone" 
                dataKey="average" 
                stroke={colors.darkGreen} 
                strokeWidth={2} 
                strokeDasharray="5 5" 
                dot={false} 
                name="Average Monthly Expenses"
              />
            </BarChart>
          </ResponsiveContainer>
          
          <div style={{ marginTop: "20px" }}>
            <h4>Expense Categories</h4>
            <div style={{ height: "200px" }}>
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={[
                      { name: "Housing", value: 40 },
                      { name: "Food", value: 20 },
                      { name: "Transportation", value: 15 },
                      { name: "Utilities", value: 10 },
                      { name: "Entertainment", value: 15 }
                    ]}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill={colors.mediumGreen}
                    dataKey="value"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {[0, 1, 2, 3, 4].map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={colors.chartColors[index % colors.chartColors.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      );
    } else if (selectedForecastType === "income") {
      return (
        <div style={{ height: "400px", width: "100%" }}>
          <h3 style={headerStyle}>Income Forecast</h3>
          <ResponsiveContainer>
            <LineChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" stroke={colors.veryLightGreen} />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="income" 
                stroke={colors.darkGreen} 
                strokeWidth={2} 
                dot={{ fill: colors.darkGreen }} 
                name="Projected Income"
              />
              <Line 
                type="monotone" 
                dataKey="baseline" 
                stroke={colors.paleGreen} 
                strokeWidth={2} 
                strokeDasharray="5 5" 
                dot={false} 
                name="Baseline Income"
              />
            </LineChart>
          </ResponsiveContainer>
          
          <div style={{ marginTop: "20px" }}>
            <h4>Income Growth</h4>
            {forecastData && forecastData.length > 0 && (
              <p>
                Your income is projected to grow from {forecastData[0].income.toLocaleString('en-US', { style: 'currency', currency: 'INR' })} to {forecastData[forecastData.length - 1].income.toLocaleString('en-US', { style: 'currency', currency: 'USD' })} over the next {months} months.
              </p>
            )}
          </div>
        </div>
      );
    }
    
    return null;
  };

  return (
    <div style={containerStyle}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "20px" }}>
        <h1 style={headerStyle}>Financial Forecasting</h1>
        <button
          style={buttonStyle}
          onClick={() => navigate("/dashboard")}
        >
          Back to Dashboard
        </button>
      </div>

      <div style={cardStyle}>
        <h2 style={headerStyle}>Generate Forecast</h2>
        
        <div style={formGroupStyle}>
          <label style={labelStyle}>Forecast Type:</label>
          <select
            style={selectStyle}
            value={selectedForecastType}
            onChange={(e) => setSelectedForecastType(e.target.value)}
          >
            <option value="savings">Savings Growth</option>
            <option value="expenses">Expense Projections</option>
            <option value="income">Income Forecast</option>
          </select>
        </div>
        
        <div style={formGroupStyle}>
          <label style={labelStyle}>Forecast Period:</label>
          <select
            style={selectStyle}
            value={months}
            onChange={(e) => setMonths(parseInt(e.target.value))}
          >
            <option value="3">3 Months</option>
            <option value="6">6 Months</option>
            <option value="12">12 Months</option>
            <option value="24">24 Months</option>
          </select>
        </div>
        
        {selectedForecastType === "savings" && (
          <div style={formGroupStyle}>
            <label style={labelStyle}>Savings Goal:</label>
            <input
              type="number"
              style={inputStyle}
              value={savingsGoal}
              onChange={(e) => setSavingsGoal(parseInt(e.target.value))}
              min="1000"
              step="1000"
            />
            <span>USD</span>
          </div>
        )}
        
        <button
          style={buttonStyle}
          onClick={generateForecast}
          disabled={loading}
        >
          {loading ? "Generating..." : "Generate Forecast"}
        </button>
      </div>

      {error && (
        <div style={{ 
          backgroundColor: "#f8d7da", 
          color: "#721c24", 
          padding: "10px", 
          borderRadius: "4px",
          marginBottom: "20px" 
        }}>
          {error}
        </div>
      )}

      {forecastData && forecastData.length > 0 && (
        <div style={cardStyle}>
          {renderChart()}
        </div>
      )}

      {/* Financial Tips Section */}
      <div style={cardStyle}>
        <h2 style={headerStyle}>Financial Tips</h2>
        <div style={{ display: "flex", flexWrap: "wrap" }}>
          <div style={{ flex: "1 1 300px", marginRight: "20px", marginBottom: "20px" }}>
            <h3 style={{ color: colors.mediumGreen }}>Savings Tips</h3>
            <ul style={{ paddingLeft: "20px" }}>
              <li>Set up automatic transfers to your savings account</li>
              <li>Follow the 50/30/20 rule for budgeting</li>
              <li>Cut unnecessary subscriptions and memberships</li>
              <li>Consider a high-yield savings account</li>
            </ul>
          </div>
          <div style={{ flex: "1 1 300px" }}>
            <h3 style={{ color: colors.mediumGreen }}>Investment Tips</h3>
            <ul style={{ paddingLeft: "20px" }}>
              <li>Start investing early, even with small amounts</li>
              <li>Diversify your investment portfolio</li>
              <li>Consider low-cost index funds for beginners</li>
              <li>Max out retirement accounts for tax benefits</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Forecasting;