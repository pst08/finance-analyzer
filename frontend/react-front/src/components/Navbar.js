import React from "react";
import { NavLink } from "react-router-dom";
import { FaHome, FaChartBar, FaCalendarAlt, FaTable } from "react-icons/fa";

function Navbar() {
  // Define the forest green theme colors
  const colors = {
    darkGreen: "#1b4332",
    mediumGreen: "#2d6a4f",
    lightGreen: "#40916c",
    hoverGreen: "#52b788",
    textLight: "#d8f3dc"
  };

  // Style for the sidebar container
  const sidebarStyle = {
    width: "240px",
    minHeight: "100vh",
    backgroundColor: colors.darkGreen,
    color: colors.textLight,
    padding: "20px 0",
    boxShadow: "2px 0 5px rgba(0,0,0,0.2)",
    position: "sticky",
    top: 0,
    display: "flex",
    flexDirection: "column"
  };

  // Style for the app title
  const titleStyle = {
    textAlign: "center",
    padding: "10px 20px 30px",
    fontSize: "18px",
    fontWeight: "bold",
    color: colors.textLight,
    borderBottom: `1px solid ${colors.mediumGreen}`,
    letterSpacing: "1.3px"
  };

  // Style for each navigation link
  const navLinkStyle = {
    display: "flex",
    alignItems: "center",
    padding: "15px 20px",
    color: colors.textLight,
    textDecoration: "none",
    transition: "background-color 0.3s, border-left 0.3s",
    fontSize: "16px",
    borderLeft: "4px solid transparent"
  };

  // Active link style
  const activeStyle = {
    backgroundColor: colors.mediumGreen,
    borderLeft: `4px solid ${colors.hoverGreen}`
  };

  // Style for the icon in each link
  const iconStyle = {
    marginRight: "12px",
    fontSize: "18px"
  };

  return (
    <div style={sidebarStyle}>
      <div style={titleStyle}>
        Personal Finance Analyzer
      </div>
      
      <nav style={{ marginTop: "20px", flex: 1 }}>
        <NavLink 
          to="/" 
          style={({ isActive }) => ({
            ...navLinkStyle,
            ...(isActive ? activeStyle : {})
          })}
          end
        >
          <FaHome style={iconStyle} />
          Home
        </NavLink>
        
        <NavLink 
          to="/insights" 
          style={({ isActive }) => ({
            ...navLinkStyle,
            ...(isActive ? activeStyle : {})
          })}
        >
          <FaChartBar style={iconStyle} />
          Insights
        </NavLink>
        
        <NavLink 
          to="/forecasting" 
          style={({ isActive }) => ({
            ...navLinkStyle,
            ...(isActive ? activeStyle : {})
          })}
        >
          <FaCalendarAlt style={iconStyle} />
          Forecasting
        </NavLink>
        
        <NavLink 
          to="/display-table" 
          style={({ isActive }) => ({
            ...navLinkStyle,
            ...(isActive ? activeStyle : {})
          })}
        >
          <FaTable style={iconStyle} />
          Display Table
        </NavLink>
      </nav>
      
      <div style={{ 
        padding: "15px 20px", 
        fontSize: "12px", 
        textAlign: "center",
        color: "rgba(216, 243, 220, 0.7)",
        borderTop: `1px solid ${colors.mediumGreen}`
      }}>
        Data Mining Project © 2023
      </div>
    </div>
  );
}

export default Navbar;