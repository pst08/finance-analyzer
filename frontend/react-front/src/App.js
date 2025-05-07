import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Insights from "./pages/Insights";
import Forecasting from "./pages/Forecasting";

function App() {
  return (
    <Router>
      <div style={{ display: 'flex' }}>
        <Navbar />
        <div style={{ flexGrow: 1, overflowY: 'auto', height: '100vh' }}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/insights" element={<Insights />} />
            <Route path="/forecasting" element={<Forecasting />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;