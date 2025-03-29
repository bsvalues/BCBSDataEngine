import React, { useState, useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';

/**
 * WhatIfAnalysis Component
 * 
 * This component provides an interactive interface for users to adjust valuation parameters
 * and see the impact on property valuations in real-time.
 * 
 * It uses sliders for parameter adjustment and Chart.js for data visualization.
 */
const WhatIfAnalysis = ({ propertyId, initialValuation }) => {
  // State to track the current property data
  const [property, setProperty] = useState(null);
  // State to track the original valuation for comparison
  const [originalValuation, setOriginalValuation] = useState(null);
  // State to track the adjusted valuation after parameter changes
  const [adjustedValuation, setAdjustedValuation] = useState(null);
  // State for loading status
  const [loading, setLoading] = useState(true);
  // State for error messages
  const [error, setError] = useState(null);

  // State for adjustable valuation parameters
  const [parameters, setParameters] = useState({
    capRate: 0.05, // Default cap rate: 5%
    squareFootageWeight: 0.3, // Weight for square footage in valuation
    locationWeight: 0.4, // Weight for location factors in valuation
    amenitiesWeight: 0.2, // Weight for property amenities in valuation
    marketTrendAdjustment: 0.0, // Adjustment for market trends (-0.1 to +0.1)
    renovationImpact: 0.0, // Impact of potential renovations (0 to 0.2)
  });

  // Reference for the chart instance
  const chartRef = useRef(null);
  // Reference for the chart canvas
  const chartCanvasRef = useRef(null);

  /**
   * Fetches property data from the API
   */
  const fetchPropertyData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // If we have initial valuation data passed as a prop, use that
      if (initialValuation) {
        setProperty(initialValuation);
        setOriginalValuation(initialValuation.estimated_value);
        setAdjustedValuation(initialValuation.estimated_value);
        setLoading(false);
        return;
      }
      
      // Otherwise fetch from API
      const response = await fetch(`/api/valuations/${propertyId}`);
      
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      
      const data = await response.json();
      setProperty(data);
      setOriginalValuation(data.estimated_value);
      setAdjustedValuation(data.estimated_value);
    } catch (err) {
      console.error('Error fetching property data:', err);
      setError('Failed to load property data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Recalculates the property valuation based on adjusted parameters
   * This uses a simplified model that you would replace with your actual valuation logic
   */
  const recalculateValuation = () => {
    if (!property) return;
    
    // Base value from original valuation
    const baseValue = originalValuation;
    
    // Get property features that affect valuation
    const { 
      square_feet = 2000, 
      location_score = 0.5, 
      amenities_score = 0.5 
    } = property.features_used || {};

    // Calculate weighted contributions of each factor
    const sqftContribution = square_feet * parameters.squareFootageWeight;
    const locationContribution = location_score * parameters.locationWeight * 100000; // Scale for demonstration
    const amenitiesContribution = amenities_score * parameters.amenitiesWeight * 50000; // Scale for demonstration
    
    // Apply cap rate adjustment (lower cap rate = higher value)
    const capRateAdjustment = (0.05 / parameters.capRate) - 1;
    
    // Apply market trend adjustment
    const marketAdjustment = parameters.marketTrendAdjustment * baseValue;
    
    // Apply renovation impact
    const renovationAdjustment = parameters.renovationImpact * baseValue;
    
    // Calculate new valuation with all adjustments
    // This is a simplified model - your actual calculation would be more sophisticated
    let newValuation = baseValue * (1 + capRateAdjustment) + marketAdjustment + renovationAdjustment;
    
    // Ensure the value doesn't go below zero
    newValuation = Math.max(newValuation, 0);
    
    // Update the adjusted valuation state
    setAdjustedValuation(newValuation);
    
    // Update the chart with new data
    updateChart(baseValue, newValuation);
  };

  /**
   * Handles changes to parameter sliders
   */
  const handleParameterChange = (e) => {
    const { name, value } = e.target;
    
    // Update the parameters state
    setParameters(prevParams => ({
      ...prevParams,
      [name]: parseFloat(value)
    }));
  };

  /**
   * Initializes the comparison chart using Chart.js
   */
  const initializeChart = () => {
    if (!chartCanvasRef.current || !property) return;
    
    // Destroy existing chart if it exists
    if (chartRef.current) {
      chartRef.current.destroy();
    }
    
    // Get the canvas context
    const ctx = chartCanvasRef.current.getContext('2d');
    
    // Create a new chart
    chartRef.current = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Original Valuation', 'Adjusted Valuation'],
        datasets: [{
          label: 'Property Valuation',
          data: [originalValuation, adjustedValuation],
          backgroundColor: [
            'rgba(54, 162, 235, 0.5)',
            'rgba(255, 99, 132, 0.5)'
          ],
          borderColor: [
            'rgba(54, 162, 235, 1)',
            'rgba(255, 99, 132, 1)'
          ],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: false,
            title: {
              display: true,
              text: 'Valuation ($)'
            },
            ticks: {
              // Format ticks as currency
              callback: (value) => '$' + value.toLocaleString()
            }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: (context) => {
                return 'Valuation: $' + context.parsed.y.toLocaleString();
              }
            }
          }
        }
      }
    });
  };

  /**
   * Updates the chart with new valuation data
   */
  const updateChart = (originalVal, newVal) => {
    if (!chartRef.current) return;
    
    // Update chart data
    chartRef.current.data.datasets[0].data = [originalVal, newVal];
    
    // Change color based on whether the new value is higher or lower
    const color = newVal >= originalVal 
      ? 'rgba(75, 192, 192, 0.5)' // Green for increase
      : 'rgba(255, 99, 132, 0.5)'; // Red for decrease
    
    chartRef.current.data.datasets[0].backgroundColor[1] = color;
    chartRef.current.data.datasets[0].borderColor[1] = color.replace('0.5', '1');
    
    // Update the chart
    chartRef.current.update();
  };
  
  /**
   * Creates a second chart showing the contribution of each factor
   */
  const initializeFactorsChart = () => {
    if (!document.getElementById('factorsChart') || !property) return;
    
    // Create a factors chart
    const factorsCtx = document.getElementById('factorsChart').getContext('2d');
    
    // Get property features
    const { 
      square_feet = 2000, 
      location_score = 0.5, 
      amenities_score = 0.5 
    } = property.features_used || {};
    
    // Calculate factor contributions
    const sqftContribution = square_feet * parameters.squareFootageWeight;
    const locationContribution = location_score * parameters.locationWeight * 100000;
    const amenitiesContribution = amenities_score * parameters.amenitiesWeight * 50000;
    const marketContribution = parameters.marketTrendAdjustment * originalValuation;
    const renovationContribution = parameters.renovationImpact * originalValuation;
    
    // Create/update factors chart
    if (window.factorsChart) {
      window.factorsChart.destroy();
    }
    
    window.factorsChart = new Chart(factorsCtx, {
      type: 'doughnut',
      data: {
        labels: [
          'Square Footage', 
          'Location', 
          'Amenities', 
          'Market Trends', 
          'Renovation Impact'
        ],
        datasets: [{
          data: [
            sqftContribution,
            locationContribution,
            amenitiesContribution,
            marketContribution,
            renovationContribution
          ],
          backgroundColor: [
            'rgba(255, 99, 132, 0.7)',
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 206, 86, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(153, 102, 255, 0.7)'
          ]
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right'
          },
          tooltip: {
            callbacks: {
              label: (context) => {
                const value = context.raw;
                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                const percentage = Math.round((value / total) * 100);
                return `${context.label}: $${value.toLocaleString()} (${percentage}%)`;
              }
            }
          }
        }
      }
    });
  };

  // Initialize property data when component mounts
  useEffect(() => {
    fetchPropertyData();
  }, [propertyId, initialValuation]);

  // Recalculate valuation when parameters change
  useEffect(() => {
    if (property) {
      recalculateValuation();
    }
  }, [parameters, property]);

  // Initialize charts when data is loaded
  useEffect(() => {
    if (property && originalValuation && adjustedValuation) {
      initializeChart();
      initializeFactorsChart();
    }
  }, [property, originalValuation, adjustedValuation]);

  // Format currency values for display
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(value);
  };

  // Calculate the percentage change for display
  const calculatePercentChange = () => {
    if (!originalValuation || !adjustedValuation) return 0;
    return ((adjustedValuation - originalValuation) / originalValuation) * 100;
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">What-If Analysis</h1>
      
      {/* Loading State */}
      {loading && (
        <div className="flex justify-center items-center p-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-gray-600">Loading property data...</span>
        </div>
      )}
      
      {/* Error State */}
      {error && (
        <div className="bg-red-50 p-4 border-l-4 border-red-500">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Main Content */}
      {!loading && !error && property && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Property Info & Parameter Controls */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Property Information</h2>
              <div className="mb-4">
                <p className="text-sm text-gray-500">Address</p>
                <p className="text-base font-medium">{property.address}</p>
              </div>
              <div className="mb-4">
                <p className="text-sm text-gray-500">Original Valuation</p>
                <p className="text-2xl font-bold text-blue-600">{formatCurrency(originalValuation)}</p>
              </div>
              <div className="mb-4">
                <p className="text-sm text-gray-500">Adjusted Valuation</p>
                <p className={`text-2xl font-bold ${adjustedValuation >= originalValuation ? 'text-green-600' : 'text-red-600'}`}>
                  {formatCurrency(adjustedValuation)}
                </p>
                <p className={`text-sm ${adjustedValuation >= originalValuation ? 'text-green-600' : 'text-red-600'}`}>
                  {adjustedValuation >= originalValuation ? '+' : ''}{calculatePercentChange().toFixed(2)}% from original
                </p>
              </div>
            </div>
            
            {/* Parameter Controls */}
            <div className="bg-white rounded-lg shadow-md p-6 mt-6">
              <h2 className="text-xl font-semibold mb-4">Adjust Parameters</h2>
              
              {/* Cap Rate Slider */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Cap Rate: {(parameters.capRate * 100).toFixed(2)}%
                </label>
                <input
                  type="range"
                  name="capRate"
                  min="0.03"
                  max="0.08"
                  step="0.001"
                  value={parameters.capRate}
                  onChange={handleParameterChange}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>3%</span>
                  <span>8%</span>
                </div>
              </div>
              
              {/* Square Footage Weight Slider */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Square Footage Weight: {(parameters.squareFootageWeight * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  name="squareFootageWeight"
                  min="0.1"
                  max="0.5"
                  step="0.01"
                  value={parameters.squareFootageWeight}
                  onChange={handleParameterChange}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>10%</span>
                  <span>50%</span>
                </div>
              </div>
              
              {/* Location Weight Slider */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Location Weight: {(parameters.locationWeight * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  name="locationWeight"
                  min="0.2"
                  max="0.6"
                  step="0.01"
                  value={parameters.locationWeight}
                  onChange={handleParameterChange}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>20%</span>
                  <span>60%</span>
                </div>
              </div>
              
              {/* Amenities Weight Slider */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Amenities Weight: {(parameters.amenitiesWeight * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  name="amenitiesWeight"
                  min="0.1"
                  max="0.4"
                  step="0.01"
                  value={parameters.amenitiesWeight}
                  onChange={handleParameterChange}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>10%</span>
                  <span>40%</span>
                </div>
              </div>
              
              {/* Market Trend Adjustment Slider */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Market Trend Adjustment: {parameters.marketTrendAdjustment >= 0 ? '+' : ''}{(parameters.marketTrendAdjustment * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  name="marketTrendAdjustment"
                  min="-0.1"
                  max="0.1"
                  step="0.01"
                  value={parameters.marketTrendAdjustment}
                  onChange={handleParameterChange}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>-10%</span>
                  <span>+10%</span>
                </div>
              </div>
              
              {/* Renovation Impact Slider */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Renovation Impact: +{(parameters.renovationImpact * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  name="renovationImpact"
                  min="0"
                  max="0.2"
                  step="0.01"
                  value={parameters.renovationImpact}
                  onChange={handleParameterChange}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>0%</span>
                  <span>+20%</span>
                </div>
              </div>
              
              {/* Reset Button */}
              <button
                onClick={() => {
                  setParameters({
                    capRate: 0.05,
                    squareFootageWeight: 0.3,
                    locationWeight: 0.4,
                    amenitiesWeight: 0.2,
                    marketTrendAdjustment: 0.0,
                    renovationImpact: 0.0,
                  });
                }}
                className="mt-4 px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500"
              >
                Reset Parameters
              </button>
            </div>
          </div>
          
          {/* Charts and Visualizations */}
          <div className="lg:col-span-2">
            {/* Valuation Comparison Chart */}
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <h2 className="text-xl font-semibold mb-4">Valuation Comparison</h2>
              <div className="h-80">
                <canvas ref={chartCanvasRef}></canvas>
              </div>
            </div>
            
            {/* Factor Contribution Chart */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Factor Contribution Analysis</h2>
              <div className="h-80">
                <canvas id="factorsChart"></canvas>
              </div>
              <div className="mt-4">
                <p className="text-sm text-gray-600">
                  This chart shows how each factor contributes to the adjusted property valuation
                  based on your current parameter settings.
                </p>
              </div>
            </div>
            
            {/* Parameter Explanation Card */}
            <div className="bg-white rounded-lg shadow-md p-6 mt-6">
              <h2 className="text-xl font-semibold mb-4">Understanding Parameters</h2>
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-blue-600">Cap Rate</h3>
                  <p className="text-sm text-gray-600">
                    The capitalization rate is the rate of return on a real estate investment property.
                    Lower cap rates typically result in higher property valuations.
                  </p>
                </div>
                <div>
                  <h3 className="font-medium text-blue-600">Square Footage, Location & Amenities Weights</h3>
                  <p className="text-sm text-gray-600">
                    These sliders control how much each factor impacts the overall valuation.
                    Increasing one factor's weight will make that aspect more significant in the calculation.
                  </p>
                </div>
                <div>
                  <h3 className="font-medium text-blue-600">Market Trend Adjustment</h3>
                  <p className="text-sm text-gray-600">
                    Accounts for projected market changes. Positive values indicate an upcoming market improvement,
                    while negative values represent a potential market downturn.
                  </p>
                </div>
                <div>
                  <h3 className="font-medium text-blue-600">Renovation Impact</h3>
                  <p className="text-sm text-gray-600">
                    Estimates the potential increase in property value from renovations or property improvements.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WhatIfAnalysis;