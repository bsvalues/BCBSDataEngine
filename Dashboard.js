import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, ArcElement, TimeScale, Filler } from 'chart.js';
import { Line, Bar, Pie } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns'; // For time scale
import { debounce } from 'lodash'; // For search optimization

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  TimeScale,
  Filler,
  Title,
  Tooltip,
  Legend
);

/**
 * Dashboard Component
 * 
 * This component fetches property valuation data from the API and displays it in a
 * filterable table and interactive visualizations. It includes options for filtering 
 * by neighborhood, price range, property type, and date, and handles loading states 
 * and errors gracefully.
 * 
 * Key Features:
 * - Data fetching with loading indicators and error handling
 * - Interactive filtering by neighborhood, price range, property type, and date
 * - Real-time search functionality with debounce optimization
 * - Tabular display of property valuations with sortable columns
 * - Advanced data visualization with Chart.js showing trends and distributions
 * - Responsive design with Tailwind CSS
 * - Detailed property information with advanced metrics
 */
const Dashboard = () => {
  // State for storing property valuations data
  const [properties, setProperties] = useState([]);
  // State for tracking loading status
  const [loading, setLoading] = useState(true);
  // State for storing any error messages
  const [error, setError] = useState(null);
  // State for tracking filter values
  const [filters, setFilters] = useState({
    neighborhood: '',
    minValue: '',
    maxValue: '',
    propertyType: '',
    fromDate: '',
    toDate: '',
    searchQuery: '',
    sortBy: 'estimated_value',
    sortDirection: 'desc'
  });
  // State for active tab in the chart section
  const [activeTab, setActiveTab] = useState('distribution');
  // State for selected property detail view
  const [selectedProperty, setSelectedProperty] = useState(null);
  // State for available neighborhoods from data
  const [availableNeighborhoods, setAvailableNeighborhoods] = useState([]);
  // State for tracking pagination
  const [pagination, setPagination] = useState({
    currentPage: 1,
    itemsPerPage: 10,
    totalItems: 0
  });
  // State for API key
  const [apiKey, setApiKey] = useState('');
  
  // Refs for chart containers and observer for dashboard metrics
  const valueDistributionChartRef = useRef(null);
  const neighborhoodChartRef = useRef(null);
  const trendChartRef = useRef(null);
  const metricsObserverRef = useRef(null);

  /**
   * Function to fetch property valuations from the enhanced API endpoint
   * Includes pagination, sorting, and filtering capabilities
   * Wrapped in useCallback to prevent recreation on each render
   */
  const fetchPropertyValuations = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Build query parameters based on active filters
      const queryParams = new URLSearchParams();
      
      if (filters.minValue) queryParams.append('min_value', filters.minValue);
      if (filters.maxValue) queryParams.append('max_value', filters.maxValue);
      if (filters.propertyType) queryParams.append('property_type', filters.propertyType);
      if (filters.fromDate) queryParams.append('from_date', filters.fromDate);
      if (filters.toDate) queryParams.append('to_date', filters.toDate);
      if (filters.sortBy) queryParams.append('sort_by', filters.sortBy);
      if (filters.sortDirection) queryParams.append('sort_direction', filters.sortDirection);
      
      // Add pagination parameters
      queryParams.append('page', pagination.currentPage);
      queryParams.append('page_size', pagination.itemsPerPage);
      
      // Headers for the API request including the API key
      const headers = {
        'Content-Type': 'application/json'
      };
      
      // Add API key header if available
      if (apiKey) {
        headers['X-API-KEY'] = apiKey;
      }
      
      // Make the API request to the enhanced valuations endpoint
      const response = await fetch(`/api/valuations?${queryParams.toString()}`, {
        method: 'GET',
        headers: headers
      });
      
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Check if the API returns data in the expected format
      if (Array.isArray(data.properties)) {
        // New API format with pagination info
        setProperties(data.properties);
        setPagination(prev => ({
          ...prev,
          totalItems: data.total_count || data.properties.length
        }));
      } else if (Array.isArray(data)) {
        // Legacy API format (just an array of properties)
        setProperties(data);
        setPagination(prev => ({
          ...prev,
          totalItems: data.length
        }));
      } else {
        throw new Error("Unexpected data format received from API");
      }
      
      console.log(`Loaded ${Array.isArray(data.properties) ? data.properties.length : data.length} properties from API`);
      
      // Extract unique neighborhoods for the filter dropdown
      extractUniqueNeighborhoods(Array.isArray(data.properties) ? data.properties : data);
      
    } catch (err) {
      console.error('Error fetching property valuations:', err);
      setError(`Failed to load property valuations: ${err.message}. Please try again later.`);
    } finally {
      setLoading(false);
    }
  }, [filters, pagination, extractUniqueNeighborhoods, apiKey]); // Add all dependencies

  /**
   * Function to extract neighborhood from address
   * @param {string} address - The property address
   * @returns {string} - Extracted neighborhood
   */
  const extractNeighborhood = (address) => {
    // This is a simple extraction - you might need more complex logic
    // based on your actual address format
    const parts = address.split(',');
    if (parts.length >= 2) {
      return parts[1].trim();
    }
    return 'Unknown';
  };

  /**
   * Extract unique neighborhoods from the property data
   * @param {Array} propertyData - Array of property objects
   */
  const extractUniqueNeighborhoods = useCallback((propertyData) => {
    const neighborhoods = propertyData.map(property => 
      extractNeighborhood(property.address)
    ).filter((value, index, self) => 
      value !== 'Unknown' && self.indexOf(value) === index
    ).sort();
    
    setAvailableNeighborhoods(neighborhoods);
  }, [extractNeighborhood]);

  /**
   * Function to handle filter changes with optimized search capability
   */
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters(prevFilters => ({
      ...prevFilters,
      [name]: value
    }));
    
    // Reset to first page when filters change
    if (name !== 'searchQuery') {
      setPagination(prev => ({
        ...prev,
        currentPage: 1
      }));
    }
    
    // If search query is changing, use the debounced search function
    if (name === 'searchQuery') {
      debouncedSearch(value);
    }
  };

  /**
   * Debounced search function to prevent excessive API calls while typing
   */
  const debouncedSearch = useCallback(
    debounce((query) => {
      // Reset to first page when search query changes
      setPagination(prev => ({
        ...prev,
        currentPage: 1
      }));
      
      // Only fetch if we're not already searching or if query length is appropriate
      if (query.length === 0 || query.length >= 3) {
        fetchPropertyValuations();
      }
    }, 500),
    [fetchPropertyValuations] // Include fetchPropertyValuations in the dependency array
  );

  /**
   * Function to apply filters
   */
  const applyFilters = (e) => {
    e.preventDefault();
    setPagination(prev => ({
      ...prev,
      currentPage: 1
    }));
    fetchPropertyValuations();
  };

  /**
   * Function to reset all filters
   */
  const resetFilters = () => {
    setFilters({
      neighborhood: '',
      minValue: '',
      maxValue: '',
      propertyType: '',
      fromDate: '',
      toDate: '',
      searchQuery: '',
      sortBy: 'estimated_value',
      sortDirection: 'desc'
    });
    
    setPagination(prev => ({
      ...prev,
      currentPage: 1
    }));
    
    // Re-fetch data with cleared filters
    fetchPropertyValuations();
  };

  /**
   * Function to format currency
   * @param {number} value - The numeric value to format as currency
   * @returns {string} - Formatted currency string
   */
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(value);
  };

  /**
   * Format percentage values for display
   * @param {number} value - Value to format as percentage
   * @returns {string} - Formatted percentage string
   */
  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  /**
   * Format date values for display
   * @param {string} dateString - ISO date string
   * @returns {string} - Formatted date string
   */
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  /**
   * Function to handle sorting column clicks
   * @param {string} columnName - The name of the column to sort by
   */
  const handleSortColumn = (columnName) => {
    setFilters(prevFilters => ({
      ...prevFilters,
      sortDirection: prevFilters.sortBy === columnName && prevFilters.sortDirection === 'asc' ? 'desc' : 'asc',
      sortBy: columnName
    }));
    fetchPropertyValuations();
  };

  /**
   * Function to handle pagination changes
   * @param {number} newPage - The new page number to navigate to
   */
  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= Math.ceil(pagination.totalItems / pagination.itemsPerPage)) {
      setPagination(prev => ({
        ...prev,
        currentPage: newPage
      }));
      fetchPropertyValuations();
    }
  };

  /**
   * Handle items per page change
   * @param {Event} e - The change event
   */
  const handleItemsPerPageChange = (e) => {
    const newItemsPerPage = parseInt(e.target.value, 10);
    setPagination(prev => ({
      ...prev,
      itemsPerPage: newItemsPerPage,
      currentPage: 1 // Reset to first page when changing items per page
    }));
    fetchPropertyValuations();
  };

  /**
   * Handle property row click to view detailed information
   * @param {Object} property - The property to view in detail
   */
  const handlePropertyClick = (property) => {
    setSelectedProperty(property);
  };

  /**
   * Close the property detail modal
   */
  const closePropertyDetail = () => {
    setSelectedProperty(null);
  };

  /**
   * Function to filter properties by search query and neighborhood
   * for client-side filtering in addition to API-based filtering
   */
  const filteredProperties = useMemo(() => {
    return properties.filter(property => {
      // Apply neighborhood filter (client-side filtering if needed)
      if (filters.neighborhood && !extractNeighborhood(property.address).toLowerCase().includes(filters.neighborhood.toLowerCase())) {
        return false;
      }
      
      // Apply search query filter on address or property_id
      if (filters.searchQuery && 
          !(property.address.toLowerCase().includes(filters.searchQuery.toLowerCase()) || 
            property.property_id.toLowerCase().includes(filters.searchQuery.toLowerCase()))) {
        return false;
      }
      
      return true;
    });
  }, [properties, filters.neighborhood, filters.searchQuery]);
  
  /**
   * Load property valuations when component mounts
   * Also set up the IntersectionObserver for dashboard metrics animations
   */
  useEffect(() => {
    fetchPropertyValuations();
    
    // Set up IntersectionObserver for dashboard metrics animations
    const options = {
      root: null,
      rootMargin: '0px',
      threshold: 0.1
    };
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('animate-fade-in');
        }
      });
    }, options);
    
    const metricsContainer = document.getElementById('dashboard-metrics');
    if (metricsContainer) {
      metricsContainer.querySelectorAll('.metric-card').forEach(card => {
        observer.observe(card);
      });
    }
    
    metricsObserverRef.current = observer;
    
    return () => {
      if (metricsObserverRef.current) {
        metricsObserverRef.current.disconnect();
      }
    };
  }, []); // Empty dependency array means this runs once on mount
  
  /**
   * Calculate summary metrics for the dashboard based on filtered properties
   */
  const dashboardMetrics = useMemo(() => {
    if (filteredProperties.length === 0) {
      return {
        totalProperties: 0,
        averageValue: 0,
        minValue: 0,
        maxValue: 0,
        medianValue: 0,
        averageConfidence: 0
      };
    }
    
    // Calculate metrics
    const totalProperties = filteredProperties.length;
    const values = filteredProperties.map(p => p.estimated_value);
    const confidences = filteredProperties.map(p => p.confidence_score);
    
    // Sort values for median calculation
    const sortedValues = [...values].sort((a, b) => a - b);
    
    const averageValue = values.reduce((sum, val) => sum + val, 0) / totalProperties;
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const medianValue = sortedValues[Math.floor(totalProperties / 2)];
    const averageConfidence = confidences.reduce((sum, val) => sum + val, 0) / totalProperties;
    
    return {
      totalProperties,
      averageValue,
      minValue,
      maxValue,
      medianValue,
      averageConfidence
    };
  }, [filteredProperties]);
  
  /**
   * Function to prepare chart data based on filtered properties
   * Using useMemo to prevent unnecessary recalculations
   */
  const preparedChartData = useMemo(() => {
    if (filteredProperties.length === 0) {
      return null;
    }
    
    // Prepare chart data based on filtered properties
    // Group properties by value ranges for distribution chart
    const valueRanges = [
      '< $200K', 
      '$200K - $300K', 
      '$300K - $400K', 
      '$400K - $500K', 
      '$500K - $750K', 
      '$750K - $1M', 
      '> $1M'
    ];
    
    const valueCounts = [0, 0, 0, 0, 0, 0, 0];
    
    filteredProperties.forEach(property => {
      const value = property.estimated_value;
      if (value < 200000) {
        valueCounts[0]++;
      } else if (value < 300000) {
        valueCounts[1]++;
      } else if (value < 400000) {
        valueCounts[2]++;
      } else if (value < 500000) {
        valueCounts[3]++;
      } else if (value < 750000) {
        valueCounts[4]++;
      } else if (value < 1000000) {
        valueCounts[5]++;
      } else {
        valueCounts[6]++;
      }
    });
    
    // Group properties by neighborhood for neighborhood chart
    const neighborhoods = {};
    
    filteredProperties.forEach(property => {
      const neighborhood = extractNeighborhood(property.address);
      if (!neighborhoods[neighborhood]) {
        neighborhoods[neighborhood] = {
          count: 0,
          totalValue: 0,
          maxValue: 0,
          minValue: Number.MAX_SAFE_INTEGER,
          confidenceSum: 0
        };
      }
      neighborhoods[neighborhood].count++;
      neighborhoods[neighborhood].totalValue += property.estimated_value;
      neighborhoods[neighborhood].maxValue = Math.max(neighborhoods[neighborhood].maxValue, property.estimated_value);
      neighborhoods[neighborhood].minValue = Math.min(neighborhoods[neighborhood].minValue, property.estimated_value);
      neighborhoods[neighborhood].confidenceSum += property.confidence_score || 0;
    });
    
    // Convert to arrays for chart data - top 8 neighborhoods by count
    const neighborhoodNames = Object.keys(neighborhoods)
      .sort((a, b) => neighborhoods[b].count - neighborhoods[a].count)
      .slice(0, 8);
    
    const neighborhoodCounts = neighborhoodNames.map(name => neighborhoods[name].count);
    const neighborhoodAvgValues = neighborhoodNames.map(name => 
      Math.round(neighborhoods[name].totalValue / neighborhoods[name].count)
    );
    const neighborhoodValueRanges = neighborhoodNames.map(name => ({
      min: neighborhoods[name].minValue,
      max: neighborhoods[name].maxValue,
      avg: Math.round(neighborhoods[name].totalValue / neighborhoods[name].count)
    }));
    const neighborhoodConfidences = neighborhoodNames.map(name => 
      neighborhoods[name].confidenceSum / neighborhoods[name].count
    );
    
    // Group data by date for trend analysis
    // Create a map of date -> average value for the trend chart
    const dateMap = {};
    
    filteredProperties.forEach(property => {
      if (property.valuation_date) {
        // Use just the date part, truncating time
        const dateKey = property.valuation_date.split('T')[0];
        if (!dateMap[dateKey]) {
          dateMap[dateKey] = {
            totalValue: 0,
            count: 0,
            confidenceSum: 0
          };
        }
        dateMap[dateKey].totalValue += property.estimated_value;
        dateMap[dateKey].count++;
        dateMap[dateKey].confidenceSum += property.confidence_score || 0;
      }
    });
    
    // Convert to arrays sorted by date for the trend chart
    const trendDates = Object.keys(dateMap).sort();
    const trendValues = trendDates.map(date => 
      Math.round(dateMap[date].totalValue / dateMap[date].count)
    );
    const trendConfidences = trendDates.map(date => 
      dateMap[date].confidenceSum / dateMap[date].count
    );
    
    // Create model distribution data
    const modelCounts = {};
    filteredProperties.forEach(property => {
      const model = property.model_used || 'Unknown';
      modelCounts[model] = (modelCounts[model] || 0) + 1;
    });
    
    const modelLabels = Object.keys(modelCounts);
    const modelData = modelLabels.map(model => modelCounts[model]);
    const modelColors = [
      'rgba(54, 162, 235, 0.7)',
      'rgba(255, 99, 132, 0.7)',
      'rgba(75, 192, 192, 0.7)',
      'rgba(255, 206, 86, 0.7)',
      'rgba(153, 102, 255, 0.7)',
      'rgba(255, 159, 64, 0.7)'
    ];
    
    // Return prepared chart data
    return {
      valueDistribution: {
        labels: valueRanges,
        datasets: [
          {
            label: 'Property Count',
            data: valueCounts,
            backgroundColor: 'rgba(54, 162, 235, 0.6)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          }
        ]
      },
      neighborhoodComparison: {
        labels: neighborhoodNames,
        datasets: [
          {
            label: 'Property Count',
            data: neighborhoodCounts,
            backgroundColor: 'rgba(153, 102, 255, 0.6)',
            borderColor: 'rgba(153, 102, 255, 1)',
            borderWidth: 1,
            yAxisID: 'y'
          },
          {
            label: 'Average Value',
            data: neighborhoodAvgValues,
            type: 'line',
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderWidth: 2,
            fill: false,
            yAxisID: 'y1'
          }
        ]
      },
      valueTrend: {
        labels: trendDates,
        datasets: [
          {
            label: 'Average Value',
            data: trendValues,
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderWidth: 2,
            fill: true,
            tension: 0.3
          },
          {
            label: 'Confidence Score',
            data: trendConfidences,
            borderColor: 'rgba(255, 159, 64, 1)',
            backgroundColor: 'rgba(255, 159, 64, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.3,
            yAxisID: 'y1'
          }
        ]
      },
      modelDistribution: {
        labels: modelLabels,
        datasets: [
          {
            data: modelData,
            backgroundColor: modelColors.slice(0, modelLabels.length),
            borderWidth: 1
          }
        ]
      },
      neighborhoodDetails: {
        names: neighborhoodNames,
        averageValues: neighborhoodAvgValues,
        valueRanges: neighborhoodValueRanges,
        confidences: neighborhoodConfidences
      }
    };
  }, [filteredProperties]);
  
  /**
   * Options for Value Distribution chart
   */
  const valueDistributionOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Property Value Distribution',
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.dataset.label || '';
            const value = context.parsed.y || 0;
            return `${label}: ${value} properties`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Number of Properties'
        }
      }
    }
  };

  /**
   * Options for Neighborhood Comparison chart
   */
  const neighborhoodComparisonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Neighborhood Comparison',
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            if (context.dataset.label === 'Property Count') {
              return `${context.dataset.label}: ${context.parsed.y} properties`;
            } else {
              return `${context.dataset.label}: ${formatCurrency(context.parsed.y)}`;
            }
          }
        }
      }
    },
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Number of Properties'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        grid: {
          drawOnChartArea: false,
        },
        title: {
          display: true,
          text: 'Average Value ($)'
        },
        ticks: {
          callback: function(value) {
            return '$' + value.toLocaleString();
          }
        }
      }
    }
  };

  /**
   * Options for Value Trend chart
   */
  const valueTrendOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Property Value Trend Over Time',
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            if (context.dataset.label === 'Average Value') {
              return `${context.dataset.label}: ${formatCurrency(context.parsed.y)}`;
            } else {
              return `${context.dataset.label}: ${formatPercentage(context.parsed.y)}`;
            }
          }
        }
      }
    },
    scales: {
      x: {
        type: 'category',
        title: {
          display: true,
          text: 'Date'
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Average Value ($)'
        },
        ticks: {
          callback: function(value) {
            return '$' + value.toLocaleString();
          }
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        min: 0,
        max: 1,
        grid: {
          drawOnChartArea: false,
        },
        title: {
          display: true,
          text: 'Confidence Score'
        },
        ticks: {
          callback: function(value) {
            return formatPercentage(value);
          }
        }
      }
    }
  };

  /**
   * Options for Model Distribution pie chart
   */
  const modelDistributionOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
      },
      title: {
        display: true,
        text: 'Distribution by Valuation Model',
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.label || '';
            const value = context.parsed || 0;
            const total = context.dataset.data.reduce((a, b) => a + b, 0);
            const percentage = Math.round((value / total) * 100);
            return `${label}: ${value} properties (${percentage}%)`;
          }
        }
      }
    }
  };

  // Handle API key input change
  const handleApiKeyChange = (e) => {
    setApiKey(e.target.value);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Advanced Property Valuations Dashboard</h1>
      
      {/* API Key Input */}
      <div className="mb-6 bg-white rounded-lg shadow-md p-4">
        <div className="flex flex-col md:flex-row items-start md:items-center">
          <div className="mb-2 md:mb-0 md:mr-4 flex-grow">
            <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700 mb-1">
              API Key
            </label>
            <input 
              type="password" 
              id="apiKey" 
              className="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md" 
              placeholder="Enter your API key"
              value={apiKey}
              onChange={handleApiKeyChange}
            />
          </div>
          <button
            className="mt-2 md:mt-6 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={fetchPropertyValuations}
            disabled={loading}
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Loading...
              </>
            ) : (
              'Apply API Key'
            )}
          </button>
        </div>
        <p className="mt-2 text-xs text-gray-500">
          The API key is required to access property valuation data. If you don't have an API key, please contact the administrator.
        </p>
        
        {/* API Key Error Message */}
        {error && error.includes('API') && (
          <div className="mt-3 bg-red-50 border-l-4 border-red-400 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">
                  API key is invalid or missing. Please check your key and try again.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Dashboard Summary Metrics */}
      <div id="dashboard-metrics" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4 mb-8">
        <div className="metric-card bg-white rounded-lg shadow-md p-4 border-l-4 border-blue-500 opacity-0 transition-opacity duration-500">
          <p className="text-sm text-gray-500 mb-1">Total Properties</p>
          <p className="text-2xl font-bold">{dashboardMetrics.totalProperties.toLocaleString()}</p>
        </div>
        <div className="metric-card bg-white rounded-lg shadow-md p-4 border-l-4 border-green-500 opacity-0 transition-opacity duration-500 delay-100">
          <p className="text-sm text-gray-500 mb-1">Average Value</p>
          <p className="text-2xl font-bold">{formatCurrency(dashboardMetrics.averageValue)}</p>
        </div>
        <div className="metric-card bg-white rounded-lg shadow-md p-4 border-l-4 border-red-500 opacity-0 transition-opacity duration-500 delay-200">
          <p className="text-sm text-gray-500 mb-1">Median Value</p>
          <p className="text-2xl font-bold">{formatCurrency(dashboardMetrics.medianValue)}</p>
        </div>
        <div className="metric-card bg-white rounded-lg shadow-md p-4 border-l-4 border-yellow-500 opacity-0 transition-opacity duration-500 delay-300">
          <p className="text-sm text-gray-500 mb-1">Minimum Value</p>
          <p className="text-2xl font-bold">{formatCurrency(dashboardMetrics.minValue)}</p>
        </div>
        <div className="metric-card bg-white rounded-lg shadow-md p-4 border-l-4 border-purple-500 opacity-0 transition-opacity duration-500 delay-400">
          <p className="text-sm text-gray-500 mb-1">Maximum Value</p>
          <p className="text-2xl font-bold">{formatCurrency(dashboardMetrics.maxValue)}</p>
        </div>
        <div className="metric-card bg-white rounded-lg shadow-md p-4 border-l-4 border-indigo-500 opacity-0 transition-opacity duration-500 delay-500">
          <p className="text-sm text-gray-500 mb-1">Avg. Confidence</p>
          <p className="text-2xl font-bold">{formatPercentage(dashboardMetrics.averageConfidence)}</p>
        </div>
      </div>
      
      {/* Search and Filter bar with improved styling */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <div className="flex flex-col md:flex-row justify-between items-center mb-4">
          <h2 className="text-xl font-semibold mb-2 md:mb-0">Property Search & Filters</h2>
          
          {/* Search box */}
          <div className="w-full md:w-1/3">
            <div className="relative">
              <input
                type="text"
                name="searchQuery"
                value={filters.searchQuery}
                onChange={handleFilterChange}
                className="w-full p-2 pl-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Search by address or ID..."
              />
              <svg className="h-5 w-5 text-gray-400 absolute left-3 top-2.5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
              </svg>
            </div>
          </div>
        </div>
        
        <form onSubmit={applyFilters}>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Neighborhood Filter with dropdown from available neighborhoods */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Neighborhood</label>
              <select
                name="neighborhood"
                value={filters.neighborhood}
                onChange={handleFilterChange}
                className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Neighborhoods</option>
                {availableNeighborhoods.map(neighborhood => (
                  <option key={neighborhood} value={neighborhood}>{neighborhood}</option>
                ))}
              </select>
            </div>
            
            {/* Min Value Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Min Value ($)</label>
              <input
                type="number"
                name="minValue"
                value={filters.minValue}
                onChange={handleFilterChange}
                className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Min value"
              />
            </div>
            
            {/* Max Value Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max Value ($)</label>
              <input
                type="number"
                name="maxValue"
                value={filters.maxValue}
                onChange={handleFilterChange}
                className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Max value"
              />
            </div>
            
            {/* Property Type Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Property Type</label>
              <select
                name="propertyType"
                value={filters.propertyType}
                onChange={handleFilterChange}
                className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Types</option>
                <option value="residential">Residential</option>
                <option value="commercial">Commercial</option>
                <option value="land">Land</option>
                <option value="multifamily">Multi-Family</option>
              </select>
            </div>
            
            {/* From Date Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">From Date</label>
              <input
                type="date"
                name="fromDate"
                value={filters.fromDate}
                onChange={handleFilterChange}
                className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            {/* To Date Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">To Date</label>
              <input
                type="date"
                name="toDate"
                value={filters.toDate}
                onChange={handleFilterChange}
                className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          
          {/* Filter Action Buttons */}
          <div className="mt-4 flex justify-end space-x-2">
            <button
              type="button"
              onClick={resetFilters}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500 transition duration-150"
            >
              Reset Filters
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-150"
            >
              Apply Filters
            </button>
          </div>
        </form>
      </div>
      
      {/* Chart Section with Tabs */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden mb-8">
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            <button
              className={`py-4 px-6 border-b-2 font-medium text-sm ${
                activeTab === 'distribution' 
                  ? 'border-blue-500 text-blue-600' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setActiveTab('distribution')}
            >
              Value Distribution
            </button>
            <button
              className={`py-4 px-6 border-b-2 font-medium text-sm ${
                activeTab === 'neighborhood' 
                  ? 'border-blue-500 text-blue-600' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setActiveTab('neighborhood')}
            >
              Neighborhood Comparison
            </button>
            <button
              className={`py-4 px-6 border-b-2 font-medium text-sm ${
                activeTab === 'trend' 
                  ? 'border-blue-500 text-blue-600' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setActiveTab('trend')}
            >
              Value Trend
            </button>
            <button
              className={`py-4 px-6 border-b-2 font-medium text-sm ${
                activeTab === 'model' 
                  ? 'border-blue-500 text-blue-600' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setActiveTab('model')}
            >
              Model Distribution
            </button>
          </nav>
        </div>
        
        {/* Chart Content */}
        <div className="p-6">
          {/* Loading State */}
          {loading && (
            <div className="flex justify-center items-center p-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
              <span className="ml-3 text-gray-600">Loading chart data...</span>
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
          
          {/* No Data State */}
          {!loading && !error && (!preparedChartData || filteredProperties.length === 0) && (
            <div className="text-center py-12">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              <h3 className="mt-2 text-sm font-medium text-gray-900">No data available</h3>
              <p className="mt-1 text-sm text-gray-500">
                Try adjusting your filters or adding more properties to the database.
              </p>
            </div>
          )}
          
          {/* Charts */}
          {!loading && !error && preparedChartData && (
            <div className="h-96">
              {activeTab === 'distribution' && (
                <Bar 
                  ref={valueDistributionChartRef}
                  data={preparedChartData.valueDistribution} 
                  options={valueDistributionOptions} 
                />
              )}
              {activeTab === 'neighborhood' && (
                <Bar 
                  ref={neighborhoodChartRef}
                  data={preparedChartData.neighborhoodComparison} 
                  options={neighborhoodComparisonOptions} 
                />
              )}
              {activeTab === 'trend' && (
                <Line 
                  ref={trendChartRef}
                  data={preparedChartData.valueTrend} 
                  options={valueTrendOptions} 
                />
              )}
              {activeTab === 'model' && (
                <div className="flex justify-center h-full">
                  <div style={{ width: '50%', height: '100%' }}>
                    <Pie 
                      data={preparedChartData.modelDistribution} 
                      options={modelDistributionOptions} 
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Neighborhood Insights */}
      {!loading && !error && preparedChartData && preparedChartData.neighborhoodDetails.names.length > 0 && (
        <div className="bg-white rounded-lg shadow-md overflow-hidden mb-8">
          <h2 className="text-xl font-semibold p-6 border-b">Neighborhood Insights</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Neighborhood
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Avg. Value
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Min Value
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Max Value
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Value Range
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Confidence
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {preparedChartData.neighborhoodDetails.names.map((name, index) => (
                  <tr key={name} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatCurrency(preparedChartData.neighborhoodDetails.averageValues[index])}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatCurrency(preparedChartData.neighborhoodDetails.valueRanges[index].min)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatCurrency(preparedChartData.neighborhoodDetails.valueRanges[index].max)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-blue-600 h-2.5 rounded-full" 
                          style={{ 
                            width: '100%',
                            background: `linear-gradient(to right, #4ade80, #60a5fa, #ef4444)`,
                          }}
                        ></div>
                      </div>
                      <div className="flex justify-between text-xs mt-1">
                        <span>{formatCurrency(preparedChartData.neighborhoodDetails.valueRanges[index].min)}</span>
                        <span>{formatCurrency(preparedChartData.neighborhoodDetails.valueRanges[index].max)}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <div className="flex items-center">
                        <div className="mr-2">
                          {formatPercentage(preparedChartData.neighborhoodDetails.confidences[index])}
                        </div>
                        <div className="w-24 bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-green-600 h-2.5 rounded-full" 
                            style={{ 
                              width: `${preparedChartData.neighborhoodDetails.confidences[index] * 100}%`,
                              backgroundColor: `${
                                preparedChartData.neighborhoodDetails.confidences[index] > 0.8 ? '#4ade80' : 
                                preparedChartData.neighborhoodDetails.confidences[index] > 0.6 ? '#facc15' : 
                                '#ef4444'
                              }`
                            }}
                          ></div>
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      
      {/* Results Section */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <div className="flex flex-col md:flex-row justify-between items-center p-6 border-b">
          <h2 className="text-xl font-semibold mb-2 md:mb-0">
            Properties ({filteredProperties.length})
          </h2>
          
          {/* Pagination Controls */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <span className="text-sm text-gray-700 mr-2">Show:</span>
              <select
                value={pagination.itemsPerPage}
                onChange={handleItemsPerPageChange}
                className="p-1 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="10">10</option>
                <option value="25">25</option>
                <option value="50">50</option>
                <option value="100">100</option>
              </select>
            </div>
            
            <nav className="flex items-center">
              <button
                onClick={() => handlePageChange(pagination.currentPage - 1)}
                disabled={pagination.currentPage === 1}
                className={`p-1 rounded-md ${
                  pagination.currentPage === 1 
                    ? 'text-gray-400 cursor-not-allowed' 
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </button>
              <span className="mx-2 text-sm text-gray-700">
                Page {pagination.currentPage} of {Math.max(1, Math.ceil(pagination.totalItems / pagination.itemsPerPage))}
              </span>
              <button
                onClick={() => handlePageChange(pagination.currentPage + 1)}
                disabled={pagination.currentPage >= Math.ceil(pagination.totalItems / pagination.itemsPerPage)}
                className={`p-1 rounded-md ${
                  pagination.currentPage >= Math.ceil(pagination.totalItems / pagination.itemsPerPage) 
                    ? 'text-gray-400 cursor-not-allowed' 
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                </svg>
              </button>
            </nav>
          </div>
        </div>
        
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
        
        {/* Empty State */}
        {!loading && !error && filteredProperties.length === 0 && (
          <div className="text-center py-12">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-gray-900">No properties found</h3>
            <p className="mt-1 text-sm text-gray-500">
              Try adjusting your filters to see more results.
            </p>
            <div className="mt-6">
              <button
                type="button"
                onClick={resetFilters}
                className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                <svg className="-ml-1 mr-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
                </svg>
                Reset Filters
              </button>
            </div>
          </div>
        )}
        
        {/* Results Table */}
        {!loading && !error && filteredProperties.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('property_id')}
                  >
                    <div className="flex items-center">
                      Property ID
                      {filters.sortBy === 'property_id' && (
                        <svg className="ml-1 h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          {filters.sortDirection === 'asc' ? (
                            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                          ) : (
                            <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" />
                          )}
                        </svg>
                      )}
                    </div>
                  </th>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('address')}
                  >
                    <div className="flex items-center">
                      Address
                      {filters.sortBy === 'address' && (
                        <svg className="ml-1 h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          {filters.sortDirection === 'asc' ? (
                            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                          ) : (
                            <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" />
                          )}
                        </svg>
                      )}
                    </div>
                  </th>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('estimated_value')}
                  >
                    <div className="flex items-center">
                      Estimated Value
                      {filters.sortBy === 'estimated_value' && (
                        <svg className="ml-1 h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          {filters.sortDirection === 'asc' ? (
                            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                          ) : (
                            <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" />
                          )}
                        </svg>
                      )}
                    </div>
                  </th>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('confidence_score')}
                  >
                    <div className="flex items-center">
                      Confidence
                      {filters.sortBy === 'confidence_score' && (
                        <svg className="ml-1 h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          {filters.sortDirection === 'asc' ? (
                            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                          ) : (
                            <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" />
                          )}
                        </svg>
                      )}
                    </div>
                  </th>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('model_used')}
                  >
                    <div className="flex items-center">
                      Model
                      {filters.sortBy === 'model_used' && (
                        <svg className="ml-1 h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          {filters.sortDirection === 'asc' ? (
                            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                          ) : (
                            <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" />
                          )}
                        </svg>
                      )}
                    </div>
                  </th>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('valuation_date')}
                  >
                    <div className="flex items-center">
                      Date
                      {filters.sortBy === 'valuation_date' && (
                        <svg className="ml-1 h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          {filters.sortDirection === 'asc' ? (
                            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                          ) : (
                            <path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" />
                          )}
                        </svg>
                      )}
                    </div>
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredProperties.map((property, index) => (
                  <tr 
                    key={property.property_id} 
                    className={`
                      ${index % 2 === 0 ? 'bg-white' : 'bg-gray-50'} 
                      hover:bg-blue-50 cursor-pointer transition duration-150
                    `}
                    onClick={() => handlePropertyClick(property)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {property.property_id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {property.address}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-semibold">
                      {formatCurrency(property.estimated_value)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="mr-2">
                          {formatPercentage(property.confidence_score)}
                        </div>
                        <div className="w-16 bg-gray-200 rounded-full h-1.5">
                          <div 
                            className="h-1.5 rounded-full" 
                            style={{ 
                              width: `${property.confidence_score * 100}%`,
                              backgroundColor: `${
                                property.confidence_score > 0.8 ? '#4ade80' : 
                                property.confidence_score > 0.6 ? '#facc15' : 
                                '#ef4444'
                              }`
                            }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                        {property.model_used || 'basic'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {property.valuation_date ? formatDate(property.valuation_date) : '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <button
                        className="text-blue-600 hover:text-blue-900"
                        onClick={(e) => {
                          e.stopPropagation();
                          handlePropertyClick(property);
                        }}
                      >
                        Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      
      {/* Property Detail Modal */}
      {selectedProperty && (
        <div className="fixed inset-0 z-10 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" aria-hidden="true" onClick={closePropertyDetail}></div>
            
            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
            
            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full md:max-w-2xl">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mx-auto flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 sm:mx-0 sm:h-10 sm:w-10">
                    <svg className="h-6 w-6 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                    </svg>
                  </div>
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left flex-grow">
                    <h3 className="text-lg leading-6 font-medium text-gray-900" id="modal-title">
                      Property Details
                    </h3>
                    <div className="mt-4 border-t border-gray-200 pt-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="text-sm font-medium text-gray-500">Property ID</h4>
                          <p className="mt-1 text-sm text-gray-900">{selectedProperty.property_id}</p>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-gray-500">Address</h4>
                          <p className="mt-1 text-sm text-gray-900">{selectedProperty.address}</p>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-gray-500">Estimated Value</h4>
                          <p className="mt-1 text-sm text-gray-900 font-semibold">{formatCurrency(selectedProperty.estimated_value)}</p>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-gray-500">Confidence Score</h4>
                          <div className="mt-1 flex items-center">
                            <span className="text-sm text-gray-900 font-semibold mr-2">{formatPercentage(selectedProperty.confidence_score)}</span>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div 
                                className="h-2 rounded-full" 
                                style={{ 
                                  width: `${selectedProperty.confidence_score * 100}%`,
                                  backgroundColor: `${
                                    selectedProperty.confidence_score > 0.8 ? '#4ade80' : 
                                    selectedProperty.confidence_score > 0.6 ? '#facc15' : 
                                    '#ef4444'
                                  }`
                                }}
                              ></div>
                            </div>
                          </div>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-gray-500">Valuation Model</h4>
                          <p className="mt-1 text-sm text-gray-900">{selectedProperty.model_used || 'basic'}</p>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-gray-500">Valuation Date</h4>
                          <p className="mt-1 text-sm text-gray-900">{selectedProperty.valuation_date ? formatDate(selectedProperty.valuation_date) : '-'}</p>
                        </div>
                      </div>
                      
                      {/* Property Features */}
                      {selectedProperty.features_used && (
                        <div className="mt-6 border-t border-gray-200 pt-4">
                          <h4 className="text-sm font-medium text-gray-900 mb-3">Property Features</h4>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                            {Object.entries(selectedProperty.features_used).map(([key, value]) => (
                              <div key={key}>
                                <h5 className="text-xs font-medium text-gray-500 capitalize">{key.replace(/_/g, ' ')}</h5>
                                <p className="mt-1 text-sm text-gray-900">
                                  {typeof value === 'number' && key.includes('square_feet') ? `${value.toLocaleString()} sq ft` : value.toString()}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* Advanced Metrics (if available) */}
                      {selectedProperty.advanced_metrics && (
                        <div className="mt-6 border-t border-gray-200 pt-4">
                          <h4 className="text-sm font-medium text-gray-900 mb-3">Advanced Metrics</h4>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                            {Object.entries(selectedProperty.advanced_metrics).map(([key, value]) => (
                              <div key={key}>
                                <h5 className="text-xs font-medium text-gray-500 capitalize">{key.replace(/_/g, ' ')}</h5>
                                <p className="mt-1 text-sm text-gray-900">
                                  {typeof value === 'number' ? 
                                    (key.includes('score') || key.includes('index') ? formatPercentage(value) : value.toLocaleString()) : 
                                    value.toString()}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {/* Comparable Properties */}
                      {selectedProperty.comparables && selectedProperty.comparables.length > 0 && (
                        <div className="mt-6 border-t border-gray-200 pt-4">
                          <h4 className="text-sm font-medium text-gray-900 mb-3">Comparable Properties</h4>
                          <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200">
                              <thead className="bg-gray-50">
                                <tr>
                                  <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Address</th>
                                  <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>
                                  <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Similarity</th>
                                </tr>
                              </thead>
                              <tbody className="bg-white divide-y divide-gray-200">
                                {selectedProperty.comparables.map((comp, i) => (
                                  <tr key={i}>
                                    <td className="px-3 py-2 whitespace-nowrap text-xs text-gray-500">{comp.address}</td>
                                    <td className="px-3 py-2 whitespace-nowrap text-xs text-gray-900">{formatCurrency(comp.value)}</td>
                                    <td className="px-3 py-2 whitespace-nowrap text-xs">
                                      <div className="flex items-center">
                                        <span className="mr-2">{formatPercentage(comp.similarity)}</span>
                                        <div className="w-12 bg-gray-200 rounded-full h-1.5">
                                          <div 
                                            className="bg-blue-600 h-1.5 rounded-full" 
                                            style={{ width: `${comp.similarity * 100}%` }}
                                          ></div>
                                        </div>
                                      </div>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <button 
                  type="button" 
                  className="mt-3 w-full inline-flex justify-center rounded-lg border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm"
                  onClick={closePropertyDetail}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Footer */}
      <footer className="mt-8 text-center text-sm text-gray-500">
        <p>BCBS Advanced Valuation System &copy; {new Date().getFullYear()}</p>
        <p className="mt-1">Developed for Benton County Washington property valuations</p>
      </footer>
    </div>
  );
};

export default Dashboard;
