// Use existing React and ReactDOM from global scope
(function() {
  // Destructure React hooks and components from the global React object
  const { useState, useEffect, useRef, useMemo, useCallback } = React;
  
  // Assuming Chart.js is already loaded and Chart is available globally
  const { Chart } = window;
  const { CategoryScale, LinearScale, PointElement, LineElement, BarElement, 
          Title, Tooltip, Legend, ArcElement, TimeScale, Filler } = Chart;
  
  // Destructure chart components from react-chartjs-2 (these will be created manually)
  // We'll create simple wrapper components for Chart.js
  
  // Register Chart.js components
  Chart.register(
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
  
  // Simple implementation of the Line chart component
  const Line = ({ data, options }) => {
    const chartRef = useRef(null);
    const canvasRef = useRef(null);
    
    useEffect(() => {
      if (canvasRef.current) {
        // Destroy existing chart if it exists
        if (chartRef.current) {
          chartRef.current.destroy();
        }
        
        // Create new chart
        const ctx = canvasRef.current.getContext('2d');
        chartRef.current = new Chart(ctx, {
          type: 'line',
          data: data,
          options: options
        });
      }
      
      // Cleanup function
      return () => {
        if (chartRef.current) {
          chartRef.current.destroy();
        }
      };
    }, [data, options]);
    
    return React.createElement('canvas', { ref: canvasRef });
  };
  
  // Simple implementation of the Bar chart component
  const Bar = ({ data, options }) => {
    const chartRef = useRef(null);
    const canvasRef = useRef(null);
    
    useEffect(() => {
      if (canvasRef.current) {
        // Destroy existing chart if it exists
        if (chartRef.current) {
          chartRef.current.destroy();
        }
        
        // Create new chart
        const ctx = canvasRef.current.getContext('2d');
        chartRef.current = new Chart(ctx, {
          type: 'bar',
          data: data,
          options: options
        });
      }
      
      // Cleanup function
      return () => {
        if (chartRef.current) {
          chartRef.current.destroy();
        }
      };
    }, [data, options]);
    
    return React.createElement('canvas', { ref: canvasRef });
  };
  
  // Simple implementation of the Pie chart component
  const Pie = ({ data, options }) => {
    const chartRef = useRef(null);
    const canvasRef = useRef(null);
    
    useEffect(() => {
      if (canvasRef.current) {
        // Destroy existing chart if it exists
        if (chartRef.current) {
          chartRef.current.destroy();
        }
        
        // Create new chart
        const ctx = canvasRef.current.getContext('2d');
        chartRef.current = new Chart(ctx, {
          type: 'pie',
          data: data,
          options: options
        });
      }
      
      // Cleanup function
      return () => {
        if (chartRef.current) {
          chartRef.current.destroy();
        }
      };
    }, [data, options]);
    
    return React.createElement('canvas', { ref: canvasRef });
  };
  
  // Simple debounce implementation
  const debounce = (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  };

  /**
   * Dashboard Component
   * 
   * This component fetches property valuation data from the API and displays it in a
   * filterable table and interactive visualizations. It includes options for filtering 
   * by neighborhood, price range, property type, and date, and handles loading states 
   * and errors gracefully.
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
    
    // Refs for chart containers and observer for dashboard metrics
    const valueDistributionChartRef = useRef(null);
    const neighborhoodChartRef = useRef(null);
    const trendChartRef = useRef(null);
    const metricsObserverRef = useRef(null);

    /**
     * Function to extract neighborhood from address
     * @param {string} address - The property address
     * @returns {string} - Extracted neighborhood
     */
    const extractNeighborhood = (address) => {
      if (!address) return 'Unknown';
      // Simple extraction - you might need more complex logic based on your actual address format
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
      if (!propertyData || !Array.isArray(propertyData)) return [];
      
      const neighborhoods = propertyData.map(property => 
        extractNeighborhood(property.address || '')
      ).filter((value, index, self) => 
        value !== 'Unknown' && self.indexOf(value) === index
      ).sort();
      
      setAvailableNeighborhoods(neighborhoods);
    }, []);

    /**
     * Function to fetch property valuations from the enhanced API endpoint
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
        queryParams.append('page', pagination.currentPage.toString());
        queryParams.append('page_size', pagination.itemsPerPage.toString());
        
        // Make the API request to the enhanced valuations endpoint
        const response = await fetch(`/api/valuations?${queryParams.toString()}`);
        
        if (!response.ok) {
          throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Check if the API returns data in the expected format
        if (data && Array.isArray(data.properties)) {
          // New API format with pagination info
          setProperties(data.properties);
          setPagination(prev => ({
            ...prev,
            totalItems: data.total_count || data.properties.length
          }));
        } else if (data && Array.isArray(data)) {
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
        setProperties([]);
      } finally {
        setLoading(false);
      }
    }, [filters, pagination, extractUniqueNeighborhoods]);

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
      [fetchPropertyValuations]
    );

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
      if (value === undefined || value === null) return '$0';
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
      if (value === undefined || value === null) return '0.0%';
      return `${(value * 100).toFixed(1)}%`;
    };

    /**
     * Format date values for display
     * @param {string} dateString - ISO date string
     * @returns {string} - Formatted date string
     */
    const formatDate = (dateString) => {
      if (!dateString) return 'N/A';
      try {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
          year: 'numeric',
          month: 'short',
          day: 'numeric'
        });
      } catch (e) {
        return 'Invalid Date';
      }
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
      if (!properties || !Array.isArray(properties)) return [];
      
      return properties.filter(property => {
        // Make sure property is valid
        if (!property) return false;
        
        // Apply neighborhood filter (client-side filtering if needed)
        if (filters.neighborhood && !extractNeighborhood(property.address || '').toLowerCase().includes(filters.neighborhood.toLowerCase())) {
          return false;
        }
        
        // Apply search query filter on address or property_id
        if (filters.searchQuery && 
            !((property.address || '').toLowerCase().includes(filters.searchQuery.toLowerCase()) || 
              (property.property_id || '').toLowerCase().includes(filters.searchQuery.toLowerCase()))) {
          return false;
        }
        
        return true;
      });
    }, [properties, filters.neighborhood, filters.searchQuery]);

    // Create tabbed chart controls
    const renderChartTabs = () => {
      return React.createElement('div', { className: 'border-b border-gray-200' },
        React.createElement('nav', { className: 'flex -mb-px' }, [
          React.createElement('button', {
            key: 'distribution',
            onClick: () => setActiveTab('distribution'),
            className: `ml-8 py-4 px-6 font-medium text-sm ${
              activeTab === 'distribution'
                ? 'border-b-2 border-primary text-primary'
                : 'text-muted hover:text-white hover:border-gray-300'
            }`
          }, 'Value Distribution'),
          
          React.createElement('button', {
            key: 'neighborhoods',
            onClick: () => setActiveTab('neighborhoods'),
            className: `py-4 px-6 font-medium text-sm ${
              activeTab === 'neighborhoods'
                ? 'border-b-2 border-primary text-primary'
                : 'text-muted hover:text-white hover:border-gray-300'
            }`
          }, 'Neighborhood Comparison'),
          
          React.createElement('button', {
            key: 'trend',
            onClick: () => setActiveTab('trend'),
            className: `py-4 px-6 font-medium text-sm ${
              activeTab === 'trend'
                ? 'border-b-2 border-primary text-primary'
                : 'text-muted hover:text-white hover:border-gray-300'
            }`
          }, 'Value Trend'),
          
          React.createElement('button', {
            key: 'models',
            onClick: () => setActiveTab('models'),
            className: `py-4 px-6 font-medium text-sm ${
              activeTab === 'models'
                ? 'border-b-2 border-primary text-primary'
                : 'text-muted hover:text-white hover:border-gray-300'
            }`
          }, 'Valuation Models')
        ])
      );
    };
    
    /**
     * Load property valuations when component mounts
     * Also set up the IntersectionObserver for dashboard metrics animations
     */
    useEffect(() => {
      // Fetch property valuations when component mounts
      fetchPropertyValuations();
      
      // Set up IntersectionObserver for dashboard metrics animations if supported
      if (typeof IntersectionObserver !== 'undefined') {
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
          const cards = metricsContainer.querySelectorAll('.metric-card');
          cards.forEach(card => {
            observer.observe(card);
          });
        }
        
        metricsObserverRef.current = observer;
        
        return () => {
          if (metricsObserverRef.current) {
            metricsObserverRef.current.disconnect();
          }
        };
      }
    }, [fetchPropertyValuations]);
    
    /**
     * Calculate summary metrics for the dashboard based on filtered properties
     */
    const dashboardMetrics = useMemo(() => {
      if (!filteredProperties || filteredProperties.length === 0) {
        return {
          totalProperties: 0,
          averageValue: 0,
          minValue: 0,
          maxValue: 0,
          medianValue: 0,
          averageConfidence: 0
        };
      }
      
      try {
        // Calculate metrics
        const totalProperties = filteredProperties.length;
        const values = filteredProperties.map(p => p.estimated_value || 0);
        const confidences = filteredProperties.map(p => p.confidence_score || 0);
        
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
      } catch (error) {
        console.error('Error calculating dashboard metrics:', error);
        return {
          totalProperties: 0,
          averageValue: 0,
          minValue: 0,
          maxValue: 0,
          medianValue: 0,
          averageConfidence: 0
        };
      }
    }, [filteredProperties]);
    
    /**
     * Function to prepare chart data based on filtered properties
     * Using useMemo to prevent unnecessary recalculations
     */
    const preparedChartData = useMemo(() => {
      if (!filteredProperties || filteredProperties.length === 0) {
        return null;
      }
      
      try {
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
          if (!property || typeof property.estimated_value !== 'number') return;
          
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
        
        // Group properties by neighborhood
        const neighborhoods = {};
        
        filteredProperties.forEach(property => {
          if (!property || !property.address) return;
          
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
          neighborhoods[neighborhood].totalValue += property.estimated_value || 0;
          neighborhoods[neighborhood].maxValue = Math.max(neighborhoods[neighborhood].maxValue, property.estimated_value || 0);
          neighborhoods[neighborhood].minValue = Math.min(neighborhoods[neighborhood].minValue, property.estimated_value || 0);
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
        const dateMap = {};
        
        filteredProperties.forEach(property => {
          if (!property || !property.valuation_date) return;
          
          // Use just the date part, truncating time
          const dateKey = property.valuation_date.split('T')[0];
          if (!dateMap[dateKey]) {
            dateMap[dateKey] = {
              totalValue: 0,
              count: 0,
              confidenceSum: 0
            };
          }
          dateMap[dateKey].totalValue += property.estimated_value || 0;
          dateMap[dateKey].count++;
          dateMap[dateKey].confidenceSum += property.confidence_score || 0;
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
          if (!property) return;
          
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
      } catch (error) {
        console.error('Error preparing chart data:', error);
        return null;
      }
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

    // Render the dashboard component
    return React.createElement(
      'div', 
      { className: 'container mx-auto px-4 py-8' },
      [
        // Heading
        React.createElement('h2', { 
          key: 'title',
          className: 'text-3xl font-bold mb-6' 
        }, 'Property Valuations Dashboard'),
        
        // Dashboard metrics summary 
        React.createElement('div', {
          key: 'metrics',
          id: 'dashboard-metrics',
          className: 'row mb-4'
        }, [
          React.createElement('div', { key: 'metric1', className: 'col-md-4 mb-3' },
            React.createElement('div', { className: 'metric-card card text-white bg-primary h-100' }, 
              React.createElement('div', { className: 'card-body text-center' }, [
                React.createElement('h5', { key: 'title', className: 'card-title' }, 'Total Properties'),
                React.createElement('p', { key: 'value', className: 'display-4' }, dashboardMetrics.totalProperties.toLocaleString()),
                React.createElement('p', { key: 'subtext', className: 'text-muted mb-0' }, 'in current filter set')
              ])
            )
          ),
          
          React.createElement('div', { key: 'metric2', className: 'col-md-4 mb-3' },
            React.createElement('div', { className: 'metric-card card text-white bg-success h-100' }, 
              React.createElement('div', { className: 'card-body text-center' }, [
                React.createElement('h5', { key: 'title', className: 'card-title' }, 'Average Value'),
                React.createElement('p', { key: 'value', className: 'display-4' }, formatCurrency(dashboardMetrics.averageValue)),
                React.createElement('p', { key: 'subtext', className: 'text-muted mb-0' }, `median: ${formatCurrency(dashboardMetrics.medianValue)}`)
              ])
            )
          ),
          
          React.createElement('div', { key: 'metric3', className: 'col-md-4 mb-3' },
            React.createElement('div', { className: 'metric-card card text-white bg-info h-100' }, 
              React.createElement('div', { className: 'card-body text-center' }, [
                React.createElement('h5', { key: 'title', className: 'card-title' }, 'Value Range'),
                React.createElement('p', { key: 'value', className: 'display-4' }, formatCurrency(dashboardMetrics.maxValue - dashboardMetrics.minValue)),
                React.createElement('p', { key: 'subtext', className: 'text-muted mb-0' }, 
                  `${formatCurrency(dashboardMetrics.minValue)} - ${formatCurrency(dashboardMetrics.maxValue)}`)
              ])
            )
          )
        ]),
        
        // Filter panel
        React.createElement('div', {
          key: 'filters',
          className: 'card mb-4'
        }, [
          React.createElement('div', { key: 'header', className: 'card-header' }, 
            React.createElement('h5', { className: 'mb-0' }, 'Filter Properties')
          ),
          React.createElement('div', { key: 'body', className: 'card-body' },
            React.createElement('form', { onSubmit: applyFilters }, [
              React.createElement('div', { key: 'filter-row1', className: 'row mb-3' }, [
                React.createElement('div', { key: 'neighborhood', className: 'col-md-4 mb-3' }, [
                  React.createElement('label', { 
                    key: 'label',
                    htmlFor: 'neighborhood', 
                    className: 'form-label' 
                  }, 'Neighborhood'),
                  React.createElement('select', {
                    key: 'select',
                    id: 'neighborhood',
                    name: 'neighborhood',
                    value: filters.neighborhood,
                    onChange: handleFilterChange,
                    className: 'form-select'
                  }, [
                    React.createElement('option', { key: 'all', value: '' }, 'All Neighborhoods'),
                    ...availableNeighborhoods.map(neighborhood => 
                      React.createElement('option', { key: neighborhood, value: neighborhood }, neighborhood)
                    )
                  ])
                ]),
                
                React.createElement('div', { key: 'propertyType', className: 'col-md-4 mb-3' }, [
                  React.createElement('label', { 
                    key: 'label',
                    htmlFor: 'propertyType', 
                    className: 'form-label' 
                  }, 'Property Type'),
                  React.createElement('select', {
                    key: 'select',
                    id: 'propertyType',
                    name: 'propertyType',
                    value: filters.propertyType,
                    onChange: handleFilterChange,
                    className: 'form-select'
                  }, [
                    React.createElement('option', { key: 'all', value: '' }, 'All Types'),
                    React.createElement('option', { key: 'single', value: 'single_family' }, 'Single Family'),
                    React.createElement('option', { key: 'multi', value: 'multi_family' }, 'Multi Family'),
                    React.createElement('option', { key: 'condo', value: 'condo' }, 'Condo'),
                    React.createElement('option', { key: 'town', value: 'townhouse' }, 'Townhouse'),
                    React.createElement('option', { key: 'land', value: 'land' }, 'Land'),
                    React.createElement('option', { key: 'comm', value: 'commercial' }, 'Commercial')
                  ])
                ]),
                
                React.createElement('div', { key: 'search', className: 'col-md-4 mb-3' }, [
                  React.createElement('label', { 
                    key: 'label',
                    htmlFor: 'searchQuery', 
                    className: 'form-label' 
                  }, 'Search'),
                  React.createElement('input', {
                    key: 'input',
                    type: 'text',
                    id: 'searchQuery',
                    name: 'searchQuery',
                    value: filters.searchQuery,
                    onChange: handleFilterChange,
                    placeholder: 'Address or Property ID',
                    className: 'form-control'
                  })
                ])
              ]),
              
              React.createElement('div', { key: 'filter-row2', className: 'row mb-3' }, [
                React.createElement('div', { key: 'minValue', className: 'col-md-3 mb-3' }, [
                  React.createElement('label', { 
                    key: 'label',
                    htmlFor: 'minValue', 
                    className: 'form-label' 
                  }, 'Min Value'),
                  React.createElement('input', {
                    key: 'input',
                    type: 'number',
                    id: 'minValue',
                    name: 'minValue',
                    value: filters.minValue,
                    onChange: handleFilterChange,
                    placeholder: 'Min',
                    className: 'form-control'
                  })
                ]),
                
                React.createElement('div', { key: 'maxValue', className: 'col-md-3 mb-3' }, [
                  React.createElement('label', { 
                    key: 'label',
                    htmlFor: 'maxValue', 
                    className: 'form-label' 
                  }, 'Max Value'),
                  React.createElement('input', {
                    key: 'input',
                    type: 'number',
                    id: 'maxValue',
                    name: 'maxValue',
                    value: filters.maxValue,
                    onChange: handleFilterChange,
                    placeholder: 'Max',
                    className: 'form-control'
                  })
                ]),
                
                React.createElement('div', { key: 'fromDate', className: 'col-md-3 mb-3' }, [
                  React.createElement('label', { 
                    key: 'label',
                    htmlFor: 'fromDate', 
                    className: 'form-label' 
                  }, 'From Date'),
                  React.createElement('input', {
                    key: 'input',
                    type: 'date',
                    id: 'fromDate',
                    name: 'fromDate',
                    value: filters.fromDate,
                    onChange: handleFilterChange,
                    className: 'form-control'
                  })
                ]),
                
                React.createElement('div', { key: 'toDate', className: 'col-md-3 mb-3' }, [
                  React.createElement('label', { 
                    key: 'label',
                    htmlFor: 'toDate', 
                    className: 'form-label' 
                  }, 'To Date'),
                  React.createElement('input', {
                    key: 'input',
                    type: 'date',
                    id: 'toDate',
                    name: 'toDate',
                    value: filters.toDate,
                    onChange: handleFilterChange,
                    className: 'form-control'
                  })
                ])
              ]),
              
              React.createElement('div', { key: 'filter-actions', className: 'text-end' }, [
                React.createElement('button', {
                  key: 'reset',
                  type: 'button',
                  onClick: resetFilters,
                  className: 'btn btn-secondary me-2'
                }, 'Reset'),
                React.createElement('button', {
                  key: 'apply',
                  type: 'submit',
                  className: 'btn btn-primary'
                }, 'Apply Filters')
              ])
            ])
          )
        ]),
        
        // Loading and error states
        loading && React.createElement('div', {
          key: 'loading',
          className: 'text-center py-5'
        }, [
          React.createElement('div', { 
            key: 'spinner',
            className: 'spinner-border text-primary mb-3',
            style: { width: '3rem', height: '3rem' }
          }),
          React.createElement('p', { key: 'text', className: 'text-muted' }, 'Loading property valuations...')
        ]),
        
        error && React.createElement('div', {
          key: 'error',
          className: 'alert alert-danger'
        }, [
          React.createElement('i', { 
            key: 'icon',
            className: 'bi bi-exclamation-triangle-fill me-2' 
          }),
          error
        ]),
        
        // Data visualizations
        !loading && !error && filteredProperties.length > 0 && React.createElement('div', {
          key: 'charts',
          className: 'card mb-4'
        }, [
          React.createElement('div', { key: 'tabs', className: 'card-header' }, 
            renderChartTabs()
          ),
          React.createElement('div', { key: 'chart-body', className: 'card-body' }, 
            React.createElement('div', {
              className: 'chart-container',
              style: { height: '400px' }
            }, 
              activeTab === 'distribution' && preparedChartData && 
                React.createElement(Bar, {
                  data: preparedChartData.valueDistribution, 
                  options: valueDistributionOptions
                }),
              
              activeTab === 'neighborhoods' && preparedChartData && 
                React.createElement(Bar, {
                  data: preparedChartData.neighborhoodComparison, 
                  options: neighborhoodComparisonOptions
                }),
              
              activeTab === 'trend' && preparedChartData && 
                React.createElement(Line, {
                  data: preparedChartData.valueTrend, 
                  options: valueTrendOptions
                }),
              
              activeTab === 'models' && preparedChartData && 
                React.createElement('div', {
                  style: { width: '70%', margin: '0 auto', height: '100%' }
                },
                  React.createElement(Pie, {
                    data: preparedChartData.modelDistribution, 
                    options: modelDistributionOptions
                  })
                )
            )
          )
        ]),
        
        // Property data table
        !loading && !error && React.createElement('div', {
          key: 'table',
          className: 'card mb-4'
        }, [
          React.createElement('div', { key: 'header', className: 'card-header' }, 
            React.createElement('h5', { className: 'mb-0' }, 'Property Valuations')
          ),
          React.createElement('div', { key: 'body', className: 'card-body p-0' },
            React.createElement('div', { className: 'table-responsive' },
              React.createElement('table', { className: 'table table-hover table-striped mb-0' }, [
                React.createElement('thead', { key: 'thead' },
                  React.createElement('tr', {}, [
                    React.createElement('th', { 
                      key: 'property_id',
                      scope: 'col',
                      className: 'cursor-pointer',
                      onClick: () => handleSortColumn('property_id')
                    }, [
                      'Property ID',
                      filters.sortBy === 'property_id' && 
                        React.createElement('span', { className: 'ms-1' },
                          filters.sortDirection === 'asc' ? '↑' : '↓'
                        )
                    ]),
                    React.createElement('th', { 
                      key: 'address',
                      scope: 'col',
                      className: 'cursor-pointer',
                      onClick: () => handleSortColumn('address')
                    }, [
                      'Address',
                      filters.sortBy === 'address' && 
                        React.createElement('span', { className: 'ms-1' },
                          filters.sortDirection === 'asc' ? '↑' : '↓'
                        )
                    ]),
                    React.createElement('th', { 
                      key: 'estimated_value',
                      scope: 'col',
                      className: 'cursor-pointer text-end',
                      onClick: () => handleSortColumn('estimated_value')
                    }, [
                      'Estimated Value',
                      filters.sortBy === 'estimated_value' && 
                        React.createElement('span', { className: 'ms-1' },
                          filters.sortDirection === 'asc' ? '↑' : '↓'
                        )
                    ]),
                    React.createElement('th', { 
                      key: 'confidence',
                      scope: 'col',
                      className: 'cursor-pointer',
                      onClick: () => handleSortColumn('confidence_score')
                    }, [
                      'Confidence',
                      filters.sortBy === 'confidence_score' && 
                        React.createElement('span', { className: 'ms-1' },
                          filters.sortDirection === 'asc' ? '↑' : '↓'
                        )
                    ]),
                    React.createElement('th', { 
                      key: 'date',
                      scope: 'col',
                      className: 'cursor-pointer',
                      onClick: () => handleSortColumn('valuation_date')
                    }, [
                      'Date',
                      filters.sortBy === 'valuation_date' && 
                        React.createElement('span', { className: 'ms-1' },
                          filters.sortDirection === 'asc' ? '↑' : '↓'
                        )
                    ])
                  ])
                ),
                React.createElement('tbody', { key: 'tbody' },
                  filteredProperties.length === 0 ? 
                    React.createElement('tr', {},
                      React.createElement('td', { colSpan: '5', className: 'text-center py-4' },
                        'No properties found matching your filters.'
                      )
                    ) :
                    filteredProperties.map(property => 
                      React.createElement('tr', { 
                        key: property.property_id || Math.random().toString(36),
                        className: 'cursor-pointer',
                        onClick: () => handlePropertyClick(property)
                      }, [
                        React.createElement('td', { key: 'id' }, property.property_id),
                        React.createElement('td', { key: 'address' }, property.address),
                        React.createElement('td', { key: 'value', className: 'text-end' }, 
                          formatCurrency(property.estimated_value)
                        ),
                        React.createElement('td', { key: 'confidence' }, 
                          React.createElement('div', { className: 'd-flex align-items-center' }, [
                            React.createElement('div', { key: 'text', className: 'me-2' },
                              formatPercentage(property.confidence_score || 0)
                            ),
                            React.createElement('div', { key: 'bar', className: 'progress flex-grow-1', style: { height: '8px' } },
                              React.createElement('div', { 
                                className: 'progress-bar',
                                style: { 
                                  width: `${(property.confidence_score || 0) * 100}%`,
                                  backgroundColor: 
                                    property.confidence_score > 0.8 ? 'var(--bs-success)' :
                                    property.confidence_score > 0.6 ? 'var(--bs-primary)' :
                                    property.confidence_score > 0.4 ? 'var(--bs-warning)' :
                                    'var(--bs-danger)'
                                }
                              })
                            )
                          ])
                        ),
                        React.createElement('td', { key: 'date' }, 
                          property.valuation_date ? formatDate(property.valuation_date) : 'N/A'
                        )
                      ])
                    )
                )
              ])
            )
          ),
          React.createElement('div', { key: 'footer', className: 'card-footer d-flex justify-content-between align-items-center' }, [
            React.createElement('div', { key: 'per-page', className: 'd-flex align-items-center' }, [
              React.createElement('span', { key: 'label', className: 'me-2' }, 'Items per page:'),
              React.createElement('select', {
                key: 'select',
                value: pagination.itemsPerPage,
                onChange: handleItemsPerPageChange,
                className: 'form-select form-select-sm',
                style: { width: 'auto' }
              }, [
                React.createElement('option', { key: '10', value: '10' }, '10'),
                React.createElement('option', { key: '25', value: '25' }, '25'),
                React.createElement('option', { key: '50', value: '50' }, '50'),
                React.createElement('option', { key: '100', value: '100' }, '100')
              ]),
              React.createElement('span', { key: 'showing', className: 'ms-4 text-muted small' }, 
                `Showing ${Math.min((pagination.currentPage - 1) * pagination.itemsPerPage + 1, pagination.totalItems)} 
                to ${Math.min(pagination.currentPage * pagination.itemsPerPage, pagination.totalItems)} 
                of ${pagination.totalItems} properties`
              )
            ]),
            
            React.createElement('div', { key: 'pagination' }, [
              React.createElement('button', {
                key: 'prev',
                onClick: () => handlePageChange(pagination.currentPage - 1),
                disabled: pagination.currentPage === 1,
                className: 'btn btn-sm btn-outline-secondary me-2'
              }, 'Previous'),
              React.createElement('button', {
                key: 'next',
                onClick: () => handlePageChange(pagination.currentPage + 1),
                disabled: pagination.currentPage * pagination.itemsPerPage >= pagination.totalItems,
                className: 'btn btn-sm btn-outline-secondary'
              }, 'Next')
            ])
          ])
        ]),
        
        // Property detail modal 
        selectedProperty && React.createElement('div', {
          key: 'modal',
          className: 'modal fade show',
          style: { display: 'block' }
        }, [
          React.createElement('div', { 
            key: 'backdrop',
            className: 'modal-backdrop fade show',
            onClick: closePropertyDetail
          }),
          React.createElement('div', { key: 'dialog', className: 'modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable' },
            React.createElement('div', { className: 'modal-content' }, [
              React.createElement('div', { key: 'header', className: 'modal-header' }, [
                React.createElement('h5', { key: 'title', className: 'modal-title' }, 'Property Details'),
                React.createElement('button', {
                  key: 'close',
                  type: 'button',
                  className: 'btn-close',
                  onClick: closePropertyDetail
                })
              ]),
              React.createElement('div', { key: 'body', className: 'modal-body' },
                React.createElement('div', { className: 'row' }, [
                  React.createElement('div', { key: 'col1', className: 'col-md-6' }, [
                    React.createElement('div', { key: 'info', className: 'mb-4' }, [
                      React.createElement('h6', { key: 'title', className: 'fw-bold mb-3' }, 'Property Information'),
                      React.createElement('div', { key: 'content', className: 'card bg-dark p-3' }, [
                        React.createElement('p', { key: 'id' }, [
                          React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Property ID'),
                          selectedProperty.property_id
                        ]),
                        React.createElement('p', { key: 'address' }, [
                          React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Address'),
                          selectedProperty.address
                        ]),
                        React.createElement('p', { key: 'neighborhood' }, [
                          React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Neighborhood'),
                          extractNeighborhood(selectedProperty.address || '')
                        ]),
                        React.createElement('p', { key: 'type' }, [
                          React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Property Type'),
                          selectedProperty.property_type || 'Unknown'
                        ]),
                        React.createElement('p', { key: 'year' }, [
                          React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Year Built'),
                          selectedProperty.year_built || 'Unknown'
                        ]),
                        React.createElement('p', { key: 'sqft', className: 'mb-0' }, [
                          React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Sq Footage'),
                          selectedProperty.sq_footage ? `${selectedProperty.sq_footage.toLocaleString()} sq ft` : 'Unknown'
                        ])
                      ])
                    ]),
                    
                    React.createElement('div', { key: 'features' }, [
                      React.createElement('h6', { key: 'title', className: 'fw-bold mb-3' }, 'Property Features'),
                      React.createElement('div', { key: 'content', className: 'card bg-dark p-3' }, [
                        React.createElement('div', { key: 'grid', className: 'row mb-3' }, [
                          React.createElement('div', { key: 'col1', className: 'col-6' }, [
                            React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Bedrooms'),
                            React.createElement('p', { key: 'value' }, selectedProperty.bedrooms || 'N/A')
                          ]),
                          React.createElement('div', { key: 'col2', className: 'col-6' }, [
                            React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Bathrooms'),
                            React.createElement('p', { key: 'value' }, selectedProperty.bathrooms || 'N/A')
                          ]),
                          React.createElement('div', { key: 'col3', className: 'col-6' }, [
                            React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Garage'),
                            React.createElement('p', { key: 'value' }, selectedProperty.garage_spaces || 'N/A')
                          ]),
                          React.createElement('div', { key: 'col4', className: 'col-6' }, [
                            React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Lot Size'),
                            React.createElement('p', { key: 'value' }, selectedProperty.lot_size ? 
                              `${selectedProperty.lot_size.toLocaleString()} sq ft` : 'N/A')
                          ])
                        ]),
                        
                        selectedProperty.features && selectedProperty.features.length > 0 &&
                          React.createElement('div', { key: 'additional' }, [
                            React.createElement('small', { 
                              key: 'label',
                              className: 'text-muted d-block mb-2' 
                            }, 'Additional Features'),
                            React.createElement('div', { key: 'badges', className: 'd-flex flex-wrap gap-2' },
                              selectedProperty.features.map((feature, index) => 
                                React.createElement('span', { 
                                  key: index,
                                  className: 'badge bg-info text-dark' 
                                }, feature)
                              )
                            )
                          ])
                      ])
                    ])
                  ]),
                  
                  React.createElement('div', { key: 'col2', className: 'col-md-6' }, [
                    React.createElement('div', { key: 'valuation', className: 'mb-4' }, [
                      React.createElement('h6', { key: 'title', className: 'fw-bold mb-3' }, 'Valuation Details'),
                      React.createElement('div', { key: 'content', className: 'card bg-dark p-3' }, [
                        React.createElement('div', { key: 'value', className: 'd-flex justify-content-between align-items-center mb-3' }, [
                          React.createElement('small', { key: 'label', className: 'text-muted' }, 'Estimated Value'),
                          React.createElement('span', { key: 'amount', className: 'fs-4 fw-bold text-primary' }, 
                            formatCurrency(selectedProperty.estimated_value)
                          )
                        ]),
                        
                        React.createElement('div', { key: 'confidence', className: 'mb-3' }, [
                          React.createElement('div', { key: 'label', className: 'd-flex justify-content-between' }, [
                            React.createElement('small', { key: 'text', className: 'text-muted' }, 'Confidence Score'),
                            React.createElement('small', { key: 'value' }, 
                              formatPercentage(selectedProperty.confidence_score || 0)
                            )
                          ]),
                          React.createElement('div', { key: 'progress', className: 'progress mt-1', style: { height: '8px' } },
                            React.createElement('div', { 
                              className: 'progress-bar',
                              style: { 
                                width: `${(selectedProperty.confidence_score || 0) * 100}%`,
                                backgroundColor: 
                                  selectedProperty.confidence_score > 0.8 ? 'var(--bs-success)' :
                                  selectedProperty.confidence_score > 0.6 ? 'var(--bs-primary)' :
                                  selectedProperty.confidence_score > 0.4 ? 'var(--bs-warning)' :
                                  'var(--bs-danger)'
                              }
                            })
                          )
                        ]),
                        
                        React.createElement('p', { key: 'method' }, [
                          React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Valuation Method'),
                          selectedProperty.model_used || 'Multiple Models'
                        ]),
                        
                        React.createElement('p', { key: 'date' }, [
                          React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Valuation Date'),
                          selectedProperty.valuation_date ? formatDate(selectedProperty.valuation_date) : 'N/A'
                        ]),
                        
                        selectedProperty.price_per_sqft && React.createElement('p', { key: 'price-sqft' }, [
                          React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 'Price Per Sq Ft'),
                          formatCurrency(selectedProperty.price_per_sqft)
                        ]),
                        
                        selectedProperty.value_change_pct && React.createElement('p', { key: 'change', className: 'mb-0' }, [
                          React.createElement('small', { key: 'label', className: 'text-muted d-block' }, 
                            'Value Change (Last 12 Months)'
                          ),
                          React.createElement('span', { 
                            key: 'value',
                            className: selectedProperty.value_change_pct > 0 ? 'text-success' : 'text-danger'
                          }, 
                            `${selectedProperty.value_change_pct > 0 ? '+' : ''}${formatPercentage(selectedProperty.value_change_pct)}`
                          )
                        ])
                      ])
                    ]),
                    
                    selectedProperty.valuation_factors && React.createElement('div', { key: 'factors' }, [
                      React.createElement('h6', { key: 'title', className: 'fw-bold mb-3' }, 'Valuation Factors'),
                      React.createElement('div', { key: 'content', className: 'card bg-dark p-3' },
                        Object.entries(selectedProperty.valuation_factors).map(([factor, impact]) => 
                          React.createElement('div', { key: factor, className: 'mb-2' }, [
                            React.createElement('div', { key: 'labels', className: 'd-flex justify-content-between' }, [
                              React.createElement('small', { key: 'factor' }, 
                                factor.split('_').map(word => 
                                  word.charAt(0).toUpperCase() + word.slice(1)
                                ).join(' ')
                              ),
                              React.createElement('small', { 
                                key: 'impact',
                                className: impact > 0 ? 'text-success' : 
                                           impact < 0 ? 'text-danger' : 'text-muted'
                              }, 
                                `${impact > 0 ? '+' : ''}${formatPercentage(impact)}`
                              )
                            ]),
                            React.createElement('div', { 
                              key: 'progress',
                              className: 'progress mt-1 position-relative',
                              style: { height: '8px' }
                            }, [
                              React.createElement('div', { 
                                key: 'center',
                                className: 'position-absolute',
                                style: {
                                  width: '2px',
                                  height: '100%',
                                  backgroundColor: 'var(--bs-secondary)',
                                  left: '50%',
                                  transform: 'translateX(-50%)'
                                }
                              }),
                              React.createElement('div', { 
                                key: 'bar',
                                className: 'position-absolute',
                                style: {
                                  height: '100%',
                                  left: impact >= 0 ? '50%' : `${50 - Math.min(Math.abs(impact) * 100, 50)}%`,
                                  width: `${Math.min(Math.abs(impact) * 100, 50)}%`,
                                  backgroundColor: impact > 0 ? 'var(--bs-success)' : 
                                                 impact < 0 ? 'var(--bs-danger)' : 'var(--bs-gray)'
                                }
                              })
                            ])
                          ])
                        )
                      )
                    ])
                  ])
                ])
              ),
              React.createElement('div', { key: 'footer', className: 'modal-footer' },
                React.createElement('button', {
                  type: 'button',
                  className: 'btn btn-primary',
                  onClick: closePropertyDetail
                }, 'Close')
              )
            ])
          )
        ])
      ]
    );
  };

  // Mount the React component
  const dashboardRoot = document.getElementById('dashboardRoot');
  if (dashboardRoot) {
    ReactDOM.render(React.createElement(Dashboard), dashboardRoot);
  }
})();