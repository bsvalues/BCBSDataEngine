/**
 * Enhanced Dashboard Component for BCBS Dash Project
 * 
 * This component fetches real-time property valuation data from various API endpoints
 * and displays it in interactive visualizations and filterable tables. It handles
 * data fetching, filtering, pagination, error handling, and responsive display.
 * 
 * Key Features:
 * - Real-time data fetching from "/api/valuations", "/api/etl-status", and "/api/agent-status" endpoints
 * - Interactive filtering by neighborhood, price range, property type, and date
 * - Dynamic table with sortable columns and pagination
 * - Advanced data visualization with Chart.js showing trends, distributions, and comparisons
 * - Comprehensive error handling with meaningful user feedback
 * - Loading indicators for all asynchronous operations
 * - ETL pipeline monitoring with progress visualization
 * - Agent status monitoring with detailed agent metrics
 * - Fully responsive design using Tailwind CSS
 * - Auto-refresh capability with configurable intervals
 * - Detailed property and agent information in modal views
 * 
 * @version 2.0.0
 * @author BCBS Dash Team
 * @last-updated 2025-03-31
 */

function Dashboard() {
  // --- State Variables ---
  
  // Property data state
  const [properties, setProperties] = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const [availableNeighborhoods, setAvailableNeighborhoods] = React.useState([]);
  const [availablePropertyTypes, setAvailablePropertyTypes] = React.useState([]);
  
  // ETL status state
  const [etlStatus, setEtlStatus] = React.useState({
    status: 'unknown',
    lastUpdate: null,
    progress: 0,
    sources: [],
    metrics: {
      recordsProcessed: 0,
      successRate: 0,
      averageProcessingTime: 0
    },
    dataQuality: {
      completeness: 0,
      accuracy: 0,
      timeliness: 0
    },
    isLoading: true,
    error: null
  });
  
  // Agent status state
  const [agentStatus, setAgentStatus] = React.useState({
    agents: [],
    isLoading: true,
    error: null,
    lastUpdate: null
  });
  
  // Selected item states for detailed views
  const [selectedProperty, setSelectedProperty] = React.useState(null);
  const [selectedAgent, setSelectedAgent] = React.useState(null);
  
  // API key state (for authentication)
  const [apiKey, setApiKey] = React.useState('');
  
  // Filter state
  const [filters, setFilters] = React.useState({
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
  
  // Pagination state
  const [pagination, setPagination] = React.useState({
    currentPage: 1,
    totalPages: 1,
    totalItems: 0,
    itemsPerPage: 10
  });
  
  // Auto-refresh settings
  const [refreshSettings, setRefreshSettings] = React.useState({
    autoRefresh: false,
    interval: 60000, // 1 minute
    lastRefreshTime: null
  });
  
  // Dashboard tab navigation
  const [activeDashboardTab, setActiveDashboardTab] = React.useState('properties');
  
  // --- Chart References ---
  const valueDistributionChartRef = React.useRef(null);
  const neighborhoodChartRef = React.useRef(null);
  const trendChartRef = React.useRef(null);
  const etlProgressChartRef = React.useRef(null);
  const agentPerformanceChartRef = React.useRef(null);
  const metricsObserverRef = React.useRef(null); // For animation triggering with Intersection Observer
  const refreshTimerRef = React.useRef(null); // For tracking auto-refresh interval
  
  /**
   * Initialization effect - Fetch all data on component mount
   * This effect runs only once when the component is first loaded
   */
  React.useEffect(() => {
    console.log('Initializing dashboard data...');
    
    // Initial data fetch
    fetchPropertyValuations();
    fetchEtlStatus();
    fetchAgentStatus();
    
    // Set API key from localStorage if available
    const savedApiKey = localStorage.getItem('bcbs_api_key');
    if (savedApiKey) {
      setApiKey(savedApiKey);
    }
  }, []);

  /**
   * Function to fetch property valuations from the enhanced API endpoint
   * Includes pagination, sorting, and filtering capabilities
   */
  const fetchPropertyValuations = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Build query parameters based on active filters
      const queryParams = new URLSearchParams();
      
      // Add filter parameters if they exist
      if (filters.neighborhood) queryParams.append('neighborhood', filters.neighborhood);
      if (filters.minValue) queryParams.append('min_value', filters.minValue);
      if (filters.maxValue) queryParams.append('max_value', filters.maxValue);
      if (filters.propertyType) queryParams.append('property_type', filters.propertyType);
      if (filters.fromDate) queryParams.append('from_date', filters.fromDate);
      if (filters.toDate) queryParams.append('to_date', filters.toDate);
      if (filters.searchQuery) queryParams.append('search', filters.searchQuery);
      if (filters.sortBy) queryParams.append('sort_by', filters.sortBy);
      if (filters.sortDirection) queryParams.append('sort_direction', filters.sortDirection);
      
      // Add pagination parameters
      queryParams.append('page', pagination.currentPage);
      queryParams.append('per_page', pagination.itemsPerPage);
      
      // Headers for the API request including the API key
      const headers = {
        'Content-Type': 'application/json'
      };
      
      // Add API key header if available
      if (apiKey) {
        headers['X-API-Key'] = apiKey;
      }
      
      // Make the API request to the enhanced valuations endpoint
      console.log(`Fetching from /api/valuations?${queryParams.toString()}`);
      const response = await fetch(`/api/valuations?${queryParams.toString()}`, {
        method: 'GET',
        headers: headers
      });
      
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Update the UI based on response data format
      if (data && typeof data === 'object') {
        if (Array.isArray(data.properties)) {
          // New API format with pagination info
          setProperties(data.properties);
          setPagination(prev => ({
            ...prev,
            totalItems: data.total_count || data.properties.length,
            totalPages: data.total_pages || Math.ceil((data.total_count || data.properties.length) / prev.itemsPerPage)
          }));
          
          // Extract metadata if available
          if (data.metadata) {
            if (Array.isArray(data.metadata.neighborhoods)) {
              setAvailableNeighborhoods(data.metadata.neighborhoods);
            }
            
            if (Array.isArray(data.metadata.property_types)) {
              setAvailablePropertyTypes(data.metadata.property_types);
            }
          }
        } else if (Array.isArray(data)) {
          // Legacy API format (just an array of properties)
          setProperties(data);
          setPagination(prev => ({
            ...prev,
            totalItems: data.length,
            totalPages: Math.ceil(data.length / prev.itemsPerPage)
          }));
          
          // Extract neighborhoods from the data
          extractUniqueNeighborhoods(data);
        } else {
          throw new Error("Unexpected data format received from API");
        }
        
        console.log(`Loaded ${Array.isArray(data.properties) ? data.properties.length : data.length} properties from API`);
      } else {
        throw new Error("Invalid response data");
      }
    } catch (err) {
      console.error('Error fetching property valuations:', err);
      setError(`Failed to load property valuations: ${err.message}. Please try again later.`);
    } finally {
      setLoading(false);
      
      // Update last refresh time
      setRefreshSettings(prev => ({
        ...prev,
        lastRefreshTime: new Date()
      }));
    }
  }, [filters, pagination.currentPage, pagination.itemsPerPage, apiKey, pagination.itemsPerPage]);

  /**
   * Extract unique neighborhoods from the property data
   * @param {Array} propertyData - Array of property objects
   */
  const extractUniqueNeighborhoods = React.useCallback((propertyData) => {
    if (!Array.isArray(propertyData) || propertyData.length === 0) return;
    
    // Extract neighborhoods, preferring the neighborhood property if available
    const neighborhoods = propertyData
      .map(property => property.neighborhood || extractNeighborhoodFromAddress(property.address))
      .filter(Boolean) // Remove null/undefined/empty values
      .filter((value, index, self) => self.indexOf(value) === index) // Remove duplicates
      .sort(); // Sort alphabetically
    
    setAvailableNeighborhoods(neighborhoods);
  }, []);

  /**
   * Function to extract neighborhood from address if needed
   * @param {string} address - The property address
   * @returns {string} - Extracted neighborhood or empty string
   */
  const extractNeighborhoodFromAddress = (address) => {
    if (!address) return '';
    
    // This is a simple extraction - you might need more complex logic
    // based on your actual address format
    const parts = address.split(',');
    if (parts.length >= 2) {
      return parts[1].trim();
    }
    
    return '';
  };

  /**
   * Function to handle filter changes
   * @param {Event} e - The input change event
   */
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    
    setFilters(prevFilters => ({
      ...prevFilters,
      [name]: value
    }));
    
    // Reset to first page when filters change except for search query
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
  const debounce = (func, wait) => {
    let timeout;
    return function(...args) {
      const context = this;
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(context, args), wait);
    };
  };
  
  const debouncedSearch = React.useCallback(
    debounce((query) => {
      // Reset to first page when search query changes
      setPagination(prev => ({
        ...prev,
        currentPage: 1
      }));
      
      // Only fetch if we're not already searching or if query length is appropriate
      if (query.length === 0 || query.length >= 2) {
        fetchPropertyValuations();
      }
    }, 500),
    [fetchPropertyValuations]
  );

  /**
   * Function to apply filters (for form submission)
   * @param {Event} e - The form submit event
   */
  const applyFilters = (e) => {
    if (e) e.preventDefault();
    
    // Reset to first page
    setPagination(prev => ({
      ...prev,
      currentPage: 1
    }));
    
    // Fetch with current filters
    fetchPropertyValuations();
  };

  /**
   * Function to reset all filters to default values
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
    
    // Reset pagination to first page
    setPagination(prev => ({
      ...prev,
      currentPage: 1
    }));
    
    // Fetch with cleared filters (deferred to next render cycle)
    setTimeout(() => fetchPropertyValuations(), 0);
  };

  /**
   * Function to handle sorting column clicks
   * @param {string} columnName - The name of the column to sort by
   */
  const handleSortColumn = (columnName) => {
    setFilters(prevFilters => {
      // Toggle sort direction if clicking the same column
      const newDirection = prevFilters.sortBy === columnName && 
                         prevFilters.sortDirection === 'asc' ? 'desc' : 'asc';
      
      return {
        ...prevFilters,
        sortBy: columnName,
        sortDirection: newDirection
      };
    });
    
    // Fetch data with new sort settings (deferred to next render cycle)
    setTimeout(() => fetchPropertyValuations(), 0);
  };

  /**
   * Function to handle pagination changes
   * @param {number} newPage - The new page number to navigate to
   */
  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= pagination.totalPages) {
      setPagination(prev => ({
        ...prev,
        currentPage: newPage
      }));
      
      // Fetch data for the new page
      setTimeout(() => fetchPropertyValuations(), 0);
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
    
    // Fetch data with new pagination settings
    setTimeout(() => fetchPropertyValuations(), 0);
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
   * Function to fetch ETL pipeline status data
   */
  const fetchEtlStatus = React.useCallback(async () => {
    setEtlStatus(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      // Headers for the API request including the API key
      const headers = {
        'Content-Type': 'application/json'
      };
      
      // Add API key header if available
      if (apiKey) {
        headers['X-API-Key'] = apiKey;
      }
      
      // Make the API request to the ETL status endpoint
      const response = await fetch('/api/etl-status', {
        method: 'GET',
        headers: headers
      });
      
      if (!response.ok) {
        throw new Error(`ETL status API request failed with status ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Transform API response to our state format
      setEtlStatus({
        status: data.status || 'unknown',
        lastUpdate: data.last_update ? new Date(data.last_update) : null,
        progress: data.progress || 0,
        sources: data.sources || [],
        metrics: {
          recordsProcessed: data.metrics?.records_processed || 0,
          successRate: data.metrics?.success_rate || 0,
          averageProcessingTime: data.metrics?.average_processing_time || 0
        },
        dataQuality: {
          completeness: data.data_quality?.completeness || 0,
          accuracy: data.data_quality?.accuracy || 0,
          timeliness: data.data_quality?.timeliness || 0
        },
        isLoading: false,
        error: null
      });
      
      // Update refresh timestamp
      setRefreshSettings(prev => ({
        ...prev,
        lastRefreshTime: new Date()
      }));
      
      console.log('ETL status data loaded successfully.');
    } catch (err) {
      console.error('Error fetching ETL status:', err);
      setEtlStatus(prev => ({
        ...prev,
        isLoading: false,
        error: `Failed to load ETL status: ${err.message}. Please try again later.`
      }));
    }
  }, [apiKey]);

  /**
   * Function to fetch agent status data
   */
  const fetchAgentStatus = React.useCallback(async () => {
    setAgentStatus(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      // Headers for the API request including the API key
      const headers = {
        'Content-Type': 'application/json'
      };
      
      // Add API key header if available
      if (apiKey) {
        headers['X-API-Key'] = apiKey;
      }
      
      // Make the API request to the agent status endpoint
      const response = await fetch('/api/agent-status', {
        method: 'GET',
        headers: headers
      });
      
      if (!response.ok) {
        throw new Error(`Agent status API request failed with status ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Transform API response to our state format
      setAgentStatus({
        agents: Array.isArray(data.agents) ? data.agents : [],
        isLoading: false,
        error: null,
        lastUpdate: new Date()
      });
      
      // Update refresh timestamp
      setRefreshSettings(prev => ({
        ...prev,
        lastRefreshTime: new Date()
      }));
      
      console.log(`Loaded status for ${data.agents?.length || 0} agents.`);
    } catch (err) {
      console.error('Error fetching agent status:', err);
      setAgentStatus(prev => ({
        ...prev,
        isLoading: false,
        error: `Failed to load agent status: ${err.message}. Please try again later.`
      }));
    }
  }, [apiKey]);

  /**
   * Function to fetch detailed information for a specific agent
   * @param {string} agentId - The ID of the agent to retrieve detailed information for
   */
  const fetchAgentDetails = React.useCallback(async (agentId) => {
    if (!agentId) return;
    
    try {
      // Headers for the API request including the API key
      const headers = {
        'Content-Type': 'application/json'
      };
      
      // Add API key header if available
      if (apiKey) {
        headers['X-API-Key'] = apiKey;
      }
      
      // Make the API request to the agent logs endpoint
      const response = await fetch(`/api/agent-logs/${agentId}`, {
        method: 'GET',
        headers: headers
      });
      
      if (!response.ok) {
        throw new Error(`Agent logs API request failed with status ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Find the agent in our current state
      const agent = agentStatus.agents.find(a => a.agent_id === agentId);
      
      if (agent) {
        // Set selected agent with the logs from the API
        setSelectedAgent({
          ...agent,
          logs: data.logs || []
        });
      } else {
        throw new Error(`Agent ${agentId} not found in current state`);
      }
      
      console.log(`Loaded ${data.logs?.length || 0} logs for agent ${agentId}`);
    } catch (err) {
      console.error(`Error fetching agent logs for ${agentId}:`, err);
      // Set error in selected agent
      setSelectedAgent(prev => ({
        ...prev,
        logsError: `Failed to load agent logs: ${err.message}. Please try again later.`
      }));
    }
  }, [apiKey, agentStatus.agents]);

  /**
   * Function to refresh all dashboard data
   */
  const refreshAllData = React.useCallback(() => {
    console.log('Refreshing all dashboard data...');
    
    // Update refresh timestamp
    setRefreshSettings(prev => ({
      ...prev,
      lastRefreshTime: new Date()
    }));
    
    // Fetch all data
    fetchPropertyValuations();
    fetchEtlStatus();
    fetchAgentStatus();
  }, [fetchPropertyValuations, fetchEtlStatus, fetchAgentStatus]);

  /**
   * Handle auto-refresh interval change
   * @param {Event} e - The change event
   */
  const handleRefreshIntervalChange = (e) => {
    const newInterval = parseInt(e.target.value, 10);
    
    setRefreshSettings(prev => ({
      ...prev,
      interval: newInterval
    }));
  };

  /**
   * Toggle auto-refresh on/off
   */
  const toggleAutoRefresh = () => {
    setRefreshSettings(prev => ({
      ...prev,
      autoRefresh: !prev.autoRefresh
    }));
  };

  /**
   * Effect for managing the auto-refresh interval
   */
  React.useEffect(() => {
    // Clear existing interval
    if (refreshTimerRef.current) {
      clearInterval(refreshTimerRef.current);
      refreshTimerRef.current = null;
    }
    
    // Set up new interval if auto-refresh is enabled
    if (refreshSettings.autoRefresh) {
      refreshTimerRef.current = setInterval(() => {
        refreshAllData();
      }, refreshSettings.interval);
    }
    
    // Cleanup function
    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [refreshSettings.autoRefresh, refreshSettings.interval, refreshAllData]);

  /**
   * Handle selecting an agent for detailed view
   * @param {Object} agent - The agent object to view in detail
   */
  const handleAgentClick = (agent) => {
    // Fetch detailed information including logs
    fetchAgentDetails(agent.agent_id);
  };

  /**
   * Close the agent detail view
   */
  const closeAgentDetail = () => {
    setSelectedAgent(null);
  };

  /**
   * Handle switching between dashboard tabs
   * @param {string} tabName - The name of the tab to switch to
   */
  const handleTabChange = (tabName) => {
    setActiveDashboardTab(tabName);
  };

  /**
   * Format percentage values for display
   * @param {number} value - Value to format as percentage
   * @returns {string} - Formatted percentage string
   */
  const formatPercentage = (value) => {
    if (typeof value !== 'number') return '0%';
    return `${(value * 100).toFixed(1)}%`;
  };

  /**
   * Format date values for display
   * @param {string} dateString - ISO date string
   * @returns {string} - Formatted date string
   */
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return 'Invalid date';
    
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  /**
   * Format date values for display with time included
   * @param {string|Date} dateTimeValue - Date object or ISO date string
   * @returns {string} - Formatted date and time string
   */
  const formatDateTime = (dateTimeValue) => {
    if (!dateTimeValue) return 'N/A';
    
    const date = dateTimeValue instanceof Date ? dateTimeValue : new Date(dateTimeValue);
    if (isNaN(date.getTime())) return 'Invalid date';
    
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  /**
   * Helper function to get status color class based on status string
   * @param {string} status - Status string
   * @returns {string} - Tailwind CSS color class
   */
  const getStatusColorClass = (status) => {
    const statusMap = {
      'active': 'text-green-500',
      'idle': 'text-blue-500',
      'error': 'text-red-500',
      'processing': 'text-blue-500',
      'queued': 'text-gray-500',
      'completed': 'text-green-500',
      'warning': 'text-yellow-500',
      'unknown': 'text-gray-500'
    };
    
    return statusMap[status?.toLowerCase()] || 'text-gray-500';
  };

  /**
   * Render the property table
   * @returns {JSX.Element} - The property table component
   */
  const renderPropertyTable = () => (
    <div className="bg-white rounded-lg shadow p-4 mt-4">
      {/* Filter form */}
      <form onSubmit={applyFilters} className="mb-4 space-y-4">
        <div className="flex flex-wrap -mx-2">
          {/* Search box */}
          <div className="px-2 w-full lg:w-1/3 mb-3">
            <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
            <input
              type="text"
              name="searchQuery"
              value={filters.searchQuery}
              onChange={handleFilterChange}
              placeholder="Search by address, ID..."
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
          
          {/* Neighborhood filter */}
          <div className="px-2 w-full sm:w-1/2 lg:w-1/6 mb-3">
            <label className="block text-sm font-medium text-gray-700 mb-1">Neighborhood</label>
            <select
              name="neighborhood"
              value={filters.neighborhood}
              onChange={handleFilterChange}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="">All</option>
              {availableNeighborhoods.map((neighborhood, idx) => (
                <option key={idx} value={neighborhood}>{neighborhood}</option>
              ))}
            </select>
          </div>
          
          {/* Property type filter */}
          <div className="px-2 w-full sm:w-1/2 lg:w-1/6 mb-3">
            <label className="block text-sm font-medium text-gray-700 mb-1">Property Type</label>
            <select
              name="propertyType"
              value={filters.propertyType}
              onChange={handleFilterChange}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="">All</option>
              {availablePropertyTypes.map((type, idx) => (
                <option key={idx} value={type}>{type}</option>
              ))}
            </select>
          </div>
          
          {/* Min value filter */}
          <div className="px-2 w-full sm:w-1/2 lg:w-1/6 mb-3">
            <label className="block text-sm font-medium text-gray-700 mb-1">Min Value</label>
            <input
              type="number"
              name="minValue"
              value={filters.minValue}
              onChange={handleFilterChange}
              placeholder="Min $"
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
          
          {/* Max value filter */}
          <div className="px-2 w-full sm:w-1/2 lg:w-1/6 mb-3">
            <label className="block text-sm font-medium text-gray-700 mb-1">Max Value</label>
            <input
              type="number"
              name="maxValue"
              value={filters.maxValue}
              onChange={handleFilterChange}
              placeholder="Max $"
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
        </div>
        
        <div className="flex flex-wrap -mx-2">
          {/* From date filter */}
          <div className="px-2 w-full sm:w-1/2 lg:w-1/4 mb-3">
            <label className="block text-sm font-medium text-gray-700 mb-1">From Date</label>
            <input
              type="date"
              name="fromDate"
              value={filters.fromDate}
              onChange={handleFilterChange}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
          
          {/* To date filter */}
          <div className="px-2 w-full sm:w-1/2 lg:w-1/4 mb-3">
            <label className="block text-sm font-medium text-gray-700 mb-1">To Date</label>
            <input
              type="date"
              name="toDate"
              value={filters.toDate}
              onChange={handleFilterChange}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
          
          {/* Filter buttons */}
          <div className="px-2 w-full lg:w-1/2 flex items-end mb-3">
            <div className="flex space-x-2">
              <button
                type="submit"
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                <svg className="h-4 w-4 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                </svg>
                Apply Filters
              </button>
              <button
                type="button"
                onClick={resetFilters}
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                <svg className="h-4 w-4 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Reset
              </button>
              <button
                type="button"
                onClick={refreshAllData}
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                <svg className="h-4 w-4 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Refresh
              </button>
            </div>
          </div>
        </div>
      </form>
      
      {/* Error message */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Loading indicator */}
      {loading ? (
        <div className="flex justify-center items-center p-12">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      ) : (
        <>
          {/* Properties table */}
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('property_id')}
                  >
                    ID
                    {filters.sortBy === 'property_id' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </th>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('address')}
                  >
                    Address
                    {filters.sortBy === 'address' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </th>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('neighborhood')}
                  >
                    Neighborhood
                    {filters.sortBy === 'neighborhood' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </th>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('property_type')}
                  >
                    Type
                    {filters.sortBy === 'property_type' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </th>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('estimated_value')}
                  >
                    Estimated Value
                    {filters.sortBy === 'estimated_value' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </th>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('confidence_score')}
                  >
                    Confidence
                    {filters.sortBy === 'confidence_score' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </th>
                  <th 
                    scope="col" 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSortColumn('valuation_date')}
                  >
                    Valuation Date
                    {filters.sortBy === 'valuation_date' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {properties.length > 0 ? (
                  properties.map((property, index) => (
                    <tr 
                      key={property.property_id || index} 
                      className="hover:bg-gray-50 cursor-pointer"
                      onClick={() => handlePropertyClick(property)}
                    >
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {property.property_id}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {property.address}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {property.neighborhood || 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {property.property_type}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-medium">
                        ${property.estimated_value.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-200 rounded-full h-2.5">
                            <div 
                              className={`h-2.5 rounded-full ${
                                property.confidence_score >= 0.8 ? 'bg-green-500' : 
                                property.confidence_score >= 0.6 ? 'bg-yellow-500' : 
                                'bg-red-500'
                              }`} 
                              style={{ width: `${property.confidence_score * 100}%` }}
                            ></div>
                          </div>
                          <span className="ml-2 text-sm text-gray-500">
                            {(property.confidence_score * 100).toFixed(0)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDate(property.valuation_date)}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="7" className="px-6 py-4 text-center text-sm text-gray-500">
                      No properties found matching your criteria.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          
          {/* Pagination controls */}
          <div className="py-3 flex items-center justify-between border-t border-gray-200 mt-4">
            <div className="flex-1 flex justify-between sm:hidden">
              <button
                onClick={() => handlePageChange(pagination.currentPage - 1)}
                disabled={pagination.currentPage === 1}
                className={`relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md ${
                  pagination.currentPage === 1 ? 'text-gray-400 bg-gray-100' : 'text-gray-700 bg-white hover:bg-gray-50'
                }`}
              >
                Previous
              </button>
              <button
                onClick={() => handlePageChange(pagination.currentPage + 1)}
                disabled={pagination.currentPage === pagination.totalPages}
                className={`ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md ${
                  pagination.currentPage === pagination.totalPages ? 'text-gray-400 bg-gray-100' : 'text-gray-700 bg-white hover:bg-gray-50'
                }`}
              >
                Next
              </button>
            </div>
            <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-gray-700">
                  Showing <span className="font-medium">{properties.length > 0 ? ((pagination.currentPage - 1) * pagination.itemsPerPage) + 1 : 0}</span> to <span className="font-medium">{Math.min(pagination.currentPage * pagination.itemsPerPage, pagination.totalItems)}</span> of{' '}
                  <span className="font-medium">{pagination.totalItems}</span> results
                </p>
              </div>
              <div>
                <div className="flex items-center space-x-4">
                  <span className="text-sm text-gray-700">Rows per page:</span>
                  <select
                    value={pagination.itemsPerPage}
                    onChange={handleItemsPerPageChange}
                    className="rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 text-sm"
                  >
                    {[10, 20, 50, 100].map(size => (
                      <option key={size} value={size}>{size}</option>
                    ))}
                  </select>
                  <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                    <button
                      onClick={() => handlePageChange(1)}
                      disabled={pagination.currentPage === 1}
                      className={`relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium ${
                        pagination.currentPage === 1 ? 'text-gray-400' : 'text-gray-500 hover:bg-gray-50'
                      }`}
                    >
                      <span className="sr-only">First Page</span>
                      <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                      </svg>
                    </button>
                    <button
                      onClick={() => handlePageChange(pagination.currentPage - 1)}
                      disabled={pagination.currentPage === 1}
                      className={`relative inline-flex items-center px-2 py-2 border border-gray-300 bg-white text-sm font-medium ${
                        pagination.currentPage === 1 ? 'text-gray-400' : 'text-gray-500 hover:bg-gray-50'
                      }`}
                    >
                      <span className="sr-only">Previous</span>
                      <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    </button>
                    
                    {/* Page number buttons */}
                    {[...Array(pagination.totalPages).keys()].map(page => {
                      // Only show 5 page numbers centered around current page
                      if (pagination.totalPages <= 7 || 
                          page + 1 === 1 || 
                          page + 1 === pagination.totalPages ||
                          (page + 1 >= pagination.currentPage - 2 && page + 1 <= pagination.currentPage + 2)) {
                        return (
                          <button
                            key={page}
                            onClick={() => handlePageChange(page + 1)}
                            className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                              pagination.currentPage === page + 1
                                ? 'z-10 bg-blue-50 border-blue-500 text-blue-600'
                                : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                            }`}
                          >
                            {page + 1}
                          </button>
                        );
                      } else if (
                        (page + 1 === pagination.currentPage - 3 && pagination.currentPage > 4) ||
                        (page + 1 === pagination.currentPage + 3 && pagination.currentPage < pagination.totalPages - 3)
                      ) {
                        // Show ellipsis
                        return (
                          <span
                            key={page}
                            className="relative inline-flex items-center px-4 py-2 border border-gray-300 bg-white text-sm font-medium text-gray-700"
                          >
                            ...
                          </span>
                        );
                      }
                      
                      return null;
                    })}
                    
                    <button
                      onClick={() => handlePageChange(pagination.currentPage + 1)}
                      disabled={pagination.currentPage === pagination.totalPages}
                      className={`relative inline-flex items-center px-2 py-2 border border-gray-300 bg-white text-sm font-medium ${
                        pagination.currentPage === pagination.totalPages ? 'text-gray-400' : 'text-gray-500 hover:bg-gray-50'
                      }`}
                    >
                      <span className="sr-only">Next</span>
                      <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                      </svg>
                    </button>
                    <button
                      onClick={() => handlePageChange(pagination.totalPages)}
                      disabled={pagination.currentPage === pagination.totalPages}
                      className={`relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium ${
                        pagination.currentPage === pagination.totalPages ? 'text-gray-400' : 'text-gray-500 hover:bg-gray-50'
                      }`}
                    >
                      <span className="sr-only">Last Page</span>
                      <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                      </svg>
                    </button>
                  </nav>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );

  /**
   * Render dashboard summary metrics
   * @returns {JSX.Element} - The dashboard summary component
   */
  const renderSummary = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
      {/* Total properties card */}
      <div className="bg-white overflow-hidden shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0 bg-blue-500 rounded-md p-3">
              <svg className="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
              </svg>
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Total Properties
                </dt>
                <dd className="flex items-baseline">
                  <div className="text-2xl font-semibold text-gray-900">
                    {pagination.totalItems.toLocaleString()}
                  </div>
                </dd>
              </dl>
            </div>
          </div>
        </div>
        <div className="bg-gray-50 px-4 py-4 sm:px-6">
          <div className="text-sm">
            <a href="#" className="font-medium text-blue-600 hover:text-blue-500">
              View all<span className="sr-only"> properties</span>
            </a>
          </div>
        </div>
      </div>
  
      {/* Average property value card */}
      <div className="bg-white overflow-hidden shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0 bg-green-500 rounded-md p-3">
              <svg className="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Average Property Value
                </dt>
                <dd className="flex items-baseline">
                  <div className="text-2xl font-semibold text-gray-900">
                    {properties.length > 0 
                      ? '$' + (properties.reduce((sum, property) => sum + property.estimated_value, 0) / properties.length).toLocaleString(undefined, {maximumFractionDigits: 0})
                      : 'N/A'}
                  </div>
                </dd>
              </dl>
            </div>
          </div>
        </div>
        <div className="bg-gray-50 px-4 py-4 sm:px-6">
          <div className="text-sm">
            <a href="#" className="font-medium text-blue-600 hover:text-blue-500">
              View trends<span className="sr-only"> for property values</span>
            </a>
          </div>
        </div>
      </div>
  
      {/* Active agents card */}
      <div className="bg-white overflow-hidden shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center">
            <div className="flex-shrink-0 bg-purple-500 rounded-md p-3">
              <svg className="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  Active Agents
                </dt>
                <dd className="flex items-baseline">
                  <div className="text-2xl font-semibold text-gray-900">
                    {agentStatus.agents.filter(agent => agent.status === 'active').length} / {agentStatus.agents.length}
                  </div>
                </dd>
              </dl>
            </div>
          </div>
        </div>
        <div className="bg-gray-50 px-4 py-4 sm:px-6">
          <div className="text-sm">
            <a href="#" className="font-medium text-blue-600 hover:text-blue-500" onClick={() => handleTabChange('agents')}>
              View agents<span className="sr-only"> status</span>
            </a>
          </div>
        </div>
      </div>
  
      {/* ETL status card */}
      <div className="bg-white overflow-hidden shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center">
            <div className={`flex-shrink-0 rounded-md p-3 ${
              etlStatus.status === 'processing' ? 'bg-blue-500' : 
              etlStatus.status === 'completed' ? 'bg-green-500' : 
              etlStatus.status === 'error' ? 'bg-red-500' : 
              'bg-gray-500'
            }`}>
              <svg className="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">
                  ETL Pipeline
                </dt>
                <dd className="flex items-baseline">
                  <div className="text-2xl font-semibold text-gray-900 capitalize">
                    {etlStatus.status}
                  </div>
                  <div className="ml-2">
                    <div className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-white bg-blue-500">
                      {etlStatus.progress}%
                    </div>
                  </div>
                </dd>
              </dl>
            </div>
          </div>
        </div>
        <div className="bg-gray-50 px-4 py-4 sm:px-6">
          <div className="text-sm">
            <a href="#" className="font-medium text-blue-600 hover:text-blue-500" onClick={() => handleTabChange('etl')}>
              View ETL status<span className="sr-only"> details</span>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
  
  /**
   * Render the main dashboard interface
   */
  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-primary shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <h1 className="text-xl font-bold text-white">BCBS Values Dashboard</h1>
            <div className="flex items-center space-x-4">
              {/* API Key input */}
              <div className="flex items-center">
                <label htmlFor="api-key" className="sr-only">API Key</label>
                <input
                  type="password"
                  id="api-key"
                  value={apiKey}
                  onChange={(e) => {
                    setApiKey(e.target.value);
                    localStorage.setItem('bcbs_api_key', e.target.value);
                  }}
                  placeholder="API Key"
                  className="rounded-md text-sm px-2 py-1 border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                />
              </div>
              
              {/* Auto-refresh toggle */}
              <div className="flex items-center">
                <span className="text-white text-sm mr-2">Auto-refresh</span>
                <button
                  type="button"
                  onClick={toggleAutoRefresh}
                  className={`relative inline-flex flex-shrink-0 h-6 w-11 border-2 border-transparent rounded-full cursor-pointer transition-colors ease-in-out duration-200 focus:outline-none ${
                    refreshSettings.autoRefresh ? 'bg-green-500' : 'bg-gray-400'
                  }`}
                  role="switch"
                  aria-checked={refreshSettings.autoRefresh}
                >
                  <span
                    className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow transform ring-0 transition ease-in-out duration-200 ${
                      refreshSettings.autoRefresh ? 'translate-x-5' : 'translate-x-0'
                    }`}
                  ></span>
                </button>
              </div>
              
              {/* Refresh interval */}
              {refreshSettings.autoRefresh && (
                <div className="flex items-center">
                  <select
                    value={refreshSettings.interval}
                    onChange={handleRefreshIntervalChange}
                    className="rounded-md text-sm px-2 py-1 border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  >
                    <option value={30000}>30s</option>
                    <option value={60000}>1m</option>
                    <option value={300000}>5m</option>
                    <option value={600000}>10m</option>
                  </select>
                </div>
              )}
              
              {/* Last refreshed indicator */}
              {refreshSettings.lastRefreshTime && (
                <div className="text-white text-xs">
                  Last updated: {formatDateTime(refreshSettings.lastRefreshTime)}
                </div>
              )}
              
              {/* Manual refresh button */}
              <button
                type="button"
                onClick={refreshAllData}
                className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded-md text-white bg-blue-500 hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                <svg className="h-4 w-4 mr-1" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Refresh
              </button>
            </div>
          </div>
        </div>
      </header>
  
      {/* Main content */}
      <main className="py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Dashboard summary metrics */}
          {renderSummary()}
          
          {/* Dashboard tabs */}
          <div className="bg-white shadow rounded-lg mb-6">
            <div className="border-b border-gray-200">
              <nav className="flex -mb-px">
                <button
                  onClick={() => handleTabChange('properties')}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeDashboardTab === 'properties'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <svg className="w-5 h-5 mr-2 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                  </svg>
                  Properties
                </button>
                <button
                  onClick={() => handleTabChange('etl')}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeDashboardTab === 'etl'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <svg className="w-5 h-5 mr-2 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                  </svg>
                  ETL Pipeline
                </button>
                <button
                  onClick={() => handleTabChange('agents')}
                  className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                    activeDashboardTab === 'agents'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <svg className="w-5 h-5 mr-2 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  Agents
                </button>
              </nav>
            </div>
          </div>
          
          {/* Tab content */}
          <div className="mb-8">
            {activeDashboardTab === 'properties' && renderPropertyTable()}
            
            {activeDashboardTab === 'etl' && (
              <div className="bg-white rounded-lg shadow p-4 mt-4">
                <h2 className="text-lg font-medium text-gray-900 mb-4">ETL Pipeline Status</h2>
                
                {etlStatus.isLoading ? (
                  <div className="flex justify-center items-center p-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
                  </div>
                ) : etlStatus.error ? (
                  <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
                    <div className="flex">
                      <div className="flex-shrink-0">
                        <svg className="h-5 w-5 text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <p className="text-sm text-red-700">{etlStatus.error}</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* ETL Progress */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="text-base font-medium text-gray-900">Overall Progress</h3>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full capitalize ${
                          etlStatus.status === 'processing' ? 'bg-blue-100 text-blue-800' : 
                          etlStatus.status === 'completed' ? 'bg-green-100 text-green-800' : 
                          etlStatus.status === 'error' ? 'bg-red-100 text-red-800' : 
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {etlStatus.status}
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-4 mb-2">
                        <div 
                          className={`h-4 rounded-full ${
                            etlStatus.status === 'completed' ? 'bg-green-500' : 
                            etlStatus.status === 'error' ? 'bg-red-500' : 
                            'bg-blue-500'
                          }`} 
                          style={{ width: `${etlStatus.progress}%` }}
                        ></div>
                      </div>
                      <div className="text-right text-sm text-gray-500">
                        {etlStatus.progress}% Complete
                      </div>
                      <div className="mt-2 text-sm text-gray-500">
                        Last updated: {formatDateTime(etlStatus.lastUpdate)}
                      </div>
                    </div>
                    
                    {/* ETL Data Sources */}
                    <div>
                      <h3 className="text-base font-medium text-gray-900 mb-4">Data Sources</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        {etlStatus.sources.map((source, idx) => (
                          <div key={idx} className="bg-white p-4 border rounded-lg">
                            <div className="flex justify-between items-center mb-2">
                              <h4 className="font-medium text-gray-700">{source.name}</h4>
                              <span className={`px-2 py-1 text-xs font-medium rounded-full capitalize ${
                                source.status === 'completed' ? 'bg-green-100 text-green-800' : 
                                source.status === 'processing' ? 'bg-blue-100 text-blue-800' : 
                                source.status === 'queued' ? 'bg-gray-100 text-gray-800' : 
                                'bg-red-100 text-red-800'
                              }`}>
                                {source.status}
                              </span>
                            </div>
                            <div className="text-sm text-gray-500">
                              Records: {source.records.toLocaleString()}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    {/* ETL Metrics */}
                    <div>
                      <h3 className="text-base font-medium text-gray-900 mb-4">Processing Metrics</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        <div className="bg-white p-4 border rounded-lg text-center">
                          <div className="text-2xl font-bold text-gray-900">
                            {etlStatus.metrics.recordsProcessed.toLocaleString()}
                          </div>
                          <div className="text-sm text-gray-500">
                            Records Processed
                          </div>
                        </div>
                        <div className="bg-white p-4 border rounded-lg text-center">
                          <div className="text-2xl font-bold text-gray-900">
                            {formatPercentage(etlStatus.metrics.successRate)}
                          </div>
                          <div className="text-sm text-gray-500">
                            Success Rate
                          </div>
                        </div>
                        <div className="bg-white p-4 border rounded-lg text-center">
                          <div className="text-2xl font-bold text-gray-900">
                            {etlStatus.metrics.averageProcessingTime.toFixed(2)}s
                          </div>
                          <div className="text-sm text-gray-500">
                            Avg. Processing Time
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Data Quality Metrics */}
                    <div>
                      <h3 className="text-base font-medium text-gray-900 mb-4">Data Quality Metrics</h3>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        <div className="bg-white p-4 border rounded-lg">
                          <div className="mb-2 flex justify-between items-center">
                            <span className="text-sm font-medium text-gray-700">Completeness</span>
                            <span className="text-sm font-medium text-gray-900">
                              {formatPercentage(etlStatus.dataQuality.completeness)}
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="h-2 rounded-full bg-blue-500" 
                              style={{ width: `${etlStatus.dataQuality.completeness * 100}%` }}
                            ></div>
                          </div>
                        </div>
                        <div className="bg-white p-4 border rounded-lg">
                          <div className="mb-2 flex justify-between items-center">
                            <span className="text-sm font-medium text-gray-700">Accuracy</span>
                            <span className="text-sm font-medium text-gray-900">
                              {formatPercentage(etlStatus.dataQuality.accuracy)}
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="h-2 rounded-full bg-green-500" 
                              style={{ width: `${etlStatus.dataQuality.accuracy * 100}%` }}
                            ></div>
                          </div>
                        </div>
                        <div className="bg-white p-4 border rounded-lg">
                          <div className="mb-2 flex justify-between items-center">
                            <span className="text-sm font-medium text-gray-700">Timeliness</span>
                            <span className="text-sm font-medium text-gray-900">
                              {formatPercentage(etlStatus.dataQuality.timeliness)}
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="h-2 rounded-full bg-purple-500" 
                              style={{ width: `${etlStatus.dataQuality.timeliness * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {activeDashboardTab === 'agents' && (
              <div className="bg-white rounded-lg shadow p-4 mt-4">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Valuation Agents Status</h2>
                
                {agentStatus.isLoading ? (
                  <div className="flex justify-center items-center p-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
                  </div>
                ) : agentStatus.error ? (
                  <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
                    <div className="flex">
                      <div className="flex-shrink-0">
                        <svg className="h-5 w-5 text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <p className="text-sm text-red-700">{agentStatus.error}</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Agent ID
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Type
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Status
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Last Active
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Queue
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Success Rate
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Processed
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {agentStatus.agents.length > 0 ? (
                          agentStatus.agents.map((agent, index) => (
                            <tr 
                              key={agent.agent_id || index} 
                              className="hover:bg-gray-50 cursor-pointer"
                              onClick={() => handleAgentClick(agent)}
                            >
                              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                {agent.agent_id}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 capitalize">
                                {agent.agent_type}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <span className={`px-2 py-1 text-xs font-medium rounded-full capitalize ${
                                  agent.status === 'active' ? 'bg-green-100 text-green-800' : 
                                  agent.status === 'idle' ? 'bg-blue-100 text-blue-800' : 
                                  agent.status === 'error' ? 'bg-red-100 text-red-800' : 
                                  'bg-gray-100 text-gray-800'
                                }`}>
                                  {agent.status}
                                </span>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {formatDateTime(agent.last_active)}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                <div className="flex items-center">
                                  <div className="w-16 bg-gray-200 rounded-full h-2.5">
                                    <div 
                                      className={`h-2.5 rounded-full ${
                                        agent.queue_size < 10 ? 'bg-green-500' : 
                                        agent.queue_size < 30 ? 'bg-yellow-500' : 
                                        'bg-red-500'
                                      }`} 
                                      style={{ width: `${Math.min(agent.queue_size / 50 * 100, 100)}%` }}
                                    ></div>
                                  </div>
                                  <span className="ml-2 text-sm text-gray-500">
                                    {agent.queue_size}
                                  </span>
                                </div>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <div className="flex items-center">
                                  <div className="w-16 bg-gray-200 rounded-full h-2.5">
                                    <div 
                                      className={`h-2.5 rounded-full ${
                                        agent.metrics?.success_rate >= 0.9 ? 'bg-green-500' : 
                                        agent.metrics?.success_rate >= 0.8 ? 'bg-yellow-500' : 
                                        'bg-red-500'
                                      }`} 
                                      style={{ width: `${(agent.metrics?.success_rate || 0) * 100}%` }}
                                    ></div>
                                  </div>
                                  <span className="ml-2 text-sm text-gray-500">
                                    {formatPercentage(agent.metrics?.success_rate || 0)}
                                  </span>
                                </div>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                {agent.metrics?.total_processed.toLocaleString()}
                              </td>
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan="7" className="px-6 py-4 text-center text-sm text-gray-500">
                              No agents found.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
  
      {/* Footer */}
      <footer className="bg-white">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
          <p className="text-center text-gray-500 text-sm">
            &copy; 2025 BCBS Values - Advanced Property Valuation Platform
          </p>
        </div>
      </footer>
  
      {/* Property Detail Modal */}
      {selectedProperty && (
        <div className="fixed z-10 inset-0 overflow-y-auto" aria-labelledby="property-detail-modal" role="dialog" aria-modal="true">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" aria-hidden="true" onClick={closePropertyDetail}></div>
  
            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
  
            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-3xl sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left w-full">
                    <h3 className="text-lg leading-6 font-medium text-gray-900" id="property-detail-title">
                      Property Details
                    </h3>
                    <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div>
                        <div className="bg-gray-50 p-4 rounded-lg">
                          <h4 className="text-base font-medium text-gray-900 mb-2">General Information</h4>
                          <dl className="grid grid-cols-1 gap-2">
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Property ID:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{selectedProperty.property_id}</dd>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Address:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{selectedProperty.address}</dd>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Neighborhood:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{selectedProperty.neighborhood || 'N/A'}</dd>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Property Type:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{selectedProperty.property_type}</dd>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Year Built:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{selectedProperty.year_built || 'N/A'}</dd>
                            </div>
                          </dl>
                        </div>
  
                        <div className="bg-gray-50 p-4 rounded-lg mt-4">
                          <h4 className="text-base font-medium text-gray-900 mb-2">Property Specifications</h4>
                          <dl className="grid grid-cols-1 gap-2">
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Land Area:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{selectedProperty.land_area ? `${selectedProperty.land_area.toLocaleString()} sq ft` : 'N/A'}</dd>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Living Area:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{selectedProperty.living_area ? `${selectedProperty.living_area.toLocaleString()} sq ft` : 'N/A'}</dd>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Bedrooms:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{selectedProperty.bedrooms || 'N/A'}</dd>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Bathrooms:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{selectedProperty.bathrooms || 'N/A'}</dd>
                            </div>
                          </dl>
                        </div>
                      </div>
  
                      <div>
                        <div className="bg-gray-50 p-4 rounded-lg">
                          <h4 className="text-base font-medium text-gray-900 mb-2">Valuation Information</h4>
                          <dl className="grid grid-cols-1 gap-2">
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Estimated Value:</dt>
                              <dd className="text-sm text-gray-900 font-bold col-span-2">${selectedProperty.estimated_value?.toLocaleString()}</dd>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Confidence Score:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">
                                <div className="flex items-center">
                                  <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                                    <div 
                                      className={`h-2.5 rounded-full ${
                                        selectedProperty.confidence_score >= 0.8 ? 'bg-green-500' : 
                                        selectedProperty.confidence_score >= 0.6 ? 'bg-yellow-500' : 
                                        'bg-red-500'
                                      }`} 
                                      style={{ width: `${selectedProperty.confidence_score * 100}%` }}
                                    ></div>
                                  </div>
                                  <span>{(selectedProperty.confidence_score * 100).toFixed(0)}%</span>
                                </div>
                              </dd>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Valuation Date:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{formatDateTime(selectedProperty.valuation_date)}</dd>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Valuation Method:</dt>
                              <dd className="text-sm text-gray-900 col-span-2 capitalize">{selectedProperty.valuation_method || 'N/A'}</dd>
                            </div>
                          </dl>
                        </div>
  
                        <div className="bg-gray-50 p-4 rounded-lg mt-4">
                          <h4 className="text-base font-medium text-gray-900 mb-2">Sales History</h4>
                          <dl className="grid grid-cols-1 gap-2">
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Last Sale Date:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{formatDate(selectedProperty.last_sale_date) || 'N/A'}</dd>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              <dt className="text-sm font-medium text-gray-500">Last Sale Price:</dt>
                              <dd className="text-sm text-gray-900 col-span-2">{selectedProperty.last_sale_price ? `$${selectedProperty.last_sale_price.toLocaleString()}` : 'N/A'}</dd>
                            </div>
                            {selectedProperty.last_sale_price && selectedProperty.estimated_value && (
                              <div className="grid grid-cols-3 gap-2">
                                <dt className="text-sm font-medium text-gray-500">Value Change:</dt>
                                <dd className="text-sm text-gray-900 col-span-2">
                                  <span className={selectedProperty.estimated_value > selectedProperty.last_sale_price ? 'text-green-600' : 'text-red-600'}>
                                    {selectedProperty.estimated_value > selectedProperty.last_sale_price ? '+' : ''}
                                    {(((selectedProperty.estimated_value - selectedProperty.last_sale_price) / selectedProperty.last_sale_price) * 100).toFixed(1)}%
                                  </span>
                                </dd>
                              </div>
                            )}
                          </dl>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <button 
                  type="button" 
                  className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-blue-600 text-base font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:ml-3 sm:w-auto sm:text-sm"
                  onClick={closePropertyDetail}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
  
      {/* Agent Detail Modal */}
      {selectedAgent && (
        <div className="fixed z-10 inset-0 overflow-y-auto" aria-labelledby="agent-detail-modal" role="dialog" aria-modal="true">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" aria-hidden="true" onClick={closeAgentDetail}></div>
  
            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>
  
            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-3xl sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left w-full">
                    <h3 className="text-lg leading-6 font-medium text-gray-900" id="agent-detail-title">
                      Agent Details
                    </h3>
                    <div className="mt-4">
                      <div className="bg-gray-50 p-4 rounded-lg mb-4">
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                          <div>
                            <span className="text-sm text-gray-500">Agent ID</span>
                            <p className="font-medium">{selectedAgent.agent_id}</p>
                          </div>
                          <div>
                            <span className="text-sm text-gray-500">Type</span>
                            <p className="font-medium capitalize">{selectedAgent.agent_type}</p>
                          </div>
                          <div>
                            <span className="text-sm text-gray-500">Status</span>
                            <p className={`font-medium capitalize ${getStatusColorClass(selectedAgent.status)}`}>
                              {selectedAgent.status}
                            </p>
                          </div>
                          <div>
                            <span className="text-sm text-gray-500">Last Active</span>
                            <p className="font-medium">{formatDateTime(selectedAgent.last_active)}</p>
                          </div>
                          <div>
                            <span className="text-sm text-gray-500">Queue Size</span>
                            <p className="font-medium">{selectedAgent.queue_size} items</p>
                          </div>
                          <div>
                            <span className="text-sm text-gray-500">Success Rate</span>
                            <p className="font-medium">{formatPercentage(selectedAgent.metrics?.success_rate || 0)}</p>
                          </div>
                          <div>
                            <span className="text-sm text-gray-500">Properties Processed</span>
                            <p className="font-medium">{selectedAgent.metrics?.total_processed.toLocaleString()}</p>
                          </div>
                          <div>
                            <span className="text-sm text-gray-500">Avg. Processing Time</span>
                            <p className="font-medium">{selectedAgent.metrics?.average_processing_time.toFixed(2)}s</p>
                          </div>
                          {selectedAgent.error && (
                            <div className="sm:col-span-2 md:col-span-3">
                              <span className="text-sm text-gray-500">Error</span>
                              <p className="font-medium text-red-600">{selectedAgent.error}</p>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      <h4 className="text-base font-medium mb-2">Agent Logs</h4>
                      {selectedAgent.logsError ? (
                        <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
                          <div className="flex">
                            <div className="flex-shrink-0">
                              <svg className="h-5 w-5 text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                              </svg>
                            </div>
                            <div className="ml-3">
                              <p className="text-sm text-red-700">{selectedAgent.logsError}</p>
                            </div>
                          </div>
                        </div>
                      ) : Array.isArray(selectedAgent.logs) && selectedAgent.logs.length > 0 ? (
                        <div className="bg-gray-50 rounded-lg p-1 h-64 overflow-y-auto">
                          <table className="min-w-full">
                            <thead className="bg-gray-100">
                              <tr>
                                <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sticky top-0 bg-gray-100">
                                  Timestamp
                                </th>
                                <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sticky top-0 bg-gray-100">
                                  Level
                                </th>
                                <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sticky top-0 bg-gray-100">
                                  Message
                                </th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-200">
                              {selectedAgent.logs.map((log, idx) => (
                                <tr key={idx}>
                                  <td className="px-3 py-2 whitespace-nowrap text-xs text-gray-500">
                                    {formatDateTime(log.timestamp)}
                                  </td>
                                  <td className="px-3 py-2 whitespace-nowrap text-xs">
                                    <span className={`px-2 py-1 text-xs font-medium rounded-full capitalize ${
                                      log.level === 'info' ? 'bg-blue-100 text-blue-800' : 
                                      log.level === 'warning' ? 'bg-yellow-100 text-yellow-800' : 
                                      log.level === 'error' ? 'bg-red-100 text-red-800' : 
                                      'bg-gray-100 text-gray-800'
                                    }`}>
                                      {log.level}
                                    </span>
                                  </td>
                                  <td className="px-3 py-2 text-xs text-gray-500">
                                    {log.message}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <div className="bg-gray-50 p-4 text-center rounded-lg">
                          <p className="text-gray-500">No logs available for this agent.</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <button 
                  type="button" 
                  className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-blue-600 text-base font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:ml-3 sm:w-auto sm:text-sm"
                  onClick={closeAgentDetail}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}