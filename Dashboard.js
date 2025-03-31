import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, ArcElement, TimeScale, Filler, RadialLinearScale, DoughnutController } from 'chart.js';
import { Line, Bar, Pie, Doughnut } from 'react-chartjs-2';
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
  Legend,
  RadialLinearScale,
  DoughnutController
);

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
const Dashboard = () => {
  // --- State Management ---
  
  // Property data state
  const [properties, setProperties] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Filter state
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
  
  // UI state
  const [activeTab, setActiveTab] = useState('distribution');
  const [selectedProperty, setSelectedProperty] = useState(null);
  const [availableNeighborhoods, setAvailableNeighborhoods] = useState([]);
  const [availablePropertyTypes, setAvailablePropertyTypes] = useState([
    'single_family', 'condo', 'townhouse', 'multi_family', 'land'
  ]);
  
  // Pagination state
  const [pagination, setPagination] = useState({
    currentPage: 1,
    itemsPerPage: 10,
    totalItems: 0,
    totalPages: 1
  });
  
  // Auth state
  const [apiKey, setApiKey] = useState('');
  
  // ETL pipeline status state
  const [etlStatus, setEtlStatus] = useState({
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
    isLoading: false,
    error: null
  });
  
  // Agent status state
  const [agentStatus, setAgentStatus] = useState({
    agents: [],
    isLoading: false,
    error: null,
    lastUpdate: null
  });
  
  // Selected agent for detailed view
  const [selectedAgent, setSelectedAgent] = useState(null);
  
  // Refresh settings
  const [refreshSettings, setRefreshSettings] = useState({
    autoRefresh: true,
    interval: 60000, // 1 minute default
    lastRefreshTime: null
  });
  
  // Dashboard tab navigation
  const [activeDashboardTab, setActiveDashboardTab] = useState('properties');
  
  // --- Chart References ---
  const valueDistributionChartRef = useRef(null);
  const neighborhoodChartRef = useRef(null);
  const trendChartRef = useRef(null);
  const etlProgressChartRef = useRef(null);
  const agentPerformanceChartRef = useRef(null);
  const metricsObserverRef = useRef(null); // For animation triggering with Intersection Observer
  const refreshTimerRef = useRef(null); // For tracking auto-refresh interval
  
  /**
   * Initialization effect - Fetch all data on component mount
   * This effect runs only once when the component is first loaded
   */
  useEffect(() => {
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
  }, [fetchPropertyValuations, fetchEtlStatus, fetchAgentStatus]);

  /**
   * Function to fetch property valuations from the enhanced API endpoint
   * Includes pagination, sorting, and filtering capabilities
   * @returns {Promise<void>}
   */
  const fetchPropertyValuations = useCallback(async () => {
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
  }, [filters, pagination.currentPage, pagination.itemsPerPage, apiKey]);

  /**
   * Extract unique neighborhoods from the property data
   * @param {Array} propertyData - Array of property objects
   */
  const extractUniqueNeighborhoods = useCallback((propertyData) => {
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
  const debouncedSearch = useCallback(
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
   * @returns {Promise<void>}
   */
  const fetchEtlStatus = useCallback(async () => {
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
   * @returns {Promise<void>}
   */
  const fetchAgentStatus = useCallback(async () => {
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
   * @returns {Promise<void>}
   */
  const fetchAgentDetails = useCallback(async (agentId) => {
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
   * This will fetch property valuations, ETL status, and agent status
   */
  const refreshAllData = useCallback(() => {
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
   * Handle selecting an agent for detailed view
   * @param {Object} agent - The agent object to view in detail
   */
  const handleAgentSelect = (agent) => {
    setSelectedAgent(agent);
    
    // Fetch detailed information for this agent
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
  const handleDashboardTabChange = (tabName) => {
    setActiveDashboardTab(tabName);
  };

  // --- Utility functions ---
  
  /**
   * Format currency values for display
   * @param {number} value - The numeric value to format as currency
   * @returns {string} - Formatted currency string
   */
  const formatCurrency = (value) => {
    if (value === null || value === undefined) return 'N/A';
    
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
    if (value === null || value === undefined) return 'N/A';
    
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
   * Format date values for display with time included
   * @param {string|Date} dateTimeValue - Date object or ISO date string
   * @returns {string} - Formatted date and time string
   */
  const formatDateTime = (dateTimeValue) => {
    if (!dateTimeValue) return 'N/A';
    
    try {
      const date = dateTimeValue instanceof Date ? dateTimeValue : new Date(dateTimeValue);
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch (e) {
      return 'Invalid Date';
    }
  };

  /**
   * Helper function to get status color class based on status string
   * @param {string} status - Status string
   * @returns {string} - Tailwind CSS color class
   */
  const getStatusColorClass = (status) => {
    if (!status) return 'bg-gray-500';
    
    const statusLower = status.toLowerCase();
    
    if (statusLower === 'active' || statusLower === 'online' || statusLower === 'completed' || statusLower === 'success') {
      return 'bg-green-500';
    } else if (statusLower === 'idle' || statusLower === 'pending' || statusLower === 'waiting') {
      return 'bg-blue-500';
    } else if (statusLower === 'warning' || statusLower === 'busy') {
      return 'bg-yellow-500';
    } else if (statusLower === 'error' || statusLower === 'failed' || statusLower === 'offline') {
      return 'bg-red-500';
    }
    
    return 'bg-gray-500';
  };

  // --- Initialize charts and load data on component mount ---
  useEffect(() => {
    // Set up intersection observer for animations
    const observerOptions = {
      root: null,
      rootMargin: '0px',
      threshold: 0.1,
    };
    
    const handleIntersection = (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          // Add animation class when element is visible
          entry.target.classList.add('animate-fade-in-up');
        }
      });
    };
    
    metricsObserverRef.current = new IntersectionObserver(handleIntersection, observerOptions);
    
    // Observe dashboard metric elements
    document.querySelectorAll('.metric-card').forEach(card => {
      metricsObserverRef.current.observe(card);
    });
    
    // Initial data loading
    refreshAllData();
    
    // Set up auto-refresh interval
    if (refreshSettings.autoRefresh) {
      refreshTimerRef.current = setInterval(() => {
        refreshAllData();
      }, refreshSettings.interval);
    }
    
    // Cleanup function
    return () => {
      // Disconnect intersection observer
      if (metricsObserverRef.current) {
        metricsObserverRef.current.disconnect();
      }
      
      // Clear refresh interval
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [refreshSettings.autoRefresh, refreshSettings.interval, refreshAllData]);

  // Update refresh timer when settings change
  useEffect(() => {
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
  
  // --- Filter properties client-side (as a backup to API filtering) ---
  const filteredProperties = useMemo(() => {
    return properties.filter(property => {
      // Skip client-side filtering if the searchQuery is empty (API handles all filtering)
      if (!filters.searchQuery) return true;
      
      // Apply search query filter on address, property_id, or neighborhood
      const searchLower = filters.searchQuery.toLowerCase();
      
      return (
        (property.address && property.address.toLowerCase().includes(searchLower)) ||
        (property.property_id && property.property_id.toLowerCase().includes(searchLower)) ||
        (property.neighborhood && property.neighborhood.toLowerCase().includes(searchLower))
      );
    });
  }, [properties, filters.searchQuery]);
  
  // --- Prepare chart data ---
  
  /**
   * Calculate summary metrics for the dashboard based on filtered properties
   */
  const summaryMetrics = useMemo(() => {
    if (!properties.length) {
      return {
        totalProperties: 0,
        averageValue: 0,
        minValue: 0,
        maxValue: 0,
        medianValue: 0
      };
    }
    
    const propertyValues = properties
      .map(p => p.estimated_value)
      .filter(v => v !== undefined && v !== null)
      .sort((a, b) => a - b);
    
    const medianIndex = Math.floor(propertyValues.length / 2);
    
    return {
      totalProperties: properties.length,
      averageValue: propertyValues.length ? 
        propertyValues.reduce((sum, value) => sum + value, 0) / propertyValues.length : 
        0,
      minValue: propertyValues.length ? propertyValues[0] : 0,
      maxValue: propertyValues.length ? propertyValues[propertyValues.length - 1] : 0,
      medianValue: propertyValues.length ? 
        (propertyValues.length % 2 === 0 ? 
          (propertyValues[medianIndex - 1] + propertyValues[medianIndex]) / 2 : 
          propertyValues[medianIndex]) : 
        0
    };
  }, [properties]);
  
  /**
   * Prepare value distribution chart data
   */
  const valueDistributionData = useMemo(() => {
    if (!properties.length) return { labels: [], datasets: [] };
    
    // Define value ranges
    const ranges = [
      { min: 0, max: 100000, label: 'Under $100K' },
      { min: 100000, max: 200000, label: '$100K-$200K' },
      { min: 200000, max: 300000, label: '$200K-$300K' },
      { min: 300000, max: 400000, label: '$300K-$400K' },
      { min: 400000, max: 500000, label: '$400K-$500K' },
      { min: 500000, max: 750000, label: '$500K-$750K' },
      { min: 750000, max: 1000000, label: '$750K-$1M' },
      { min: 1000000, max: Infinity, label: 'Over $1M' }
    ];
    
    // Count properties in each range
    const rangeCounts = ranges.map(range => {
      return properties.filter(p => 
        p.estimated_value >= range.min && p.estimated_value < range.max
      ).length;
    });
    
    return {
      labels: ranges.map(r => r.label),
      datasets: [{
        label: 'Properties',
        data: rangeCounts,
        backgroundColor: 'rgba(52, 152, 219, 0.8)',
        borderColor: 'rgba(52, 152, 219, 1)',
        borderWidth: 1
      }]
    };
  }, [properties]);
  
  /**
   * Prepare neighborhood comparison chart data
   */
  const neighborhoodComparisonData = useMemo(() => {
    if (!properties.length) return { labels: [], datasets: [] };
    
    // Group properties by neighborhood and calculate average values
    const neighborhoodGroups = {};
    
    properties.forEach(property => {
      const neighborhood = property.neighborhood || 'Unknown';
      
      if (!neighborhoodGroups[neighborhood]) {
        neighborhoodGroups[neighborhood] = {
          count: 0,
          totalValue: 0
        };
      }
      
      neighborhoodGroups[neighborhood].count++;
      neighborhoodGroups[neighborhood].totalValue += property.estimated_value || 0;
    });
    
    // Calculate averages and sort by value (descending)
    const neighborhoods = Object.keys(neighborhoodGroups)
      .map(name => ({
        name,
        avgValue: neighborhoodGroups[name].totalValue / neighborhoodGroups[name].count,
        count: neighborhoodGroups[name].count
      }))
      .sort((a, b) => b.avgValue - a.avgValue)
      .slice(0, 10); // Top 10 neighborhoods
    
    return {
      labels: neighborhoods.map(n => n.name),
      datasets: [{
        label: 'Average Value',
        data: neighborhoods.map(n => n.avgValue),
        backgroundColor: 'rgba(46, 204, 113, 0.8)',
        borderColor: 'rgba(46, 204, 113, 1)',
        borderWidth: 1
      }]
    };
  }, [properties]);
  
  /**
   * Prepare value trend chart data (by valuation date)
   */
  const valueTrendData = useMemo(() => {
    if (!properties.length) return { labels: [], datasets: [] };
    
    // Group properties by month
    const monthlyData = {};
    
    properties.forEach(property => {
      if (!property.valuation_date) return;
      
      const date = new Date(property.valuation_date);
      const monthKey = `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}`;
      
      if (!monthlyData[monthKey]) {
        monthlyData[monthKey] = {
          count: 0,
          totalValue: 0,
          date: new Date(date.getFullYear(), date.getMonth(), 1)
        };
      }
      
      monthlyData[monthKey].count++;
      monthlyData[monthKey].totalValue += property.estimated_value || 0;
    });
    
    // Calculate averages and sort by date
    const monthlyAverages = Object.values(monthlyData)
      .map(month => ({
        date: month.date,
        avgValue: month.totalValue / month.count
      }))
      .sort((a, b) => a.date - b.date);
    
    return {
      labels: monthlyAverages.map(m => m.date),
      datasets: [{
        label: 'Average Value',
        data: monthlyAverages.map(m => ({ x: m.date, y: m.avgValue })),
        backgroundColor: 'rgba(52, 152, 219, 0.2)',
        borderColor: 'rgba(52, 152, 219, 1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4
      }]
    };
  }, [properties]);
  
  /**
   * Prepare model distribution pie chart data
   */
  const modelDistributionData = useMemo(() => {
    if (!properties.length) return { labels: [], datasets: [] };
    
    // Count properties by valuation method
    const methodCounts = {};
    
    properties.forEach(property => {
      const method = property.valuation_method || 'unknown';
      methodCounts[method] = (methodCounts[method] || 0) + 1;
    });
    
    // Prepare chart data
    return {
      labels: Object.keys(methodCounts),
      datasets: [{
        label: 'Valuation Methods',
        data: Object.values(methodCounts),
        backgroundColor: [
          'rgba(52, 152, 219, 0.8)', // Blue
          'rgba(46, 204, 113, 0.8)', // Green
          'rgba(155, 89, 182, 0.8)', // Purple
          'rgba(52, 73, 94, 0.8)',   // Dark Blue
          'rgba(22, 160, 133, 0.8)', // Teal
          'rgba(39, 174, 96, 0.8)',  // Emerald
          'rgba(41, 128, 185, 0.8)'  // Light Blue
        ],
        borderColor: [
          'rgba(52, 152, 219, 1)',
          'rgba(46, 204, 113, 1)',
          'rgba(155, 89, 182, 1)',
          'rgba(52, 73, 94, 1)',
          'rgba(22, 160, 133, 1)',
          'rgba(39, 174, 96, 1)',
          'rgba(41, 128, 185, 1)'
        ],
        borderWidth: 1
      }]
    };
  }, [properties]);
  
  /**
   * Prepare agent status chart data
   */
  const agentStatusData = useMemo(() => {
    if (!agentStatus.agents.length) return { labels: [], datasets: [] };
    
    // Count agents by status
    const statusCounts = {};
    
    agentStatus.agents.forEach(agent => {
      const status = agent.status || 'unknown';
      statusCounts[status] = (statusCounts[status] || 0) + 1;
    });
    
    // Prepare chart data
    return {
      labels: Object.keys(statusCounts),
      datasets: [{
        label: 'Agent Status',
        data: Object.values(statusCounts),
        backgroundColor: [
          'rgba(46, 204, 113, 0.8)',  // Green (idle/active)
          'rgba(52, 152, 219, 0.8)',  // Blue (waiting/busy)
          'rgba(243, 156, 18, 0.8)',  // Yellow (warning)
          'rgba(231, 76, 60, 0.8)',   // Red (error/offline)
          'rgba(149, 165, 166, 0.8)'  // Gray (unknown)
        ],
        borderColor: [
          'rgba(46, 204, 113, 1)',
          'rgba(52, 152, 219, 1)',
          'rgba(243, 156, 18, 1)',
          'rgba(231, 76, 60, 1)',
          'rgba(149, 165, 166, 1)'
        ],
        borderWidth: 1
      }]
    };
  }, [agentStatus.agents]);
  
  // --- Chart Options ---
  
  /**
   * Options for Value Distribution chart
   */
  const valueDistributionOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            return `${context.dataset.label}: ${context.raw} (${((context.raw / properties.length) * 100).toFixed(1)}%)`;
          }
        }
      },
      title: {
        display: true,
        text: 'Property Value Distribution',
        font: {
          size: 16
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
      },
      x: {
        title: {
          display: true,
          text: 'Value Range'
        }
      }
    }
  };
  
  /**
   * Options for Neighborhood Comparison chart
   */
  const neighborhoodOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            return `${context.dataset.label}: ${formatCurrency(context.raw)}`;
          }
        }
      },
      title: {
        display: true,
        text: 'Average Value by Neighborhood',
        font: {
          size: 16
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Average Value'
        },
        ticks: {
          callback: (value) => {
            return formatCurrency(value);
          }
        }
      },
      x: {
        title: {
          display: true,
          text: 'Neighborhood'
        }
      }
    }
  };
  
  /**
   * Options for Value Trend chart
   */
  const trendOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            return `${context.dataset.label}: ${formatCurrency(context.raw.y)}`;
          },
          title: (tooltipItems) => {
            const date = new Date(tooltipItems[0].raw.x);
            return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long' });
          }
        }
      },
      title: {
        display: true,
        text: 'Value Trends Over Time',
        font: {
          size: 16
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Average Value'
        },
        ticks: {
          callback: (value) => {
            return formatCurrency(value);
          }
        }
      },
      x: {
        type: 'time',
        time: {
          unit: 'month',
          tooltipFormat: 'MMM yyyy',
          displayFormats: {
            month: 'MMM yyyy'
          }
        },
        title: {
          display: true,
          text: 'Month'
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
        labels: {
          boxWidth: 15,
          padding: 15
        }
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const value = context.raw;
            const percentage = ((value / properties.length) * 100).toFixed(1);
            return `${context.label}: ${value} (${percentage}%)`;
          }
        }
      },
      title: {
        display: true,
        text: 'Valuation Methods Used',
        font: {
          size: 16
        }
      }
    }
  };
  
  /**
   * Options for Agent Status doughnut chart
   */
  const agentStatusOptions = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: '50%',
    plugins: {
      legend: {
        position: 'right',
        labels: {
          boxWidth: 15,
          padding: 15
        }
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const value = context.raw;
            const percentage = ((value / agentStatus.agents.length) * 100).toFixed(1);
            return `${context.label}: ${value} (${percentage}%)`;
          }
        }
      },
      title: {
        display: true,
        text: 'Agent Status Distribution',
        font: {
          size: 16
        }
      }
    }
  };
  
  /**
   * Options for ETL Progress gauge chart
   */
  const etlProgressOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      tooltip: {
        callbacks: {
          label: (context) => {
            return `Progress: ${(context.raw * 100).toFixed(1)}%`;
          }
        }
      }
    },
    scales: {
      r: {
        angleLines: {
          display: false
        },
        suggestedMin: 0,
        suggestedMax: 1
      }
    }
  };
  
  // --- Render UI Components ---
  
  /**
   * Render the filter section
   * @returns {JSX.Element} - The filter section component
   */
  const renderFilterSection = () => (
    <div className="bg-white rounded-lg shadow-md p-4 mb-6">
      <h2 className="text-xl font-semibold mb-4">Filters</h2>
      <form onSubmit={applyFilters} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Neighborhood filter */}
        <div>
          <label htmlFor="neighborhood" className="block text-sm font-medium text-gray-700 mb-1">
            Neighborhood
          </label>
          <select
            id="neighborhood"
            name="neighborhood"
            value={filters.neighborhood}
            onChange={handleFilterChange}
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">All Neighborhoods</option>
            {availableNeighborhoods.map((neighborhood, index) => (
              <option key={index} value={neighborhood}>{neighborhood}</option>
            ))}
          </select>
        </div>
        
        {/* Property Type filter */}
        <div>
          <label htmlFor="propertyType" className="block text-sm font-medium text-gray-700 mb-1">
            Property Type
          </label>
          <select
            id="propertyType"
            name="propertyType"
            value={filters.propertyType}
            onChange={handleFilterChange}
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">All Property Types</option>
            {availablePropertyTypes.map((type, index) => (
              <option key={index} value={type}>
                {type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </option>
            ))}
          </select>
        </div>
        
        {/* Price Range filters */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label htmlFor="minValue" className="block text-sm font-medium text-gray-700 mb-1">
              Min Value
            </label>
            <input
              type="number"
              id="minValue"
              name="minValue"
              value={filters.minValue}
              onChange={handleFilterChange}
              placeholder="Min $"
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label htmlFor="maxValue" className="block text-sm font-medium text-gray-700 mb-1">
              Max Value
            </label>
            <input
              type="number"
              id="maxValue"
              name="maxValue"
              value={filters.maxValue}
              onChange={handleFilterChange}
              placeholder="Max $"
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
        
        {/* Date Range filters */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label htmlFor="fromDate" className="block text-sm font-medium text-gray-700 mb-1">
              From Date
            </label>
            <input
              type="date"
              id="fromDate"
              name="fromDate"
              value={filters.fromDate}
              onChange={handleFilterChange}
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label htmlFor="toDate" className="block text-sm font-medium text-gray-700 mb-1">
              To Date
            </label>
            <input
              type="date"
              id="toDate"
              name="toDate"
              value={filters.toDate}
              onChange={handleFilterChange}
              className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
        
        {/* Search filter */}
        <div className="md:col-span-2 lg:col-span-1">
          <label htmlFor="searchQuery" className="block text-sm font-medium text-gray-700 mb-1">
            Search
          </label>
          <div className="relative">
            <input
              type="text"
              id="searchQuery"
              name="searchQuery"
              value={filters.searchQuery}
              onChange={handleFilterChange}
              placeholder="Search address, ID, etc."
              className="w-full border border-gray-300 rounded-md pl-10 pr-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <i className="fas fa-search text-gray-400"></i>
            </div>
          </div>
        </div>
        
        {/* Action buttons */}
        <div className="md:col-span-2 lg:col-span-3 flex justify-end gap-2 mt-2">
          <button
            type="button"
            onClick={resetFilters}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500"
          >
            Reset Filters
          </button>
          <button
            type="submit"
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Apply Filters
          </button>
        </div>
      </form>
    </div>
  );
  
  /**
   * Render the property table
   * @returns {JSX.Element} - The property table component
   */
  const renderPropertyTable = () => (
    <div className="bg-white rounded-lg shadow-md overflow-hidden mb-6">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th 
                scope="col" 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
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
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
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
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
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
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
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
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSortColumn('valuation_date')}
              >
                Valuation Date
                {filters.sortBy === 'valuation_date' && (
                  <span className="ml-1">
                    {filters.sortDirection === 'asc' ? '↑' : '↓'}
                  </span>
                )}
              </th>
              <th 
                scope="col" 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSortColumn('valuation_method')}
              >
                Method
                {filters.sortBy === 'valuation_method' && (
                  <span className="ml-1">
                    {filters.sortDirection === 'asc' ? '↑' : '↓'}
                  </span>
                )}
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {loading ? (
              <tr>
                <td colSpan="6" className="px-6 py-4 text-center">
                  <div className="flex justify-center items-center">
                    <svg className="animate-spin h-5 w-5 mr-3 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Loading properties...
                  </div>
                </td>
              </tr>
            ) : error ? (
              <tr>
                <td colSpan="6" className="px-6 py-4 text-center text-red-500">
                  {error}
                </td>
              </tr>
            ) : filteredProperties.length === 0 ? (
              <tr>
                <td colSpan="6" className="px-6 py-4 text-center">
                  No properties found with the current filters. Try adjusting your criteria.
                </td>
              </tr>
            ) : (
              filteredProperties.map((property, index) => (
                <tr 
                  key={property.property_id || index} 
                  className="hover:bg-gray-50 cursor-pointer transition-colors"
                  onClick={() => handlePropertyClick(property)}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{property.property_id}</div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm text-gray-900">{property.address}</div>
                    <div className="text-xs text-gray-500">
                      {property.neighborhood && `${property.neighborhood}, `}
                      {property.city && `${property.city}, `}
                      {property.state} {property.zip_code}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">
                      {property.property_type?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">
                      {formatCurrency(property.estimated_value)}
                    </div>
                    {property.confidence_score !== undefined && (
                      <div className="text-xs text-gray-500">
                        {formatPercentage(property.confidence_score)} confidence
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">
                      {formatDate(property.valuation_date)}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
                      {property.valuation_method?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      
      {/* Pagination */}
      <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
        <div className="flex-1 flex justify-between items-center">
          <div>
            <p className="text-sm text-gray-700">
              Showing <span className="font-medium">{filteredProperties.length > 0 ? ((pagination.currentPage - 1) * pagination.itemsPerPage) + 1 : 0}</span> to <span className="font-medium">{Math.min(pagination.currentPage * pagination.itemsPerPage, pagination.totalItems)}</span> of <span className="font-medium">{pagination.totalItems}</span> properties
            </p>
          </div>
          <div className="flex items-center">
            <label htmlFor="itemsPerPage" className="mr-2 text-sm text-gray-700">Items per page:</label>
            <select
              id="itemsPerPage"
              value={pagination.itemsPerPage}
              onChange={handleItemsPerPageChange}
              className="border border-gray-300 rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="10">10</option>
              <option value="25">25</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </select>
            
            <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px ml-4" aria-label="Pagination">
              <button
                onClick={() => handlePageChange(pagination.currentPage - 1)}
                disabled={pagination.currentPage === 1}
                className={`relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium ${pagination.currentPage === 1 ? 'text-gray-300 cursor-not-allowed' : 'text-gray-500 hover:bg-gray-50'}`}
              >
                <span className="sr-only">Previous</span>
                <i className="fas fa-chevron-left"></i>
              </button>
              
              {/* Page buttons */}
              {[...Array(Math.min(pagination.totalPages, 5))].map((_, i) => {
                let pageNumber;
                
                // For small page counts or first pages
                if (pagination.totalPages <= 5 || pagination.currentPage <= 3) {
                  pageNumber = i + 1;
                } else if (pagination.currentPage >= pagination.totalPages - 2) {
                  // For last pages
                  pageNumber = pagination.totalPages - 4 + i;
                } else {
                  // For middle pages
                  pageNumber = pagination.currentPage - 2 + i;
                }
                
                return (
                  <button
                    key={pageNumber}
                    onClick={() => handlePageChange(pageNumber)}
                    className={`relative inline-flex items-center px-4 py-2 border border-gray-300 bg-white text-sm font-medium ${pagination.currentPage === pageNumber ? 'bg-blue-50 text-blue-600 z-10' : 'text-gray-500 hover:bg-gray-50'}`}
                  >
                    {pageNumber}
                  </button>
                );
              })}
              
              <button
                onClick={() => handlePageChange(pagination.currentPage + 1)}
                disabled={pagination.currentPage === pagination.totalPages || pagination.totalPages === 0}
                className={`relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium ${pagination.currentPage === pagination.totalPages || pagination.totalPages === 0 ? 'text-gray-300 cursor-not-allowed' : 'text-gray-500 hover:bg-gray-50'}`}
              >
                <span className="sr-only">Next</span>
                <i className="fas fa-chevron-right"></i>
              </button>
            </nav>
          </div>
        </div>
      </div>
    </div>
  );
  
  /**
   * Render dashboard summary metrics
   * @returns {JSX.Element} - The dashboard summary component
   */
  const renderSummaryMetrics = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
      <div className="metric-card bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center">
          <div className="p-3 rounded-full bg-blue-100 text-blue-500 mr-4">
            <i className="fas fa-home text-xl"></i>
          </div>
          <div>
            <p className="text-sm text-gray-500">Total Properties</p>
            <p className="text-2xl font-semibold">{summaryMetrics.totalProperties.toLocaleString()}</p>
          </div>
        </div>
      </div>
      
      <div className="metric-card bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center">
          <div className="p-3 rounded-full bg-green-100 text-green-500 mr-4">
            <i className="fas fa-dollar-sign text-xl"></i>
          </div>
          <div>
            <p className="text-sm text-gray-500">Average Value</p>
            <p className="text-2xl font-semibold">{formatCurrency(summaryMetrics.averageValue)}</p>
          </div>
        </div>
      </div>
      
      <div className="metric-card bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center">
          <div className="p-3 rounded-full bg-purple-100 text-purple-500 mr-4">
            <i className="fas fa-chart-pie text-xl"></i>
          </div>
          <div>
            <p className="text-sm text-gray-500">Median Value</p>
            <p className="text-2xl font-semibold">{formatCurrency(summaryMetrics.medianValue)}</p>
          </div>
        </div>
      </div>
      
      <div className="metric-card bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center">
          <div className="p-3 rounded-full bg-yellow-100 text-yellow-500 mr-4">
            <i className="fas fa-history text-xl"></i>
          </div>
          <div>
            <p className="text-sm text-gray-500">Last Refresh</p>
            <p className="text-2xl font-semibold">
              {refreshSettings.lastRefreshTime ? 
                new Date(refreshSettings.lastRefreshTime).toLocaleTimeString() : 
                'Never'}
            </p>
          </div>
        </div>
        <div className="mt-1 text-xs text-gray-500 flex items-center">
          <span className={`mr-1 h-2 w-2 rounded-full ${refreshSettings.autoRefresh ? 'bg-green-500' : 'bg-gray-400'}`}></span>
          <span>Auto-refresh: {refreshSettings.autoRefresh ? 'On' : 'Off'}</span>
        </div>
      </div>
    </div>
  );
  
  /**
   * Render property data visualization charts
   * @returns {JSX.Element} - The charts component
   */
  const renderPropertyCharts = () => (
    <div className="bg-white rounded-lg shadow-md p-4 mb-6">
      <div className="flex border-b mb-4">
        <button
          className={`px-4 py-2 font-medium ${activeTab === 'distribution' ? 'text-blue-500 border-b-2 border-blue-500' : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => setActiveTab('distribution')}
        >
          Value Distribution
        </button>
        <button
          className={`px-4 py-2 font-medium ${activeTab === 'neighborhood' ? 'text-blue-500 border-b-2 border-blue-500' : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => setActiveTab('neighborhood')}
        >
          Neighborhood Comparison
        </button>
        <button
          className={`px-4 py-2 font-medium ${activeTab === 'trend' ? 'text-blue-500 border-b-2 border-blue-500' : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => setActiveTab('trend')}
        >
          Value Trends
        </button>
        <button
          className={`px-4 py-2 font-medium ${activeTab === 'models' ? 'text-blue-500 border-b-2 border-blue-500' : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => setActiveTab('models')}
        >
          Valuation Models
        </button>
      </div>
      
      <div className="h-80">
        {loading ? (
          <div className="h-full flex justify-center items-center">
            <svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
        ) : error ? (
          <div className="h-full flex justify-center items-center text-red-500">
            {error}
          </div>
        ) : properties.length === 0 ? (
          <div className="h-full flex justify-center items-center text-gray-500">
            No data available for visualization.
          </div>
        ) : (
          <>
            {/* Value Distribution Chart */}
            {activeTab === 'distribution' && (
              <div className="h-full" ref={valueDistributionChartRef}>
                <Bar data={valueDistributionData} options={valueDistributionOptions} />
              </div>
            )}
            
            {/* Neighborhood Comparison Chart */}
            {activeTab === 'neighborhood' && (
              <div className="h-full" ref={neighborhoodChartRef}>
                <Bar data={neighborhoodComparisonData} options={neighborhoodOptions} />
              </div>
            )}
            
            {/* Value Trend Chart */}
            {activeTab === 'trend' && (
              <div className="h-full" ref={trendChartRef}>
                <Line data={valueTrendData} options={trendOptions} />
              </div>
            )}
            
            {/* Model Distribution Chart */}
            {activeTab === 'models' && (
              <div className="h-full flex justify-center">
                <div className="w-3/4 h-full">
                  <Pie data={modelDistributionData} options={modelDistributionOptions} />
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
  
  /**
   * Render ETL status section
   * @returns {JSX.Element} - The ETL status component
   */
  const renderEtlStatus = () => (
    <div className="bg-white rounded-lg shadow-md p-4 mb-6">
      <h2 className="text-xl font-semibold mb-4">ETL Pipeline Status</h2>
      
      {etlStatus.isLoading ? (
        <div className="flex justify-center items-center h-40">
          <svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        </div>
      ) : etlStatus.error ? (
        <div className="bg-red-50 text-red-500 p-4 rounded-md">
          {etlStatus.error}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <div className="mb-4">
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium">Status: </span>
                <span className={`text-sm font-semibold ${
                  etlStatus.status === 'running' ? 'text-green-500' :
                  etlStatus.status === 'pending' ? 'text-blue-500' :
                  etlStatus.status === 'failed' ? 'text-red-500' :
                  etlStatus.status === 'completed' ? 'text-green-700' :
                  'text-gray-500'
                }`}>
                  {etlStatus.status?.charAt(0).toUpperCase() + etlStatus.status?.slice(1) || 'Unknown'}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    etlStatus.status === 'failed' ? 'bg-red-500' :
                    etlStatus.status === 'completed' ? 'bg-green-500' :
                    'bg-blue-500'
                  }`}
                  style={{ width: `${Math.round(etlStatus.progress * 100)}%` }}
                ></div>
              </div>
              <div className="text-right text-xs text-gray-500 mt-1">
                {Math.round(etlStatus.progress * 100)}% complete
              </div>
            </div>
            
            <div className="text-sm text-gray-600 mb-4">
              <div><span className="font-medium">Last Updated:</span> {formatDateTime(etlStatus.lastUpdate)}</div>
              <div><span className="font-medium">Records Processed:</span> {etlStatus.metrics.recordsProcessed.toLocaleString()}</div>
              <div><span className="font-medium">Success Rate:</span> {formatPercentage(etlStatus.metrics.successRate)}</div>
              <div><span className="font-medium">Avg. Processing Time:</span> {etlStatus.metrics.averageProcessingTime.toFixed(2)}ms</div>
            </div>
            
            <div className="mb-2">
              <h3 className="text-sm font-semibold mb-2">Data Quality Metrics</h3>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span>Completeness</span>
                    <span>{formatPercentage(etlStatus.dataQuality.completeness)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: `${etlStatus.dataQuality.completeness * 100}%` }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span>Accuracy</span>
                    <span>{formatPercentage(etlStatus.dataQuality.accuracy)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div className="bg-green-500 h-1.5 rounded-full" style={{ width: `${etlStatus.dataQuality.accuracy * 100}%` }}></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span>Timeliness</span>
                    <span>{formatPercentage(etlStatus.dataQuality.timeliness)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div className="bg-purple-500 h-1.5 rounded-full" style={{ width: `${etlStatus.dataQuality.timeliness * 100}%` }}></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="text-sm font-semibold mb-2">Data Sources</h3>
            {etlStatus.sources && etlStatus.sources.length > 0 ? (
              <div className="overflow-y-auto max-h-60">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Source</th>
                      <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                      <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Records</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {etlStatus.sources.map((source, index) => (
                      <tr key={index}>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900">{source.name}</td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm">
                          <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            source.status === 'completed' ? 'bg-green-100 text-green-800' :
                            source.status === 'running' ? 'bg-blue-100 text-blue-800' :
                            source.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                            source.status === 'failed' ? 'bg-red-100 text-red-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {source.status}
                          </span>
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500">{source.records?.toLocaleString() || 0}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-sm text-gray-500">No data sources available.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
  
  /**
   * Render agent status section
   * @returns {JSX.Element} - The agent status component
   */
  const renderAgentStatus = () => (
    <div className="bg-white rounded-lg shadow-md p-4 mb-6">
      <h2 className="text-xl font-semibold mb-4">Agent Status</h2>
      
      {agentStatus.isLoading ? (
        <div className="flex justify-center items-center h-40">
          <svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        </div>
      ) : agentStatus.error ? (
        <div className="bg-red-50 text-red-500 p-4 rounded-md">
          {agentStatus.error}
        </div>
      ) : agentStatus.agents.length === 0 ? (
        <p className="text-center text-gray-500 py-8">No agent data available.</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agent</th>
                    <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                    <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                    <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Queue</th>
                    <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Success Rate</th>
                    <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Update</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {agentStatus.agents.map((agent, index) => (
                    <tr 
                      key={agent.agent_id || index}
                      onClick={() => handleAgentSelect(agent)}
                      className="hover:bg-gray-50 cursor-pointer transition-colors"
                    >
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">{agent.agent_name}</div>
                        <div className="text-xs text-gray-500">{agent.agent_id}</div>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{agent.agent_type}</div>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          agent.status === 'idle' ? 'bg-green-100 text-green-800' :
                          agent.status === 'busy' ? 'bg-blue-100 text-blue-800' :
                          agent.status === 'error' ? 'bg-red-100 text-red-800' :
                          agent.status === 'offline' ? 'bg-gray-100 text-gray-800' :
                          'bg-yellow-100 text-yellow-800'
                        }`}>
                          {agent.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                        {agent.queue_size || 0}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                        {formatPercentage(agent.success_rate || 0)}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                        {formatDateTime(agent.last_heartbeat)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          <div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-sm font-semibold mb-3">Agent Status Distribution</h3>
              <div className="h-48">
                <Doughnut data={agentStatusData} options={agentStatusOptions} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
  
  /**
   * Render property detail modal
   * @returns {JSX.Element|null} - The property detail modal or null if no property is selected
   */
  const renderPropertyDetailModal = () => {
    if (!selectedProperty) return null;
    
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-full overflow-hidden">
          <div className="flex justify-between items-center p-4 border-b">
            <h2 className="text-xl font-semibold">Property Details</h2>
            <button 
              onClick={closePropertyDetail}
              className="text-gray-500 hover:text-gray-700 focus:outline-none"
            >
              <i className="fas fa-times text-xl"></i>
            </button>
          </div>
          
          <div className="p-4 overflow-y-auto max-h-[calc(100vh-10rem)]">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">Property Information</h3>
                <div className="space-y-2">
                  <p><span className="font-medium">ID:</span> {selectedProperty.property_id}</p>
                  <p><span className="font-medium">Address:</span> {selectedProperty.address}</p>
                  <p><span className="font-medium">Location:</span> {selectedProperty.city}, {selectedProperty.state} {selectedProperty.zip_code}</p>
                  <p><span className="font-medium">Neighborhood:</span> {selectedProperty.neighborhood || 'Not specified'}</p>
                  <p><span className="font-medium">Property Type:</span> {selectedProperty.property_type?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
                  <p><span className="font-medium">Bedrooms:</span> {selectedProperty.bedrooms || 'N/A'}</p>
                  <p><span className="font-medium">Bathrooms:</span> {selectedProperty.bathrooms || 'N/A'}</p>
                  <p><span className="font-medium">Square Feet:</span> {selectedProperty.square_feet ? `${selectedProperty.square_feet.toLocaleString()} sq ft` : 'N/A'}</p>
                  <p><span className="font-medium">Year Built:</span> {selectedProperty.year_built || 'N/A'}</p>
                  <p><span className="font-medium">Lot Size:</span> {selectedProperty.lot_size ? `${selectedProperty.lot_size.toLocaleString()} acres` : 'N/A'}</p>
                  {selectedProperty.latitude && selectedProperty.longitude && (
                    <p><span className="font-medium">Coordinates:</span> {selectedProperty.latitude.toFixed(6)}, {selectedProperty.longitude.toFixed(6)}</p>
                  )}
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-2">Valuation Details</h3>
                <div className="mb-3">
                  <span className="text-sm text-gray-600">Estimated Value</span>
                  <div className="text-3xl font-bold text-blue-600">
                    {formatCurrency(selectedProperty.estimated_value)}
                  </div>
                  <span className="text-sm text-gray-500">
                    as of {formatDate(selectedProperty.valuation_date)}
                  </span>
                </div>
                
                <div className="space-y-2">
                  <p><span className="font-medium">Valuation Method:</span> {selectedProperty.valuation_method?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
                  <p><span className="font-medium">Confidence Score:</span> {formatPercentage(selectedProperty.confidence_score)}</p>
                  
                  {selectedProperty.adj_r2_score !== undefined && (
                    <p><span className="font-medium">Adjusted R² Score:</span> {selectedProperty.adj_r2_score.toFixed(4)}</p>
                  )}
                  
                  {selectedProperty.rmse !== undefined && (
                    <p><span className="font-medium">RMSE:</span> {selectedProperty.rmse.toLocaleString()}</p>
                  )}
                  
                  {selectedProperty.mae !== undefined && (
                    <p><span className="font-medium">MAE:</span> {selectedProperty.mae.toLocaleString()}</p>
                  )}
                  
                  {selectedProperty.last_sale_price && (
                    <p>
                      <span className="font-medium">Last Sale Price:</span> {formatCurrency(selectedProperty.last_sale_price)}
                      {selectedProperty.last_sale_date && ` (${formatDate(selectedProperty.last_sale_date)})`}
                    </p>
                  )}
                </div>
                
                {selectedProperty.gis_adjustments && (
                  <div className="mt-4">
                    <h4 className="text-md font-semibold mb-2">GIS Adjustments</h4>
                    <div className="space-y-2">
                      <p><span className="font-medium">Base Value:</span> {formatCurrency(selectedProperty.gis_adjustments.base_value)}</p>
                      <p><span className="font-medium">Neighborhood Quality:</span> {formatPercentage(selectedProperty.gis_adjustments.quality_adjustment)}</p>
                      <p><span className="font-medium">Proximity Factors:</span> {formatPercentage(selectedProperty.gis_adjustments.proximity_adjustment)}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            {selectedProperty.feature_importance && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-2">Feature Importance</h3>
                <div className="space-y-2">
                  {Object.entries(selectedProperty.feature_importance)
                    .sort(([, a], [, b]) => b - a)
                    .map(([feature, importance], index) => (
                      <div key={index}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm">{feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                          <span className="text-sm font-medium">{formatPercentage(importance)}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="bg-blue-500 h-2 rounded-full" style={{ width: `${importance * 100}%` }}></div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            )}
            
            {selectedProperty.comparable_properties && selectedProperty.comparable_properties.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-2">Comparable Properties</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Address</th>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Distance</th>
                        <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Similarity</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {selectedProperty.comparable_properties.map((comp, index) => (
                        <tr key={index}>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900">{comp.address}</td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900">{formatCurrency(comp.value)}</td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500">{comp.distance ? `${comp.distance.toFixed(2)} miles` : 'N/A'}</td>
                          <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500">{formatPercentage(comp.similarity)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
          
          <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse border-t">
            <button
              type="button"
              className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm"
              onClick={closePropertyDetail}
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  };
  
  /**
   * Render agent detail modal
   * @returns {JSX.Element|null} - The agent detail modal or null if no agent is selected
   */
  const renderAgentDetailModal = () => {
    if (!selectedAgent) return null;
    
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 overflow-y-auto">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-full overflow-hidden">
          <div className="flex justify-between items-center p-4 border-b">
            <h2 className="text-xl font-semibold">Agent Details: {selectedAgent.agent_name}</h2>
            <button 
              onClick={closeAgentDetail}
              className="text-gray-500 hover:text-gray-700 focus:outline-none"
            >
              <i className="fas fa-times text-xl"></i>
            </button>
          </div>
          
          <div className="p-4 overflow-y-auto max-h-[calc(100vh-10rem)]">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div>
                <h3 className="text-lg font-semibold mb-2">Agent Information</h3>
                <div className="space-y-2">
                  <p><span className="font-medium">ID:</span> {selectedAgent.agent_id}</p>
                  <p><span className="font-medium">Type:</span> {selectedAgent.agent_type}</p>
                  <p>
                    <span className="font-medium">Status:</span> 
                    <span className={`ml-2 px-2 py-0.5 text-xs font-medium rounded-full ${
                      selectedAgent.status === 'idle' ? 'bg-green-100 text-green-800' :
                      selectedAgent.status === 'busy' ? 'bg-blue-100 text-blue-800' :
                      selectedAgent.status === 'error' ? 'bg-red-100 text-red-800' :
                      selectedAgent.status === 'offline' ? 'bg-gray-100 text-gray-800' :
                      'bg-yellow-100 text-yellow-800'
                    }`}>
                      {selectedAgent.status}
                    </span>
                  </p>
                  <p><span className="font-medium">Last Heartbeat:</span> {formatDateTime(selectedAgent.last_heartbeat)}</p>
                  <p><span className="font-medium">Current Task:</span> {selectedAgent.current_task || 'None'}</p>
                  <p><span className="font-medium">Queue Size:</span> {selectedAgent.queue_size || 0}</p>
                  <p><span className="font-medium">Success Rate:</span> {formatPercentage(selectedAgent.success_rate)}</p>
                  <p><span className="font-medium">Error Count:</span> {selectedAgent.error_count || 0}</p>
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-2">Performance Metrics</h3>
                
                {/* Agent Success Rate Visual */}
                <div className="mb-4">
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Success Rate</span>
                    <span className="text-sm font-medium">{formatPercentage(selectedAgent.success_rate)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className={`h-2.5 rounded-full ${
                        selectedAgent.success_rate >= 0.9 ? 'bg-green-500' :
                        selectedAgent.success_rate >= 0.7 ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${(selectedAgent.success_rate || 0) * 100}%` }}
                    ></div>
                  </div>
                </div>
                
                {/* Agent Queue Size Visual */}
                <div className="mb-4">
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium">Queue Utilization</span>
                    <span className="text-sm font-medium">{selectedAgent.queue_size || 0} tasks</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className={`h-2.5 rounded-full ${
                        (selectedAgent.queue_size || 0) >= 10 ? 'bg-red-500' :
                        (selectedAgent.queue_size || 0) >= 5 ? 'bg-yellow-500' :
                        'bg-blue-500'
                      }`}
                      style={{ width: `${Math.min((selectedAgent.queue_size || 0) * 10, 100)}%` }}
                    ></div>
                  </div>
                </div>
                
                {selectedAgent.metrics && (
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <div className="bg-gray-50 rounded-lg p-3">
                      <div className="text-sm text-gray-500">Avg. Response Time</div>
                      <div className="text-xl font-semibold">{selectedAgent.metrics.avg_response_time?.toFixed(2) || 'N/A'} ms</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <div className="text-sm text-gray-500">Tasks Completed</div>
                      <div className="text-xl font-semibold">{selectedAgent.metrics.tasks_completed?.toLocaleString() || 'N/A'}</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <div className="text-sm text-gray-500">Memory Usage</div>
                      <div className="text-xl font-semibold">{selectedAgent.metrics.memory_usage ? `${(selectedAgent.metrics.memory_usage / (1024 * 1024)).toFixed(1)} MB` : 'N/A'}</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <div className="text-sm text-gray-500">CPU Usage</div>
                      <div className="text-xl font-semibold">{selectedAgent.metrics.cpu_usage ? `${(selectedAgent.metrics.cpu_usage * 100).toFixed(1)}%` : 'N/A'}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            <h3 className="text-lg font-semibold mb-2">Agent Logs</h3>
            {selectedAgent.logsError ? (
              <div className="bg-red-50 text-red-500 p-4 rounded-md">
                {selectedAgent.logsError}
              </div>
            ) : selectedAgent.logs && selectedAgent.logs.length > 0 ? (
              <div className="bg-gray-50 rounded-lg overflow-hidden border border-gray-200">
                <div className="max-h-80 overflow-y-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-100 sticky top-0">
                      <tr>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Level</th>
                        <th scope="col" className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Message</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {selectedAgent.logs.map((log, index) => (
                        <tr key={index}>
                          <td className="px-4 py-2 whitespace-nowrap text-xs text-gray-500">
                            {formatDateTime(log.timestamp)}
                          </td>
                          <td className="px-4 py-2 whitespace-nowrap">
                            <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                              log.level === 'error' ? 'bg-red-100 text-red-800' :
                              log.level === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                              log.level === 'info' ? 'bg-blue-100 text-blue-800' :
                              log.level === 'debug' ? 'bg-gray-100 text-gray-800' :
                              'bg-purple-100 text-purple-800'
                            }`}>
                              {log.level}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-xs text-gray-900">
                            {log.message}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <p className="text-gray-500 text-center py-4">No logs available for this agent.</p>
            )}
          </div>
          
          <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse border-t">
            <button
              type="button"
              className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm"
              onClick={closeAgentDetail}
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  };
  
  /**
   * Render dashboard tabs navigation
   * @returns {JSX.Element} - The dashboard tabs component
   */
  const renderDashboardTabs = () => (
    <div className="bg-white rounded-lg shadow-md p-4 mb-6">
      <div className="flex flex-wrap border-b mb-4">
        <button
          className={`px-4 py-2 font-medium ${activeDashboardTab === 'properties' ? 'text-blue-500 border-b-2 border-blue-500' : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => handleDashboardTabChange('properties')}
        >
          <i className="fas fa-home mr-2"></i>
          Properties
        </button>
        <button
          className={`px-4 py-2 font-medium ${activeDashboardTab === 'etl' ? 'text-blue-500 border-b-2 border-blue-500' : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => handleDashboardTabChange('etl')}
        >
          <i className="fas fa-database mr-2"></i>
          ETL Status
        </button>
        <button
          className={`px-4 py-2 font-medium ${activeDashboardTab === 'agents' ? 'text-blue-500 border-b-2 border-blue-500' : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => handleDashboardTabChange('agents')}
        >
          <i className="fas fa-robot mr-2"></i>
          Agents
        </button>
        
        {/* Push to right */}
        <div className="flex-grow"></div>
        
        {/* Refresh controls */}
        <div className="flex items-center">
          <div className="mr-2">
            <select
              value={refreshSettings.interval}
              onChange={handleRefreshIntervalChange}
              className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="30000">Refresh: 30s</option>
              <option value="60000">Refresh: 1m</option>
              <option value="300000">Refresh: 5m</option>
              <option value="600000">Refresh: 10m</option>
            </select>
          </div>
          
          <button
            onClick={toggleAutoRefresh}
            className={`mr-2 px-2 py-1 text-sm rounded-md ${refreshSettings.autoRefresh ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'}`}
            title={refreshSettings.autoRefresh ? 'Auto-refresh is on' : 'Auto-refresh is off'}
          >
            <i className={`fas fa-${refreshSettings.autoRefresh ? 'sync' : 'sync-alt'} mr-1`}></i>
            {refreshSettings.autoRefresh ? 'Auto' : 'Manual'}
          </button>
          
          <button
            onClick={refreshAllData}
            className="px-2 py-1 text-sm bg-blue-100 text-blue-700 rounded-md"
            title="Refresh all data now"
          >
            <i className="fas fa-redo-alt"></i>
          </button>
        </div>
      </div>
    </div>
  );
  
  // --- Main Render ---
  return (
    <div className="bg-gray-100 min-h-screen p-4">
      <h1 className="text-2xl font-bold mb-6">Property Valuation Dashboard</h1>
      
      {/* Dashboard tabs */}
      {renderDashboardTabs()}
      
      {/* Summary metrics */}
      {renderSummaryMetrics()}
      
      {/* Properties view */}
      {activeDashboardTab === 'properties' && (
        <>
          {/* Filters */}
          {renderFilterSection()}
          
          {/* Charts */}
          {renderPropertyCharts()}
          
          {/* Property Table */}
          {renderPropertyTable()}
        </>
      )}
      
      {/* ETL Status view */}
      {activeDashboardTab === 'etl' && renderEtlStatus()}
      
      {/* Agent Status view */}
      {activeDashboardTab === 'agents' && renderAgentStatus()}
      
      {/* Property Detail Modal */}
      {renderPropertyDetailModal()}
      
      {/* Agent Detail Modal */}
      {renderAgentDetailModal()}
    </div>
  );
};

export default Dashboard;