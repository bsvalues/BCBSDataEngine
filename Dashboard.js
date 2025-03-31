import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend, 
  ArcElement, 
  TimeScale, 
  Filler, 
  RadialLinearScale, 
  DoughnutController 
} from 'chart.js';
import { Line, Bar, Pie, Doughnut } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns'; // For time scale
import { format } from 'date-fns'; // For date formatting
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
 * @version 3.0.0
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

    // Setup auto-refresh timer if enabled
    if (refreshSettings.autoRefresh) {
      startAutoRefreshTimer();
    }

    // Setup intersection observer for chart animations
    setupIntersectionObserver();

    // Cleanup function for component unmount
    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
      if (metricsObserverRef.current) {
        metricsObserverRef.current.disconnect();
      }
    };
  }, []);

  /**
   * Effect to handle changes to auto-refresh settings
   */
  useEffect(() => {
    if (refreshSettings.autoRefresh) {
      startAutoRefreshTimer();
    } else if (refreshTimerRef.current) {
      clearInterval(refreshTimerRef.current);
    }

    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [refreshSettings.autoRefresh, refreshSettings.interval]);

  /**
   * Set up the auto-refresh timer for periodic data updates
   */
  const startAutoRefreshTimer = () => {
    if (refreshTimerRef.current) {
      clearInterval(refreshTimerRef.current);
    }

    refreshTimerRef.current = setInterval(() => {
      console.log('Auto-refreshing dashboard data...');
      refreshAllData();
    }, refreshSettings.interval);
  };

  /**
   * Set up intersection observer for triggering animations when charts come into view
   */
  const setupIntersectionObserver = () => {
    const options = {
      root: null,
      rootMargin: '0px',
      threshold: 0.1
    };

    metricsObserverRef.current = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          // Trigger chart animation when it becomes visible
          const chartId = entry.target.id;
          if (chartId === 'value-distribution-chart' && valueDistributionChartRef.current) {
            valueDistributionChartRef.current.update();
          } else if (chartId === 'neighborhood-chart' && neighborhoodChartRef.current) {
            neighborhoodChartRef.current.update();
          } else if (chartId === 'trend-chart' && trendChartRef.current) {
            trendChartRef.current.update();
          } else if (chartId === 'etl-progress-chart' && etlProgressChartRef.current) {
            etlProgressChartRef.current.update();
          } else if (chartId === 'agent-performance-chart' && agentPerformanceChartRef.current) {
            agentPerformanceChartRef.current.update();
          }
        }
      });
    }, options);

    // Observe each chart container
    const chartContainers = document.querySelectorAll('.chart-container');
    chartContainers.forEach(container => {
      metricsObserverRef.current.observe(container);
    });
  };

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
      
      // Add a timestamp to prevent caching
      queryParams.append('_t', Date.now());
      
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
      const startTime = performance.now();
      const response = await fetch(`/api/valuations?${queryParams.toString()}`, {
        method: 'GET',
        headers: headers
      });
      
      const endTime = performance.now();
      console.log(`API request completed in ${(endTime - startTime).toFixed(2)}ms`);
      
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
      const startTime = performance.now();
      const response = await fetch('/api/etl-status', {
        method: 'GET',
        headers: headers
      });
      
      const endTime = performance.now();
      console.log(`ETL status API request completed in ${(endTime - startTime).toFixed(2)}ms`);
      
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
      const startTime = performance.now();
      const response = await fetch('/api/agent-status', {
        method: 'GET',
        headers: headers
      });
      
      const endTime = performance.now();
      console.log(`Agent status API request completed in ${(endTime - startTime).toFixed(2)}ms`);
      
      if (!response.ok) {
        throw new Error(`Agent status API request failed with status ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      setAgentStatus({
        agents: Array.isArray(data.agents) ? data.agents : 
                (Array.isArray(data) ? data : []),
        isLoading: false,
        error: null,
        lastUpdate: new Date()
      });
      
      // Update refresh timestamp
      setRefreshSettings(prev => ({
        ...prev,
        lastRefreshTime: new Date()
      }));
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
    
    setSelectedAgent(prev => prev ? { ...prev, isLoading: true, error: null } : { id: agentId, isLoading: true, error: null });
    
    try {
      // Headers for the API request including the API key
      const headers = {
        'Content-Type': 'application/json'
      };
      
      // Add API key header if available
      if (apiKey) {
        headers['X-API-Key'] = apiKey;
      }
      
      // Make the API request to the agent details endpoint
      const response = await fetch(`/api/agent-status/${agentId}`, {
        method: 'GET',
        headers: headers
      });
      
      if (!response.ok) {
        throw new Error(`Agent details API request failed with status ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Update the selected agent with detailed information
      setSelectedAgent({
        ...data,
        isLoading: false,
        error: null
      });
    } catch (err) {
      console.error(`Error fetching agent details for ${agentId}:`, err);
      setSelectedAgent(prev => ({ 
        ...prev, 
        isLoading: false, 
        error: `Failed to load agent details: ${err.message}. Please try again later.` 
      }));
    }
  }, [apiKey]);
  
  /**
   * Function to refresh all dashboard data
   * This will fetch property valuations, ETL status, and agent status
   */
  const refreshAllData = useCallback(() => {
    fetchPropertyValuations();
    fetchEtlStatus();
    fetchAgentStatus();
    
    // If an agent is selected, also refresh its details
    if (selectedAgent?.id) {
      fetchAgentDetails(selectedAgent.id);
    }
    
    // Update last refresh time
    setRefreshSettings(prev => ({
      ...prev,
      lastRefreshTime: new Date()
    }));
  }, [fetchPropertyValuations, fetchEtlStatus, fetchAgentStatus, fetchAgentDetails, selectedAgent]);
  
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
  const handleAgentClick = (agent) => {
    fetchAgentDetails(agent.id);
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
  
  /**
   * Format percentage values for display
   * @param {number} value - Value to format as percentage
   * @returns {string} - Formatted percentage string
   */
  const formatPercentage = (value) => {
    if (value === undefined || value === null) return '0%';
    
    // Ensure value is a number
    const numValue = parseFloat(value);
    if (isNaN(numValue)) return '0%';
    
    // Format as percentage with 1 decimal place
    return (numValue * 100).toFixed(1) + '%';
  };
  
  /**
   * Format date values for display
   * @param {string} dateString - ISO date string
   * @returns {string} - Formatted date string
   */
  const formatDate = (dateString) => {
    if (!dateString) return '';
    
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return '';
    
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
    if (!dateTimeValue) return '';
    
    const date = dateTimeValue instanceof Date ? dateTimeValue : new Date(dateTimeValue);
    if (isNaN(date.getTime())) return '';
    
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };
  
  /**
   * Helper function to get status color class based on status string
   * @param {string} status - Status string
   * @returns {string} - Tailwind CSS color class
   */
  const getStatusColorClass = (status) => {
    if (!status) return 'bg-gray-300';
    
    const statusLower = status.toLowerCase();
    
    if (statusLower.includes('error') || statusLower.includes('fail') || statusLower === 'down') {
      return 'bg-red-500';
    } else if (statusLower.includes('warn') || statusLower === 'degraded' || statusLower === 'partial') {
      return 'bg-yellow-500';
    } else if (statusLower.includes('success') || statusLower === 'up' || statusLower === 'active' || statusLower === 'completed') {
      return 'bg-green-500';
    } else if (statusLower.includes('process') || statusLower === 'running' || statusLower === 'pending') {
      return 'bg-blue-500';
    }
    
    return 'bg-gray-500';
  };
  
  /**
   * Format a future date as a relative time string (e.g., "in 3 hours")
   * @param {Date} date - The future date to format
   * @returns {string} - Formatted relative time string
   */
  const formatRelativeTime = (date) => {
    if (!date || isNaN(date.getTime())) return 'Unknown';
    
    const now = new Date();
    const diffMs = date - now;
    
    // Invalid if in the past or too far in the future
    if (diffMs < 0) return 'Already completed';
    if (diffMs > 1000 * 60 * 60 * 24 * 30) return 'Over a month';
    
    const diffMinutes = Math.round(diffMs / (1000 * 60));
    const diffHours = Math.round(diffMs / (1000 * 60 * 60));
    const diffDays = Math.round(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffMinutes < 60) {
      return `in ${diffMinutes} minute${diffMinutes === 1 ? '' : 's'}`;
    } else if (diffHours < 24) {
      return `in ${diffHours} hour${diffHours === 1 ? '' : 's'}`;
    } else {
      return `in ${diffDays} day${diffDays === 1 ? '' : 's'}`;
    }
  };
  
  /**
   * Format currency values for display
   * @param {number} value - Value to format as currency
   * @returns {string} - Formatted currency string
   */
  const formatCurrency = (value) => {
    if (value === undefined || value === null) return '$0';
    
    // Ensure value is a number
    const numValue = parseFloat(value);
    if (isNaN(numValue)) return '$0';
    
    // Format as USD currency
    return new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(numValue);
  };

  // --- Chart Data Preparation ---

  /**
   * Prepare value distribution chart data
   */
  const valueDistributionChartData = useMemo(() => {
    if (!properties || properties.length === 0) {
      return {
        labels: [],
        datasets: [{
          label: 'No data available',
          data: [],
          backgroundColor: 'rgba(200, 200, 200, 0.5)'
        }]
      };
    }

    // Create value buckets
    const bucketSize = 100000; // $100k buckets
    const buckets = {};
    const minValue = 0;
    const maxValue = 2000000; // $2M cap

    // Initialize buckets
    for (let i = minValue; i < maxValue; i += bucketSize) {
      buckets[i] = 0;
    }

    // Count properties in each bucket
    properties.forEach(property => {
      const value = property.estimated_value || property.value || 0;
      const bucketIndex = Math.floor(value / bucketSize) * bucketSize;
      if (bucketIndex < maxValue) {
        buckets[bucketIndex] = (buckets[bucketIndex] || 0) + 1;
      }
    });

    // Convert to chart format
    const labels = Object.keys(buckets).map(bucket => formatCurrency(parseInt(bucket)));
    const data = Object.values(buckets);

    return {
      labels,
      datasets: [{
        label: 'Property Count',
        data,
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 1
      }]
    };
  }, [properties]);

  /**
   * Prepare neighborhood comparison chart data
   */
  const neighborhoodChartData = useMemo(() => {
    if (!properties || properties.length === 0) {
      return {
        labels: [],
        datasets: [{
          label: 'No data available',
          data: [],
          backgroundColor: 'rgba(200, 200, 200, 0.5)'
        }]
      };
    }

    // Group properties by neighborhood and calculate average value
    const neighborhoodData = {};
    properties.forEach(property => {
      const neighborhood = property.neighborhood || extractNeighborhoodFromAddress(property.address) || 'Unknown';
      if (!neighborhoodData[neighborhood]) {
        neighborhoodData[neighborhood] = {
          total: 0,
          count: 0
        };
      }
      neighborhoodData[neighborhood].total += property.estimated_value || property.value || 0;
      neighborhoodData[neighborhood].count += 1;
    });

    // Calculate averages and sort by average value
    const neighborhoods = Object.keys(neighborhoodData).map(name => ({
      name,
      average: neighborhoodData[name].count > 0 
        ? neighborhoodData[name].total / neighborhoodData[name].count 
        : 0
    }));

    // Sort by average value (descending)
    neighborhoods.sort((a, b) => b.average - a.average);

    // Take top 10 neighborhoods
    const top10 = neighborhoods.slice(0, 10);

    return {
      labels: top10.map(n => n.name),
      datasets: [{
        label: 'Avg. Property Value',
        data: top10.map(n => n.average),
        backgroundColor: 'rgba(16, 185, 129, 0.5)',
        borderColor: 'rgba(16, 185, 129, 1)',
        borderWidth: 1
      }]
    };
  }, [properties]);

  /**
   * Prepare value trend chart data (by valuation date)
   */
  const valueTrendChartData = useMemo(() => {
    if (!properties || properties.length === 0) {
      return {
        labels: [],
        datasets: [{
          label: 'No data available',
          data: [],
          backgroundColor: 'rgba(200, 200, 200, 0.5)'
        }]
      };
    }

    // Group properties by valuation date and calculate average value and confidence
    const dateMap = {};
    properties.forEach(property => {
      if (!property.valuation_date) return;
      
      const date = formatDate(property.valuation_date);
      if (!dateMap[date]) {
        dateMap[date] = {
          total: 0,
          count: 0,
          confidenceTotal: 0
        };
      }
      dateMap[date].total += property.estimated_value || property.value || 0;
      dateMap[date].confidenceTotal += property.confidence_score || 0;
      dateMap[date].count += 1;
    });

    // Convert to arrays for chart
    const dates = Object.keys(dateMap).sort();
    const averages = dates.map(date => 
      dateMap[date].count > 0 ? dateMap[date].total / dateMap[date].count : 0
    );
    const confidenceScores = dates.map(date => 
      dateMap[date].count > 0 ? (dateMap[date].confidenceTotal / dateMap[date].count) * 100 : 0
    );

    return {
      labels: dates,
      datasets: [
        {
          label: 'Avg. Valuation',
          data: averages,
          fill: false,
          borderColor: 'rgba(139, 92, 246, 1)',
          backgroundColor: 'rgba(139, 92, 246, 0.5)',
          tension: 0.1,
          yAxisID: 'y'
        },
        {
          label: 'Confidence Score (%)', 
          data: confidenceScores,
          fill: false,
          borderColor: 'rgba(16, 185, 129, 1)',
          backgroundColor: 'rgba(16, 185, 129, 0.5)',
          borderDash: [5, 5],
          tension: 0.1,
          yAxisID: 'y1'
        }
      ]
    };
  }, [properties]);

  /**
   * Prepare model distribution pie chart data
   */
  const modelDistributionChartData = useMemo(() => {
    if (!properties || properties.length === 0) {
      return {
        labels: ['No data'],
        datasets: [{
          data: [1],
          backgroundColor: ['rgba(200, 200, 200, 0.5)'],
          borderColor: ['rgba(200, 200, 200, 1)'],
          borderWidth: 1
        }]
      };
    }

    // Count properties by model type
    const modelCounts = {};
    properties.forEach(property => {
      const model = property.valuation_model || 'Unknown';
      modelCounts[model] = (modelCounts[model] || 0) + 1;
    });

    // Convert to arrays for chart
    const models = Object.keys(modelCounts);
    const counts = models.map(model => modelCounts[model]);

    // Colors for each model type
    const backgroundColors = [
      'rgba(59, 130, 246, 0.6)',
      'rgba(16, 185, 129, 0.6)',
      'rgba(245, 158, 11, 0.6)',
      'rgba(239, 68, 68, 0.6)',
      'rgba(139, 92, 246, 0.6)',
      'rgba(236, 72, 153, 0.6)',
      'rgba(75, 85, 99, 0.6)'
    ];

    return {
      labels: models,
      datasets: [{
        data: counts,
        backgroundColor: backgroundColors.slice(0, models.length),
        borderColor: backgroundColors.map(color => color.replace('0.6', '1')),
        borderWidth: 1
      }]
    };
  }, [properties]);

  /**
   * Prepare agent status chart data
   */
  const agentStatusChartData = useMemo(() => {
    if (!agentStatus.agents || agentStatus.agents.length === 0) {
      return {
        labels: ['No data'],
        datasets: [{
          data: [1],
          backgroundColor: ['rgba(200, 200, 200, 0.5)'],
          borderColor: ['rgba(200, 200, 200, 1)'],
          borderWidth: 1
        }]
      };
    }

    // Count agents by status
    const statusCounts = {};
    agentStatus.agents.forEach(agent => {
      const status = agent.status || 'Unknown';
      statusCounts[status] = (statusCounts[status] || 0) + 1;
    });

    // Convert to arrays for chart
    const statuses = Object.keys(statusCounts);
    const counts = statuses.map(status => statusCounts[status]);

    // Status-specific colors
    const statusColors = {
      'active': 'rgba(16, 185, 129, 0.6)',
      'idle': 'rgba(59, 130, 246, 0.6)',
      'processing': 'rgba(245, 158, 11, 0.6)',
      'error': 'rgba(239, 68, 68, 0.6)',
      'offline': 'rgba(75, 85, 99, 0.6)',
      'unknown': 'rgba(200, 200, 200, 0.6)'
    };

    const backgroundColors = statuses.map(status => 
      statusColors[status.toLowerCase()] || 'rgba(200, 200, 200, 0.6)'
    );

    return {
      labels: statuses,
      datasets: [{
        data: counts,
        backgroundColor: backgroundColors,
        borderColor: backgroundColors.map(color => color.replace('0.6', '1')),
        borderWidth: 1
      }]
    };
  }, [agentStatus.agents]);

  /**
   * Options for Neighborhood Comparison chart
   */
  const neighborhoodChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Average Property Value by Neighborhood',
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `Avg. Value: ${formatCurrency(context.raw)}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(value) {
            return formatCurrency(value);
          }
        }
      }
    }
  };

  /**
   * Options for Value Trend chart
   * Enhanced with dual y-axis for valuation and confidence score
   */
  const valueTrendChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Valuation Trends & Confidence Over Time',
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            if (context.dataset.label === 'Avg. Valuation') {
              return `Avg. Value: ${formatCurrency(context.raw)}`;
            } else if (context.dataset.label === 'Confidence Score (%)') {
              return `Confidence: ${context.raw.toFixed(1)}%`;
            }
            return context.dataset.label + ': ' + context.raw;
          }
        }
      }
    },
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        beginAtZero: true,
        title: {
          display: true,
          text: 'Valuation ($)',
          color: 'rgba(139, 92, 246, 1)'
        },
        ticks: {
          callback: function(value) {
            return formatCurrency(value);
          },
          color: 'rgba(139, 92, 246, 1)'
        },
        grid: {
          drawOnChartArea: true
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Confidence Score (%)',
          color: 'rgba(16, 185, 129, 1)'
        },
        ticks: {
          callback: function(value) {
            return value + '%';
          },
          color: 'rgba(16, 185, 129, 1)'
        },
        grid: {
          drawOnChartArea: false
        }
      }
    }
  };

  /**
   * Options for Model Distribution pie chart
   */
  const modelDistributionChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Property Valuations by Model Type',
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const total = context.dataset.data.reduce((sum, val) => sum + val, 0);
            const percentage = ((context.raw / total) * 100).toFixed(1);
            return `${context.label}: ${context.raw} (${percentage}%)`;
          }
        }
      }
    }
  };

  /**
   * Options for Agent Status doughnut chart
   */
  const agentStatusChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Agent Status Distribution',
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const total = context.dataset.data.reduce((sum, val) => sum + val, 0);
            const percentage = ((context.raw / total) * 100).toFixed(1);
            return `${context.label}: ${context.raw} (${percentage}%)`;
          }
        }
      }
    }
  };

  /**
   * Options for ETL Progress gauge chart
   */
  const etlProgressChartOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'ETL Pipeline Progress',
        font: {
          size: 16
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `Progress: ${context.raw}%`;
          }
        }
      }
    },
    circumference: 180,
    rotation: -90,
    cutout: '70%'
  };

  // --- UI Rendering Functions ---

  /**
   * Render the filter section with all filter inputs
   */
  const renderFilterSection = () => (
    <div className="bg-white rounded-lg shadow-md p-4 mb-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Filters</h2>
      
      <form onSubmit={applyFilters} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Neighborhood filter */}
        <div className="mb-3">
          <label htmlFor="neighborhood" className="block text-sm font-medium text-gray-700 mb-1">
            Neighborhood
          </label>
          <select
            id="neighborhood"
            name="neighborhood"
            value={filters.neighborhood}
            onChange={handleFilterChange}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">All Neighborhoods</option>
            {availableNeighborhoods.map(neighborhood => (
              <option key={neighborhood} value={neighborhood}>
                {neighborhood}
              </option>
            ))}
          </select>
        </div>
        
        {/* Price range filters */}
        <div className="mb-3">
          <label htmlFor="minValue" className="block text-sm font-medium text-gray-700 mb-1">
            Minimum Value
          </label>
          <input
            type="number"
            id="minValue"
            name="minValue"
            value={filters.minValue}
            onChange={handleFilterChange}
            placeholder="Min Value"
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        <div className="mb-3">
          <label htmlFor="maxValue" className="block text-sm font-medium text-gray-700 mb-1">
            Maximum Value
          </label>
          <input
            type="number"
            id="maxValue"
            name="maxValue"
            value={filters.maxValue}
            onChange={handleFilterChange}
            placeholder="Max Value"
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        {/* Property type filter */}
        <div className="mb-3">
          <label htmlFor="propertyType" className="block text-sm font-medium text-gray-700 mb-1">
            Property Type
          </label>
          <select
            id="propertyType"
            name="propertyType"
            value={filters.propertyType}
            onChange={handleFilterChange}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">All Types</option>
            {availablePropertyTypes.map(type => (
              <option key={type} value={type}>
                {type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </option>
            ))}
          </select>
        </div>
        
        {/* Date range filters */}
        <div className="mb-3">
          <label htmlFor="fromDate" className="block text-sm font-medium text-gray-700 mb-1">
            From Date
          </label>
          <input
            type="date"
            id="fromDate"
            name="fromDate"
            value={filters.fromDate}
            onChange={handleFilterChange}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        <div className="mb-3">
          <label htmlFor="toDate" className="block text-sm font-medium text-gray-700 mb-1">
            To Date
          </label>
          <input
            type="date"
            id="toDate"
            name="toDate"
            value={filters.toDate}
            onChange={handleFilterChange}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        {/* Search query */}
        <div className="mb-3">
          <label htmlFor="searchQuery" className="block text-sm font-medium text-gray-700 mb-1">
            Search
          </label>
          <input
            type="text"
            id="searchQuery"
            name="searchQuery"
            value={filters.searchQuery}
            onChange={handleFilterChange}
            placeholder="Search by address, ID, etc."
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        {/* Filter actions */}
        <div className="mb-3 flex items-end space-x-2 col-span-full">
          <button
            type="submit"
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
          >
            Apply Filters
          </button>
          
          <button
            type="button"
            onClick={resetFilters}
            className="px-4 py-2 bg-gray-300 text-gray-800 rounded-md hover:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50"
          >
            Reset
          </button>
          
          <button
            type="button"
            onClick={refreshAllData}
            className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50 ml-auto"
          >
            <span className="flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
              </svg>
              Refresh
            </span>
          </button>
        </div>
      </form>
      
      {/* Auto-refresh controls */}
      <div className="flex items-center justify-end space-x-4 mt-2 text-sm text-gray-600">
        <div className="flex items-center">
          <input
            type="checkbox"
            id="autoRefresh"
            checked={refreshSettings.autoRefresh}
            onChange={toggleAutoRefresh}
            className="mr-2"
          />
          <label htmlFor="autoRefresh">Auto-refresh</label>
        </div>
        
        <div className="flex items-center">
          <select
            value={refreshSettings.interval}
            onChange={handleRefreshIntervalChange}
            disabled={!refreshSettings.autoRefresh}
            className="p-1 border border-gray-300 rounded-md"
          >
            <option value={10000}>10 seconds</option>
            <option value={30000}>30 seconds</option>
            <option value={60000}>1 minute</option>
            <option value={300000}>5 minutes</option>
          </select>
        </div>
        
        {refreshSettings.lastRefreshTime && (
          <span className="text-xs">
            Last updated: {formatDateTime(refreshSettings.lastRefreshTime)}
          </span>
        )}
      </div>
    </div>
  );

  /**
   * Render the property table
   * @returns {JSX.Element} - The property table component
   */
  const renderPropertyTable = () => (
    <div className="bg-white rounded-lg shadow-md p-4 mb-6 overflow-x-auto">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Property Valuations</h2>
      
      {loading ? (
        <div className="flex justify-center items-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          <span className="ml-3">Loading property data...</span>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <p>{error}</p>
          <button
            onClick={fetchPropertyValuations}
            className="mt-2 px-3 py-1 bg-red-600 text-white rounded-md hover:bg-red-700 text-sm"
          >
            Retry
          </button>
        </div>
      ) : properties.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <p>No properties found matching your criteria.</p>
          <button
            onClick={resetFilters}
            className="mt-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
          >
            Reset Filters
          </button>
        </div>
      ) : (
        <>
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th 
                  scope="col" 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                  onClick={() => handleSortColumn('property_id')}
                >
                  <div className="flex items-center">
                    ID
                    {filters.sortBy === 'property_id' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
                <th 
                  scope="col" 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                  onClick={() => handleSortColumn('address')}
                >
                  <div className="flex items-center">
                    Address
                    {filters.sortBy === 'address' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
                <th 
                  scope="col" 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                  onClick={() => handleSortColumn('estimated_value')}
                >
                  <div className="flex items-center">
                    Estimated Value
                    {filters.sortBy === 'estimated_value' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
                <th 
                  scope="col" 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                  onClick={() => handleSortColumn('confidence')}
                >
                  <div className="flex items-center">
                    Confidence
                    {filters.sortBy === 'confidence' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
                <th 
                  scope="col" 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                  onClick={() => handleSortColumn('valuation_date')}
                >
                  <div className="flex items-center">
                    Date
                    {filters.sortBy === 'valuation_date' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
                <th 
                  scope="col" 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                  onClick={() => handleSortColumn('valuation_model')}
                >
                  <div className="flex items-center">
                    Model
                    {filters.sortBy === 'valuation_model' && (
                      <span className="ml-1">
                        {filters.sortDirection === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {properties.map((property) => (
                <tr 
                  key={property.id || property.property_id}
                  onClick={() => handlePropertyClick(property)}
                  className="hover:bg-gray-100 cursor-pointer transition duration-150"
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {property.id || property.property_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {property.address}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {formatCurrency(property.estimated_value || property.value)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <div className="flex items-center">
                      <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                        <div
                          className="bg-blue-600 h-2.5 rounded-full"
                          style={{ width: `${(property.confidence || 0) * 100}%` }}
                        ></div>
                      </div>
                      <span>{formatPercentage(property.confidence)}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {formatDate(property.valuation_date)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <span className="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {property.valuation_model || 'Unknown'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {/* Pagination */}
          <div className="flex items-center justify-between mt-4">
            <div className="flex items-center">
              <span className="text-sm text-gray-700">
                Showing {((pagination.currentPage - 1) * pagination.itemsPerPage) + 1} to {Math.min(pagination.currentPage * pagination.itemsPerPage, pagination.totalItems)} of {pagination.totalItems} properties
              </span>
              
              <div className="ml-4">
                <label htmlFor="itemsPerPage" className="mr-2 text-sm text-gray-700">Per page:</label>
                <select
                  id="itemsPerPage"
                  value={pagination.itemsPerPage}
                  onChange={handleItemsPerPageChange}
                  className="p-1 border border-gray-300 rounded-md text-sm"
                >
                  <option value={10}>10</option>
                  <option value={25}>25</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                </select>
              </div>
            </div>
            
            <div>
              <nav className="flex items-center">
                <button
                  onClick={() => handlePageChange(1)}
                  disabled={pagination.currentPage === 1}
                  className="px-3 py-1 rounded-md text-sm font-medium text-gray-700 bg-gray-200 hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed mr-1"
                >
                  First
                </button>
                <button
                  onClick={() => handlePageChange(pagination.currentPage - 1)}
                  disabled={pagination.currentPage === 1}
                  className="px-3 py-1 rounded-md text-sm font-medium text-gray-700 bg-gray-200 hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed mr-1"
                >
                  Previous
                </button>
                
                <span className="mx-2 text-sm text-gray-700">
                  Page {pagination.currentPage} of {pagination.totalPages}
                </span>
                
                <button
                  onClick={() => handlePageChange(pagination.currentPage + 1)}
                  disabled={pagination.currentPage === pagination.totalPages}
                  className="px-3 py-1 rounded-md text-sm font-medium text-gray-700 bg-gray-200 hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed ml-1"
                >
                  Next
                </button>
                <button
                  onClick={() => handlePageChange(pagination.totalPages)}
                  disabled={pagination.currentPage === pagination.totalPages}
                  className="px-3 py-1 rounded-md text-sm font-medium text-gray-700 bg-gray-200 hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed ml-1"
                >
                  Last
                </button>
              </nav>
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
  const renderSummaryMetrics = () => (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {/* Total Properties */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0 bg-blue-500 rounded-full p-3">
            <svg className="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
            </svg>
          </div>
          <div className="ml-4">
            <div className="text-sm font-medium text-gray-500">Total Properties</div>
            <div className="text-xl font-semibold text-gray-900">
              {loading ? (
                <div className="animate-pulse w-16 h-6 bg-gray-200 rounded"></div>
              ) : (
                pagination.totalItems.toLocaleString()
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Average Value */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0 bg-green-500 rounded-full p-3">
            <svg className="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="ml-4">
            <div className="text-sm font-medium text-gray-500">Average Value</div>
            <div className="text-xl font-semibold text-gray-900">
              {loading ? (
                <div className="animate-pulse w-24 h-6 bg-gray-200 rounded"></div>
              ) : (
                formatCurrency(
                  properties.length > 0
                    ? properties.reduce((sum, property) => sum + (property.estimated_value || property.value || 0), 0) / properties.length
                    : 0
                )
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Average Confidence */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0 bg-yellow-500 rounded-full p-3">
            <svg className="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div className="ml-4">
            <div className="text-sm font-medium text-gray-500">Average Confidence</div>
            <div className="text-xl font-semibold text-gray-900">
              {loading ? (
                <div className="animate-pulse w-16 h-6 bg-gray-200 rounded"></div>
              ) : (
                formatPercentage(
                  properties.length > 0
                    ? properties.reduce((sum, property) => sum + (property.confidence || 0), 0) / properties.length
                    : 0
                )
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Neighborhoods */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center">
          <div className="flex-shrink-0 bg-purple-500 rounded-full p-3">
            <svg className="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <div className="ml-4">
            <div className="text-sm font-medium text-gray-500">Neighborhoods</div>
            <div className="text-xl font-semibold text-gray-900">
              {loading ? (
                <div className="animate-pulse w-16 h-6 bg-gray-200 rounded"></div>
              ) : (
                availableNeighborhoods.length
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  /**
   * Render property data visualization charts
   * @returns {JSX.Element} - The charts component
   */
  const renderCharts = () => (
    <div className="mb-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Data Visualization</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Value Distribution Chart */}
        <div className="bg-white rounded-lg shadow-md p-4 chart-container" id="value-distribution-chart">
          <h3 className="text-md font-semibold text-gray-700 mb-2">Value Distribution</h3>
          {loading ? (
            <div className="animate-pulse flex flex-col items-center justify-center h-64 bg-gray-100 rounded">
              <div className="w-12 h-12 rounded-full bg-gray-300"></div>
              <div className="mt-2 w-24 h-4 bg-gray-300 rounded"></div>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-64 bg-red-50 rounded">
              <p className="text-red-500">Error loading chart data</p>
            </div>
          ) : (
            <Bar 
              data={valueDistributionChartData} 
              options={{
                responsive: true,
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
                        return `Count: ${context.raw}`;
                      }
                    }
                  }
                },
                scales: {
                  x: {
                    title: {
                      display: true,
                      text: 'Property Value'
                    }
                  },
                  y: {
                    title: {
                      display: true,
                      text: 'Number of Properties'
                    },
                    beginAtZero: true
                  }
                }
              }}
              ref={valueDistributionChartRef}
            />
          )}
        </div>
        
        {/* Neighborhood Comparison Chart */}
        <div className="bg-white rounded-lg shadow-md p-4 chart-container" id="neighborhood-chart">
          <h3 className="text-md font-semibold text-gray-700 mb-2">Neighborhood Comparison</h3>
          {loading ? (
            <div className="animate-pulse flex flex-col items-center justify-center h-64 bg-gray-100 rounded">
              <div className="w-12 h-12 rounded-full bg-gray-300"></div>
              <div className="mt-2 w-24 h-4 bg-gray-300 rounded"></div>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-64 bg-red-50 rounded">
              <p className="text-red-500">Error loading chart data</p>
            </div>
          ) : (
            <Bar 
              data={neighborhoodChartData} 
              options={neighborhoodChartOptions}
              ref={neighborhoodChartRef}
            />
          )}
        </div>
        
        {/* Valuation Trend Chart */}
        <div className="bg-white rounded-lg shadow-md p-4 chart-container" id="trend-chart">
          <h3 className="text-md font-semibold text-gray-700 mb-2">Valuation Trends</h3>
          {loading ? (
            <div className="animate-pulse flex flex-col items-center justify-center h-64 bg-gray-100 rounded">
              <div className="w-12 h-12 rounded-full bg-gray-300"></div>
              <div className="mt-2 w-24 h-4 bg-gray-300 rounded"></div>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-64 bg-red-50 rounded">
              <p className="text-red-500">Error loading chart data</p>
            </div>
          ) : (
            <Line 
              data={valueTrendChartData} 
              options={valueTrendChartOptions}
              ref={trendChartRef}
            />
          )}
        </div>
        
        {/* Model Distribution Chart */}
        <div className="bg-white rounded-lg shadow-md p-4 chart-container" id="model-distribution-chart">
          <h3 className="text-md font-semibold text-gray-700 mb-2">Model Distribution</h3>
          {loading ? (
            <div className="animate-pulse flex flex-col items-center justify-center h-64 bg-gray-100 rounded">
              <div className="w-12 h-12 rounded-full bg-gray-300"></div>
              <div className="mt-2 w-24 h-4 bg-gray-300 rounded"></div>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-64 bg-red-50 rounded">
              <p className="text-red-500">Error loading chart data</p>
            </div>
          ) : (
            <Pie 
              data={modelDistributionChartData} 
              options={modelDistributionChartOptions}
            />
          )}
        </div>
      </div>
    </div>
  );

  /**
   * Render ETL status section
   * @returns {JSX.Element} - The ETL status component
   */
  const renderEtlStatus = () => (
    <div className="bg-white rounded-lg shadow-md p-4 mb-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">ETL Pipeline Status</h2>
      
      {etlStatus.isLoading ? (
        <div className="flex justify-center items-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          <span className="ml-3">Loading ETL status...</span>
        </div>
      ) : etlStatus.error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <p>{etlStatus.error}</p>
          <button
            onClick={fetchEtlStatus}
            className="mt-2 px-3 py-1 bg-red-600 text-white rounded-md hover:bg-red-700 text-sm"
          >
            Retry
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* ETL Status Summary */}
          <div className="md:col-span-2">
            <div className="mb-4">
              <div className="flex items-center mb-2">
                <h3 className="text-md font-semibold text-gray-700">Current Status</h3>
                <span className={`ml-2 px-2 py-1 rounded-full text-xs font-medium ${
                  etlStatus.status.toLowerCase() === 'completed' ? 'bg-green-100 text-green-800' :
                  etlStatus.status.toLowerCase() === 'running' ? 'bg-blue-100 text-blue-800' :
                  etlStatus.status.toLowerCase() === 'failed' ? 'bg-red-100 text-red-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {etlStatus.status}
                </span>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Last Update</p>
                  <p className="text-base font-medium text-gray-900">
                    {etlStatus.lastUpdate ? formatDateTime(etlStatus.lastUpdate) : 'N/A'}
                  </p>
                </div>
                
                <div>
                  <p className="text-sm text-gray-500">Progress</p>
                  <div className="flex items-center">
                    <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                      <div
                        className={`h-2.5 rounded-full ${
                          etlStatus.status.toLowerCase() === 'completed' ? 'bg-green-500' :
                          etlStatus.status.toLowerCase() === 'running' ? 'bg-blue-500' :
                          etlStatus.status.toLowerCase() === 'failed' ? 'bg-red-500' :
                          'bg-gray-500'
                        }`}
                        style={{ width: `${etlStatus.progress * 100}%` }}
                      ></div>
                    </div>
                    <span>{(etlStatus.progress * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="mb-4">
              <h3 className="text-md font-semibold text-gray-700 mb-2">Processing Metrics</h3>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Records Processed</p>
                  <p className="text-base font-medium text-gray-900">
                    {etlStatus.metrics.recordsProcessed.toLocaleString()}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    {etlStatus.status.toLowerCase() === 'running' && 
                      `~${Math.round(etlStatus.metrics.recordsProcessed / 
                        (((new Date()) - new Date(etlStatus.lastUpdate)) / 60000) || 1)} records/min`
                    }
                  </p>
                </div>
                
                <div>
                  <p className="text-sm text-gray-500">Success Rate</p>
                  <p className="text-base font-medium text-gray-900">
                    {formatPercentage(etlStatus.metrics.successRate)}
                  </p>
                  <p className={`text-xs ${etlStatus.metrics.successRate < 0.9 ? 'text-yellow-500' : 
                    etlStatus.metrics.successRate < 0.8 ? 'text-red-500' : 'text-green-500'} mt-1`}>
                    {etlStatus.metrics.successRate < 0.9 ? 'Needs Attention' : 
                     etlStatus.metrics.successRate < 0.8 ? 'Critical' : 'Good'}
                  </p>
                </div>
                
                <div>
                  <p className="text-sm text-gray-500">Avg. Processing Time</p>
                  <p className="text-base font-medium text-gray-900">
                    {etlStatus.metrics.averageProcessingTime.toFixed(2)}ms
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    {etlStatus.metrics.averageProcessingTime > 0 && 
                      `Est. completion: ${formatRelativeTime(
                        new Date(Date.now() + (100 - etlStatus.progress * 100) / 
                          (etlStatus.progress * 100) * 
                          ((new Date()) - (etlStatus.lastUpdate ? new Date(etlStatus.lastUpdate) : new Date())))
                      )}`
                    }
                  </p>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-semibold text-gray-700 mb-2">Data Quality</h3>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Completeness</p>
                  <div className="flex items-center">
                    <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                      <div
                        className="bg-green-500 h-2.5 rounded-full"
                        style={{ width: `${etlStatus.dataQuality.completeness * 100}%` }}
                      ></div>
                    </div>
                    <span>{formatPercentage(etlStatus.dataQuality.completeness)}</span>
                  </div>
                </div>
                
                <div>
                  <p className="text-sm text-gray-500">Accuracy</p>
                  <div className="flex items-center">
                    <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                      <div
                        className="bg-green-500 h-2.5 rounded-full"
                        style={{ width: `${etlStatus.dataQuality.accuracy * 100}%` }}
                      ></div>
                    </div>
                    <span>{formatPercentage(etlStatus.dataQuality.accuracy)}</span>
                  </div>
                </div>
                
                <div>
                  <p className="text-sm text-gray-500">Timeliness</p>
                  <div className="flex items-center">
                    <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                      <div
                        className="bg-green-500 h-2.5 rounded-full"
                        style={{ width: `${etlStatus.dataQuality.timeliness * 100}%` }}
                      ></div>
                    </div>
                    <span>{formatPercentage(etlStatus.dataQuality.timeliness)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* ETL Progress Chart */}
          <div className="chart-container flex flex-col items-center justify-center" id="etl-progress-chart">
            <Doughnut
              data={{
                labels: ['Complete', 'Remaining'],
                datasets: [{
                  data: [etlStatus.progress * 100, 100 - (etlStatus.progress * 100)],
                  backgroundColor: [
                    etlStatus.status.toLowerCase() === 'completed' ? 'rgba(16, 185, 129, 0.6)' :
                    etlStatus.status.toLowerCase() === 'running' ? 'rgba(59, 130, 246, 0.6)' :
                    etlStatus.status.toLowerCase() === 'failed' ? 'rgba(239, 68, 68, 0.6)' :
                    'rgba(107, 114, 128, 0.6)',
                    'rgba(229, 231, 235, 0.6)'
                  ],
                  borderColor: [
                    etlStatus.status.toLowerCase() === 'completed' ? 'rgba(16, 185, 129, 1)' :
                    etlStatus.status.toLowerCase() === 'running' ? 'rgba(59, 130, 246, 1)' :
                    etlStatus.status.toLowerCase() === 'failed' ? 'rgba(239, 68, 68, 1)' :
                    'rgba(107, 114, 128, 1)',
                    'rgba(229, 231, 235, 1)'
                  ],
                  borderWidth: 1
                }]
              }}
              options={etlProgressChartOptions}
              ref={etlProgressChartRef}
            />
            <div className="text-center mt-2">
              <p className="text-xl font-bold">{(etlStatus.progress * 100).toFixed(0)}%</p>
              <p className="text-sm text-gray-500">ETL Pipeline Progress</p>
            </div>
          </div>
        </div>
      )}
      
      {/* ETL Data Sources */}
      {!etlStatus.isLoading && !etlStatus.error && etlStatus.sources.length > 0 && (
        <div className="mt-6">
          <h3 className="text-md font-semibold text-gray-700 mb-2">Data Sources</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Source
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Records
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Last Update
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {etlStatus.sources.map((source, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {source.name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColorClass(source.status)} bg-opacity-20`}>
                        {source.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {source.records.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatDateTime(source.last_update)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
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
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Agent Status</h2>
      
      {agentStatus.isLoading ? (
        <div className="flex justify-center items-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          <span className="ml-3">Loading agent status...</span>
        </div>
      ) : agentStatus.error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <p>{agentStatus.error}</p>
          <button
            onClick={fetchAgentStatus}
            className="mt-2 px-3 py-1 bg-red-600 text-white rounded-md hover:bg-red-700 text-sm"
          >
            Retry
          </button>
        </div>
      ) : agentStatus.agents.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <p>No agent data available.</p>
          <button
            onClick={fetchAgentStatus}
            className="mt-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
          >
            Refresh
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Agent Status Summary */}
          <div className="md:col-span-2">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Agent
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Success Rate
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Last Activity
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {agentStatus.agents.map((agent) => (
                    <tr 
                      key={agent.id}
                      onClick={() => handleAgentClick(agent)}
                      className="hover:bg-gray-100 cursor-pointer transition duration-150"
                    >
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {agent.name}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        <span className="px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                          {agent.type}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          agent.status === 'active' ? 'bg-green-100 text-green-800' :
                          agent.status === 'idle' ? 'bg-blue-100 text-blue-800' :
                          agent.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                          agent.status === 'error' ? 'bg-red-100 text-red-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {agent.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        <div className="flex items-center">
                          <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                            <div
                              className={`h-2.5 rounded-full ${
                                (agent.success_rate || 0) > 0.9 ? 'bg-green-500' :
                                (agent.success_rate || 0) > 0.8 ? 'bg-yellow-500' :
                                'bg-red-500'
                              }`}
                              style={{ width: `${(agent.success_rate || 0) * 100}%` }}
                            ></div>
                          </div>
                          <span>{formatPercentage(agent.success_rate)}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatDateTime(agent.last_activity)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          {/* Agent Status Chart and Summary */}
          <div className="flex flex-col">
            <div className="chart-container mb-4" id="agent-performance-chart">
              <h3 className="text-md font-semibold text-gray-700 mb-2">Agent Status Distribution</h3>
              <Doughnut
                data={agentStatusChartData}
                options={agentStatusChartOptions}
                ref={agentPerformanceChartRef}
              />
            </div>
            
            {/* Agent Summary Stats */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-md font-semibold text-gray-700 mb-2">Agent Health Summary</h3>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm">
                    <span>Active Agents</span>
                    <span className="font-medium">{
                      agentStatus.agents.filter(a => a.status === 'active').length
                    }/{agentStatus.agents.length}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
                    <div
                      className="bg-green-500 h-2.5 rounded-full"
                      style={{ 
                        width: `${(agentStatus.agents.filter(a => a.status === 'active').length / 
                          (agentStatus.agents.length || 1)) * 100}%` 
                      }}
                    ></div>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between text-sm">
                    <span>Average Success Rate</span>
                    <span className="font-medium">{
                      formatPercentage(
                        agentStatus.agents.reduce((sum, agent) => sum + (agent.success_rate || 0), 0) / 
                        (agentStatus.agents.length || 1)
                      )
                    }</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
                    <div
                      className={`h-2.5 rounded-full ${
                        (agentStatus.agents.reduce((sum, agent) => sum + (agent.success_rate || 0), 0) / 
                          (agentStatus.agents.length || 1)) > 0.9 ? 'bg-green-500' :
                        (agentStatus.agents.reduce((sum, agent) => sum + (agent.success_rate || 0), 0) / 
                          (agentStatus.agents.length || 1)) > 0.8 ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}
                      style={{ 
                        width: `${(agentStatus.agents.reduce((sum, agent) => sum + (agent.success_rate || 0), 0) / 
                          (agentStatus.agents.length || 1)) * 100}%` 
                      }}
                    ></div>
                  </div>
                </div>
                
                <div className="pt-2">
                  <p className="text-xs text-gray-500">Last Updated: {formatDateTime(agentStatus.lastUpdate)}</p>
                  <p className="text-xs text-gray-500">
                    {agentStatus.agents.filter(a => a.status === 'error').length > 0 && 
                      `${agentStatus.agents.filter(a => a.status === 'error').length} agents with errors detected`}
                  </p>
                </div>
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
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
          <div className="p-6">
            <div className="flex justify-between items-start">
              <h2 className="text-xl font-semibold text-gray-800">Property Details</h2>
              <button
                onClick={closePropertyDetail}
                className="text-gray-500 hover:text-gray-700 focus:outline-none"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Property Info */}
              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-2">Basic Information</h3>
                <div className="space-y-2">
                  <p><span className="font-medium">ID:</span> {selectedProperty.id || selectedProperty.property_id}</p>
                  <p><span className="font-medium">Address:</span> {selectedProperty.address}</p>
                  <p><span className="font-medium">Neighborhood:</span> {selectedProperty.neighborhood || extractNeighborhoodFromAddress(selectedProperty.address) || 'N/A'}</p>
                  <p><span className="font-medium">Property Type:</span> {selectedProperty.property_type?.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) || 'N/A'}</p>
                  <p><span className="font-medium">Living Area:</span> {selectedProperty.living_area ? `${selectedProperty.living_area.toLocaleString()} sq ft` : 'N/A'}</p>
                  <p><span className="font-medium">Bedrooms:</span> {selectedProperty.bedrooms || 'N/A'}</p>
                  <p><span className="font-medium">Bathrooms:</span> {selectedProperty.bathrooms || 'N/A'}</p>
                  <p><span className="font-medium">Year Built:</span> {selectedProperty.year_built || 'N/A'}</p>
                  <p><span className="font-medium">Lot Size:</span> {selectedProperty.lot_size ? `${selectedProperty.lot_size.toLocaleString()} sq ft` : 'N/A'}</p>
                </div>
              </div>
              
              {/* Valuation Info */}
              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-2">Valuation Details</h3>
                <div className="space-y-2">
                  <div className="mb-4">
                    <p className="text-sm text-gray-500">Estimated Value</p>
                    <p className="text-2xl font-bold text-blue-600">
                      {formatCurrency(selectedProperty.estimated_value || selectedProperty.value)}
                    </p>
                  </div>
                  
                  <div className="mb-4">
                    <p className="text-sm text-gray-500">Confidence Score</p>
                    <div className="flex items-center">
                      <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                        <div
                          className="bg-blue-600 h-2.5 rounded-full"
                          style={{ width: `${(selectedProperty.confidence || 0) * 100}%` }}
                        ></div>
                      </div>
                      <span>{formatPercentage(selectedProperty.confidence)}</span>
                    </div>
                  </div>
                  
                  <p><span className="font-medium">Valuation Date:</span> {formatDate(selectedProperty.valuation_date) || 'N/A'}</p>
                  <p><span className="font-medium">Valuation Model:</span> {selectedProperty.valuation_model || 'N/A'}</p>
                  
                  {selectedProperty.price_range && (
                    <p><span className="font-medium">Price Range:</span> {formatCurrency(selectedProperty.price_range.min)} - {formatCurrency(selectedProperty.price_range.max)}</p>
                  )}
                  
                  {/* Comparable properties if available */}
                  {selectedProperty.comparables && selectedProperty.comparables.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-md font-medium text-gray-700 mb-2">Comparable Properties</h4>
                      <ul className="space-y-2">
                        {selectedProperty.comparables.map((comp, idx) => (
                          <li key={idx} className="text-sm">
                            {comp.address}: {formatCurrency(comp.value)} ({formatDate(comp.sale_date)})
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {/* Feature importance if available */}
                  {selectedProperty.feature_importance && Object.keys(selectedProperty.feature_importance).length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-md font-medium text-gray-700 mb-2">Feature Importance</h4>
                      <div className="space-y-2">
                        {Object.entries(selectedProperty.feature_importance)
                          .sort(([, a], [, b]) => b - a)
                          .map(([feature, importance]) => (
                            <div key={feature} className="flex items-center">
                              <span className="w-1/3 text-sm">{feature.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                              <div className="w-2/3 flex items-center">
                                <div className="w-full bg-gray-200 rounded-full h-2 mr-2">
                                  <div
                                    className="bg-purple-500 h-2 rounded-full"
                                    style={{ width: `${importance * 100}%` }}
                                  ></div>
                                </div>
                                <span className="text-xs">{(importance * 100).toFixed(1)}%</span>
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            {/* Additional notes if available */}
            {selectedProperty.notes && (
              <div className="mt-6">
                <h3 className="text-lg font-medium text-gray-800 mb-2">Notes</h3>
                <p className="text-gray-700">{selectedProperty.notes}</p>
              </div>
            )}
            
            <div className="mt-6 flex justify-end">
              <button
                onClick={closePropertyDetail}
                className="px-4 py-2 bg-gray-300 text-gray-800 rounded-md hover:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50"
              >
                Close
              </button>
            </div>
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
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
          <div className="p-6">
            <div className="flex justify-between items-start">
              <h2 className="text-xl font-semibold text-gray-800">Agent Details</h2>
              <button
                onClick={closeAgentDetail}
                className="text-gray-500 hover:text-gray-700 focus:outline-none"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            {selectedAgent.isLoading ? (
              <div className="flex justify-center items-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
                <span className="ml-3">Loading agent details...</span>
              </div>
            ) : selectedAgent.error ? (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded my-4">
                <p>{selectedAgent.error}</p>
                <button
                  onClick={() => fetchAgentDetails(selectedAgent.id)}
                  className="mt-2 px-3 py-1 bg-red-600 text-white rounded-md hover:bg-red-700 text-sm"
                >
                  Retry
                </button>
              </div>
            ) : (
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Agent Info */}
                <div>
                  <h3 className="text-lg font-medium text-gray-800 mb-2">Basic Information</h3>
                  <div className="space-y-2">
                    <p><span className="font-medium">ID:</span> {selectedAgent.id}</p>
                    <p><span className="font-medium">Name:</span> {selectedAgent.name}</p>
                    <p><span className="font-medium">Type:</span> {selectedAgent.type}</p>
                    <p>
                      <span className="font-medium">Status:</span>
                      <span className={`ml-2 px-2 py-1 rounded-full text-xs font-medium ${
                        selectedAgent.status === 'active' ? 'bg-green-100 text-green-800' :
                        selectedAgent.status === 'idle' ? 'bg-blue-100 text-blue-800' :
                        selectedAgent.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                        selectedAgent.status === 'error' ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {selectedAgent.status}
                      </span>
                    </p>
                    <p><span className="font-medium">Version:</span> {selectedAgent.version || 'N/A'}</p>
                    <p><span className="font-medium">Last Activity:</span> {formatDateTime(selectedAgent.last_activity) || 'N/A'}</p>
                    <p><span className="font-medium">Success Rate:</span> {formatPercentage(selectedAgent.success_rate)}</p>
                    <p><span className="font-medium">Uptime:</span> {selectedAgent.uptime ? `${(selectedAgent.uptime / 3600).toFixed(1)} hours` : 'N/A'}</p>
                  </div>
                </div>
                
                {/* Performance Metrics */}
                <div>
                  <h3 className="text-lg font-medium text-gray-800 mb-2">Performance Metrics</h3>
                  
                  {selectedAgent.metrics ? (
                    <div className="space-y-4">
                      <div>
                        <p className="text-sm text-gray-500">Requests Processed</p>
                        <p className="text-xl font-medium text-gray-900">
                          {selectedAgent.metrics.requests_processed?.toLocaleString() || 0}
                        </p>
                      </div>
                      
                      <div>
                        <p className="text-sm text-gray-500">Average Response Time</p>
                        <p className="text-xl font-medium text-gray-900">
                          {selectedAgent.metrics.avg_response_time?.toFixed(2) || 0} ms
                        </p>
                      </div>
                      
                      <div>
                        <p className="text-sm text-gray-500">Error Rate</p>
                        <div className="flex items-center">
                          <div className="w-full bg-gray-200 rounded-full h-2.5 mr-2">
                            <div
                              className={`h-2.5 rounded-full ${
                                (selectedAgent.metrics.error_rate || 0) < 0.01 ? 'bg-green-500' :
                                (selectedAgent.metrics.error_rate || 0) < 0.05 ? 'bg-yellow-500' :
                                'bg-red-500'
                              }`}
                              style={{ width: `${(selectedAgent.metrics.error_rate || 0) * 100}%` }}
                            ></div>
                          </div>
                          <span>{formatPercentage(selectedAgent.metrics.error_rate)}</span>
                        </div>
                      </div>
                      
                      <div>
                        <p className="text-sm text-gray-500">Resource Utilization</p>
                        <div className="grid grid-cols-2 gap-2 mt-1">
                          <div>
                            <p className="text-xs text-gray-500">CPU</p>
                            <div className="flex items-center">
                              <div className="w-full bg-gray-200 rounded-full h-2 mr-2">
                                <div
                                  className={`h-2 rounded-full ${
                                    (selectedAgent.metrics.cpu_utilization || 0) < 50 ? 'bg-green-500' :
                                    (selectedAgent.metrics.cpu_utilization || 0) < 80 ? 'bg-yellow-500' :
                                    'bg-red-500'
                                  }`}
                                  style={{ width: `${selectedAgent.metrics.cpu_utilization || 0}%` }}
                                ></div>
                              </div>
                              <span className="text-xs">{selectedAgent.metrics.cpu_utilization || 0}%</span>
                            </div>
                          </div>
                          
                          <div>
                            <p className="text-xs text-gray-500">Memory</p>
                            <div className="flex items-center">
                              <div className="w-full bg-gray-200 rounded-full h-2 mr-2">
                                <div
                                  className={`h-2 rounded-full ${
                                    (selectedAgent.metrics.memory_utilization || 0) < 50 ? 'bg-green-500' :
                                    (selectedAgent.metrics.memory_utilization || 0) < 80 ? 'bg-yellow-500' :
                                    'bg-red-500'
                                  }`}
                                  style={{ width: `${selectedAgent.metrics.memory_utilization || 0}%` }}
                                ></div>
                              </div>
                              <span className="text-xs">{selectedAgent.metrics.memory_utilization || 0}%</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-gray-500">No performance metrics available for this agent.</p>
                  )}
                </div>
              </div>
            )}
            
            {/* Agent Logs */}
            {selectedAgent && selectedAgent.logs && selectedAgent.logs.length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-medium text-gray-800 mb-2">Recent Logs</h3>
                <div className="bg-gray-900 text-gray-200 p-4 rounded-md h-64 overflow-y-auto font-mono text-sm">
                  {selectedAgent.logs.map((log, idx) => (
                    <div key={idx} className={`mb-1 ${
                      log.level === 'error' ? 'text-red-400' :
                      log.level === 'warn' ? 'text-yellow-400' :
                      log.level === 'info' ? 'text-blue-400' :
                      'text-gray-400'
                    }`}>
                      <span className="mr-2">[{formatDateTime(log.timestamp)}]</span>
                      <span className="mr-2">[{log.level.toUpperCase()}]</span>
                      <span>{log.message}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            <div className="mt-6 flex justify-end">
              <button
                onClick={() => fetchAgentDetails(selectedAgent.id)}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 mr-2"
              >
                Refresh
              </button>
              <button
                onClick={closeAgentDetail}
                className="px-4 py-2 bg-gray-300 text-gray-800 rounded-md hover:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50"
              >
                Close
              </button>
            </div>
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
    <div className="mb-6">
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => handleDashboardTabChange('properties')}
            className={`pb-4 px-1 ${
              activeDashboardTab === 'properties'
                ? 'border-b-2 border-blue-500 text-blue-600 font-medium'
                : 'border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } focus:outline-none transition duration-150 ease-in-out`}
          >
            Properties
          </button>
          <button
            onClick={() => handleDashboardTabChange('etl')}
            className={`pb-4 px-1 ${
              activeDashboardTab === 'etl'
                ? 'border-b-2 border-blue-500 text-blue-600 font-medium'
                : 'border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } focus:outline-none transition duration-150 ease-in-out`}
          >
            ETL Pipeline
          </button>
          <button
            onClick={() => handleDashboardTabChange('agents')}
            className={`pb-4 px-1 ${
              activeDashboardTab === 'agents'
                ? 'border-b-2 border-blue-500 text-blue-600 font-medium'
                : 'border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } focus:outline-none transition duration-150 ease-in-out`}
          >
            Agents
          </button>
        </nav>
      </div>
    </div>
  );

  // Main component render
  return (
    <div className="min-h-screen bg-gray-100 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-2xl md:text-3xl font-bold text-gray-900">BCBS Property Valuation Dashboard</h1>
          <p className="mt-2 text-sm text-gray-600">
            Real-time property valuation metrics and analysis
          </p>
        </header>
        
        {/* Filter section for all views */}
        {renderFilterSection()}
        
        {/* Dashboard tabs */}
        {renderDashboardTabs()}
        
        {/* Dashboard content based on active tab */}
        {activeDashboardTab === 'properties' && (
          <>
            {renderSummaryMetrics()}
            {renderCharts()}
            {renderPropertyTable()}
          </>
        )}
        
        {activeDashboardTab === 'etl' && renderEtlStatus()}
        
        {activeDashboardTab === 'agents' && renderAgentStatus()}
        
        {/* Modals */}
        {renderPropertyDetailModal()}
        {renderAgentDetailModal()}
      </div>
    </div>
  );
};

export default Dashboard;