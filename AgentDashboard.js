import React, { useState, useEffect, useCallback } from 'react';

/**
 * AgentDashboard Component
 * 
 * This component displays real-time status of the BS Army of Agents,
 * fetching data from the /api/agent-status endpoint and providing 
 * visual indicators of each agent's current state.
 * 
 * Enhancements include:
 * - Interactive agent selection to view detailed metrics and logs
 * - Visual status indicators with color-coding and icons
 * - Real-time data fetching with manual refresh capability
 * - Responsive design using Tailwind CSS
 */
const AgentDashboard = () => {
  // State to store agent data
  const [agentData, setAgentData] = useState(null);
  // State to track loading status
  const [loading, setLoading] = useState(true);
  // State to track any errors
  const [error, setError] = useState(null);
  // State to track last refresh time
  const [lastRefresh, setLastRefresh] = useState(null);
  // State to track selected agent for detailed view
  const [selectedAgent, setSelectedAgent] = useState(null);
  // State to store historical logs for the selected agent
  const [agentLogs, setAgentLogs] = useState([]);
  // State to track if logs are loading
  const [loadingLogs, setLoadingLogs] = useState(false);

  /**
   * Function to fetch agent status data from the API
   * This function requests real-time data from the /api/agent-status endpoint
   */
  const fetchAgentStatus = useCallback(async () => {
    setLoading(true);
    try {
      // Fetch agent status data with a timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      const response = await fetch('/api/agent-status', {
        signal: controller.signal,
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      setAgentData(data);
      setLastRefresh(new Date());
      setError(null);
      
      // If we have a selected agent, update its data
      if (selectedAgent) {
        const updatedAgent = data.agents.find(agent => agent.agent_id === selectedAgent.agent_id);
        if (updatedAgent) {
          setSelectedAgent(updatedAgent);
        }
      }
    } catch (err) {
      console.error('Error fetching agent status:', err);
      setError(`Failed to fetch agent status: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedAgent]);

  /**
   * Function to fetch historical logs for a specific agent
   * @param {string} agentId - The ID of the agent to fetch logs for
   */
  const fetchAgentLogs = async (agentId) => {
    setLoadingLogs(true);
    try {
      const response = await fetch(`/api/agent-logs/${agentId}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      setAgentLogs(data.logs || []);
    } catch (err) {
      console.error(`Error fetching logs for agent ${agentId}:`, err);
      setAgentLogs([{ 
        timestamp: new Date().toISOString(),
        level: 'error',
        message: `Failed to load logs: ${err.message}`
      }]);
    } finally {
      setLoadingLogs(false);
    }
  };

  /**
   * Handler for selecting an agent to view detailed information
   * @param {Object} agent - The agent object to select
   */
  const handleAgentSelect = (agent) => {
    if (selectedAgent && selectedAgent.agent_id === agent.agent_id) {
      // If user clicks on the already selected agent, deselect it
      setSelectedAgent(null);
      setAgentLogs([]);
    } else {
      setSelectedAgent(agent);
      // Fetch logs for the selected agent
      fetchAgentLogs(agent.agent_id);
    }
  };

  // Fetch data on component mount
  useEffect(() => {
    fetchAgentStatus();
    
    // Set up polling interval (every 30 seconds)
    const intervalId = setInterval(fetchAgentStatus, 30000);
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, [fetchAgentStatus]);

  /**
   * Helper function to determine status color based on agent status
   * @param {string} status - The agent's status
   * @returns {string} - Tailwind CSS color class
   */
  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'active':
        return 'bg-green-500';
      case 'idle':
        return 'bg-blue-500';
      case 'busy':
        return 'bg-yellow-500';
      case 'error':
        return 'bg-red-500';
      case 'offline':
        return 'bg-gray-500';
      default:
        return 'bg-gray-300';
    }
  };

  /**
   * Helper function to render a status icon based on agent status
   * @param {string} status - The agent's status
   * @returns {JSX.Element} - SVG icon element
   */
  const getStatusIcon = (status) => {
    switch (status.toLowerCase()) {
      case 'active':
        return (
          <svg className="h-5 w-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        );
      case 'idle':
        return (
          <svg className="h-5 w-5 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
          </svg>
        );
      case 'busy':
        return (
          <svg className="h-5 w-5 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-13a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V5z" clipRule="evenodd" />
          </svg>
        );
      case 'error':
        return (
          <svg className="h-5 w-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        );
      case 'offline':
        return (
          <svg className="h-5 w-5 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M3.707 2.293a1 1 0 00-1.414 1.414l14 14a1 1 0 001.414-1.414l-1.473-1.473A10.014 10.014 0 0019.542 10C18.268 5.943 14.478 3 10 3a9.958 9.958 0 00-4.512 1.074l-1.78-1.781zm4.261 4.26l1.514 1.515a2.003 2.003 0 012.45 2.45l1.514 1.514a4 4 0 00-5.478-5.478z" clipRule="evenodd" />
            <path d="M12.454 16.697L9.75 13.992a4 4 0 01-3.742-3.741L2.335 6.578A9.98 9.98 0 00.458 10c1.274 4.057 5.065 7 9.542 7 .847 0 1.669-.105 2.454-.303z" />
          </svg>
        );
      default:
        return (
          <svg className="h-5 w-5 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
          </svg>
        );
    }
  };

  /**
   * Helper function to format the date for display
   * @param {Date|string} date - Date to format
   * @returns {string} - Formatted date string
   */
  const formatDate = (date) => {
    if (!date) return 'Never';
    return new Date(date).toLocaleString();
  };

  /**
   * Render a visual progress bar for the queue size
   * This shows a color-coded bar representing the queue fullness
   * @param {number} current - Current queue size
   * @param {number} max - Maximum expected queue size
   * @returns {JSX.Element} - Rendered progress bar
   */
  const renderQueueProgressBar = (current, max = 10) => {
    const percentage = Math.min((current / max) * 100, 100);
    const barColor = current > max * 0.8 ? 'bg-red-500' : 
                     current > max * 0.5 ? 'bg-yellow-500' : 'bg-blue-500';
    
    return (
      <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
        <div 
          className={`${barColor} h-2.5 rounded-full transition-all duration-300`} 
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    );
  };

  /**
   * Render a radial progress chart for the success rate
   * @param {number} rate - Success rate percentage
   * @returns {JSX.Element} - Rendered radial progress chart
   */
  const renderSuccessRateChart = (rate) => {
    const circumference = 2 * Math.PI * 40; // Circle circumference (r=40)
    const offset = circumference - (rate / 100) * circumference;
    const color = rate > 90 ? 'text-green-500' : 
                 rate > 70 ? 'text-blue-500' : 
                 rate > 50 ? 'text-yellow-500' : 'text-red-500';
                 
    return (
      <div className="relative inline-flex items-center justify-center">
        <svg className="w-20 h-20 transform -rotate-90">
          <circle 
            className="text-gray-300"
            strokeWidth="5"
            stroke="currentColor"
            fill="transparent"
            r="40"
            cx="50"
            cy="50"
          />
          <circle 
            className={color}
            strokeWidth="5"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            stroke="currentColor"
            fill="transparent"
            r="40"
            cx="50"
            cy="50"
          />
        </svg>
        <span className="absolute text-xl font-semibold">{rate}%</span>
      </div>
    );
  };

  /**
   * Get CSS class for log level to colorize log entries
   * @param {string} level - Log level (info, warn, error, etc.)
   * @returns {string} - Tailwind CSS class for the log level
   */
  const getLogLevelClass = (level) => {
    switch (level.toLowerCase()) {
      case 'error':
        return 'text-red-500';
      case 'warn':
      case 'warning':
        return 'text-yellow-500';
      case 'info':
        return 'text-blue-500';
      case 'success':
        return 'text-green-500';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div className="container mx-auto p-4">
      {/* Dashboard Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-white">
          BS Army of Agents Dashboard
        </h1>
        <div className="flex items-center space-x-4">
          {/* Last refresh timestamp */}
          <span className="text-sm text-gray-600 dark:text-gray-300">
            Last updated: {lastRefresh ? formatDate(lastRefresh) : 'Never'}
          </span>
          
          {/* Manual refresh button - triggers API fetch on click */}
          <button
            onClick={fetchAgentStatus}
            disabled={loading}
            aria-label="Refresh agent data"
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors duration-300 disabled:bg-blue-300 disabled:cursor-not-allowed flex items-center"
          >
            {loading ? (
              <span className="flex items-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Refreshing...
              </span>
            ) : (
              <>
                <svg className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Refresh
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error message displayed when API fetch fails */}
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6 dark:bg-red-900 dark:text-red-200" role="alert">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      )}

      {/* System Status Summary - Shows overview metrics */}
      {agentData && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6 transition-all duration-300 hover:shadow-lg">
          <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">System Status</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            {/* System status card */}
            <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg transition-transform duration-300 hover:scale-105">
              <div className="flex items-center mb-2">
                <svg className="h-5 w-5 text-blue-600 dark:text-blue-300 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                </svg>
                <p className="text-sm text-gray-600 dark:text-gray-300">Status</p>
              </div>
              <p className="text-2xl font-bold text-blue-600 dark:text-blue-300">{agentData.system_status}</p>
            </div>
            
            {/* Active agents card */}
            <div className="bg-green-50 dark:bg-green-900 p-4 rounded-lg transition-transform duration-300 hover:scale-105">
              <div className="flex items-center mb-2">
                <svg className="h-5 w-5 text-green-600 dark:text-green-300 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3zM6 8a2 2 0 11-4 0 2 2 0 014 0zM16 18v-3a5.972 5.972 0 00-.75-2.906A3.005 3.005 0 0119 15v3h-3zM4.75 12.094A5.973 5.973 0 004 15v3H1v-3a3 3 0 013.75-2.906z" />
                </svg>
                <p className="text-sm text-gray-600 dark:text-gray-300">Active Agents</p>
              </div>
              <p className="text-2xl font-bold text-green-600 dark:text-green-300">{agentData.active_agents}</p>
            </div>
            
            {/* Tasks in progress card */}
            <div className="bg-yellow-50 dark:bg-yellow-900 p-4 rounded-lg transition-transform duration-300 hover:scale-105">
              <div className="flex items-center mb-2">
                <svg className="h-5 w-5 text-yellow-600 dark:text-yellow-300 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M3 5a2 2 0 012-2h10a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2V5zm11 1H6v8l4-2 4 2V6z" clipRule="evenodd" />
                </svg>
                <p className="text-sm text-gray-600 dark:text-gray-300">Tasks In Progress</p>
              </div>
              <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-300">{agentData.tasks_in_progress}</p>
            </div>
            
            {/* Tasks completed card */}
            <div className="bg-purple-50 dark:bg-purple-900 p-4 rounded-lg transition-transform duration-300 hover:scale-105">
              <div className="flex items-center mb-2">
                <svg className="h-5 w-5 text-purple-600 dark:text-purple-300 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clipRule="evenodd" />
                </svg>
                <p className="text-sm text-gray-600 dark:text-gray-300">Tasks Completed Today</p>
              </div>
              <p className="text-2xl font-bold text-purple-600 dark:text-purple-300">{agentData.tasks_completed_today}</p>
            </div>
          </div>
        </div>
      )}

      {/* Selected Agent Detail View - Shows when an agent is selected */}
      {selectedAgent && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6 transition-all duration-300">
          <div className="flex justify-between items-start mb-4">
            <div>
              <div className="flex items-center">
                {getStatusIcon(selectedAgent.status)}
                <h2 className="text-xl font-semibold ml-2 text-gray-800 dark:text-white">{selectedAgent.name}</h2>
              </div>
              <p className="text-sm text-gray-500 dark:text-gray-400">ID: {selectedAgent.agent_id}</p>
            </div>
            {/* Close button for the detail view */}
            <button 
              onClick={() => setSelectedAgent(null)}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-300 dark:hover:text-gray-100"
              aria-label="Close details"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Agent Stats and Metrics */}
            <div>
              <h3 className="text-lg font-medium mb-4 text-gray-700 dark:text-gray-200">Performance Metrics</h3>
              
              <div className="flex flex-col md:flex-row items-center justify-around mb-6">
                {/* Success Rate Chart */}
                <div className="text-center mb-4 md:mb-0">
                  <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">Success Rate</p>
                  {renderSuccessRateChart(selectedAgent.performance_metrics.success_rate)}
                </div>
                
                {/* Tasks Completed */}
                <div className="text-center">
                  <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">Tasks Completed</p>
                  <div className="bg-blue-100 dark:bg-blue-800 rounded-lg p-3">
                    <span className="text-2xl font-bold text-blue-600 dark:text-blue-300">
                      {selectedAgent.performance_metrics.tasks_completed}
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Additional metrics table */}
              <div className="overflow-hidden bg-gray-50 dark:bg-gray-700 rounded-lg">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-600">
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-600">
                    <tr>
                      <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">Status</td>
                      <td className="px-4 py-3 text-sm font-medium">
                        <span className={`px-2 py-1 rounded-full capitalize ${
                          selectedAgent.status.toLowerCase() === 'active' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                          selectedAgent.status.toLowerCase() === 'idle' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                          selectedAgent.status.toLowerCase() === 'busy' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                          selectedAgent.status.toLowerCase() === 'error' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                          'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                        }`}>
                          {selectedAgent.status}
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">Current Task</td>
                      <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                        {selectedAgent.current_task || 'No active task'}
                      </td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">Queue Size</td>
                      <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                        {selectedAgent.queue_size} tasks
                      </td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">Avg. Completion Time</td>
                      <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                        {selectedAgent.performance_metrics.avg_completion_time}s
                      </td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">Last Active</td>
                      <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                        {formatDate(selectedAgent.last_active)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            
            {/* Agent Logs Section */}
            <div>
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium text-gray-700 dark:text-gray-200">Recent Logs</h3>
                {/* Refresh logs button */}
                <button 
                  onClick={() => fetchAgentLogs(selectedAgent.agent_id)}
                  disabled={loadingLogs}
                  className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 disabled:text-gray-400 text-sm flex items-center"
                >
                  {loadingLogs ? (
                    <svg className="animate-spin h-4 w-4 mr-1" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  ) : (
                    <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                  )}
                  Refresh Logs
                </button>
              </div>
              
              {/* Logs display area */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 max-h-80 overflow-y-auto">
                {loadingLogs ? (
                  <div className="flex justify-center items-center p-6">
                    <svg className="animate-spin h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  </div>
                ) : agentLogs.length > 0 ? (
                  <ul className="space-y-2">
                    {agentLogs.map((log, index) => (
                      <li key={index} className="p-2 text-sm border-b border-gray-200 dark:border-gray-600 last:border-0">
                        <div className="flex justify-between">
                          <span className={`font-semibold ${getLogLevelClass(log.level)}`}>
                            {log.level.toUpperCase()}
                          </span>
                          <span className="text-gray-500 dark:text-gray-400 text-xs">
                            {formatDate(log.timestamp)}
                          </span>
                        </div>
                        <p className="mt-1 text-gray-700 dark:text-gray-300">
                          {log.message}
                        </p>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                    <svg className="mx-auto h-10 w-10 text-gray-400 dark:text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <p className="mt-2">No logs available for this agent</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Agents List - Main table showing all agents */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Agents</h2>
        
        {/* Loading state when initially fetching data */}
        {loading && !agentData && (
          <div className="flex justify-center items-center p-10">
            <svg className="animate-spin h-10 w-10 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
        )}
        
        {/* Agents table showing list of all agents */}
        {agentData && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Status
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Agent
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Current Task
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Last Active
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Queue
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Performance
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {agentData.agents.map((agent) => (
                  <tr 
                    key={agent.agent_id} 
                    className={`hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors duration-150 ${
                      selectedAgent && selectedAgent.agent_id === agent.agent_id 
                        ? 'bg-blue-50 dark:bg-blue-900/30' 
                        : ''
                    }`}
                  >
                    {/* Status indicator with icon and color */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className={`h-3 w-3 rounded-full mr-2 ${getStatusColor(agent.status)}`}></div>
                        <span className="text-sm text-gray-900 dark:text-gray-200 capitalize flex items-center">
                          {getStatusIcon(agent.status)}
                          <span className="ml-1">{agent.status}</span>
                        </span>
                      </div>
                    </td>
                    
                    {/* Agent name and ID */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div>
                          <div className="text-sm font-medium text-gray-900 dark:text-gray-200">{agent.name}</div>
                          <div className="text-xs text-gray-500 dark:text-gray-400">{agent.agent_id}</div>
                        </div>
                      </div>
                    </td>
                    
                    {/* Current task with truncation for long task names */}
                    <td className="px-6 py-4">
                      <div className="text-sm text-gray-900 dark:text-gray-200 max-w-xs truncate">
                        {agent.current_task || (
                          <span className="text-gray-500 dark:text-gray-400 italic">No active task</span>
                        )}
                      </div>
                    </td>
                    
                    {/* Last active timestamp */}
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {formatDate(agent.last_active)}
                    </td>
                    
                    {/* Queue size with visual progress bar */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900 dark:text-gray-200 mb-1">
                        {agent.queue_size} tasks in queue
                      </div>
                      {renderQueueProgressBar(agent.queue_size)}
                    </td>
                    
                    {/* Performance metrics */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        <div className="flex items-center mb-1">
                          <span className="w-24">Success Rate:</span>
                          <span className={`font-medium ${
                            agent.performance_metrics.success_rate > 90 ? 'text-green-600 dark:text-green-400' :
                            agent.performance_metrics.success_rate > 70 ? 'text-blue-600 dark:text-blue-400' :
                            agent.performance_metrics.success_rate > 50 ? 'text-yellow-600 dark:text-yellow-400' :
                            'text-red-600 dark:text-red-400'
                          }`}>{agent.performance_metrics.success_rate}%</span>
                        </div>
                        <div className="flex items-center mb-1">
                          <span className="w-24">Avg. Completion:</span>
                          <span>{agent.performance_metrics.avg_completion_time}s</span>
                        </div>
                        <div className="flex items-center">
                          <span className="w-24">Tasks Completed:</span>
                          <span>{agent.performance_metrics.tasks_completed}</span>
                        </div>
                      </div>
                    </td>
                    
                    {/* Actions column with details button */}
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <button
                        onClick={() => handleAgentSelect(agent)}
                        className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300 flex items-center"
                      >
                        <svg className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        
        {/* Empty state when no agents are available */}
        {!loading && (!agentData || agentData.agents.length === 0) && (
          <div className="text-center py-10">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-gray-200">No agents found</h3>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">No agent data is currently available.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentDashboard;