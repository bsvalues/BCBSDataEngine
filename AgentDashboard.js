import React, { useState, useEffect } from 'react';

/**
 * AgentDashboard Component
 * 
 * This component displays real-time status of the BS Army of Agents,
 * fetching data from the /api/agent-status endpoint and providing 
 * visual indicators of each agent's current state.
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

  /**
   * Function to fetch agent status data from the API
   */
  const fetchAgentStatus = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/agent-status');
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      setAgentData(data);
      setLastRefresh(new Date());
      setError(null);
    } catch (err) {
      console.error('Error fetching agent status:', err);
      setError(`Failed to fetch agent status: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Fetch data on component mount
  useEffect(() => {
    fetchAgentStatus();
    
    // Set up polling interval (every 30 seconds)
    const intervalId = setInterval(fetchAgentStatus, 30000);
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

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
   * Helper function to format the date for display
   * @param {Date} date - Date to format
   * @returns {string} - Formatted date string
   */
  const formatDate = (date) => {
    if (!date) return 'Never';
    return new Date(date).toLocaleString();
  };

  /**
   * Render a visual progress bar for the queue size
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
          className={`${barColor} h-2.5 rounded-full`} 
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    );
  };

  return (
    <div className="container mx-auto p-4">
      {/* Dashboard Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-white">
          BS Army of Agents Dashboard
        </h1>
        <div className="flex items-center space-x-4">
          {/* Last refresh timestamp */}
          <span className="text-sm text-gray-600 dark:text-gray-300">
            Last updated: {lastRefresh ? formatDate(lastRefresh) : 'Never'}
          </span>
          
          {/* Manual refresh button */}
          <button
            onClick={fetchAgentStatus}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors duration-300 disabled:bg-blue-300 disabled:cursor-not-allowed"
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
              'Refresh'
            )}
          </button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      )}

      {/* System Status Summary */}
      {agentData && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">System Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-300">Status</p>
              <p className="text-2xl font-bold text-blue-600 dark:text-blue-300">{agentData.system_status}</p>
            </div>
            <div className="bg-green-50 dark:bg-green-900 p-4 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-300">Active Agents</p>
              <p className="text-2xl font-bold text-green-600 dark:text-green-300">{agentData.active_agents}</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900 p-4 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-300">Tasks In Progress</p>
              <p className="text-2xl font-bold text-yellow-600 dark:text-yellow-300">{agentData.tasks_in_progress}</p>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900 p-4 rounded-lg">
              <p className="text-sm text-gray-600 dark:text-gray-300">Tasks Completed Today</p>
              <p className="text-2xl font-bold text-purple-600 dark:text-purple-300">{agentData.tasks_completed_today}</p>
            </div>
          </div>
        </div>
      )}

      {/* Agents List */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Agents</h2>
        
        {loading && !agentData && (
          <div className="flex justify-center items-center p-10">
            <svg className="animate-spin h-10 w-10 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
        )}
        
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
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {agentData.agents.map((agent) => (
                  <tr key={agent.agent_id}>
                    {/* Status indicator */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className={`h-3 w-3 rounded-full mr-2 ${getStatusColor(agent.status)}`}></div>
                        <span className="text-sm text-gray-900 dark:text-gray-200 capitalize">{agent.status}</span>
                      </div>
                    </td>
                    
                    {/* Agent name and ID */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900 dark:text-gray-200">{agent.name}</div>
                          <div className="text-xs text-gray-500 dark:text-gray-400">{agent.agent_id}</div>
                        </div>
                      </div>
                    </td>
                    
                    {/* Current task */}
                    <td className="px-6 py-4">
                      <div className="text-sm text-gray-900 dark:text-gray-200">
                        {agent.current_task || 'No active task'}
                      </div>
                    </td>
                    
                    {/* Last active timestamp */}
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {formatDate(agent.last_active)}
                    </td>
                    
                    {/* Queue size with progress bar */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900 dark:text-gray-200 mb-1">
                        {agent.queue_size} tasks in queue
                      </div>
                      {renderQueueProgressBar(agent.queue_size)}
                    </td>
                    
                    {/* Performance metrics */}
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        <div>Success Rate: {agent.performance_metrics.success_rate}%</div>
                        <div>Avg. Completion: {agent.performance_metrics.avg_completion_time}s</div>
                        <div>Tasks Completed: {agent.performance_metrics.tasks_completed}</div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        
        {/* Empty state */}
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