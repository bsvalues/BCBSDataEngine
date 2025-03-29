import React, { useState, useEffect } from 'react';

/**
 * Dashboard Component
 * 
 * This component fetches property valuation data from the API and displays it in a
 * filterable table. It includes options for filtering by neighborhood or date,
 * and handles loading states and errors gracefully.
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
    toDate: ''
  });

  // Function to fetch property valuations from the API
  const fetchPropertyValuations = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Build query parameters based on active filters
      const queryParams = new URLSearchParams();
      
      if (filters.minValue) queryParams.append('min_value', filters.minValue);
      if (filters.maxValue) queryParams.append('max_value', filters.maxValue);
      if (filters.propertyType) queryParams.append('property_type', filters.propertyType);
      
      // Make the API request
      const response = await fetch(`/api/valuations?${queryParams.toString()}`);
      
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      
      const data = await response.json();
      setProperties(data);
    } catch (err) {
      console.error('Error fetching property valuations:', err);
      setError('Failed to load property valuations. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Function to handle filter changes
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters(prevFilters => ({
      ...prevFilters,
      [name]: value
    }));
  };

  // Function to apply filters
  const applyFilters = (e) => {
    e.preventDefault();
    fetchPropertyValuations();
  };

  // Function to reset all filters
  const resetFilters = () => {
    setFilters({
      neighborhood: '',
      minValue: '',
      maxValue: '',
      propertyType: '',
      fromDate: '',
      toDate: ''
    });
    // Re-fetch data with cleared filters
    fetchPropertyValuations();
  };

  // Function to format currency
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(value);
  };

  // Function to extract neighborhood from address
  const extractNeighborhood = (address) => {
    // This is a simple extraction - you might need more complex logic
    // based on your actual address format
    const parts = address.split(',');
    if (parts.length >= 2) {
      return parts[1].trim();
    }
    return 'Unknown';
  };

  // Function to filter properties by neighborhood if that filter is active
  const filteredProperties = properties.filter(property => {
    // Apply neighborhood filter (client-side filtering since API doesn't support it)
    if (filters.neighborhood && !extractNeighborhood(property.address).toLowerCase().includes(filters.neighborhood.toLowerCase())) {
      return false;
    }
    
    // Apply date filters if they exist (assuming valuation_date is available)
    if (filters.fromDate && new Date(property.valuation_date) < new Date(filters.fromDate)) {
      return false;
    }
    
    if (filters.toDate && new Date(property.valuation_date) > new Date(filters.toDate)) {
      return false;
    }
    
    return true;
  });
  
  // Load property valuations when component mounts
  useEffect(() => {
    fetchPropertyValuations();
  }, []); // Empty dependency array means this runs once on mount

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Property Valuations Dashboard</h1>
      
      {/* Filters Section */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Filters</h2>
        <form onSubmit={applyFilters}>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Neighborhood Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Neighborhood</label>
              <input
                type="text"
                name="neighborhood"
                value={filters.neighborhood}
                onChange={handleFilterChange}
                className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="e.g. Richland"
              />
            </div>
            
            {/* Min Value Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Min Value ($)</label>
              <input
                type="number"
                name="minValue"
                value={filters.minValue}
                onChange={handleFilterChange}
                className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          
          {/* Filter Action Buttons */}
          <div className="mt-4 flex justify-end space-x-2">
            <button
              type="button"
              onClick={resetFilters}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500"
            >
              Reset Filters
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              Apply Filters
            </button>
          </div>
        </form>
      </div>
      
      {/* Results Section */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <h2 className="text-xl font-semibold p-6 border-b">
          Properties ({filteredProperties.length})
        </h2>
        
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
        
        {/* Results Table */}
        {!loading && !error && (
          <>
            {filteredProperties.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Property ID
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Address
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Neighborhood
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Estimated Value
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Confidence
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Model
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Valuation Date
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {filteredProperties.map((property) => (
                      <tr key={property.property_id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {property.property_id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {property.address}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {extractNeighborhood(property.address)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-medium">
                          {formatCurrency(property.estimated_value)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="relative w-full h-2 bg-gray-200 rounded">
                              <div
                                className="absolute top-0 left-0 h-2 bg-green-500 rounded"
                                style={{ width: `${property.confidence_score * 100}%` }}
                              ></div>
                            </div>
                            <span className="ml-2 text-sm text-gray-500">
                              {Math.round(property.confidence_score * 100)}%
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {property.model_used}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {new Date(property.valuation_date).toLocaleDateString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <a
                            href={`/property/${property.property_id}`}
                            className="text-blue-600 hover:text-blue-900"
                          >
                            View Details
                          </a>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-6 text-center text-gray-500">
                No properties found matching your filters. Try adjusting your search criteria.
              </div>
            )}
          </>
        )}
      </div>
      
      {/* Summary Stats (visible when data is loaded) */}
      {!loading && !error && filteredProperties.length > 0 && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Average Value Card */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-sm text-gray-500 uppercase tracking-wider mb-1">Average Value</h3>
            <p className="text-2xl font-bold text-gray-900">
              {formatCurrency(
                filteredProperties.reduce((acc, p) => acc + p.estimated_value, 0) / filteredProperties.length
              )}
            </p>
          </div>
          
          {/* Highest Value Card */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-sm text-gray-500 uppercase tracking-wider mb-1">Highest Value</h3>
            <p className="text-2xl font-bold text-gray-900">
              {formatCurrency(
                Math.max(...filteredProperties.map(p => p.estimated_value))
              )}
            </p>
          </div>
          
          {/* Lowest Value Card */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-sm text-gray-500 uppercase tracking-wider mb-1">Lowest Value</h3>
            <p className="text-2xl font-bold text-gray-900">
              {formatCurrency(
                Math.min(...filteredProperties.map(p => p.estimated_value))
              )}
            </p>
          </div>
          
          {/* Average Confidence Card */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-sm text-gray-500 uppercase tracking-wider mb-1">Avg. Confidence</h3>
            <div className="flex items-center">
              <div className="relative w-full h-3 bg-gray-200 rounded">
                <div
                  className="absolute top-0 left-0 h-3 bg-green-500 rounded"
                  style={{
                    width: `${(filteredProperties.reduce((acc, p) => acc + p.confidence_score, 0) / filteredProperties.length) * 100}%`
                  }}
                ></div>
              </div>
              <span className="ml-2 text-lg font-bold text-gray-900">
                {Math.round(
                  (filteredProperties.reduce((acc, p) => acc + p.confidence_score, 0) / filteredProperties.length) * 100
                )}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;