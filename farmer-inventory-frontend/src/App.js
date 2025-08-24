// App.js
import React, { useState, useEffect, useCallback } from 'react';
import { 
  Camera, 
  Play, 
  Square, 
  Package, 
  Plus, 
  RefreshCw, 
  Settings, 
  AlertTriangle,
  CheckCircle,
  Eye,
  Activity
} from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:5000/api'; // Update to match backend port

// Configure axios
const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
});

// Utility function for API calls
const apiCall = async (endpoint, method = 'GET', data = null) => {
  try {
    const config = { method };
    if (data && method !== 'GET') {
      config.data = data;
    }
    const response = await api.request({ ...config, url: endpoint });
    return response.data;
  } catch (error) {
    console.error('API call failed:', error);
    throw new Error(error.response?.data?.message || error.message || 'API call failed');
  }
};

// Alert Component
const Alert = ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  const bgColor = type === 'success' ? 'bg-green-100 text-green-800 border-green-200' : 'bg-red-100 text-red-800 border-red-200';
  
  return (
    <div className={`${bgColor} border rounded-lg p-4 mb-4 flex items-center gap-2`}>
      {type === 'success' ? <CheckCircle size={20} /> : <AlertTriangle size={20} />}
      <span>{message}</span>
      <button onClick={onClose} className="ml-auto text-lg font-bold">×</button>
    </div>
  );
};

// Status Indicator Component
const StatusIndicator = ({ online }) => (
  <div className="flex items-center gap-2">
    <div className={`w-3 h-3 rounded-full ${online ? 'bg-green-500 shadow-lg shadow-green-300' : 'bg-red-500'}`} />
    <span className="text-sm font-medium">{online ? 'Camera Online' : 'Camera Offline'}</span>
  </div>
);

// Inventory Item Component
const InventoryItem = ({ item }) => {
  const getItemEmoji = (name) => {
    const emojiMap = {
      apple: '🍎', orange: '🍊', banana: '🍌',
      carrot: '🥕', broccoli: '🥦', tomato: '🍅', seeds: '🌱'
    };
    return emojiMap[name.toLowerCase()] || '📦';
  };

  return (
    <div className={`bg-gray-50 rounded-xl p-4 mb-3 border-l-4 transition-all duration-300 hover:shadow-md ${
      item.low_stock ? 'border-red-500 bg-red-50 animate-pulse' : 'border-blue-500'
    }`}>
      <div className="font-semibold text-lg mb-2 flex items-center gap-2">
        <span>{getItemEmoji(item.name)}</span>
        <span className="capitalize">{item.name}</span>
      </div>
      
      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="bg-white rounded-lg p-2 text-center">
          <div className="text-xs text-gray-500 mb-1">Current Stock</div>
          <div className="font-bold text-gray-800">{item.stock}</div>
        </div>
        <div className="bg-white rounded-lg p-2 text-center">
          <div className="text-xs text-gray-500 mb-1">Threshold</div>
          <div className="font-bold text-gray-800">{item.threshold}</div>
        </div>
        <div className="bg-white rounded-lg p-2 text-center">
          <div className="text-xs text-gray-500 mb-1">Status</div>
          <div className={`font-bold ${item.low_stock ? 'text-red-500' : 'text-green-500'}`}>
            {item.low_stock ? '⚠️ Low' : '✅ OK'}
          </div>
        </div>
      </div>
      
      <div className="text-xs text-gray-500">
        Last updated: {new Date(item.last_updated).toLocaleString()}
      </div>
    </div>
  );
};

// Control Card Component
const ControlCard = ({ title, children, icon: Icon }) => (
  <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
    <div className="flex items-center gap-2 mb-4">
      <Icon size={24} className="text-blue-600" />
      <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
    </div>
    {children}
  </div>
);

// Main Dashboard Component
const FarmerInventoryDashboard = () => {
  const [inventory, setInventory] = useState([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [alert, setAlert] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backendHealth, setBackendHealth] = useState(false);
  
  // Form states
  const [newItem, setNewItem] = useState({ name: 'apple', stock: 0, threshold: 10 });
  const [stockUpdate, setStockUpdate] = useState({ item_name: '', quantity_change: 0 });
  
  // YOLO settings
  const [yoloSettings, setYoloSettings] = useState({ confidence: 0.5, line_position: 300 });

  // Check backend health
  const checkBackendHealth = useCallback(async () => {
  try {
    const response = await apiCall('/health');
    if (response.status === 'ok') {
      setBackendHealth(true);
    }
  } catch (error) {
    setBackendHealth(false);
    console.error('Backend health check failed:', error);
  }
}, []);

  // Load inventory data
  const loadInventory = useCallback(async () => {
  try {
    const data = await apiCall('/inventory');
    setInventory(data);
    if (data.length > 0 && !stockUpdate.item_name) {
      setStockUpdate(prev => ({ ...prev, item_name: data[0].name }));
    }
  } catch (error) {
    console.error('Error loading inventory:', error);
    if (backendHealth) {
      showAlert('Failed to load inventory data');
    }
  }
}, [stockUpdate.item_name, backendHealth]);

  // Initialize and auto-refresh
  useEffect(() => {
    checkBackendHealth();
    const healthInterval = setInterval(checkBackendHealth, 10000); // Check every 10 seconds
    
    return () => clearInterval(healthInterval);
  }, [checkBackendHealth]);

  useEffect(() => {
    if (backendHealth) {
      loadInventory();
      const interval = setInterval(loadInventory, 5000);
      return () => clearInterval(interval);
    }
  }, [loadInventory, backendHealth]);

  const showAlert = (message, type = 'error') => {
    setAlert({ message, type });
  };

  const handleStartMonitoring = async () => {
    setLoading(true);
    try {
      const result = await apiCall('/start_monitoring', 'POST');
      if (result.success) {
        setIsMonitoring(true);
        showAlert('Camera monitoring started successfully!', 'success');
      } else {
        showAlert('Failed to start camera monitoring. Please check camera connection.');
      }
    } catch (error) {
      showAlert('Error starting monitoring: ' + error.message);
    }
    setLoading(false);
  };

  const handleStopMonitoring = async () => {
    setLoading(true);
    try {
      await apiCall('/stop_monitoring', 'POST');
      setIsMonitoring(false);
      showAlert('Camera monitoring stopped.', 'success');
    } catch (error) {
      showAlert('Error stopping monitoring: ' + error.message);
    }
    setLoading(false);
  };

  const handleAddItem = async () => {
    try {
      const result = await apiCall('/add_item', 'POST', newItem);
      if (result.success) {
        showAlert(`Item "${newItem.name}" added successfully!`, 'success');
        setNewItem({ name: 'apple', stock: 0, threshold: 10 });
        loadInventory();
      } else {
        showAlert('Failed to add item. Please try again.');
      }
    } catch (error) {
      showAlert('Error adding item: ' + error.message);
    }
  };

  const handleUpdateStock = async () => {
    try {
      const result = await apiCall('/update_stock', 'POST', stockUpdate);
      if (result.success) {
        showAlert(`Stock updated! New stock: ${result.new_stock}`, 'success');
        setStockUpdate(prev => ({ ...prev, quantity_change: 0 }));
        loadInventory();
      } else {
        showAlert('Failed to update stock. Please try again.');
      }
    } catch (error) {
      showAlert('Error updating stock: ' + error.message);
    }
  };

  const handleYoloSettingsChange = async (key, value) => {
    const newSettings = { ...yoloSettings, [key]: value };
    setYoloSettings(newSettings);
    
    try {
      await apiCall('/update_yolo_settings', 'POST', {
        confidence: key === 'confidence' ? value : yoloSettings.confidence,
        line_position: key === 'line_position' ? value : yoloSettings.line_position
      });
    } catch (error) {
      console.error('Error updating YOLO settings:', error);
    }
  };

  const itemOptions = ['apple', 'orange', 'banana', 'carrot', 'broccoli', 'tomato', 'seeds'];

  // Show backend connection status
  if (!backendHealth) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-red-50 to-red-100 flex items-center justify-center">
        <div className="bg-white rounded-2xl p-8 shadow-xl text-center max-w-md">
          <AlertTriangle size={64} className="mx-auto text-red-500 mb-4" />
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Backend Connection Error</h2>
          <p className="text-gray-600 mb-4">
            Cannot connect to the Flask backend server. Please ensure:
          </p>
          <ul className="text-left text-sm text-gray-600 mb-6 space-y-1">
            <li>• Backend server is running on port 5000</li>
            <li>• Python dependencies are installed</li>
            <li>• No firewall blocking the connection</li>
          </ul>
          <button
            onClick={checkBackendHealth}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl p-8 mb-6 text-center shadow-xl">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🤖 Advanced Farmer Inventory Management
          </h1>
          <p className="text-gray-600 text-lg">
            Powered by YOLOv11 • Real-time Object Detection & Tracking
          </p>
          <div className="flex justify-center items-center gap-4 mt-4">
            <StatusIndicator online={backendHealth} />
            <span className="text-sm text-gray-500">Backend Status</span>
          </div>
        </div>

        {/* Alert */}
        {alert && (
          <Alert 
            message={alert.message} 
            type={alert.type} 
            onClose={() => setAlert(null)} 
          />
        )}

        {/* Control Panel */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
          {/* Camera Controls */}
          <ControlCard title="Camera Controls" icon={Camera}>
            <div className="mb-4">
              <StatusIndicator online={isMonitoring} />
            </div>
            
            <div className="flex gap-2 mb-4">
              <button
                onClick={handleStartMonitoring}
                disabled={loading || isMonitoring}
                className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <Play size={16} />
                {loading ? 'Starting...' : 'Start'}
              </button>
              <button
                onClick={handleStopMonitoring}
                disabled={loading || !isMonitoring}
                className="flex-1 bg-red-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-red-700 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <Square size={16} />
                Stop
              </button>
            </div>

            {/* YOLO Settings */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <Settings size={16} />
                <span className="font-medium text-sm">YOLOv11 Settings</span>
              </div>
              
              <div className="mb-3">
                <label className="block text-xs text-gray-600 mb-1">
                  Confidence: {yoloSettings.confidence}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={yoloSettings.confidence}
                  onChange={(e) => handleYoloSettingsChange('confidence', parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
              </div>
              
              <div>
                <label className="block text-xs text-gray-600 mb-1">
                  Detection Line: {yoloSettings.line_position}
                </label>
                <input
                  type="range"
                  min="100"
                  max="500"
                  step="10"
                  value={yoloSettings.line_position}
                  onChange={(e) => handleYoloSettingsChange('line_position', parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
              </div>
            </div>
          </ControlCard>

          {/* Add New Item */}
          <ControlCard title="Add New Item" icon={Plus}>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Item Name</label>
                <select
                  value={newItem.name}
                  onChange={(e) => setNewItem(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {itemOptions.map(item => (
                    <option key={item} value={item}>
                      {item.charAt(0).toUpperCase() + item.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Initial Stock</label>
                <input
                  type="number"
                  min="0"
                  value={newItem.stock}
                  onChange={(e) => setNewItem(prev => ({ ...prev, stock: parseInt(e.target.value) || 0 }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Min Threshold</label>
                <input
                  type="number"
                  min="1"
                  value={newItem.threshold}
                  onChange={(e) => setNewItem(prev => ({ ...prev, threshold: parseInt(e.target.value) || 1 }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              
              <button
                onClick={handleAddItem}
                className="w-full bg-green-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-green-700 flex items-center justify-center gap-2"
              >
                <Plus size={16} />
                Add Item
              </button>
            </div>
          </ControlCard>

          {/* Manual Stock Update */}
          <ControlCard title="Manual Stock Update" icon={RefreshCw}>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Item</label>
                <select
                  value={stockUpdate.item_name}
                  onChange={(e) => setStockUpdate(prev => ({ ...prev, item_name: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {inventory.map(item => (
                    <option key={item.name} value={item.name}>
                      {item.name.charAt(0).toUpperCase() + item.name.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Quantity Change</label>
                <input
                  type="number"
                  value={stockUpdate.quantity_change}
                  onChange={(e) => setStockUpdate(prev => ({ ...prev, quantity_change: parseInt(e.target.value) || 0 }))}
                  placeholder="Use negative for removal"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              
              <button
                onClick={handleUpdateStock}
                className="w-full bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 flex items-center justify-center gap-2"
              >
                <RefreshCw size={16} />
                Update Stock
              </button>
            </div>
          </ControlCard>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Video Feed */}
          <div className="lg:col-span-2 bg-white rounded-xl p-6 shadow-xl">
            <div className="flex items-center gap-2 mb-4">
              <Eye size={24} className="text-blue-600" />
              <h3 className="text-lg font-semibold text-gray-800">Live Camera Feed</h3>
            </div>
            
            <div className="relative bg-black rounded-lg overflow-hidden aspect-video mb-4">
              {isMonitoring ? (
                <img
                  src="http://localhost:5000/api/video_feed"
                  alt="Live camera feed"
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    console.error('Video stream error');
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'flex';
                  }}
                />
              ) : null}
              <div 
                className="flex items-center justify-center h-full text-gray-400 text-lg"
                style={{ display: isMonitoring ? 'none' : 'flex' }}
              >
                📷 Camera feed will appear here when monitoring starts
              </div>
            </div>
            
            <div className="text-center text-gray-600 text-sm">
              <p>🎯 Green line shows detection boundary</p>
              <p>📦 Objects crossing the line will be automatically counted</p>
            </div>
          </div>

          {/* Inventory List */}
          <div className="bg-white rounded-xl p-6 shadow-xl">
            <div className="flex items-center gap-2 mb-4">
              <Package size={24} className="text-blue-600" />
              <h3 className="text-lg font-semibold text-gray-800">Current Inventory</h3>
            </div>
            
            <div className="max-h-96 overflow-y-auto">
              {inventory.length === 0 ? (
                <div className="text-center text-gray-400 py-8">
                  <Activity size={48} className="mx-auto mb-4 opacity-50" />
                  <p>No inventory items found.</p>
                  <p>Add some items to get started!</p>
                </div>
              ) : (
                inventory.map(item => (
                  <InventoryItem key={item.name} item={item} />
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FarmerInventoryDashboard;