# Model Parameter Communication (MPC) Protocol

## Overview

The MPC Protocol is a JSON-based protocol for exchanging model parameters and calibration data between different modules of the BCBS_Values property valuation system. It ensures consistent interpretation of parameters across different components and provides a standardized way to serialize, validate, and deserialize model configurations.

## Key Features

1. **Versioned Schema**: All parameter packages include version information to ensure compatibility.
2. **Parameter Validation**: Built-in validation ensures parameters meet constraints (min/max values, allowed options, etc.).
3. **Default Values**: Every parameter has a default value to ensure consistent behavior when explicit values aren't provided.
4. **Calibration Data**: Includes statistical information (means, standard deviations, etc.) to ensure consistent feature scaling.
5. **Checksum Verification**: Ensures data integrity during transmission between modules.
6. **Comprehensive Metadata**: Each parameter includes descriptions, data types, and constraints for self-documentation.

## Components

The MPC protocol includes the following main components:

### 1. ModelParameter

Represents a single model parameter with metadata:

```python
@dataclass
class ModelParameter:
    name: str
    value: Any
    data_type: str  # float, int, string, boolean, array, object
    description: str
    default_value: Any
    min_value: Optional[Union[float, int]] = None
    max_value: Optional[Union[float, int]] = None
    required: bool = True
    options: Optional[List[Any]] = None
    unit: Optional[str] = None  # E.g., "meters", "dollars", "years"
```

### 2. CalibrationData

Contains statistical information derived from training data:

```python
@dataclass
class CalibrationData:
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    target_mean: Optional[float] = None
    target_std: Optional[float] = None
    feature_mins: Dict[str, float] = field(default_factory=dict)
    feature_maxs: Dict[str, float] = field(default_factory=dict)
    feature_medians: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    calibration_date: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
```

### 3. MPCPackage

The top-level container for parameters and calibration data:

```python
@dataclass
class MPCPackage:
    protocol_version: str = MPC_PROTOCOL_VERSION
    model_type: str = "basic"
    model_version: str = "1.0.0"
    parameters: List[ModelParameter] = field(default_factory=list)
    calibration_data: Optional[CalibrationData] = None
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    source_module: str = "valuation_engine"
    description: str = ""
    checksum: str = ""  # Will be computed during serialization
```

## Usage

### Basic Usage

```python
from src.mpc_protocol import (
    ModelParameter, 
    CalibrationData, 
    serialize_parameters,
    deserialize_parameters,
    get_default_parameters
)

# Get default parameters for a model type
params = get_default_parameters("enhanced_gis")

# Customize parameters as needed
for param in params:
    if param.name == "gis_feature_weight":
        param.value = 0.7

# Create calibration data
calibration = CalibrationData(
    feature_means={"square_feet": 2000.0, "bedrooms": 3.2},
    feature_stds={"square_feet": 800.0, "bedrooms": 0.8},
    target_mean=350000.0,
    target_std=120000.0
)

# Serialize parameters and calibration data
json_str = serialize_parameters(
    parameters=params,
    calibration_data=calibration,
    model_type="enhanced_gis"
)

# Deserialize parameters in another module
deserialized_params, deserialized_calibration = deserialize_parameters(json_str)
```

### Applying Calibration

```python
from src.mpc_protocol import apply_calibration

# Normalize a feature value using calibration data
raw_value = 2500  # square feet
normalized = apply_calibration(raw_value, "square_feet", calibration_data, "normalize")

# Denormalize a value back to the original scale
denormalized = apply_calibration(normalized, "square_feet", calibration_data, "denormalize")

# The function safely handles None values for calibration_data
# If no calibration is available, the original value is returned unchanged
raw_value = 3000
result = apply_calibration(raw_value, "square_feet", None, "normalize")  # Returns 3000
```

## Supported Model Types

The MPC protocol supports the following model types:

1. `basic`: Basic linear regression model
2. `multiple_regression`: Enhanced multiple regression with feature selection
3. `advanced_linear`: Advanced linear model with regularization options
4. `advanced_lightgbm`: LightGBM gradient boosting model
5. `advanced_ensemble`: Ensemble model combining linear and LightGBM models
6. `enhanced_gis`: Enhanced GIS model with advanced spatial features

Each model type has its own set of default parameters appropriate for that model.

## Example JSON Output

```json
{
  "protocol_version": "1.0.0",
  "model_type": "enhanced_gis",
  "model_version": "1.0.0",
  "parameters": [
    {
      "name": "test_size",
      "value": 0.2,
      "data_type": "float",
      "description": "Proportion of data to use for testing",
      "default_value": 0.2,
      "min_value": 0.1,
      "max_value": 0.5,
      "required": true,
      "options": null,
      "unit": null
    },
    {
      "name": "random_state",
      "value": 42,
      "data_type": "int",
      "description": "Random seed for reproducibility",
      "default_value": 42,
      "min_value": null,
      "max_value": null,
      "required": true,
      "options": null,
      "unit": null
    },
    {
      "name": "use_gis_features",
      "value": true,
      "data_type": "boolean",
      "description": "Whether to incorporate GIS features in the model",
      "default_value": true,
      "min_value": null,
      "max_value": null,
      "required": true,
      "options": null,
      "unit": null
    },
    // Additional parameters...
  ],
  "calibration_data": {
    "feature_means": {
      "square_feet": 2250.0,
      "bedrooms": 3.5,
      "bathrooms": 2.2,
      "year_built": 1990.0,
      "lot_size": 9500.0
    },
    "feature_stds": {
      "square_feet": 850.0,
      "bedrooms": 0.9,
      "bathrooms": 0.7,
      "year_built": 20.0,
      "lot_size": 3500.0
    },
    // Additional calibration data...
  },
  "timestamp": "2025-03-30T12:34:56.789012",
  "source_module": "valuation_engine",
  "description": "Example enhanced_gis parameter package",
  "checksum": "a1b2c3d4e5f6..."
}
```

## Testing

The MPC protocol includes a comprehensive test script that demonstrates its usage:

```bash
./test_mpc_protocol.py
```

This script tests:
- Basic serialization and deserialization
- Parameter validation
- Calibration data handling
- Cross-module communication
- Enhanced GIS parameter handling

## Integration with Valuation Engine

The MPC protocol integrates with the BCBS_Values valuation engine by:

1. **Parameter Management**: Providing a standardized way to configure model parameters
2. **Calibration Exchange**: Enabling consistent feature scaling across components
3. **Model Selection**: Supporting different model types with appropriate parameters
4. **Versioning**: Ensuring compatibility as models evolve
5. **Validation**: Preventing invalid parameter combinations

## Handling Special Cases

The MPC protocol includes special handling for various edge cases:

1. **None Values in Options**: When None is a valid option in parameter constraints, the validation logic properly handles this by explicitly checking for None values.
2. **Optional Calibration Data**: Functions that use calibration data gracefully handle the case when no calibration data is available, returning the original values without transformation.
3. **Empty or Missing Values**: The protocol is designed to handle missing or empty values throughout the process, using defaults when necessary.
4. **Checksum Verification**: Validates data integrity to ensure parameters haven't been corrupted during transmission.
5. **Type Conversion**: Handles type conversion during serialization and deserialization to ensure data consistency.

## Future Enhancements

Planned improvements to the MPC protocol include:

1. **Schema Validation**: JSON Schema validation for additional verification
2. **Backward Compatibility**: Support for migrating between protocol versions
3. **Binary Format**: Optional binary serialization for efficiency
4. **Parameter Relationships**: Support for interdependent parameters
5. **Extended Metadata**: Additional metadata for UI generation and documentation
6. **Enhanced Error Reporting**: Providing more detailed error messages for parameter validation failures
7. **Value Transformation Rules**: Supporting more complex value transformations beyond simple scaling