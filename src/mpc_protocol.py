"""
Model Parameter Communication (MPC) Protocol

This module implements a JSON-based protocol for exchanging model parameters
and calibration data between the valuation engine and other modules in the 
BCBS_Values system. The protocol ensures consistent interpretation of parameters
across different components of the system.

The protocol defines:
1. Parameter schema with versioning
2. Serialization/deserialization functions
3. Parameter validation functions
4. Default values for all parameters
5. Calibration data exchange format

Usage:
    from src.mpc_protocol import serialize_parameters, deserialize_parameters
    
    # Serialize parameters to send to another module
    params_json = serialize_parameters(model_parameters, calibration_data)
    
    # Deserialize received parameters
    model_parameters, calibration_data = deserialize_parameters(params_json)
"""

import json
import logging
import datetime
import hashlib
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict

# Configure logging
logger = logging.getLogger(__name__)

# Protocol version
MPC_PROTOCOL_VERSION = "1.0.0"

# Supported model types for parameter exchange
SUPPORTED_MODEL_TYPES = [
    "basic",
    "multiple_regression", 
    "advanced_linear", 
    "advanced_lightgbm",
    "advanced_ensemble", 
    "enhanced_gis"
]

@dataclass
class ModelParameter:
    """
    Represents a single model parameter with metadata.
    
    This class defines the structure for individual parameters used in
    valuation models, including metadata about valid ranges, default values,
    and descriptions.
    """
    name: str
    value: Any
    data_type: str  # One of: float, int, string, boolean, array, object
    description: str
    default_value: Any
    min_value: Optional[Union[float, int]] = None
    max_value: Optional[Union[float, int]] = None
    required: bool = True
    options: Optional[List[Any]] = None
    unit: Optional[str] = None  # E.g., "meters", "dollars", "years"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the parameter to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelParameter':
        """Create a ModelParameter instance from a dictionary."""
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate that the parameter value meets constraints."""
        # Check data type
        type_map = {
            "float": float,
            "int": int,
            "string": str,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_type = type_map.get(self.data_type)
        if expected_type and not isinstance(self.value, expected_type):
            try:
                # Attempt type conversion for basic types
                if self.data_type in ["float", "int", "string", "boolean"]:
                    self.value = expected_type(self.value)
                else:
                    logger.error(f"Parameter {self.name} has invalid type: expected {self.data_type}, got {type(self.value).__name__}")
                    return False
            except (ValueError, TypeError):
                logger.error(f"Parameter {self.name} has invalid type and couldn't be converted: expected {self.data_type}, got {type(self.value).__name__}")
                return False
                
        # Check min/max constraints for numeric types
        if self.data_type in ["float", "int"]:
            if self.min_value is not None and self.value < self.min_value:
                logger.error(f"Parameter {self.name} is below minimum value: {self.value} < {self.min_value}")
                return False
            if self.max_value is not None and self.value > self.max_value:
                logger.error(f"Parameter {self.name} exceeds maximum value: {self.value} > {self.max_value}")
                return False
                
        # Check options constraint
        if self.options is not None:
            # Special handling for None values in options list
            # This is important when None is a valid option in the options list
            # For example, when a parameter is optional and can be either None or one of several specific values
            if self.value is None and None in self.options:
                return True
            elif self.value not in self.options:
                logger.error(f"Parameter {self.name} has invalid value: {self.value}. Must be one of: {self.options}")
                return False
            
        return True


@dataclass
class CalibrationData:
    """
    Represents calibration data for a model.
    
    Calibration data includes statistical information derived from training
    data that helps properly interpret and scale model outputs. This ensures
    consistent behavior across different components.
    """
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert calibration data to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationData':
        """Create a CalibrationData instance from a dictionary."""
        return cls(**data)


@dataclass
class MPCPackage:
    """
    A complete package of model parameters and calibration data.
    
    This is the top-level container for all data exchanged via the MPC protocol.
    It includes versioning, metadata, model parameters, and calibration data.
    """
    protocol_version: str = MPC_PROTOCOL_VERSION
    model_type: str = "basic"
    model_version: str = "1.0.0"
    parameters: List[ModelParameter] = field(default_factory=list)
    calibration_data: Optional[CalibrationData] = None
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    source_module: str = "valuation_engine"
    description: str = ""
    checksum: str = ""  # Will be computed during serialization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the package to a dictionary."""
        result = {
            "protocol_version": self.protocol_version,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "parameters": [p.to_dict() for p in self.parameters],
            "timestamp": self.timestamp,
            "source_module": self.source_module,
            "description": self.description
        }
        
        if self.calibration_data:
            result["calibration_data"] = self.calibration_data.to_dict()
            
        # Exclude checksum from dictionary for checksum calculation
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MPCPackage':
        """Create an MPCPackage instance from a dictionary."""
        # Deep copy to avoid modifying the input
        data_copy = json.loads(json.dumps(data))
        
        # Extract and convert parameters
        parameters_data = data_copy.pop("parameters", [])
        parameters = [ModelParameter.from_dict(p) for p in parameters_data]
        
        # Extract and convert calibration data if present
        calibration_data = None
        if "calibration_data" in data_copy:
            calibration_data = CalibrationData.from_dict(data_copy.pop("calibration_data"))
            
        # Create instance with remaining data
        return cls(
            parameters=parameters,
            calibration_data=calibration_data,
            **data_copy
        )
    
    def compute_checksum(self, dict_data: Dict[str, Any]) -> str:
        """Compute a checksum for the package data."""
        # Create a deterministic JSON string (sorted keys)
        json_str = json.dumps(dict_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def validate(self) -> bool:
        """Validate the entire package."""
        # Check protocol version
        if self.protocol_version != MPC_PROTOCOL_VERSION:
            logger.error(f"Unsupported protocol version: {self.protocol_version}")
            return False
            
        # Check model type
        if self.model_type not in SUPPORTED_MODEL_TYPES:
            logger.error(f"Unsupported model type: {self.model_type}")
            return False
            
        # Validate each parameter
        for param in self.parameters:
            if not param.validate():
                return False
                
        return True


# Default parameter definitions for different model types
def get_default_parameters(model_type: str) -> List[ModelParameter]:
    """
    Get default parameters for a specific model type.
    
    This function returns a list of ModelParameter objects with default
    values appropriate for the specified model type. These defaults ensure
    consistent behavior when explicit parameters are not provided.
    
    Args:
        model_type: The type of model for which to get default parameters
        
    Returns:
        A list of ModelParameter objects with default values
    """
    # Basic shared parameters across all models
    base_params = [
        ModelParameter(
            name="test_size",
            value=0.2,
            data_type="float",
            description="Proportion of data to use for testing",
            default_value=0.2,
            min_value=0.1,
            max_value=0.5
        ),
        ModelParameter(
            name="random_state",
            value=42,
            data_type="int",
            description="Random seed for reproducibility",
            default_value=42
        ),
        ModelParameter(
            name="use_gis_features",
            value=True,
            data_type="boolean",
            description="Whether to incorporate GIS features in the model",
            default_value=True
        )
    ]
    
    # Model-specific parameters
    if model_type == "basic":
        return base_params
        
    elif model_type == "multiple_regression":
        return base_params + [
            ModelParameter(
                name="feature_selection",
                value=True,
                data_type="boolean",
                description="Whether to perform feature selection",
                default_value=True
            ),
            ModelParameter(
                name="max_features",
                value=10,
                data_type="int",
                description="Maximum number of features to select",
                default_value=10,
                min_value=1,
                max_value=50
            )
        ]
        
    elif model_type == "advanced_linear":
        return base_params + [
            ModelParameter(
                name="feature_selection",
                value="auto",
                data_type="string",
                description="Method for feature selection",
                default_value="auto",
                options=["auto", "k-best", "rfe", None]
            ),
            ModelParameter(
                name="poly_degree",
                value=2,
                data_type="int",
                description="Degree of polynomial features",
                default_value=2,
                min_value=1,
                max_value=3
            ),
            ModelParameter(
                name="regularization",
                value=None,
                data_type="string",
                description="Type of regularization to use",
                default_value=None,
                options=[None, "ridge", "lasso", "elastic"],
                required=False
            ),
            ModelParameter(
                name="alpha",
                value=1.0,
                data_type="float",
                description="Regularization strength parameter",
                default_value=1.0,
                min_value=0.0,
                max_value=10.0
            ),
            ModelParameter(
                name="normalize_features",
                value=True,
                data_type="boolean",
                description="Whether to normalize features before training",
                default_value=True
            ),
            ModelParameter(
                name="feature_selection_method",
                value="f_regression",
                data_type="string",
                description="Method for feature importance estimation",
                default_value="f_regression",
                options=["f_regression", "mutual_info", "model_specific"]
            ),
            ModelParameter(
                name="cv_folds",
                value=5,
                data_type="int",
                description="Number of cross-validation folds",
                default_value=5,
                min_value=2,
                max_value=10
            )
        ]
        
    elif model_type == "advanced_lightgbm":
        return base_params + [
            ModelParameter(
                name="lightgbm_params",
                value={
                    "objective": "regression",
                    "metric": "rmse",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9
                },
                data_type="object",
                description="Parameters for the LightGBM model",
                default_value={
                    "objective": "regression",
                    "metric": "rmse",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9
                }
            ),
            ModelParameter(
                name="normalize_features",
                value=True,
                data_type="boolean",
                description="Whether to normalize features before training",
                default_value=True
            ),
            ModelParameter(
                name="feature_selection",
                value="lightgbm",
                data_type="string",
                description="Method for feature selection",
                default_value="lightgbm",
                options=["lightgbm", "auto", "k-best", "rfe", None]
            ),
            ModelParameter(
                name="early_stopping_rounds",
                value=50,
                data_type="int",
                description="Number of rounds with no improvement to stop training",
                default_value=50,
                min_value=10,
                max_value=200
            ),
            ModelParameter(
                name="num_boost_round",
                value=1000,
                data_type="int",
                description="Maximum number of boosting iterations",
                default_value=1000,
                min_value=100,
                max_value=5000
            )
        ]
        
    elif model_type == "advanced_ensemble":
        return base_params + [
            ModelParameter(
                name="linear_weight",
                value=0.4,
                data_type="float",
                description="Weight for the linear model in the ensemble",
                default_value=0.4,
                min_value=0.0,
                max_value=1.0
            ),
            ModelParameter(
                name="lightgbm_params",
                value={
                    "objective": "regression",
                    "metric": "rmse",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9
                },
                data_type="object",
                description="Parameters for the LightGBM model in the ensemble",
                default_value={
                    "objective": "regression",
                    "metric": "rmse",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9
                }
            ),
            ModelParameter(
                name="poly_degree",
                value=2,
                data_type="int",
                description="Degree of polynomial features for the linear component",
                default_value=2,
                min_value=1,
                max_value=3
            ),
            ModelParameter(
                name="normalize_features",
                value=True,
                data_type="boolean",
                description="Whether to normalize features before training",
                default_value=True
            )
        ]
        
    elif model_type == "enhanced_gis":
        return base_params + [
            ModelParameter(
                name="use_gis_features",
                value=True,
                data_type="boolean",
                description="Whether to incorporate GIS features in the model",
                default_value=True
            ),
            ModelParameter(
                name="model_type",
                value="ensemble",
                data_type="string",
                description="Base model type to use with enhanced GIS features",
                default_value="ensemble",
                options=["linear", "lightgbm", "ensemble"]
            ),
            ModelParameter(
                name="gis_feature_weight",
                value=0.6,
                data_type="float",
                description="Weight to apply to GIS features in the model",
                default_value=0.6,
                min_value=0.0,
                max_value=1.0
            ),
            ModelParameter(
                name="distance_decay_factor",
                value=0.2,
                data_type="float",
                description="Decay factor for distance-based features",
                default_value=0.2,
                min_value=0.01,
                max_value=1.0
            ),
            ModelParameter(
                name="enhanced_gis_features",
                value=["flood_zone_risk", "school_quality_score", "view_score", 
                       "traffic_noise_level", "housing_density", "growth_potential"],
                data_type="array",
                description="List of enhanced GIS features to include",
                default_value=["flood_zone_risk", "school_quality_score", "view_score", 
                              "traffic_noise_level", "housing_density", "growth_potential"]
            )
        ]
    
    # Default to basic parameters if model type not recognized
    logger.warning(f"Unknown model type: {model_type}, using basic parameters")
    return base_params


def serialize_parameters(
    parameters: List[ModelParameter], 
    calibration_data: Optional[CalibrationData] = None,
    model_type: str = "basic",
    model_version: str = "1.0.0",
    source_module: str = "valuation_engine",
    description: str = ""
) -> str:
    """
    Serialize model parameters and calibration data to JSON.
    
    This function takes model parameters and optional calibration data,
    validates them, packages them according to the MPC protocol, and
    serializes the package to a JSON string.
    
    Args:
        parameters: List of ModelParameter objects
        calibration_data: Optional CalibrationData object
        model_type: The type of model these parameters are for
        model_version: The version of the model
        source_module: The module that generated these parameters
        description: A human-readable description of the parameter set
        
    Returns:
        A JSON string containing the serialized parameters
    """
    # Create package
    package = MPCPackage(
        protocol_version=MPC_PROTOCOL_VERSION,
        model_type=model_type,
        model_version=model_version,
        parameters=parameters,
        calibration_data=calibration_data,
        source_module=source_module,
        description=description,
        timestamp=datetime.datetime.now().isoformat()
    )
    
    # Validate the package
    if not package.validate():
        logger.error("Parameter validation failed during serialization")
        raise ValueError("Invalid parameters cannot be serialized")
    
    # Convert to dictionary for serialization
    package_dict = package.to_dict()
    
    # Calculate checksum
    checksum = package.compute_checksum(package_dict)
    package_dict["checksum"] = checksum
    
    # Serialize to JSON
    return json.dumps(package_dict, indent=2)


def deserialize_parameters(json_str: str) -> Tuple[List[ModelParameter], Optional[CalibrationData]]:
    """
    Deserialize parameters from a JSON string.
    
    This function takes a JSON string containing serialized parameters,
    deserializes it according to the MPC protocol, validates the
    checksum and parameter values, and returns the parameters and
    calibration data.
    
    Args:
        json_str: JSON string containing serialized parameters
        
    Returns:
        A tuple containing (parameters, calibration_data)
    """
    try:
        # Parse the JSON string
        data = json.loads(json_str)
        
        # Extract checksum for verification
        received_checksum = data.pop("checksum", None)
        
        # Create a package instance
        package = MPCPackage.from_dict(data)
        
        # Verify checksum if one was provided
        if received_checksum:
            calculated_checksum = package.compute_checksum(data)
            if calculated_checksum != received_checksum:
                logger.error("Checksum verification failed during deserialization")
                raise ValueError("Checksum verification failed, data may be corrupted")
        
        # Validate the package
        if not package.validate():
            logger.error("Parameter validation failed during deserialization")
            raise ValueError("Deserialized parameters are invalid")
        
        return package.parameters, package.calibration_data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise ValueError(f"Failed to parse JSON: {e}")
    except Exception as e:
        logger.error(f"Deserialization error: {e}")
        raise ValueError(f"Failed to deserialize parameters: {e}")


def apply_calibration(
    value: float, 
    feature_name: str, 
    calibration_data: Optional[CalibrationData], 
    operation: str = "normalize"
) -> float:
    """
    Apply calibration to a feature value.
    
    This function applies normalization or denormalization to a feature value
    based on the provided calibration data. This ensures consistent scaling
    across different components of the system.
    
    Args:
        value: The raw feature value
        feature_name: The name of the feature
        calibration_data: The calibration data containing means and standard deviations
        operation: The operation to perform: "normalize" or "denormalize"
        
    Returns:
        The calibrated value
    """
    # Handle the case when calibration_data is None
    # This is important for functions that accept calibration_data as an optional parameter
    # When no calibration data is available, we simply return the original value unchanged
    if calibration_data is None:
        logger.warning("No calibration data provided")
        return value
        
    if feature_name not in calibration_data.feature_means or feature_name not in calibration_data.feature_stds:
        logger.warning(f"No calibration data available for feature {feature_name}")
        return value
        
    mean = calibration_data.feature_means.get(feature_name, 0)
    std = calibration_data.feature_stds.get(feature_name, 1)
    
    # Avoid division by zero
    if std == 0:
        logger.warning(f"Standard deviation is zero for feature {feature_name}")
        std = 1
    
    if operation == "normalize":
        # Convert to z-score
        return (value - mean) / std
    elif operation == "denormalize":
        # Convert from z-score to original scale
        return (value * std) + mean
    else:
        logger.error(f"Unknown calibration operation: {operation}")
        return value


def get_parameter_by_name(parameters: List[ModelParameter], name: str) -> Optional[ModelParameter]:
    """
    Find a parameter by name.
    
    This helper function searches for a parameter with the specified name
    in a list of parameters.
    
    Args:
        parameters: List of ModelParameter objects
        name: The name of the parameter to find
        
    Returns:
        The ModelParameter object if found, otherwise None
    """
    for param in parameters:
        if param.name == name:
            return param
    return None


def create_example_package(model_type: str = "enhanced_gis") -> str:
    """
    Create an example parameter package for demonstration.
    
    This function creates a sample parameter package with default values
    for the specified model type, including sample calibration data.
    
    Args:
        model_type: The type of model to create a package for
        
    Returns:
        A JSON string with the serialized parameter package
    """
    # Get default parameters for the specified model type
    parameters = get_default_parameters(model_type)
    
    # Create sample calibration data
    calibration_data = CalibrationData(
        feature_means={
            "square_feet": 2250.0,
            "bedrooms": 3.5,
            "bathrooms": 2.2,
            "year_built": 1990.0,
            "lot_size": 9500.0
        },
        feature_stds={
            "square_feet": 850.0,
            "bedrooms": 0.9,
            "bathrooms": 0.7,
            "year_built": 20.0,
            "lot_size": 3500.0
        },
        target_mean=350000.0,
        target_std=125000.0,
        feature_mins={
            "square_feet": 800.0,
            "bedrooms": 1.0,
            "bathrooms": 1.0,
            "year_built": 1950.0,
            "lot_size": 2500.0
        },
        feature_maxs={
            "square_feet": 5000.0,
            "bedrooms": 6.0,
            "bathrooms": 5.0,
            "year_built": 2023.0,
            "lot_size": 25000.0
        },
        feature_importance={
            "square_feet": 0.45,
            "year_built": 0.22,
            "bathrooms": 0.15,
            "bedrooms": 0.10,
            "lot_size": 0.08
        },
        sample_size=1200
    )
    
    # Serialize the package
    return serialize_parameters(
        parameters=parameters,
        calibration_data=calibration_data,
        model_type=model_type,
        model_version="1.0.0",
        source_module="valuation_engine",
        description=f"Example {model_type} parameter package"
    )


if __name__ == "__main__":
    # Quick demonstration of the protocol
    example_json = create_example_package("enhanced_gis")
    print("Example MPC Protocol Package (enhanced_gis):")
    print(example_json)
    
    # Deserialize the package
    deserialized_params, deserialized_calibration = deserialize_parameters(example_json)
    print(f"\nSuccessfully deserialized {len(deserialized_params)} parameters")
    
    # Show an example of parameter retrieval
    gis_weight_param = get_parameter_by_name(deserialized_params, "gis_feature_weight")
    if gis_weight_param:
        print(f"\nGIS Feature Weight: {gis_weight_param.value} (default: {gis_weight_param.default_value})")
        
    # Show calibration example
    if deserialized_calibration:
        raw_sqft = 3000
        normalized = apply_calibration(raw_sqft, "square_feet", deserialized_calibration, "normalize")
        denormalized = apply_calibration(normalized, "square_feet", deserialized_calibration, "denormalize")
        print(f"\nCalibration Example:")
        print(f"Raw square_feet: {raw_sqft}")
        print(f"Normalized: {normalized:.4f}")
        print(f"Denormalized: {denormalized:.2f}")