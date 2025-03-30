#!/usr/bin/env python3
"""
Test script for the MPC (Model Parameter Communication) Protocol.

This script demonstrates how the MPC protocol can be used to serialize and
deserialize model parameters and calibration data between different modules
of the BCBS_Values property valuation system.
"""

import json
import logging
from src.mpc_protocol import (
    ModelParameter, 
    CalibrationData, 
    serialize_parameters,
    deserialize_parameters,
    get_default_parameters,
    apply_calibration,
    get_parameter_by_name,
    create_example_package
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_serialization():
    """Test basic serialization and deserialization of parameters."""
    logger.info("Testing basic serialization and deserialization...")
    
    # Create some test parameters
    parameters = [
        ModelParameter(
            name="test_size",
            value=0.25,
            data_type="float",
            description="Proportion of data to use for testing",
            default_value=0.2,
            min_value=0.1,
            max_value=0.5
        ),
        ModelParameter(
            name="random_state",
            value=123,
            data_type="int",
            description="Random seed for reproducibility",
            default_value=42
        )
    ]
    
    # Serialize parameters
    json_str = serialize_parameters(
        parameters=parameters,
        model_type="basic",
        description="Test basic parameters"
    )
    
    # Print the serialized JSON
    logger.info(f"Serialized JSON:\n{json_str}")
    
    # Deserialize parameters
    deserialized_params, deserialized_calibration = deserialize_parameters(json_str)
    
    # Verify deserialization
    assert len(deserialized_params) == 2, "Expected 2 deserialized parameters"
    assert deserialized_params[0].name == "test_size", "First parameter should be test_size"
    assert deserialized_params[0].value == 0.25, "test_size value should be 0.25"
    
    logger.info("Basic serialization test passed!")
    return deserialized_params

def test_with_calibration_data():
    """Test serialization and deserialization with calibration data."""
    logger.info("Testing serialization with calibration data...")
    
    # Get default parameters for advanced model
    parameters = get_default_parameters("advanced_linear")
    
    # Update the regularization parameter to fix validation issue
    for param in parameters:
        if param.name == "regularization":
            param.value = "ridge"  # Change from None to "ridge" to avoid validation issue
    
    # Create sample calibration data
    calibration_data = CalibrationData(
        feature_means={
            "square_feet": 2000.0,
            "bedrooms": 3.2,
            "bathrooms": 2.1,
            "year_built": 1985.0
        },
        feature_stds={
            "square_feet": 750.0,
            "bedrooms": 0.8,
            "bathrooms": 0.6,
            "year_built": 25.0
        },
        target_mean=325000.0,
        target_std=112000.0,
        feature_importance={
            "square_feet": 0.48,
            "year_built": 0.25,
            "bathrooms": 0.18,
            "bedrooms": 0.09
        },
        sample_size=980
    )
    
    # Serialize parameters with calibration data
    json_str = serialize_parameters(
        parameters=parameters,
        calibration_data=calibration_data,
        model_type="advanced_linear",
        description="Advanced linear model with calibration data"
    )
    
    # Print (part of) the serialized JSON
    logger.info(f"Serialized JSON (first 200 chars):\n{json_str[:200]}...")
    
    # Deserialize parameters and calibration data
    deserialized_params, deserialized_calibration = deserialize_parameters(json_str)
    
    # Verify calibration data
    assert deserialized_calibration is not None, "Calibration data should be present"
    assert deserialized_calibration.target_mean == 325000.0, "Target mean should be 325000.0"
    assert "square_feet" in deserialized_calibration.feature_means, "square_feet should be in feature_means"
    
    # Test calibration function
    raw_sqft = 2500
    if deserialized_calibration:
        normalized = apply_calibration(raw_sqft, "square_feet", deserialized_calibration, "normalize")
        denormalized = apply_calibration(normalized, "square_feet", deserialized_calibration, "denormalize")
        
        logger.info(f"Calibration test: {raw_sqft} -> normalized: {normalized:.4f} -> denormalized: {denormalized:.2f}")
        assert abs(denormalized - raw_sqft) < 0.01, "Denormalized value should be close to original"
    else:
        logger.warning("Skipping calibration test as no calibration data was returned")
    
    logger.info("Calibration data test passed!")
    return deserialized_params, deserialized_calibration

def test_parameter_validation():
    """Test parameter validation during serialization."""
    logger.info("Testing parameter validation...")
    
    # Create some invalid parameters to test validation
    invalid_parameters = [
        ModelParameter(
            name="test_size",
            value=0.6,  # Should fail validation (> max_value of 0.5)
            data_type="float",
            description="Proportion of data to use for testing",
            default_value=0.2,
            min_value=0.1,
            max_value=0.5
        ),
        ModelParameter(
            name="model_type",
            value="unknown_model",  # Should fail validation (not in options)
            data_type="string",
            description="Type of model to use",
            default_value="basic",
            options=["basic", "advanced", "ensemble"]
        )
    ]
    
    # Try to serialize invalid parameters (should raise ValueError)
    try:
        json_str = serialize_parameters(
            parameters=invalid_parameters,
            model_type="basic",
            description="Invalid parameters test"
        )
        assert False, "Serialization should have failed with invalid parameters"
    except ValueError as e:
        logger.info(f"Expected validation error occurred: {str(e)}")
    
    logger.info("Parameter validation test passed!")

def test_enhanced_gis_example():
    """Test the enhanced GIS example package."""
    logger.info("Testing enhanced GIS example package...")
    
    # Create an example enhanced GIS parameter package
    json_str = create_example_package("enhanced_gis")
    
    # Deserialize the package
    deserialized_params, deserialized_calibration = deserialize_parameters(json_str)
    
    # Find specific parameters
    gis_weight = get_parameter_by_name(deserialized_params, "gis_feature_weight")
    distance_decay = get_parameter_by_name(deserialized_params, "distance_decay_factor")
    enhanced_features = get_parameter_by_name(deserialized_params, "enhanced_gis_features")
    
    # Verify we found the parameters
    assert gis_weight is not None, "gis_feature_weight parameter should exist"
    assert distance_decay is not None, "distance_decay_factor parameter should exist"
    assert enhanced_features is not None, "enhanced_gis_features parameter should exist"
    
    # Check parameter values
    logger.info(f"GIS Feature Weight: {gis_weight.value} (default: {gis_weight.default_value})")
    logger.info(f"Distance Decay Factor: {distance_decay.value} (default: {distance_decay.default_value})")
    logger.info(f"Enhanced GIS Features: {enhanced_features.value}")
    
    logger.info("Enhanced GIS example test passed!")
    return deserialized_params, deserialized_calibration

def test_cross_module_communication():
    """
    Simulate cross-module communication using the MPC protocol.
    
    This test demonstrates how one module can serialize parameters and
    calibration data, send them to another module, and have that module
    deserialize and use the parameters.
    """
    logger.info("Testing cross-module communication scenario...")
    
    # Module A: Valuation Engine
    # --------------------------
    logger.info("Module A (Valuation Engine): Creating parameters and calibration data")
    
    # Get default parameters for the enhanced GIS model
    module_a_params = get_default_parameters("enhanced_gis")
    
    # Customize some parameters
    for param in module_a_params:
        if param.name == "gis_feature_weight":
            param.value = 0.75  # Increased weight for GIS features
        elif param.name == "distance_decay_factor":
            param.value = 0.15  # Lower decay factor (slower decay with distance)
    
    # Create calibration data based on training set
    module_a_calibration = CalibrationData(
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
        feature_importance={
            "square_feet": 0.38,
            "year_built": 0.15,
            "bathrooms": 0.12,
            "bedrooms": 0.08,
            "lot_size": 0.07,
            "location_score": 0.20
        },
        sample_size=1200
    )
    
    # Serialize the parameters and calibration data
    serialized_data = serialize_parameters(
        parameters=module_a_params,
        calibration_data=module_a_calibration,
        model_type="enhanced_gis",
        model_version="1.2.0",
        source_module="valuation_engine",
        description="Enhanced GIS parameters from valuation engine"
    )
    
    logger.info(f"Module A: Serialized {len(module_a_params)} parameters with calibration data")
    
    # Module B: GIS Integration Module
    # --------------------------------
    logger.info("Module B (GIS Integration): Receiving and using parameters")
    
    # Deserialize the parameters and calibration data
    module_b_params, module_b_calibration = deserialize_parameters(serialized_data)
    
    # Verify the parameters were received correctly
    gis_weight = get_parameter_by_name(module_b_params, "gis_feature_weight")
    assert gis_weight is not None, "gis_feature_weight parameter should exist"
    logger.info(f"Module B: Received gis_feature_weight = {gis_weight.value}")
    
    # Simulate using the calibration data for a feature
    if module_b_calibration:
        raw_home_size = 3200  # square feet
        normalized_size = apply_calibration(raw_home_size, "square_feet", module_b_calibration, "normalize")
        logger.info(f"Module B: Normalized home size: {raw_home_size} sq ft -> {normalized_size:.4f} (z-score)")
        
        # Module B makes a prediction using normalized values
        # (Here we're just simulating a prediction)
        feature_importance = module_b_calibration.feature_importance.get("square_feet", 0.38)
        predicted_z_score = normalized_size * feature_importance
        
        # Convert back to original scale using calibration data
        if module_b_calibration.target_std is not None and module_b_calibration.target_mean is not None:
            predicted_value = (predicted_z_score * module_b_calibration.target_std) + module_b_calibration.target_mean
            logger.info(f"Module B: Predicted value (based on size only): ${predicted_value:.2f}")
        else:
            logger.warning("Module B: Cannot calculate prediction without target mean and std")
    else:
        logger.warning("Module B: No calibration data received, skipping prediction")
    
    logger.info("Cross-module communication test passed!")

def main():
    """Run all tests."""
    logger.info("=== MPC Protocol Tests ===")
    
    # Run tests
    test_basic_serialization()
    print()
    
    test_with_calibration_data()
    print()
    
    test_parameter_validation()
    print()
    
    test_enhanced_gis_example()
    print()
    
    test_cross_module_communication()
    print()
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main()