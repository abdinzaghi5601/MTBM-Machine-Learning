# AVN1200 Machine Learning Components

## Overview

This folder contains the original AVN1200 Microtunnelling Machine Learning components and data files. These files represent the foundation work that was later expanded into the comprehensive AVN800 framework.

## Files Description

### Core ML Implementation
- **`steering_accuracy_ml.py`** - Original AVN1200 steering accuracy prediction model
  - Random Forest-based steering correction prediction
  - Feature engineering for AVN1200 steering system
  - Comprehensive model for predicting required steering corrections

### Data Files
- **`measure_protocol_original_.xls.csv`** - AVN1200 measurement protocol data
  - Original drive data from AVN1200 operations
  - Contains sensor readings, steering positions, and performance metrics
  - Base dataset used for training the ML models

### Documentation & Requirements
- **`steering_accuracy_code_explanation.txt`** - Detailed explanation of the steering ML code
  - Line-by-line code breakdown and explanations
  - Feature engineering methodology
  - Model training and evaluation process

- **`ml_requirements.txt`** - Core machine learning dependencies
  - Essential Python packages for AVN1200 ML implementation
  - Focused on steering accuracy model requirements

## Machine Specifications

- **Machine Type**: AVN1200 Microtunnelling Boring Machine
- **Focus Area**: Steering accuracy and trajectory control
- **Data Source**: Real operational data from tunneling projects

## Relationship to AVN800 Framework

These AVN1200 components served as the foundation for the expanded AVN800 Drive Protocol ML Framework found in the parent directory. The AVN800 framework includes:

- Enhanced multi-model architecture (steering + efficiency + ground + risk)
- Real-time optimization capabilities
- Advanced feature engineering
- Comprehensive monitoring and analytics

## Usage

The AVN1200 components can be used independently for:

1. **Steering Analysis**: Understanding basic steering accuracy patterns
2. **Historical Reference**: Comparing with newer implementations
3. **Educational Purposes**: Learning ML application in tunneling
4. **Data Processing**: Processing AVN1200-specific measurement protocols

## Technical Notes

- Compatible with the AVN1200 measurement protocol format
- Optimized for AVN1200 4-cylinder steering system
- Performance benchmarks specific to AVN1200 operational parameters

---

**Note**: For current production use, refer to the comprehensive AVN800 framework in the parent directory. These AVN1200 files are maintained for reference and compatibility purposes.