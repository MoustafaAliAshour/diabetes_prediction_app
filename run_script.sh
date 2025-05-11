#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Diabetes Prediction App Setup...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python could not be found. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip could not be found. Please install pip and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
fi

# Check if model files exist
if [ ! -f "models/xgboost_diabetes_model.pkl" ] || [ ! -f "models/preprocessor_pipeline.pkl" ]; then
    echo -e "${YELLOW}Warning: Model files not found in the models directory.${NC}"
    echo "Please ensure you have the following files in the models directory:"
    echo "  - models/xgboost_diabetes_model.pkl"
    echo "  - models/preprocessor_pipeline.pkl"
    echo ""
    echo "Attempting to start the app anyway..."
fi

# Start Streamlit app
echo -e "${GREEN}Starting Diabetes Prediction App...${NC}"
streamlit run app.py

# Deactivate virtual environment on exit
deactivate