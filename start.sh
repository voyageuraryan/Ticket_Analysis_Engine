#!/bin/bash
# SAP Module Classifier - Linux/Mac Startup Script
# This script launches the Streamlit application

echo "================================================================================"
echo "SAP MODULE CLASSIFIER - STARTUP"
echo "================================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/4] Checking Python installation..."
python3 --version
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "[2/4] Activating virtual environment..."
    source venv/bin/activate
else
    echo "[2/4] No virtual environment found. Using system Python..."
fi
echo ""

# Check if dependencies are installed
echo "[3/4] Checking dependencies..."
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "WARNING: Dependencies not installed. Installing now..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
else
    echo "Dependencies OK"
fi
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Please create .env file with your API keys:"
    echo "  GOOGLE_API_KEY=your_gemini_key"
    echo "  GROQ_API_KEY=your_groq_key"
    echo ""
    echo "Press Enter to continue anyway, or Ctrl+C to exit..."
    read
fi

# Display menu
echo "[4/4] Select application to launch:"
echo ""
echo "  1. Original App (Classification only)"
echo "  2. Enhanced App (Classification + Employee Management + Assignment)"
echo "  3. Exit"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Launching Original Streamlit App..."
        echo "Access the app at: http://localhost:8501"
        echo "Press Ctrl+C to stop the server"
        echo ""
        streamlit run app/streamlit_app.py
        ;;
    2)
        echo ""
        echo "Launching Enhanced Streamlit App..."
        echo "Access the app at: http://localhost:8501"
        echo "Press Ctrl+C to stop the server"
        echo ""
        streamlit run app/streamlit_app_enhanced.py
        ;;
    3)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac
