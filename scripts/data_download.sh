#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_message "$RED" "❌ Error: $1 is not installed"
        exit 1
    fi
}

# Check prerequisites
print_message "$YELLOW" "🔍 Checking prerequisites..."
check_command "uv"
check_command "curl"
check_command "wget"

# Create data directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed

# Function to download a file
download_file() {
    local url=$1
    local output_file=$2
    local description=$3

    print_message "$YELLOW" "📥 Downloading $description..."
    if [[ $url == *.zip ]]; then
        wget -q "$url" -O "$output_file"
        print_message "$YELLOW" "📦 Extracting $description..."
        unzip -q "$output_file" -d "data/raw"
        rm "$output_file"
    else
        wget -q "$url" -O "$output_file"
    fi
    print_message "$GREEN" "✅ Downloaded $description"
}

# Function to download from Kaggle
download_kaggle() {
    local dataset=$1
    local output_dir=$2
    local description=$3

    print_message "$YELLOW" "📥 Downloading $description from Kaggle..."
    uv run --with kaggle kaggle datasets download -d "$dataset" -p "$output_dir" --unzip
    print_message "$GREEN" "✅ Downloaded $description"
}

# Function to download from UCI
download_uci() {
    local dataset=$1
    local output_file=$2
    local description=$3

    print_message "$YELLOW" "📥 Downloading $description from UCI..."
    curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/$dataset" -o "$output_file"
    print_message "$GREEN" "✅ Downloaded $description"
}

# Example datasets (uncomment and modify as needed)

# # Iris dataset
# download_file \
#     "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv" \
#     "data/raw/iris.csv" \
#     "Iris dataset"

# # Titanic dataset
# download_kaggle \
#     "titanic" \
#     "data/raw/titanic" \
#     "Titanic dataset"

# # Wine dataset from UCI
# download_uci \
#     "wine/wine.data" \
#     "data/raw/wine.data" \
#     "Wine dataset"

print_message "$GREEN" "✅ Data download complete!"
print_message "$YELLOW" "📝 Next steps:"
echo "1. Review the downloaded data in data/raw/"
echo "2. Process the data using your data processing scripts"
echo "3. Update the data versioning with DVC if needed"
