# Check if data directory exists, if not download dataset
if [ ! -d "data" ]; then
    echo "Data directory not found. Downloading dataset..."
    wget http://images.cocodataset.org/zips/unlabeled2017.zip
    unzip unlabeled2017.zip
    mv unlabeled2017 data
    rm unlabeled2017.zip
fi

# Check if depth-anything-3 directory exists, if not clone the repository and install
if [ ! -d "depth-anything-3" ]; then
    echo "depth-anything-3 directory not found. Cloning repository and installing..."
    git clone https://github.com/ByteDance-Seed/depth-anything-3.git
    cd depth-anything-3
    pip install -e .
    cd ..
fi
