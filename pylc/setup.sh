# Clone pybind repository.
git clone --recursive https://github.com/pybind/pybind11.git

cd build

# Cmake.
cmake -DPYTHON_EXECUTABLE=`which python` ..

# Make.
make

cd ..