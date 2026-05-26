CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra
# Add -fopenmp to CXXFLAGS and -lm to LDFLAGS if desired for faster CPU runs.
LDFLAGS  =

TARGET  = renderer
SRCS    = main.cpp
HEADERS = vec3.h geodesic.h camera.h shader.h renderer.h

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) output.ppm
