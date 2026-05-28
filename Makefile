CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -I/opt/homebrew/opt/jpeg-turbo/include
# Add -fopenmp to CXXFLAGS and -lm to LDFLAGS if desired for faster CPU runs.
LDFLAGS  = -L/opt/homebrew/opt/jpeg-turbo/lib -ljpeg

TARGET  = renderer
SRCS    = main.cpp
HEADERS = vec3.h geodesic.h camera.h shader.h renderer.h

TEST_HEADERS = $(HEADERS) tests/test_util.h
TEST_BINS    = tests/test_geodesic tests/test_shading

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

# --- tests -------------------------------------------------------------
tests/test_geodesic: tests/test_geodesic.cpp $(TEST_HEADERS)
	$(CXX) $(CXXFLAGS) -I. -o $@ $< $(LDFLAGS)

tests/test_shading: tests/test_shading.cpp $(TEST_HEADERS)
	$(CXX) $(CXXFLAGS) -I. -o $@ $< $(LDFLAGS)

test: $(TEST_BINS)
	@./tests/test_geodesic
	@echo ""
	@./tests/test_shading

clean:
	rm -f $(TARGET) output.ppm output.jpg $(TEST_BINS)
