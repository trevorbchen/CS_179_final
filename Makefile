CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra
LDFLAGS  = -ljpeg

# --- Portable libjpeg-turbo discovery -----------------------------------
# On macOS the library lives under Homebrew; on Linux it is on the default
# include/lib search paths (verified: /usr/include/jpeglib.h, -ljpeg).
# Detect the platform so the same Makefile builds the CPU baseline on both.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  BREW_JPEG := $(shell brew --prefix jpeg-turbo 2>/dev/null)
  ifneq ($(BREW_JPEG),)
    CXXFLAGS += -I$(BREW_JPEG)/include
    LDFLAGS  := -L$(BREW_JPEG)/lib $(LDFLAGS)
  endif
endif
# Add -fopenmp to CXXFLAGS and -lm to LDFLAGS if desired for faster CPU runs.

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
