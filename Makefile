CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra
# Add -fopenmp to CXXFLAGS and -lm to LDFLAGS if desired for faster CPU runs.
LDFLAGS  =

TARGET  = renderer
HEADERS = vec3.h geodesic.h camera.h shader.h renderer.h config.h stb_image.h
OBJS    = main.o stb_image_impl.o

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

main.o: main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c main.cpp -o $@

# stb_image is a vendored third-party single-header lib; build its
# implementation TU with warnings off so -Wextra stays clean for our code.
stb_image_impl.o: stb_image_impl.cpp stb_image.h
	$(CXX) -O2 -std=c++17 -w -c stb_image_impl.cpp -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS) output.ppm
