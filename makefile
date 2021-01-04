CXX=g++
CXXFLAGS=-std=c++17 -O3 -DNDEBUG

all: mwe2vec

mwe2vec : src/mwe2vec.cpp
	$(CXX) src/mwe2vec.cpp -o mwe2vec $(CXXFLAGS) -pthread -licuuc

clean:
	rm -rf mwe2vec
