
CXX      = nvc++
CXXFLAGS = -O2 -std=c++14 -Minfo=accel
LIBS     = -lboost_program_options

SRC      = solver.cpp

all: solver_host solver_multi solver_gpu stages

solver_host: $(SRC)
	$(CXX) $(CXXFLAGS) -DSTAGE=4 -acc=host        -o $@ $(SRC) $(LIBS)
solver_multi: $(SRC)
	$(CXX) $(CXXFLAGS) -DSTAGE=4 -acc=multicore   -o $@ $(SRC) $(LIBS)
solver_gpu: $(SRC)
	$(CXX) $(CXXFLAGS) -DSTAGE=2 -acc=gpu -gpu=cc70 -o $@ $(SRC) $(LIBS)

stages: solver_gpu_s1 solver_gpu_s2 solver_gpu_s3 solver_gpu_s4

solver_gpu_s1: $(SRC)
	$(CXX) $(CXXFLAGS) -DSTAGE=1 -acc=gpu -gpu=cc70 -o $@ $(SRC) $(LIBS)
solver_gpu_s2: $(SRC)
	$(CXX) $(CXXFLAGS) -DSTAGE=2 -acc=gpu -gpu=cc70 -o $@ $(SRC) $(LIBS)
solver_gpu_s3: $(SRC)
	$(CXX) $(CXXFLAGS) -DSTAGE=3 -acc=gpu -gpu=cc70 -o $@ $(SRC) $(LIBS)
solver_gpu_s4: $(SRC)
	$(CXX) $(CXXFLAGS) -DSTAGE=4 -acc=gpu -gpu=cc70 -o $@ $(SRC) $(LIBS)

