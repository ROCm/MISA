CC=g++

BUILD_FP16_CONFIGS := 1

ifdef BUILD_FP16_CONFIGS
CFLAGS := -std=c++11 -I/opt/rocm/include -I../ -g -DUSE_PRECISION_FP16
else
CFLAGS := -std=c++11 -I/opt/rocm/include -I../ -g 
endif 

PROGRAMS :=  reorder_configs  generate_configs produce_header

all: $(PROGRAMS)

# Step

reorder_configs: reorder_configs.o
	$(CC) -o $@ $< 
 
generate_configs: generate_configs.o
	$(CC) -o $@ $< 

produce_header: produce_header.o
	$(CC) -o $@ $< 


%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $< 


clean:
	rm -f *.o $(PROGRAMS)

