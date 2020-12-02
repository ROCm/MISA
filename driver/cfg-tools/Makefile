CC=g++

CFLAGS := -std=c++11 -I../ -g

PROGRAMS := tailor_configs  reorder_configs  generate_configs

all: $(PROGRAMS)

# Step

tailor_configs:  tailor_configs.o
	$(CC) -o $@ $< 

reorder_configs: reorder_configs.o
	$(CC) -o $@ $< 
 
generate_configs: generate_configs.o
	$(CC) -o $@ $< 

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $< 


clean:
	rm -f *.o $(PROGRAMS)

