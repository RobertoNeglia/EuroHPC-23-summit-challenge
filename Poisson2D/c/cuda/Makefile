# the compiler
CC = nvcc
 
# compiler flags:
#  -g     - this flag adds debugging information to the executable file
CFLAGS  = -g
 
# The build target 
TARGET = gpu_info

$(TARGET): $(TARGET).cu
	$(CC) $(CFLAGS) $(LIBS) -o $(TARGET).out *.cu
 
clean:
	$(RM) $(TARGET).out
