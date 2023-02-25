# the compiler
CC = nvcc
 
# compiler flags:
#  -g     - this flag adds debugging information to the executable file
CFLAGS  = -g
 
# The build target 
TARGET = main
LIBS = -lm

$(TARGET): $(TARGET).cu
	$(CC) $(CFLAGS) $(LIBS) -o $(TARGET) *.cu
 
clean:
	$(RM) $(TARGET)
