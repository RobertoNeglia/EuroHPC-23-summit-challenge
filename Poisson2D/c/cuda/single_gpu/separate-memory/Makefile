# the compiler
CC = nvcc
 
# compiler flags:
#  -g     - this flag adds debugging information to the executable file
CFLAGS  = -Xcompiler -O3
 
# The build target 
TARGET = main
LIBS =

$(TARGET): $(TARGET).cu
	$(CC) $(CFLAGS) $(LIBS) -o $(TARGET).out *.cu
 
clean:
	$(RM) $(TARGET).out
