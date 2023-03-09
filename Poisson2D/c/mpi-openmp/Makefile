# the compiler
CC = mpicc
 
# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings
CFLAGS  = -g -Wall -O3
 
# The build target 
TARGET = main
LIBS = -lm -fopenmp

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) *.c $(LIBS) -o $(TARGET).out
 
clean:
	$(RM) $(TARGET).out
