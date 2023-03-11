# the compiler
CC = gcc
 
# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings
CFLAGS  = -pg -Wall -O3
 
# The build target 
TARGET = main
LIBS = -lm

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) *.c $(LIBS) -o $(TARGET).out
 
clean:
	$(RM) $(TARGET).out
