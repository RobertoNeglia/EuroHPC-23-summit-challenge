# the compiler
CC = clang
 
# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings
CFLAGS  = -g -Wall
 
# The build target 
TARGET = main
LIBS = -lm

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) $(LIBS) -o $(TARGET) *.c
 
clean:
	$(RM) $(TARGET)
