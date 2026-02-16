CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
LIBS = -lraylib -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
else
LIBS = -lraylib -lm -ldl -lpthread -lX11
endif

TARGET = perceptrome_scope
SOURCES = main.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
