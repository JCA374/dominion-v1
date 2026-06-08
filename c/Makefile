CC = gcc
CFLAGS = -O3 -shared -fPIC -Wall
OUT = ../dominion.so

$(OUT): dominion.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(OUT)

.PHONY: clean
