CC = gcc
CFLAGS = -O3 -shared -fPIC -Wall

dominion.so: dominion.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f dominion.so

.PHONY: clean
