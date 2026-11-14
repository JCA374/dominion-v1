# dominion.so = the shared engine (engine/c, git submodule) + the GA agent (c/ga_agent.c).
# ga/c_bridge.py also builds it automatically when missing or stale.
dominion.so: engine/c/engine.c engine/c/engine.h c/ga_agent.c
	gcc -O3 -shared -fPIC -Wall -Wextra -Wno-unused-parameter -I engine/c -o $@ engine/c/engine.c c/ga_agent.c

clean:
	rm -f dominion.so

.PHONY: clean
