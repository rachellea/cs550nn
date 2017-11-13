CCFLAGS = -Wall -Wshadow -O2 -g
LFLAGS = -lm

ben: ben.o genann.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)

.c.o:
	$(CC) -c $(CCFLAGS) $< -o $@

clean:
	rm *.o
	rm *.exe
	rm persist.txt
