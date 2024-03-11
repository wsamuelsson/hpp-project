WFLAGS=-Wall -Wextra 
CFLAGS=-fopenmp -O3 -march=native -lm


matrix: matrix.o
	gcc $(WFLAGS) -o matrix matrix.o $(CFLAGS)

matrix.o: matrix.c
	gcc $(CFLAGS) -c matrix.c 

clean:
	rm -f ./matrix *.o