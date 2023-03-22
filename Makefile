main: sudoku.cu
	nvcc sudoku.cu -o sudoku
clean:
	rm sudoku
.PHONY:
	clean