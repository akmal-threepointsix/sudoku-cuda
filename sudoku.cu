// CUDA runtime
#include <cuda_runtime.h>

// C++ libraries
#include <iostream>
#include <fstream>
#include <sstream>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

// CUDA error checking. Source: https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
static const char* _cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}
template <typename T>
void check(T result, char const* const func, const char* const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		exit(EXIT_FAILURE);
	}
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
// End

class Sudoku {
public:
	Sudoku() {
		memset(grid, 0, GRID_DIM * GRID_DIM);
	}

	void loadFromFile(std::ifstream& sudokuStream) {
		for (int row = 0; row < GRID_DIM; row++) {
			string rowString;
			std::getline(sudokuStream, rowString);
			for (int column = 0; column < GRID_DIM; column++) {
				grid[row * GRID_DIM + column] = rowString[column] - '0';
			}
		}
	}

	void print() {
		for (int i = 0; i < TOTAL_CELLS; i++) {
			if (i % 27 == 0) cout << "|" << endl << "|--------------------";
			if (i % 9 == 0) cout << "|" << endl;
			if (i % 3 == 0) cout << "|";
			if (grid[i]) {
				cout << (int)grid[i] << " ";
			}
			else {
				cout << ". ";
			}
		}
		cout << "|" << endl;
	}

	static const int GRID_DIM = 9;
	static const int BOX_DIM = 3;
	static const int TOTAL_CELLS = GRID_DIM * GRID_DIM;
	unsigned char grid[GRID_DIM * GRID_DIM];
};

enum class SudokuStatus {
	Added, // Added some digits to the sudoku
	NothingAdded, // Didn't add any digit (but there are valid options)
	Failed, // Current sudoku cannot be solved because of some incorrect guess
	Solved // Sudoku is completely and correctly solved
};

// Structure of Arrays
struct UsedDigitsBitmasks {
	uint16_t rowContains[9];
	uint16_t columnContains[9];
	uint16_t boxContains[9];
};

// Find used digits in rows, columns and boxes. Save used digits in appropriate bitmasks
// Side effects: Set sudokuStatus=failed, if some row, column or box contains two same digits
__device__ void findUsedDigits(UsedDigitsBitmasks* sm_usedDigitsBitmasks, uint8_t* currentBlockGrid, SudokuStatus& sudokuStatus) {
	if (threadIdx.x < Sudoku::GRID_DIM) { // Rows
		sm_usedDigitsBitmasks->rowContains[threadIdx.x] = 0;
		for (uint8_t row = 0; row < Sudoku::GRID_DIM; row++) {
			int digit = currentBlockGrid[threadIdx.x * Sudoku::GRID_DIM + row];
			if (digit) {
				if (sm_usedDigitsBitmasks->rowContains[threadIdx.x] >> (digit - 1) & 1) { // Two same digits in one row
					sudokuStatus = SudokuStatus::Failed;
				}
				sm_usedDigitsBitmasks->rowContains[threadIdx.x] |= 1 << (digit - 1);
			}
		}
	}
	else if (Sudoku::GRID_DIM <= threadIdx.x && threadIdx.x < Sudoku::GRID_DIM * 2) { // Columns
		const uint8_t orderIdx = threadIdx.x - Sudoku::GRID_DIM; // [0, 8]
		sm_usedDigitsBitmasks->columnContains[orderIdx] = 0;
		for (uint8_t column = 0; column < Sudoku::GRID_DIM; column++) {
			uint8_t digit = currentBlockGrid[column * Sudoku::GRID_DIM + orderIdx];
			if (digit) {
				if (sm_usedDigitsBitmasks->columnContains[orderIdx] >> (digit - 1) & 1) { // Two same digits in one column
					sudokuStatus = SudokuStatus::Failed;
				}
				sm_usedDigitsBitmasks->columnContains[orderIdx] |= 1 << (digit - 1);
			}
		}
	}
	else if (Sudoku::GRID_DIM * 2 <= threadIdx.x && threadIdx.x < Sudoku::GRID_DIM * 3) { // Boxes
		const uint8_t orderIdx = threadIdx.x - Sudoku::GRID_DIM * 2; // [0, 8]
		sm_usedDigitsBitmasks->boxContains[orderIdx] = 0;
		for (uint8_t row = (orderIdx / 3) * 3; row < ((orderIdx / 3 + 1) * 3); row++) {
			for (uint8_t column = (orderIdx % 3) * 3; column < ((orderIdx % 3 + 1) * 3); column++) {
				uint8_t digit = currentBlockGrid[row * Sudoku::GRID_DIM + column];
				if (digit) {
					if ((sm_usedDigitsBitmasks->boxContains[orderIdx] >> (digit - 1)) & 1) { // Two same digits in one box
						sudokuStatus = SudokuStatus::Failed;
					}
					sm_usedDigitsBitmasks->boxContains[orderIdx] |= 1 << (digit - 1);
				}
			}
		}
	}
}

__device__ void tryAddDigit(uint16_t& allPossibleDigits, UsedDigitsBitmasks* sm_usedDigitsBitmasks, uint8_t* currentBlockGrid, SudokuStatus& sm_sudokuStatus) {
	sm_sudokuStatus = SudokuStatus::NothingAdded;

	uint8_t row = threadIdx.x / Sudoku::GRID_DIM;
	uint8_t column = threadIdx.x % Sudoku::GRID_DIM;
	allPossibleDigits = (sm_usedDigitsBitmasks->rowContains[row]
		| sm_usedDigitsBitmasks->columnContains[column]
		| sm_usedDigitsBitmasks->boxContains[(row / 3) * 3 + (column / 3)]);

	uint16_t unusedDigit = 0;
	for (uint8_t possibleDigit = 0; possibleDigit < Sudoku::GRID_DIM; possibleDigit++) {
		const bool isUnused = (allPossibleDigits & (1 << possibleDigit)) == 0;
		if (isUnused) {
			if (unusedDigit != 0) { // More than one unused digit
				unusedDigit = 10;
				break;
			}
			else {
				unusedDigit = possibleDigit + 1;
			}
		}
	}
	if (unusedDigit == 0) { // Could not find any unused digit
		sm_sudokuStatus = SudokuStatus::Failed;
	}
	else if (unusedDigit <= 9) { // Found exactly one unused digit
		currentBlockGrid[threadIdx.x] = unusedDigit;
		sm_sudokuStatus = SudokuStatus::Added;
	}
}

// Idea source: https://github.com/evcu/cuda-sudoku-solver
// Fill cells that can be filled by only one digit
// Side effects: If there are no such cells, then find cells with the minimum number of possible digits, 
// create new sudokus for each possible digit and repeat the process for new sudokus
__global__ void fillSudoku(unsigned char* gm_allSudokuGrids, unsigned char* gm_solvedSudoku, uint32_t* gm_isBlockActive, bool* gm_isSolved) {
	uint8_t* currentBlockGrid = gm_allSudokuGrids + (Sudoku::TOTAL_CELLS * blockIdx.x); // Get appropriate grid with pointer arithmetics

	const bool isBlockActive = gm_isBlockActive[blockIdx.x] == 1;
	const bool isThreadActive = threadIdx.x < 81; // 81 threads for 81 cells

	__shared__ SudokuStatus sm_sudokuStatus;
	__shared__ UsedDigitsBitmasks sm_usedDigitsBitmasks; // Bitmasks for used digits
	__shared__ int sm_minPossibleDigits;
	__shared__ int sm_schedulingThread;

	if (isBlockActive && isThreadActive) {
		const bool isFirstThread = threadIdx.x == 0;
		if (isFirstThread) {
			sm_sudokuStatus = SudokuStatus::Added;
		}
		__syncthreads();

		uint16_t allPossibleDigits;
		while (sm_sudokuStatus == SudokuStatus::Added) { // This loop fills cells that can be filled with only one digit
			findUsedDigits(&sm_usedDigitsBitmasks, currentBlockGrid, sm_sudokuStatus);
			__syncthreads();

			if (sm_sudokuStatus != SudokuStatus::Failed) {
				if (isFirstThread) {
					sm_sudokuStatus = SudokuStatus::Solved; // Assume that the sudoku is solved
				}
				__syncthreads();
				allPossibleDigits = 0;
				if (currentBlockGrid[threadIdx.x] == 0) {
					tryAddDigit(allPossibleDigits, &sm_usedDigitsBitmasks, currentBlockGrid, sm_sudokuStatus);
				}
			}
			__syncthreads();
		}

		const bool isSolved = sm_sudokuStatus == SudokuStatus::Solved;
		const bool isFailed = sm_sudokuStatus == SudokuStatus::Failed;
		const bool cannotAdd = sm_sudokuStatus == SudokuStatus::NothingAdded;

		if (isSolved && isFirstThread) { // Success. Sudoku was completely and correctly solved
			memcpy(gm_solvedSudoku, currentBlockGrid, Sudoku::TOTAL_CELLS);
			*gm_isSolved = true;
		}
		else if (isFailed && isFirstThread) { // Failed. Stop solving this particular sudoku
			gm_isBlockActive[blockIdx.x] = 0;
		}
		else if (cannotAdd) { // Fork
			if (isFirstThread) {
				sm_minPossibleDigits = 9;
				sm_schedulingThread = blockDim.x;
			}
			__syncthreads();

			uint8_t possibleDigitsCount = 0;
			if (allPossibleDigits != 0) {
				for (int possibleDigit = 0; possibleDigit < 9; possibleDigit++) {
					if ((allPossibleDigits & (1 << possibleDigit)) == 0) {
						possibleDigitsCount++;
					}
				}
				atomicMin(&sm_minPossibleDigits, possibleDigitsCount);
			}
			__syncthreads();

			if (possibleDigitsCount == sm_minPossibleDigits) {
				atomicMin(&sm_schedulingThread, threadIdx.x);
			}
			__syncthreads();

			if (sm_schedulingThread == threadIdx.x) {
				//Find a suitable block to schedule the fork for each extra value.
				for (int i = 0, k = 1; i < 9; i++) {
					if ((allPossibleDigits & (1 << i)) == 0) {
						if (k == 1) {
							// first possibility stays with the current block
							currentBlockGrid[threadIdx.x] = i + 1;
						}
						else {
							// look for suitable block
							for (int j = 0; j < gridDim.x; j++) {
								atomicCAS(gm_isBlockActive + j, 0, gridDim.x * blockIdx.x + threadIdx.x + 2);
								if (gm_isBlockActive[j] == (gridDim.x * blockIdx.x + threadIdx.x + 2)) {
									memcpy(gm_allSudokuGrids + j * 81, currentBlockGrid, Sudoku::TOTAL_CELLS);
									gm_allSudokuGrids[j * 81 + threadIdx.x] = i + 1;
									gm_isBlockActive[j] = 1;
									break;
								}
							}
						}
						k++;
					}
				}
			}
			__syncthreads();
		}
	}
}

void SolveSudoku(Sudoku* sudoku) {
	// Initialize variables
	const int h_threadsNum = 96; // Must be multiple of 32 and greater than 81
	const int h_blocksNum = 20000; // Some random big number

	unsigned char* d_allSudokuGrids;
	checkCudaErrors(cudaMalloc(&d_allSudokuGrids, Sudoku::TOTAL_CELLS * h_blocksNum)); // All grids are stored here
	checkCudaErrors(cudaMemcpy(d_allSudokuGrids, (*sudoku).grid, Sudoku::TOTAL_CELLS, cudaMemcpyHostToDevice));

	unsigned char* d_solvedSudoku;
	checkCudaErrors(cudaMalloc(&d_solvedSudoku, Sudoku::TOTAL_CELLS));

	uint32_t* d_isBlockActive;
	checkCudaErrors(cudaMalloc(&d_isBlockActive, h_blocksNum * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset(d_isBlockActive, 0, h_blocksNum * sizeof(uint32_t)));
	checkCudaErrors(cudaMemset(d_isBlockActive, 1, 1)); // d_isBlockActive[0] = true;

	bool* d_isSolved;
	checkCudaErrors(cudaMalloc(&d_isSolved, 1));
	checkCudaErrors(cudaMemset(d_isSolved, false, 1)); // *d_isSolved = false;

	bool h_isSolved = false;
	// Solving sudoku
	while (!h_isSolved) { // Limit it by 100 iterations, so it never gets stuck?
		fillSudoku<<<h_blocksNum, h_threadsNum>>>(d_allSudokuGrids, d_solvedSudoku, d_isBlockActive, d_isSolved);
		checkCudaErrors(cudaDeviceSynchronize()); // is it necessary?
		checkCudaErrors(cudaMemcpy(&h_isSolved, d_isSolved, 1, cudaMemcpyDeviceToHost));
	}

	// Copying solved sudoku to host
	checkCudaErrors(cudaMemcpy((*sudoku).grid, d_solvedSudoku, Sudoku::TOTAL_CELLS, cudaMemcpyDeviceToHost));

	// Cleanup
	checkCudaErrors(cudaFree(d_allSudokuGrids));
	checkCudaErrors(cudaFree(d_solvedSudoku));
	checkCudaErrors(cudaFree(d_isBlockActive));
	checkCudaErrors(cudaFree(d_isSolved));
}

int main(int argc, char** argv) {
	// Usage
	if (argc < 2) {
		cerr << "Usage: sudoku filename" << endl;
		return -1;
	}

	// Load Sudoku from file
	std::ifstream sudokuFile(argv[1]);
	if (sudokuFile.fail()) {
		cout << argv[1] << " not found!" << endl;
		return false;
	}

	int sudokuCounter = 0;
	double totalTimeSec = 0;
	clock_t startTime, endTime;
	string newline;
	do {
		Sudoku h_sudoku;
		h_sudoku.loadFromFile(sudokuFile);
		// Print unsolved sudoku
		h_sudoku.print();

		startTime = clock();
		SolveSudoku(&h_sudoku);
		endTime = clock();

		// Print solved sudoku
		h_sudoku.print();

		totalTimeSec += (double(endTime - startTime) / CLOCKS_PER_SEC);
		sudokuCounter++;
	} while (std::getline(sudokuFile, newline));
	cout << endl;
	cout << "Solved " << sudokuCounter << " sudokus in " << totalTimeSec << " seconds!" << endl;

	return 0;
}