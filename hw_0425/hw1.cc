#include <vector>
#define BIN_TO_GRAY(N) ((N >> 1) ^ N)

void convertArrayBinToGray(int* input, int* output, int length) {
  for (int i = 0; i < length; ++i) {
    output[i] = BIN_TO_GRAY(input[i]);
  }
}

void extractBitPlane(unsigned short planeBit, int* input, int* output, int length) {
  for (int i = 0; i < length; ++i) {
    output[i] = (input[i] >> planeBit) & 1;
  }
}

double compute1DRunLengthEntropy(int symbolCount, int* input, int length) {
  std::vector<std::vector<int>> runLengthMap;
  runLengthMap.resize(symbolCount);

  int currSymbol = input[0];
  int runLength = 0;

  for (int i = 0; i < length; ++i) {
    if (input[i] == currSymbol) {
      runLength++;
    }
    else {
      runLengthMap[currSymbol].push_back(runLength);
      currSymbol = input[i];
      runLength = 1;
    }
  }

  runLengthMap[currSymbol].push_back(runLength);

  double totalEntropies = 0;
  double totalAvgLength = 0;

  for (int i = 0; i < symbolCount; ++i) {
    std::vector<int> currentRunLength = runLengthMap[i];
    int* ptrRunLength = new int[currentRunLength.size()];

    for (int i = 0; i < currentRunLength.size(); ++i) {
      ptrRunLength[i] = currentRunLength[i];
    }

    double entropy = computeEntropy(ptrRunLength, currentRunLength.size());
    double avgLength = 0;

    for (int j = 0; j < currentRunLength.size(); ++j) {
      avgLength += currentRunLength[j];
    }
    avgLength /= currentRunLength.size();
    // totalEntropies += entropy / avgLength;
    totalEntropies += entropy;
    totalAvgLength += avgLength;
    delete [] ptrRunLength;
  }
  return totalEntropies / totalAvgLength;
}
