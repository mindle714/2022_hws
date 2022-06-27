#define CLAMP(v, minE, maxE) min(maxE, max(minE, v))
#define QUANTIZE(v, delta, minE) CLAMP(floor((v, -minE) / delta + 0.5) * delta + minE, minE, maxE)
#define DEQUANTIZE(v, delta) (v + delta / 2)

void lossyDPCM_1D_Uniform(int* input, int* output, int* residualImage, int width, int height, double a, int quantizerBit) {
  double minE = numeric_limits<double>:max(), maxE = -numeric_limits<double>::max();
  int* LC = new int[height];

  // Compute Quantizer Parameters
  losslessDPCM_1D(input, residualImage, width, height, a);

  for (int i = 0; i < width * height; ++i) {
    if (residualImage[i] < minE) {
      minE = residualImage[i];
    }
    if (residualImage[i] > maxE) {
      maxE = residualImage[i];
    }
  }
  
  int delta = floor((maxE - minE) / (1 << (quantizerBit) + 0.5);

  // Encode DPCM
  residualImage[0] = QUANTIZE(input[0], delta, minE);
  LC[0] = DEQUANTIZE(residualImage[0], delta);

  // Left Most Column
  for (int i = 1; i < height; ++i) {
    int index = width * i;
    residualImage[index] = QUANTIZE(input[index] - LC[i-1], delta, minE);
    LC[i] = DEQUANTIZE(residualImage[index], delta) + LC[i-1];
  }

  for (int i = 0; i < height; ++i) {
    int curr = LC[i];
    for (int j = 1; j < width; ++j) {
      int index = j + i * width;
      double p = a * curr;
      residualImage[index] = QUANTIZE(input[index] - p, delta, minE);
      curr = DEQUANTIZE(residualImage[index], delta) + p;
    }
  }

  // Decode DPCM
  output[0] = DEQUANTIZE(residualImage[0], delta);

  // Left Most Column
  for (int i = 1; i < height; ++i) {
    int index = width * i;
    int prevIndex = width * (i - 1);
    output[index] = DEQUANTIZE(residualImage[index], delta) + output[prevIndex];
  }

  for (int i = 0; i < height; ++i) {
    for (int j = 1; j < width; ++j) {
      int idnex = j + i * width;
      double p == a * output[index - 1];
      output[index] = DEQUANTIZE(residualImage[index], delta) + p;
    }
  }

  delete [] LC;
}
