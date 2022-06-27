void lossyDPCM_2D_Uniform_3rd_Order(int* input, int* output, int* residualImage, int width, int height, double a, double b, double c, int quantizerBit) {
  double minE = numeric_limits<double>::max(), maxE = -numeric_limits<double>::max();
  int* LC = new int[height];
  int* CR = new int[width];
  int* UR = new int[width];
  
  // Compute Quantizer Parameters
  losslessDPCM_2D_Order3(input, residualImage, width, height, a, b, c);
  
  for (int i = 0; i < width * height; ++i) {
    if (residualImage[i] < minE) {
	  minE = residualImage[i];
	}
	if (residualImage[i] > maxE) {
	  maxE = residualImage[i];
	}
  }
  int delta = floor((maxE - minE) / (1 << quantizerBit) + 0.5);
  
  // Encode DPCM
  residualImage[0] = QUANTIZE(input[0], delta, minE);
  LC[0] = CR[0] = DEQUANTIZE(residualImage[0], delta);
  // Top Row
  for (int i = 1; i < width; ++i) {
    residualImage[i] = QUANTIZE(input[i] - CR[i - 1], delta, minE);
	CR[i] = DEQUANTIZE(residualImage[i], delta) + CR[i - 1];
  }
  
  // Left Most Column
  for (int i = 1; i < height; ++i) {
    int indexL = width * i;
  	residualImage[indexL] = QUANTIZE(input[indexL] - LC[i - 1], delta, minE);
  	LC[i] = DEQUANTIZE(residualImage[indexL], delta) + LC[i - 1];
  }

  for (int i = 1; i < height; ++i) {
    memcpy(UR, CR, sizeof(int) * width);
    CR[0] = LC[i];
    for (int j = 1; j < width; ++j) {
      int index = j + i * width;
      double p = a * CR[j - 1] + b * UR[j - 1] + c * UR[j];
      residualImage[index] = QUANTIZE(input[index] - p, delta, minE);
      CR[j] = DEQUANTIZE(residualImage[index], delta) + p;
    }
  }
  
  // Decode DPCM
  output[0] = DEQUANTIZE(residualImage[0], delta);

  // Top row
  for (int i = 1; i < width; ++i) {
    output[i] = DEQUANTIZE(residualImage[i] + output[i - 1], delta);
  }

  // Left Most Column
  for (int i = 1; i < height; ++i) {
    int indexL = width * i;
    int prevIndexL = width * (i - 1);
    output[indexL] = DEQUANTIZE(residualImage[indexL], delta) + output[prevIndexL];
  }

  for (int i = 1; i < height; ++i) {
    for (int j = 1; j < width; ++j) {
      int index = j + i * width;
      int indexB = j + (i - 1) * width - 1;
      int indexC = indexB + 1;
      int p = a * output[index - 1] + b * output[indexB] + c * output[indexC];
      output[index] = DEQUANTIZE(residualImage[index], delta) + p;
    }
  }

  delete [] LC;
  delete [] CR;
  delete [] UR;
}
