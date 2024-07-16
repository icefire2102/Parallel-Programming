#include <stdio.h>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include<C:\Users\yashs\Desktop\stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "C:\Users\yashs\Desktop\stb_image_write.h"
#include <omp.h>
void writeJPEG(const int* image, int width, int height, const char* filename) {
    stbi_write_jpg(filename, width, height, 1, image, 100);
}

void histogramEqualizationAndWrite(const int* inputImage, int* outputImage, int width, int height) {
    int min_val = inputImage[0];
    int max_val = inputImage[0];
    int i, j;
#pragma omp parallel for shared(min_val, max_val) private(i)
    for (i = 0; i < width * height; i++) {
#pragma omp critical
        {
            if (inputImage[i] < min_val) {
                min_val = inputImage[i];
            }
            if (inputImage[i] > max_val) {
                max_val = inputImage[i];
            }
        }
    }
    float scale = 1.0 / (max_val - min_val);
#pragma omp parallel for shared(inputImage, outputImage, min_val, scale) private(i, j)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            int index = i * width + j;
            outputImage[index] = (int)((inputImage[index] - min_val) * scale);
        }
    }
    writeJPEG(outputImage, width, height, "histogramEqual.jpg");
}


void applyCustomKernelAndWrite(const int* inputImage, int* outputImage, const float* kernel, int kernelSize, int width, int height) {
#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0;
            for (int ky = -kernelSize / 2; ky <= kernelSize / 2; ky++) {
                for (int kx = -kernelSize / 2; kx <= kernelSize / 2; kx++) {
                    int cx = x + kx;
                    int cy = y + ky;
                    if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                        sum += inputImage[cy * width + cx] * kernel[(ky + kernelSize / 2) * kernelSize + (kx + kernelSize / 2)];
                    }
                }
            }
            outputImage[y * width + x] = (int)sum;
        }
    }
    writeJPEG(outputImage, width, height, "custom_kernel.jpg");
}
void edgeDetectionAndWrite(const int* inputImage, int* outputImage, int width, int height) {
#pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int dx = inputImage[(y + 1) * width + (x - 1)] - inputImage[(y - 1) * width + (x - 1)]
                + 2 * inputImage[(y + 1) * width + x] - 2 * inputImage[(y - 1) * width + x]
                + inputImage[(y + 1) * width + (x + 1)] - inputImage[(y - 1) * width + (x + 1)];
            int dy = inputImage[(y - 1) * width + (x + 1)] - inputImage[(y - 1) * width + (x - 1)]
                + 2 * inputImage[y * width + (x + 1)] - 2 * inputImage[y * width + (x - 1)]
                + inputImage[(y + 1) * width + (x + 1)] - inputImage[(y + 1) * width + (x - 1)];
            outputImage[y * width + x] = abs(dx) + abs(dy);
        }
    }
    writeJPEG(outputImage, width, height, "edge_detection.jpg");
}

void blurImageAndWrite(const int* inputImage, int* outputImage, int width, int height) {
#pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    sum += inputImage[(y + dy) * width + (x + dx)];
                }
            }
            outputImage[y * width + x] = sum / 9;
        }
    }
    writeJPEG(outputImage, width, height, "blur_image.jpg");
}


unsigned char** readImage(const char* fileName, int* width, int* height) {
    int channels;
    unsigned char* imageData = stbi_load(fileName, width, height, &channels, STBI_rgb);
    if (!imageData)
        return NULL;

    unsigned char** imageMatrix = (unsigned char**)malloc(*height * sizeof(unsigned char*));
    if (!imageMatrix) {
        stbi_image_free(imageData);
        return NULL;
    }

    for (int i = 0; i < *height; ++i) {
        imageMatrix[i] = (unsigned char*)malloc(*width * sizeof(unsigned char));
        if (!imageMatrix[i]) {
            for (int j = 0; j < i; ++j)
                free(imageMatrix[j]);
            free(imageMatrix);
            stbi_image_free(imageData);
            return NULL;
        }
        if (channels == 3)
        {
            printf("%d", channels);
            for (int j = 0; j < *width; ++j) {
                unsigned char r = imageData[(i * (*width) + j) * channels];
                unsigned char g = imageData[(i * (*width) + j) * channels + 1];
                unsigned char b = imageData[(i * (*width) + j) * channels + 2];
                imageMatrix[i][j] = (unsigned char)(0.2126f * r + 0.7152f * g + 0.0722f * b);
            }
        }
    }

    stbi_image_free(imageData);

    return imageMatrix;
}
int main() {
    const char* fileName = "gray.jpg";
    int width, height, channels;

    unsigned char** imageMatrix = readImage(fileName, &width, &height);
    if (!imageMatrix) {
        printf("Error: Unable to read image.\n");
        return 1;
    }
    int* outputImage1 = (int*)malloc(width * height * sizeof(int));
    int* outputImage2 = (int*)malloc(width * height * sizeof(int));
    int* outputImage3 = (int*)malloc(width * height * sizeof(int));
    int* outputImage4 = (int*)malloc(width * height * sizeof(int));
    int* imageMatrix1 = (int*)malloc(width * height * sizeof(int));
    int* imageMatrix2 = (int*)malloc(width * height * sizeof(int));
    int* imageMatrix3 = (int*)malloc(width * height * sizeof(int));
    memcpy(imageMatrix1, imageMatrix, width * height * sizeof(int));
    memcpy(imageMatrix2, imageMatrix, width * height * sizeof(int));
    memcpy(imageMatrix3, imageMatrix, width * height * sizeof(int));
#pragma omp parallel shared(imageMatrix)
    {
#pragma omp sections
        {
#pragma omp section
            {
#pragma omp section
                {
                    blurImageAndWrite(imageMatrix, outputImage1, width, height);
                    printf("Blurring completed!! /n Image stored at blur_image.jpg");
                }
#pragma omp section
                {
                    histogramEqualizationAndWrite(imageMatrix1, outputImage2, width, height);
                    printf("Histogram Equalization completed!! /n Image stored at histogramEqual.jpg");
                }

#pragma omp section
                {
                    edgeDetectionAndWrite(imageMatrix2, outputImage3, width, height);
                    printf("Edge Detection completed!! /n Image stored at edge_detection.jpg");
                }

#pragma omp section
                {
                    float customKernel[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
                    applyCustomKernelAndWrite(imageMatrix3, outputImage4, customKernel, 3, width, height);
                    printf("Custom Kernel completed!! /n Image stored at custom_kernel.jpg");
                }
            }
        }
    }

    free(imageMatrix);
    free(imageMatrix1);
    free(imageMatrix2);
    free(imageMatrix3);
    free(outputImage1);
    free(outputImage2);
    free(outputImage3);
    free(outputImage4);

    return 0;
}



