#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vector>

int loadMNISTData() {
    /*
     * CONSTANTS DEFINED FROM MNIST WEBSITE
     * http://yann.lecun.com/exdb/mnist/
     */

    const int MAGIC_OFFSET = 0;
    const int OFFSET_SIZE = 4;

    const int LABEL_MAGIC = 2049;
    const int IMAGE_MAGIC = 2051;

    const int NUMBER_ITEMS_OFFSET = 4;
    const int ITEMS_SIZE = 4;

    const int NUMBER_OF_ROWS_OFFSET = 8;
    const int ROWS_SIZE = 4;
    const int ROWS = 28;

    const int NUMBER_OF_COLUMNS_OFFSET = 12;
    const int COLUMNS_SIZE = 4;
    const int COLUMNS = 28;

    const int IMAGE_OFFSET = 16;
    const int IMAGE_SIZE = ROWS * COLUMNS;

}

int main(int argc, char const *argv[])
{
    printf("Hello, World!\n");

    return 0;
}