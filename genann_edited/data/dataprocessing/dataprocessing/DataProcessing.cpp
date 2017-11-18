#include <iostream>
#include <fstream>
#include <vector>
#include <errno.h>
#include <cstdint>
using namespace std;

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

    vector<int> labels(0);
    vector<vector<uint8_t>> images(0);


    // Open file input streams and check for validity
    // string labelFileName = "t10k-labels.idx1-ubyte";
    // string imageFileName = "t10k-images.idx3-ubyte";
    string labelFileName = "train-labels.idx1-ubyte";
    string imageFileName = "train-images.idx3-ubyte";

    ifstream labelInputStream(labelFileName.c_str());
    ifstream imageInputStream(imageFileName.c_str());

    if(!labelInputStream) {
        cerr << "error: open labels file for input failed!" << endl;
        cout << errno << endl;
        return 1;
    }

    if(!imageInputStream) {
        cerr << "error: open images file for input failed!" << endl;
        cout << errno << endl;
        return 1;
    }


    // Read data from the files into vectors of uint8_t
    vector<uint8_t> labelBytes(0);
    vector<uint8_t> imageBytes(0);
    char byte;

    while(labelInputStream.get(byte)) {
        uint8_t a = (uint8_t)byte;
        labelBytes.push_back(a);
    }

    while(imageInputStream.get(byte)) {
        uint8_t a = (uint8_t)byte;
        imageBytes.push_back(a);
    }


    // Get the magic numbers from the file and check against constant values
    vector<uint8_t> vLabelMagic(labelBytes.begin(), labelBytes.begin()+OFFSET_SIZE);
    vector<uint8_t> vImageMagic(imageBytes.begin(), imageBytes.begin()+OFFSET_SIZE);

    int iLabelMagic = int(vLabelMagic[0] << 24 | vLabelMagic[1] << 16 |
                          vLabelMagic[2] << 8 | vLabelMagic[3]);
    int iImageMagic = int(vImageMagic[0] << 24 | vImageMagic[1] << 16 |
                          vImageMagic[2] << 8 | vImageMagic[3]);

    if(iLabelMagic != LABEL_MAGIC) {
        cerr << "Bad magic number in label file!" << endl;
        return 2;
    }

    if(iImageMagic != IMAGE_MAGIC) {
        cerr << "Bad magic number in image file!" << endl;
        return 2;
    }


    // Get the number of labels/images and make sure there's the same number of each
    vector<uint8_t> vNumberOfLabels(labelBytes.begin()+NUMBER_ITEMS_OFFSET,
                                    labelBytes.begin()+NUMBER_ITEMS_OFFSET+ITEMS_SIZE);
    vector<uint8_t> vNumberOfImages(imageBytes.begin()+NUMBER_ITEMS_OFFSET,
                                    imageBytes.begin()+NUMBER_ITEMS_OFFSET+ITEMS_SIZE);

    int iNumberOfLabels = int(vNumberOfLabels[0] << 24 | vNumberOfLabels[1] << 16 |
                              vNumberOfLabels[2] << 8 | vNumberOfLabels[3]);
    int iNumberOfImages = int(vNumberOfImages[0] << 24 | vNumberOfImages[1] << 16 |
                              vNumberOfImages[2] << 8 | vNumberOfImages[3]);

    if(iNumberOfLabels != iNumberOfImages) {
        cerr << "The number of labels and images do not match!" << endl;
        return 3;
    }


    // Get the number of rows and colums for the images and make sure the size
    // matches the constant values
    vector<uint8_t> vNumRows(imageBytes.begin()+NUMBER_OF_ROWS_OFFSET,
                             imageBytes.begin()+NUMBER_OF_ROWS_OFFSET+ROWS_SIZE);
    vector<uint8_t> vNumCols(imageBytes.begin()+NUMBER_OF_COLUMNS_OFFSET,
                             imageBytes.begin()+NUMBER_OF_COLUMNS_OFFSET+COLUMNS_SIZE);

    int iNumRows = int(vNumRows[0] << 24 | vNumRows[1] << 16 |
                       vNumRows[2] << 8 | vNumRows[3]);
    int iNumCols = int(vNumCols[0] << 24 | vNumCols[1] << 16 |
                       vNumCols[2] << 8 | vNumCols[3]);

    if(iNumRows != ROWS || iNumCols != COLUMNS) {
        cerr << "Bad image.  Rows and columns do not equal " << ROWS << "x" << COLUMNS << endl;
        return 4;
    }


    // Put images and labels into an vector
    for(int i = 0; i < iNumberOfLabels; i++) {
        int label = labelBytes[OFFSET_SIZE + ITEMS_SIZE + i];
        vector<uint8_t> vImageData(imageBytes.begin()+(i*IMAGE_SIZE)+IMAGE_OFFSET,
                                   imageBytes.begin()+(i*IMAGE_SIZE)+IMAGE_OFFSET+IMAGE_SIZE);

        // Do something with the data...
        // cout << vImageData.size() << endl;
        labels.push_back(label);
        images.push_back(vImageData);
    }


    // Close file input streams and return
    labelInputStream.close();
    imageInputStream.close();
    return 0;
}

int main(int argc, char const *argv[])
{
    loadMNISTData();
    return 0;
}