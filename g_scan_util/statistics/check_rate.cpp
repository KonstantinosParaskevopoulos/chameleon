#include <iostream>
#include <dirent.h>
#include <stdio.h>


using namespace std;

int countFiles(const char* path)
{
    DIR* dir = opendir(path);

    if (dir == NULL)
        return 0;

    struct dirent* entry;
    int count = 0;

    while ((entry = readdir(dir)) != NULL)
    {
        if (string(entry->d_name) != "." &&
            string(entry->d_name) != "..")
        {
            count++;
        }
    }

    closedir(dir);

    return count;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        cout << "Usage: " << argv[0] << " <base_path>" << " <mode (0): Print only Rate | (1): Print Full Output>" << endl;
        return 1;
    }

    string basePath = argv[1];
    int sel = atoi(argv[2]);

    string encrPath = basePath + "/encr";
    string randPath = basePath + "/rand";

    int ENCR_FILES = countFiles(encrPath.c_str());
    int RAND_FILES = countFiles(randPath.c_str());


    double ENCR_TO_RAND = (double)ENCR_FILES / RAND_FILES;

    if(sel==1){
        cout << "Encryption Training Data Size (ENCR) " << ENCR_FILES << endl;
        cout << "Random Op. Training Data Size (RAND) " << RAND_FILES << endl;
        cout << "Training Dataset Rate is: " << ENCR_TO_RAND << " Encryption to 1 Random" << endl;}
    else if(sel==0){
        if(ENCR_TO_RAND<=1.01 && ENCR_TO_RAND>=0.99){
            cout << "1" << endl;
        }
        else{
            cout << "0" << endl;
        }
    }   
    else{
        cout << "Error!"<< endl;
    }
    return 0;
}