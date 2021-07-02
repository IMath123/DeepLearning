#ifndef MESH
#define MESH 

#include <string>
#include <vector>

using namespace std;


class mesh{
    public:
        int len_v, len_f;
        vector<float> v, vn;
        vector<unsigned int> f, fn;

        mesh(const string &obj_bin_filename);

};

#endif /* ifndef MESH */
