#ifndef MESH_CPP
#define MESH_CPP 

#include "mesh.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

void load_obj_bin(const string &obj_bin_filename, int &len_v, int &len_f, vector<float> &v, vector<float> &vn, vector<unsigned int> &f, vector<unsigned int> &fn) {
    len_v = 0;
    len_f = 0;
    std::ifstream fin(obj_bin_filename, ios::binary);
    if (!fin) {
        printf("error\n");
        return;
    }


    int len[2];
    fin.read((char *) len, 2 * sizeof(int));
    printf("len = (%i, %i)\n", len[0], len[1]);

    // v = (float *) malloc(len[0] * 3 * sizeof(float));
    // vn = (float *) malloc(len[0] * 3 * sizeof(float));
    // f = (unsigned int *) malloc(len[1] * 3 * sizeof(unsigned int));
    // fn = (unsigned int *) malloc(len[1] * 3 * sizeof(unsigned int));
    v.resize(len[0] * 3);
    vn.resize(len[0] * 3);
    f.resize(len[1] * 3);
    fn.resize(len[1] * 3);

    fin.read((char *) v.data(), len[0] * 3 * sizeof(float));
    fin.read((char *) vn.data(), len[0] * 3 * sizeof(float));
    fin.read((char *) f.data(), len[1] * 3 * sizeof(unsigned int));
    fin.read((char *) fn.data(), len[1] * 3 * sizeof(unsigned int));

    // for (int i = 0; i < 10; ++i) {
    //     printf("%f %f %i %i\n", v[3 * i], vn[3 * i], f[3 * i], fn[3 * i]);
    // }

    len_v = len[0];
    len_f = len[1];
}

mesh::mesh(const string &obj_bin_filename) {
    load_obj_bin(obj_bin_filename, this->len_v, this->len_f, this->v, this->vn, this->f, this->fn);
}
#endif /* ifndef MESH_CPP */
