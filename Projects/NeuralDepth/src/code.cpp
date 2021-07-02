#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "utils.cpp"
#include "struct/mesh.hpp"

using namespace std;

int main() {
    const string obj_bin_filename = "/home/dj/anaconda3/lib/python3.8/site-packages/imath/Projects/NeuralDepth/temp.bin";

    mesh Mesh(obj_bin_filename);
    printf("%i %i\n", Mesh.len_v, Mesh.len_f);

    const unsigned int resolution_h = 640;
    const unsigned int resolution_w = 480;

    segement Segement(resolution_h, resolution_w);
    float obj_position[3 * 4] = {100, 0, 0, 0,
                                 0, 100, 0, 0,
                                 0, 0, 100, 1000000};
    float camera_position[3 * 4] = {1, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 1, 0};
    float camera_internal_param[2 * 3] = {
        50, 0, resolution_w / 2,
        0, 50, resolution_h / 2,
    };

    // for (int i = 0; i < 10000; ++i) {
    //     obj_position[3] = i * 10;
    //     // obj_position[7] = i * 10;
    //     // obj_position[11] = i * 0.2;
	//     Segement.refresh();
    //     render(Mesh, obj_position, camera_position, camera_internal_param, Segement);
    // }
	Segement.refresh();
	render(Mesh, obj_position, camera_position, camera_internal_param, Segement);
    
}
