#ifndef UTILS
#define UTILS 

#include <opencv2/core/types.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "struct/mesh.hpp"

using namespace std;

class segement{
public:
    unsigned int H;
    unsigned int W;
    cv::Mat depth;
    cv::Mat normal;
    cv::Mat barry;
    cv::Mat triangle;

    segement(unsigned int H, unsigned int W) {
        this->H = H;
        this->W = W;
        
        this->depth.create(H, W, CV_32FC1);
        this->normal.create(H, W, CV_32FC3);
        this->barry.create(H, W, CV_32FC3);
        this->triangle.create(H, W, CV_32SC1);
    }

    void refresh() {
        depth = 0;
        normal = 0;
        barry = 0;
        triangle = 0;
    }
};


float max(float a, float b) {
    if (a > b) {
        return a;
    }
    return b;
}
float min(float a, float b) {
    if (a < b) {
        return a;
    }
    return b;
}
float max(float a, float b, float c) {
    return max(a, max(b, c));
}
float min(float a, float b, float c) {
    return min(a, min(b, c));
}

float cross(float x1, float y1, float x2, float y2) {
    return x1 * y2 - x2 * y1;
}

bool get_barry_coord(float p_x, float p_y, float *A, float *B, float *C, float &a, float &b, float &c) {
    float S_C = cross(B[0] - A[0], B[1] - A[1], p_x - A[0], p_y - A[1]);
    float S_A = cross(C[0] - B[0], C[1] - B[1], p_x - B[0], p_y - B[1]);
    bool sign_C = S_C > 0 ? 1 : 0;
    bool sign_A = S_A > 0 ? 1 : 0;
    if (sign_A != sign_C) {
        return false;
    }
    float S_B = cross(A[0] - C[0], A[1] - C[1], p_x - C[0], p_y - C[1]);
    bool sign_B = S_B > 0 ? 1 : 0;
    if (sign_B != sign_C) {
        return false;
    }

    float sum = S_A + S_B + S_C;
    a = S_A / sum;
    b = S_B / sum;
    c = S_C / sum;
}

void render(const mesh &Mesh, float *obj_position, float *camera_position, float *camera_internal_param, segement &Segement) {
    float RT[12];

    RT[4 * 0 + 0] = camera_position[4 * 0 + 0] * obj_position[4 * 0 + 0] + camera_position[4 * 0 + 1] * obj_position[4 * 1 + 0] + camera_position[4 * 0 + 2] * obj_position[4 * 2 + 0];
    RT[4 * 1 + 0] = camera_position[4 * 1 + 0] * obj_position[4 * 0 + 0] + camera_position[4 * 1 + 1] * obj_position[4 * 1 + 0] + camera_position[4 * 1 + 2] * obj_position[4 * 2 + 0];
    RT[4 * 2 + 0] = camera_position[4 * 2 + 0] * obj_position[4 * 0 + 0] + camera_position[4 * 2 + 1] * obj_position[4 * 1 + 0] + camera_position[4 * 2 + 2] * obj_position[4 * 2 + 0];

    RT[4 * 0 + 1] = camera_position[4 * 0 + 0] * obj_position[4 * 0 + 1] + camera_position[4 * 0 + 1] * obj_position[4 * 1 + 1] + camera_position[4 * 0 + 2] * obj_position[4 * 2 + 1];
    RT[4 * 1 + 1] = camera_position[4 * 1 + 0] * obj_position[4 * 0 + 1] + camera_position[4 * 1 + 1] * obj_position[4 * 1 + 1] + camera_position[4 * 1 + 2] * obj_position[4 * 2 + 1];
    RT[4 * 2 + 1] = camera_position[4 * 2 + 0] * obj_position[4 * 0 + 1] + camera_position[4 * 2 + 1] * obj_position[4 * 1 + 1] + camera_position[4 * 2 + 2] * obj_position[4 * 2 + 1];

    RT[4 * 0 + 2] = camera_position[4 * 0 + 0] * obj_position[4 * 0 + 2] + camera_position[4 * 0 + 1] * obj_position[4 * 1 + 2] + camera_position[4 * 0 + 2] * obj_position[4 * 2 + 2];
    RT[4 * 1 + 2] = camera_position[4 * 1 + 0] * obj_position[4 * 0 + 2] + camera_position[4 * 1 + 1] * obj_position[4 * 1 + 2] + camera_position[4 * 1 + 2] * obj_position[4 * 2 + 2];
    RT[4 * 2 + 2] = camera_position[4 * 2 + 0] * obj_position[4 * 0 + 2] + camera_position[4 * 2 + 1] * obj_position[4 * 1 + 2] + camera_position[4 * 2 + 2] * obj_position[4 * 2 + 2];

    RT[4 * 0 + 3] = camera_position[4 * 0 + 0] * obj_position[4 * 0 + 3] + camera_position[4 * 0 + 1] * obj_position[4 * 1 + 3] + camera_position[4 * 0 + 2] * obj_position[4 * 2 + 3] + camera_position[4 * 0 + 3];
    RT[4 * 1 + 3] = camera_position[4 * 1 + 0] * obj_position[4 * 0 + 3] + camera_position[4 * 1 + 1] * obj_position[4 * 1 + 3] + camera_position[4 * 1 + 2] * obj_position[4 * 2 + 3] + camera_position[4 * 1 + 3];
    RT[4 * 2 + 3] = camera_position[4 * 2 + 0] * obj_position[4 * 0 + 3] + camera_position[4 * 2 + 1] * obj_position[4 * 1 + 3] + camera_position[4 * 2 + 2] * obj_position[4 * 2 + 3] + camera_position[4 * 2 + 3];

    float *affined_v = (float *) malloc(Mesh.len_v * 3 * sizeof(float));
    float *affined_v_2d = (float *) malloc(Mesh.len_v * 2 * sizeof(float));
    for (int i = 0; i < Mesh.len_v; ++i) {

        affined_v[3 * i + 0] = RT[4 * 0 + 0] * Mesh.v[3 * i + 0] + RT[4 * 0 + 1] * Mesh.v[3 * i + 1] + RT[4 * 0 + 2] * Mesh.v[3 * i + 2] + RT[4 * 0 + 3];
        affined_v[3 * i + 1] = RT[4 * 1 + 0] * Mesh.v[3 * i + 0] + RT[4 * 1 + 1] * Mesh.v[3 * i + 1] + RT[4 * 1 + 2] * Mesh.v[3 * i + 2] + RT[4 * 1 + 3];
        affined_v[3 * i + 2] = RT[4 * 2 + 0] * Mesh.v[3 * i + 0] + RT[4 * 2 + 1] * Mesh.v[3 * i + 1] + RT[4 * 2 + 2] * Mesh.v[3 * i + 2] + RT[4 * 2 + 3];

        float x_2d, y_2d;
        float *point_3d = affined_v + 3 * i;
        x_2d = camera_internal_param[3 * 0 + 0] * point_3d[0] / point_3d[2] + camera_internal_param[3 * 0 + 2];
        y_2d = camera_internal_param[3 * 1 + 1] * point_3d[1] / point_3d[2] + camera_internal_param[3 * 1 + 2];

        affined_v_2d[2 * i + 0] = x_2d;
        affined_v_2d[2 * i + 1] = y_2d;
    }

    float barry_1, barry_2, barry_3;
    bool in_triangle;
    for (unsigned int i = 0; i < Mesh.len_f; ++i) {
        float *point_2d_1 = affined_v_2d + 2 * (Mesh.f[3*i + 0] - 1);
        float *point_2d_2 = affined_v_2d + 2 * (Mesh.f[3*i + 1] - 1);
        float *point_2d_3 = affined_v_2d + 2 * (Mesh.f[3*i + 2] - 1);

        float *point_3d_1 = affined_v + 3 * (Mesh.f[3*i + 0] - 1);
        float *point_3d_2 = affined_v + 3 * (Mesh.f[3*i + 1] - 1);
        float *point_3d_3 = affined_v + 3 * (Mesh.f[3*i + 2] - 1);

        float x_min, x_max, y_min, y_max;
        x_min = min(point_2d_1[0], point_2d_2[0], point_2d_3[0]);
        y_min = min(point_2d_1[1], point_2d_2[1], point_2d_3[1]);
        x_max = max(point_2d_1[0], point_2d_2[0], point_2d_3[0]);
        y_max = max(point_2d_1[1], point_2d_2[1], point_2d_3[1]);

        if (x_min < 0) {
            x_min = 0;
        }
        if (y_min < 0) {
            y_min = 0;
        }
        if (x_max >= Segement.W) {
            x_max = Segement.W - 1;
        }
        if (y_max >= Segement.H) {
            y_max = Segement.H - 1;
        }

        for (int x = int(x_min); x < int(x_max) + 1; ++x) {
            for (int y = int(y_min); y < int(y_max) + 1; ++y) {
                in_triangle = get_barry_coord(x, y, point_2d_1, point_2d_2, point_2d_3, barry_1, barry_2, barry_3);
                if (in_triangle) {
                    float *point_3d_1 = affined_v + 3 * (Mesh.f[3*i + 0] - 1);
                    float *point_3d_2 = affined_v + 3 * (Mesh.f[3*i + 1] - 1);
                    float *point_3d_3 = affined_v + 3 * (Mesh.f[3*i + 2] - 1);
                    float z = point_3d_1[2] * barry_1 + point_3d_2[2] * barry_2 + point_3d_3[2] * barry_3;

                    if (Segement.depth.at<float>(y, x) == 0 || z < Segement.depth.at<float>(y, x)) {
                        Segement.depth.at<float>(y, x) = z;
                        Segement.triangle.at<unsigned int>(y, x) = i;

                    }
               }
            }
        }
    }

    free((void *) affined_v);
    free((void *) affined_v_2d);

    cv::imshow("hehe", Segement.depth/1000);
    cv::waitKey(0);
}

#endif /* ifndef UTILS */
