import numpy as np
import torch as P
import imath
# from imath.Projects.FaceReconstruct3D import FaceReconstruct


'''
说明:本项目所使用的欧拉角的顺序是ZXY, 即Z-roll, X-pitch, Y-yaw
'''


def GetRotateMatrix3D(nx, ny, nz, angle):
    rad = angle * np.pi / 180.
    q0 = np.cos(rad/2)
    q1 = nx * np.sin(rad / 2)
    q2 = ny * np.sin(rad / 2)
    q3 = nz * np.sin(rad / 2)

    R = P.FloatTensor([[1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
                  [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
                  [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]])

    return R


def RotateDepthMap(image, R, blank_thre=0.2):
    pc = imath.DepthMap2PointCloud(image, 1, -90, 90, -105, 105, -60, 60, 0)
    mean = P.mean(pc, dim=0)

    pc = P.matmul(pc - mean, R) + mean

    new_image = imath.PointCloud2DepthMap(pc, 'direct', 112, 96, -90, 90, -105, 105, -60, 60, keep_blank=True)
    # rec_image = FaceReconstruct(P.where(new_image > blank_thre, new_image, P.Tensor([128 / 255.])))

    rec_image = P.where(new_image < blank_thre, rec_image, new_image)

    return rec_image

def EulerAngle2Quaterion(yaw, pitch, roll):
    c_y = P.cos(yaw/2)
    s_y = P.sin(yaw/2)
    c_p = P.cos(pitch/2)
    s_p = P.sin(pitch/2)
    c_r = P.cos(roll/2)
    s_r = P.sin(roll/2)

    nx = c_y * s_p * c_r + s_y * c_p * s_r
    ny = s_y * c_p * c_r - c_y * s_p * s_r
    nz = -s_y * s_p * c_r + c_y * c_p * s_r
    w = c_y * c_p * c_r + s_y * s_p * s_r

    return nx, ny, nz, w

def EulerAngle2RotateMatrix(yaw, pitch, roll, is_batch=False):
    c1 = P.cos(yaw)
    s1 = P.sin(yaw)
    c2 = P.cos(pitch)
    s2 = P.sin(pitch)
    c3 = P.cos(roll)
    s3 = P.sin(roll)

    R11 = c1*c3 + s1*s2*s3
    R12 = c3*s1*s2 - c1*s3
    R13 = c2*s1
    R21 = c2*s3
    R22 = c2*c3
    R23 = -s2
    R31 = c1*s2*s3 - s1*c3
    R32 = s1*s3 + c1*c3*s2
    R33 = c1*c2

    R = P.cat([R11, R12, R13, R21, R22, R23, R31, R32, R33], dim=-1)

    if is_batch:
        R = R.view(-1, 3, 3)

    return R.view(3, 3)

def Quaterion2RotateMatrix(nx, ny, nz, w):

    def diag(a, b):
        return 1 - 2 * P.pow(a, 2) - 2 * P.pow(b, 2)

    def tr_add(a, b, c, d):
        return 2 * a * b + 2 * c * d

    def tr_sub(a, b, c, d):
        return 2 * a * b - 2 * c * d

    w = w
    x = nx
    y = ny
    z = nz
    # print w,x,y,z
    m = [[diag(y, z), tr_sub(x, y, z, w), tr_add(x, z, y, w)],
         [tr_add(x, y, z, w), diag(x, z), tr_sub(y, z, x, w)],
         [tr_sub(x, z, y, w), tr_add(y, z, x, w), diag(x, y)]]

    return P.stack([P.stack(m[i], dim=-1) for i in range(3)], dim=-2)


def RotateMatrix2EulerAngle(R):
    sy = 1 - P.pow(R[1, 2], 2)
    singular = sy.item() < 1e-6

    if not singular:
        x = P.atan2(R[0, 2], R[2, 2])
        y = P.asin(-R[1, 2])
        z = P.atan2(R[1, 0], R[1, 1])
    else:
        if R[1, 2] > 0:
            x = P.atan2(-R[0, 1], R[0, 0])
            y = P.asin(-R[1, 2])
            z = 0.
        else:
            x = P.atan2(R[0, 1], R[0, 0])
            y = P.asin(-R[1, 2])
            z = 0.

    return x, y, z

def Quaterion2EulerAngle(x, y, z, w):
    roll = P.atan2(2*(w*z + y*x), 1 - 2*(x*x + z*z))
    pitch = P.asin(2*(w*x - z*y))
    yaw = P.atan2(2*(w*y + x*z), 1 - 2*(x*x + y*y))

    return yaw, pitch, roll

if __name__ == '__main__':
    yaw = imath.Tensor(1.5)
    pitch = imath.Tensor(0.3152)
    roll = imath.Tensor(-2.)

    nx, ny, nz, w = EulerAngle2Quaterion(yaw, pitch, roll)
    print('nxyzw', nx, ny, nz, w)

    nx, ny, nz, w = P.Tensor([0.15044376, 0.09441211, 0.03597149, 0.9834425])
    r2d = 180/ np.pi
    yaw_prime, pitch_prime, roll_prime = Quaterion2EulerAngle(nx, ny, nz, w)
    print('asasa', yaw_prime*r2d, pitch_prime*r2d, roll_prime*r2d)

    R_f_euler = EulerAngle2RotateMatrix(yaw, pitch, roll)
    R_f_quertion = Quaterion2RotateMatrix(nx, ny, nz, w)

    print('R_f_E', R_f_euler)
    print('R_f_Q', R_f_quertion)

    E_f_R = RotateMatrix2EulerAngle(R_f_quertion.view(3, 3))

    print('E_f_R', E_f_R)
