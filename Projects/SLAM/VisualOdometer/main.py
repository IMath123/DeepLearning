import numpy as np
import cv2 as cv

def get_match_points(img1, img2):
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf =cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # ## Create flann matcher
    # FLANN_INDEX_KDTREE = 1 # bug: flann enums are missing
    # flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # #matcher = cv.FlannBasedMatcher_create()
    # matcher = cv.FlannBasedMatcher(flann_params, {})

    ## Ratio test
    print(len(matches))
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.7 * m2.distance:# 两个特征向量之间的欧氏距离，越小表明匹配度越高。
            matchesMask[i] = [1, 0]
            pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
            # print(kpts1)
            print(i, pt1, pt2)
            if i % 5 ==0:
                cv.circle(img1, (int(pt1[0]),int(pt1[1])), 5, (255,0,255), -1)
                cv.circle(img2, (int(pt2[0]),int(pt2[1])), 5, (255,0,255), -1)
    # 匹配点为蓝点, 坏点为红点
    draw_params = dict(matchColor = (255, 0,0),
            singlePointColor = (0,0,255),
            matchesMask = matchesMask,
            flags = 0)
    res = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv.imshow("Result", res)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    img1 = cv.imread("~/Downloads/t3d_net_in.png")
    img2 = cv.imread("~/to_pan/FR3D/test_data/face.png")

    get_match_points(img1, img2)
