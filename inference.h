#pragma once


#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm> // 包含头文件

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "layer.h"
#include "net.h"
#include "benchmark.h"

using namespace std;
using namespace cv;

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

struct ArmorObject
{
    Point2f apex[4];
    cv::Rect_<float> rect;
    int cls;
    int color;
    int area;
    float prob;
    std::vector<cv::Point2f> pts;
};

//struct BuffObject
//{
//    Point2f apex[5];
//    cv::Rect_<float> rect;
//    int cls;
//    int color;
//    float prob;
//    std::vector<cv::Point2f> pts;
//};


class ArmorDetector
{
public:
    ArmorDetector();
    ~ArmorDetector();
    bool detect(Mat& src, vector<ArmorObject>& objects);
    bool initModel(const char* param_path, const char* bin_path);
        
private:



    ncnn::Net net;

    Eigen::Matrix<float, 3, 3> transfrom_matrix;
};



float calcTriangleArea(cv::Point2f pts[3])
{
    auto a = sqrt(pow((pts[0] - pts[1]).x, 2) + pow((pts[0] - pts[1]).y, 2));
    auto b = sqrt(pow((pts[1] - pts[2]).x, 2) + pow((pts[1] - pts[2]).y, 2));
    auto c = sqrt(pow((pts[2] - pts[0]).x, 2) + pow((pts[2] - pts[0]).y, 2));

    auto p = (a + b + c) / 2.f;

    return sqrt(p * (p - a) * (p - b) * (p - c));
}

/**
 * @brief 计算四边形面积
 *
 * @param pts 四边形顶点
 * @return float 面积
 */
float calcTetragonArea(cv::Point2f pts[4])
{
    return calcTriangleArea(&pts[0]) + calcTriangleArea(&pts[1]);
}

