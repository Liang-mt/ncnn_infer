// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inference.h"

// static constexpr int INPUT_W = 640;    // Width of input
// static constexpr int INPUT_H = 384;    // Height of input
static constexpr int INPUT_W = 416;    // Width of input
static constexpr int INPUT_H = 416;    // Height of input
static constexpr int NUM_CLASSES = 8;  // Number of classes
static constexpr int NUM_COLORS = 4;   // Number of color
static constexpr int TOPK = 128;       // TopK
static constexpr float NMS_THRESH = 0.3;
static constexpr float BBOX_CONF_THRESH = 0.75;
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;

static inline int argmax(const float* ptr, int len)
{
    int max_arg = 0;
    for (int i = 1; i < len; i++) {
        if (ptr[i] > ptr[max_arg]) max_arg = i;
    }
    return max_arg;
}

/**
 * @brief Resize the image using letterbox
 * @param img Image before resize
 * @param transform_matrix Transform Matrix of Resize
 * @return Image after resize
 */
inline cv::Mat scaledResize(cv::Mat& img, Eigen::Matrix<float, 3, 3>& transform_matrix)
{
    float r = min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;

    int dw = INPUT_W - unpad_w;
    int dh = INPUT_H - unpad_h;

    dw /= 2;
    dh /= 2;

    transform_matrix << 1.0 / r, 0, -dw / r,
        0, 1.0 / r, -dh / r,
        0, 0, 1;

    Mat re;
    cv::resize(img, re, Size(unpad_w, unpad_h));
    Mat out;
    cv::copyMakeBorder(re, out, dh, dh, dw, dw, BORDER_CONSTANT);

    return out;
}

/**
 * @brief Generate grids and stride.
 * @param target_w Width of input.
 * @param target_h Height of input.
 * @param strides A vector of stride.
 * @param grid_strides Grid stride generated in this function.
 */
static void generate_grids_and_stride(const int target_w, const int target_h,
    std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;

        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                //这行改过
                grid_strides.push_back(GridAndStride{ g0, g1, stride });
            }
        }
    }
}

/**
 * @brief Generate Proposal
 * @param grid_strides Grid strides
 * @param feat_ptr Original predition result.
 * @param prob_threshold Confidence Threshold.
 * @param objects Objects proposed.
 */
static void generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float* feat_ptr,
    Eigen::Matrix<float, 3, 3>& transform_matrix, float prob_threshold,
    std::vector<ArmorObject>& objects)
{

    const int num_anchors = grid_strides.size();
    //Travel all the anchors
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (9 + NUM_COLORS + NUM_CLASSES);

        // yolox/models/yolo_head.py decode logic
        //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
        //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        float x_1 = (feat_ptr[basic_pos + 0] + grid0) * stride;
        float y_1 = (feat_ptr[basic_pos + 1] + grid1) * stride;
        float x_2 = (feat_ptr[basic_pos + 2] + grid0) * stride;
        float y_2 = (feat_ptr[basic_pos + 3] + grid1) * stride;
        float x_3 = (feat_ptr[basic_pos + 4] + grid0) * stride;
        float y_3 = (feat_ptr[basic_pos + 5] + grid1) * stride;
        float x_4 = (feat_ptr[basic_pos + 6] + grid0) * stride;
        float y_4 = (feat_ptr[basic_pos + 7] + grid1) * stride;

        int box_color = argmax(feat_ptr + basic_pos + 9, NUM_COLORS);
        int box_class = argmax(feat_ptr + basic_pos + 9 + NUM_COLORS, NUM_CLASSES);

        float box_objectness = (feat_ptr[basic_pos + 8]);

        float color_conf = (feat_ptr[basic_pos + 9 + box_color]);
        float cls_conf = (feat_ptr[basic_pos + 9 + NUM_COLORS + box_class]);

        // float box_prob = (box_objectness + cls_conf + color_conf) / 3.0;
        float box_prob = box_objectness;

        if (box_prob >= prob_threshold)
        {
            ArmorObject obj;

            Eigen::Matrix<float, 3, 4> apex_norm;
            Eigen::Matrix<float, 3, 4> apex_dst;

            apex_norm << x_1, x_2, x_3, x_4,
                y_1, y_2, y_3, y_4,
                1, 1, 1, 1;

            apex_dst = transform_matrix * apex_norm;

            for (int i = 0; i < 4; i++)
            {
                obj.apex[i] = cv::Point2f(apex_dst(0, i), apex_dst(1, i));
                obj.pts.push_back(obj.apex[i]);
            }

            vector<cv::Point2f> tmp(obj.apex, obj.apex + 4);
            obj.rect = cv::boundingRect(tmp);

            obj.cls = box_class;
            obj.color = box_color;
            obj.prob = box_prob;

            objects.push_back(obj);
        }

    } // point anchor loop
}

/**
 * @brief Calculate intersection area between two objects.
 * @param a Object a.
 * @param b Object b.
 * @return Area of intersection.
 */
static inline float intersection_area(const ArmorObject& a, const ArmorObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<ArmorObject>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}


static void qsort_descent_inplace(std::vector<ArmorObject>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}


static void nms_sorted_bboxes(std::vector<ArmorObject>& faceobjects, std::vector<int>& picked,
    float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        ArmorObject& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            ArmorObject& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float iou = inter_area / union_area;
            if (iou > nms_threshold || isnan(iou))
            {
                keep = 0;
                //Stored for Merge
                if (iou > MERGE_MIN_IOU && abs(a.prob - b.prob) < MERGE_CONF_ERROR
                    && a.cls == b.cls && a.color == b.color)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        b.pts.push_back(a.apex[i]);
                    }
                }
                // cout<<b.pts_x.size()<<endl;
            }
        }

        if (keep)
            picked.push_back(i);
    }
}

/**
 * @brief Decode outputs.
 * @param prob Original predition output.
 * @param objects Vector of objects predicted.
 * @param img_w Width of Image.
 * @param img_h Height of Image.
 */
static void decodeOutputs(const float* prob, std::vector<ArmorObject>& objects,
    Eigen::Matrix<float, 3, 3>& transform_matrix, const int img_w, const int img_h)
{
    std::vector<ArmorObject> proposals;
    std::vector<int> strides = { 8, 16, 32 };
    std::vector<GridAndStride> grid_strides;

    generate_grids_and_stride(INPUT_W, INPUT_H, strides, grid_strides);
    generateYoloxProposals(grid_strides, prob, transform_matrix, BBOX_CONF_THRESH, proposals);
    qsort_descent_inplace(proposals);

    if (proposals.size() >= TOPK)
        proposals.resize(TOPK);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);
    int count = picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
    }
}

float* blobFromImage(cv::Mat& img) {
    float* blob = new float[img.total() * 3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}


ArmorDetector::ArmorDetector()
{

}

ArmorDetector::~ArmorDetector()
{
}

//TODO:change to your dir
bool ArmorDetector::initModel(const char* param_path, const char* bin_path)
{
    // Load the ncnn model
    // 这个 ncnn 模型 0 是红，1 是蓝, 但是 onnx 是反过来的，因为刚接触 ncnn，好多地方不太清楚，有了解的望指正

    //这个设为TRUE，开启gpu推理(测试R7 4800U的AMD核显可以)，FALSE是cpu推理
    net.opt.use_vulkan_compute = TRUE;

    net.load_param(param_path);
    net.load_model(bin_path);

    return true;
}

bool ArmorDetector::detect(Mat& img, std::vector<ArmorObject>& objects)
{
    // Pre-process [bgr2rgb & resize]
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img = scaledResize(img, transfrom_matrix);

    // Convert to ncnn::Mat
    ncnn::Mat in = ncnn::Mat::from_pixels(pr_img.data, ncnn::Mat::PIXEL_BGR2RGB, pr_img.cols, pr_img.rows);

    // Prepare input
    ncnn::Extractor ex = net.create_extractor();
    ex.input("images", in); 

    // Run inference
    ncnn::Mat out;
    ex.extract("output", out);

    // Post-process [nms]
    decodeOutputs(out, objects, transfrom_matrix, img_w, img_h);

    // Additional calculations
    for (auto& object : objects)
    {
        if (object.pts.size() >= 8)
        {
            auto N = object.pts.size();
            cv::Point2f pts_final[4];

            for (int i = 0; i < N; i++)
            {
                pts_final[i % 4] += object.pts[i];
            }

            for (int i = 0; i < 4; i++)
            {
                pts_final[i].x = pts_final[i].x / (N / 4);
                pts_final[i].y = pts_final[i].y / (N / 4);
            }

            object.apex[0] = pts_final[0];
            object.apex[1] = pts_final[1];
            object.apex[2] = pts_final[2];
            object.apex[3] = pts_final[3];
        }
        object.area = (int)(calcTetragonArea(object.apex));
    }

    if (objects.size() != 0)
        return true;
    else return false;
}




int main()
{
    const cv::Scalar colors[3] = { {255, 0, 0}, {0, 0, 255}, {0, 255, 0} };
    std::string text;

    const char* param_path = "./weights/opt-0625-001.param";
    const char* bin_path = "./weights/opt-0625-001.bin";

    ArmorDetector detector;
    detector.initModel(param_path, bin_path);
    std::vector<ArmorObject> objects;

    cv::VideoCapture cap("./video/3.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Couldn't open the video file." << std::endl;
        return -1;
    }

    cv::Mat frame;
    double fps;
    cv::TickMeter tm;

    while (cap.read(frame))
    {
        tm.start();

        if (detector.detect(frame, objects))
        {
            for (const auto& b : objects)
            {
                cv::line(frame, b.apex[0], b.apex[1], colors[2], 2);
                cv::line(frame, b.apex[1], b.apex[2], colors[2], 2);
                cv::line(frame, b.apex[2], b.apex[3], colors[2], 2);
                cv::line(frame, b.apex[3], b.apex[0], colors[2], 2);

                if (b.color == 0)
                {
                    text = "R" + std::to_string(b.cls);
                }
                else if (b.color == 1)
                {
                    text = "B" + std::to_string(b.cls);
                }
                cv::putText(frame, std::to_string(b.cls), b.pts[0], cv::FONT_HERSHEY_SIMPLEX, 2, colors[2]);
                cv::putText(frame, std::to_string(b.prob), b.pts[3], cv::FONT_HERSHEY_SIMPLEX, 2, colors[2]);
            }
        }

        tm.stop();
        fps = 1.0 / tm.getTimeSec();
        tm.reset();

        cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, colors[2]);

        cv::imshow("Video", frame);
        if (cv::waitKey(1) == 27) // ESC键退出
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}