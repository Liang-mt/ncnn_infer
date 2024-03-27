# 装甲板识别-----ncnn推理

本代码为沈阳航空航天大学2022年开源模型的onnxruntime推理

参考开源

[tup-robomaster/TUP-InfantryVision-2022: 沈阳航空航天大学T-UP战队2022赛季步兵视觉识别程序 (github.com)](https://github.com/tup-robomaster/TUP-InfantryVision-2022)



video视频文件请访问网盘链接下载

链接：https://pan.baidu.com/s/1QykXf3QvKQdGDIeRvCxdzw?pwd=0000 
提取码：0000 



安装所需依赖

```
需要安装ncnn，vulkan，opencv，Eigen相关库
```

运行代码

```
因为我是在windows上vs2019进行的代码编写，所以没有进行相关的cmakelist编写，具体可参考ncnn相关代码cmakelist编写方式
```



补充：

1.可在main.cpp里面对类别和颜色数量进行修改

```c++
static constexpr int INPUT_W = 416;    // Width of input
static constexpr int INPUT_H = 416;    // Height of input
static constexpr int NUM_CLASSES = 8;  // Number of classes
static constexpr int NUM_COLORS = 4;   // Number of color
static constexpr int TOPK = 128;       // TopK
static constexpr float NMS_THRESH = 0.3;
static constexpr float BBOX_CONF_THRESH = 0.75;
static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.9;

```

2.可根据自己需求在main.cpp里面做以下的修改

```c++
//自身模型路径
const char* param_path = "./weights/opt-0625-001.param";
const char* bin_path = "./weights/opt-0625-001.bin";
//自身视频路径
cv::VideoCapture cap("./video/3.mp4");

```

3.

```c++
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
```
