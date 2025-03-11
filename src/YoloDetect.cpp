//by haolan.Z in 2024/12/06

#include <YoloDetect.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

YoloDetection::YoloDetection()
{

    try{
        mModule = torch::jit::load("/home/robot/CV_WS/yolov5/yolov5s.torchscript");
        mModule.to(mDevice);
        mModule.eval();
        std::cout << "model ok" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error Loading the model/n" << e.what() << std::endl;
        throw std::runtime_error("can't load the model");
    }
    
    std::ifstream f("coco.names");
    std::string name = "";
    while (std::getline(f, name))
    {
        mClassnames.push_back(name);
    }
    mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", "bird",
                     "cat", "dog", "horse", "sheep", "crow", "bear"};
}

YoloDetection::~YoloDetection()
{

}

bool YoloDetection::Detect()
{

    cv::Mat img;

    if(mRGB.empty())
    {
        std::cout << "Read RGB failed!"<< std::endl;
        return false;
    }

    //input tensor
    cv::resize(mRGB, img, cv::Size(640, 640));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols,3},torch::kByte);
    imgTensor = imgTensor.permute({2,0,1}).toType(torch::kFloat).div(255).unsqueeze(0).to(mDevice);
    // imgTensor = imgTensor.toType(torch::kFloat);
    // imgTensor = imgTensor.div(255);
    // imgTensor = imgTensor.unsqueeze(0);


    torch::Tensor preds = mModule.forward({imgTensor}).toTuple()->elements()[0].toTensor().to(torch::kCPU);
    std::vector<torch::Tensor> dets = YoloDetection::non_max_suppression(preds.to(torch::kCPU), 0.4, 0.5);
    if (dets.size() > 0)
    {
        //Visualize
        for (size_t i=0; i < dets[0].sizes()[0]; ++ i)
        {
            float left = dets[0][i][0].item().toFloat() * mRGB.cols / 640;
            float top = dets[0][i][1].item().toFloat() * mRGB.rows / 640;
            float right = dets[0][i][2].item().toFloat() * mRGB.cols / 640;
            float bottom = dets[0][i][3].item().toFloat() * mRGB.rows / 640;
            int classID = dets[0][i][5].item().toInt();

            cv::Rect2i DetectArea(left, top, (right - left), (bottom - top));
            mmDetectMap[mClassnames[classID]].push_back(DetectArea);

            if (count(mvDynamicNames.begin(), mvDynamicNames.end(), mClassnames[classID]))
            {
                cv::Rect2i DynamicArea(left, top, (right - left), (bottom - top));
                mvDynamicArea.push_back(DynamicArea);
            }
            
        } 
        if (mvDynamicArea.size() == 0)
        {
            cv::Rect2i tDynamicArea(1,1,1,1);
            mvDynamicArea.push_back(tDynamicArea);
        }
    }
    return true;
}

vector<torch::Tensor> YoloDetection::non_max_suppression(torch::Tensor preds, float score_thresh, float iou_thresh)
{
    std::vector<torch::Tensor> output;
    for (size_t i=0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        //filter by scores
        torch::Tensor scores = pred.select(1, 4) * std::get<0>( torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));
        pred = torch::index_select(pred, 0 ,torch::nonzero(scores > score_thresh).select(1, 0));
        if (pred.sizes()[0] == 0) continue;

        // center_x, center_y, w, h to left, top, right, bottom
        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2) ;
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3) ;

        //computing scores and classes
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
        pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
        pred.select(1, 5) = std::get<1>(max_tuple);

        torch::Tensor dets = pred.slice(1, 0, 6);

        torch::Tensor keep = torch::empty({dets.sizes()[0]});
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
        std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor v = std::get<0>(indexes_tuple);
        torch::Tensor indexes = std::get<1>(indexes_tuple);
        int count = 0;
        while (indexes.sizes()[0] > 0)
        {
            keep[count] = (indexes[0].item().toInt());
            count += 1;

            //computing overlaps
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
            torch::Tensor tops = torch::empty(indexes.sizes()[0] -1);
            torch::Tensor rights = torch::empty(indexes.sizes()[0] -1);
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] -1);
            torch::Tensor widths = torch::empty(indexes.sizes()[0] -1);
            torch::Tensor heights = torch::empty(indexes.sizes()[0] -1);
            for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i)
            {
                lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
                tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
                rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
                bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
                widths[i] =  std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
                heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());  
            }
            torch::Tensor overlaps = widths * heights;

            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }
        keep = keep.toType(torch::kInt64);
        output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
    }
    return output;
} 

void YoloDetection::GetImage(cv::Mat &RGB)
{
    mRGB = RGB;
}

void YoloDetection::ClearImage()
{
    mRGB = 0;
}

void YoloDetection::ClearArea()
{
    mvPersonArea.clear();
}
