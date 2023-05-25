#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
# include "NvInferRuntime.h"
#include <fstream>
#include <map>
# include "NvInferRuntimeCommon.h"
#include "cuda_runtime_api.h"
#include <numeric>
#include "NvInferPlugin.h"
#include <NvInfer.h>
//#include "logging.h"

#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000
#define DEVICE 0           // GPU id
#define BATCH_SIZE 1
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;
const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";
static const int INPUT_W = 768;
static const int INPUT_H = 768;
static const int OUTPUT_SIZE = 3 * 768 *768;
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kVERBOSE)
            std::cout << msg << std::endl;
    }
};


static Logger gLogger;

std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

ICudaEngine* createEngine(unsigned int maxBatchSize,IBuilder* builder,IBuilderConfig* config,DataType dt, std::string wts_path){

    std::map<std::string,Weights> weightMap = loadWeights(wts_path);

    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME,dt,Dims3{3, INPUT_H,INPUT_W});
    assert(data);
 
    Weights emptywts{DataType::kFLOAT, nullptr,0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data,64,DimsHW{9,9},weightMap["conv1.weight"],weightMap["conv1.bias"]);
    conv1->setStrideNd(DimsHW{1,1});
    conv1->setPaddingNd(DimsHW{4,4});
    conv1 -> setNbGroups(1);

    IActivationLayer* relu1=network->addActivation(*conv1->getOutput(0),ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0),32,DimsHW{1,1},weightMap["conv2.weight"],weightMap["conv2.bias"]);
    conv2->setStrideNd(DimsHW{1,1});
    conv2->setPaddingNd(DimsHW{0,0});
    conv2 -> setNbGroups(1);

    IActivationLayer* relu2=network->addActivation(*conv2->getOutput(0),ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* out = network->addConvolutionNd(*relu2->getOutput(0),3,DimsHW{5,5},weightMap["conv3.weight"],weightMap["conv3.bias"]);
    out->setStrideNd(DimsHW{1,1});
    out->setPaddingNd(DimsHW{2,2});
    out -> setNbGroups(1);

    out->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout<<"set name out"<<std::endl;
    network->markOutput(*out->getOutput(0));

    //build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(10*(1<<20));

    ICudaEngine* engine = builder->buildEngineWithConfig(*network,*config);
    assert(engine!=nullptr);
    std::cout<<"build out"<<std::endl;

    //Don't need the network any more
    network->destroy();

    //Release host memory
    for(auto& mem:weightMap){
        free((void*)(mem.second.values) );
    }
    return engine;
}

void APITomodel(unsigned int maxBatchSize,IHostMemory** modelStream, std::string wts_path){

    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    //Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize,builder,config,DataType::kFLOAT, wts_path);
    assert(engine!= nullptr);

    //Serialize the engine
    (*modelStream) = engine->serialize();

    //Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, const int output_size) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

float* blobFromImage(cv::Mat& img){
    float* blob = new float[INPUT_W*INPUT_H*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++)
    {
        for (size_t  h = 0; h < img_h; h++)
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


int main(int argc, char** argv)
{
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream = nullptr;
    size_t size = 0;
    std::string engine_name = "srcnn_x3.engine";
    std::string wts_path = "../srcnn_x3.wts";
    if(argc == 2 && std::string(argv[1]) == "-s"){
        engine_name = "srcnn_x3.engine";
        IHostMemory* modelStream{nullptr};
        APITomodel(BATCH_SIZE,&modelStream, wts_path);
        assert(modelStream!= nullptr);
        std::ofstream p(engine_name,std::ios::binary);
        if(!p){
            std::cerr<<"could not open plan output file"<<std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()),modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else if (argc == 3 && std::string(argv[1]) == "-d") 
    {
        std::ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
        }
    }
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }


    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);
    
    cv::Mat img = cv::imread(argv[2]);
    int rows = img.rows;
    int cols = img.cols;

    cv::Mat dst;
    cv::resize(img, dst, cv::Size(),3,3,cv::INTER_CUBIC);
    float* blob;
    blob = blobFromImage(dst);
    static float* prob = new float[OUTPUT_SIZE];
    // run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, blob, prob, OUTPUT_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cv::Mat out_image(768,768,CV_32FC3);
    for (size_t c = 0; c <3; c++)
    {
        for (size_t  h = 0; h < INPUT_H; h++)
        {
            for (size_t w = 0; w < INPUT_W; w++)
            {
                out_image.at<cv::Vec3f>(h, w)[c] = prob[c * INPUT_W * INPUT_H + h * INPUT_W + w];                        
            }
        }
    }

    cv::imwrite("./output.png", out_image);
    delete blob;

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}