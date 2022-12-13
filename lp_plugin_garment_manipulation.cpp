#include "lp_plugin_garment_manipulation.h"

#include "lp_renderercam.h"
#include "lp_openmesh.h"
#include "renderer/lp_glselector.h"
#include "renderer/lp_glrenderer.h"

#include <math.h>
#include <fstream>
#include <filesystem>
#include <example.hpp>
#include <QVBoxLayout>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QOpenGLShaderProgram>
#include <QOpenGLExtraFunctions>
#include <QLabel>
#include <QMatrix4x4>
#include <QPushButton>
#include <QtConcurrent/QtConcurrent>
#include <QFileDialog>
#include <QPainter>

#include "gym_arm.cpp"

/**
 * @brief BulletPhysics Headers
 */
//#include <BulletSoftBody/btSoftBody.h>
//#include <Bullet3Dynamics/b3CpuRigidBodyPipeline.h>

//tensorboard --logdir /home/cpii/projects/log/cloth_test/trial1/tfevents.pb
const std::string kLogFile = "/home/cpii/projects/log/cloth_test/trial1/tfevents.pb";
const std::string kLogFile2 = "/home/cpii/projects/log/sac_model5_rectangle/trial15/tfevents.pb";
const QString dataPath("/home/cpii/projects/data");
const QString memoryPath("/media/cpii/JohnData/SAC/ClothData_square/memory");
const QString modelPath("/media/cpii/JohnData/SAC/ClothData_square/model");
const QString testPath("/media/cpii/JohnData/SAC/ClothData_square/test");
const QString olddatasavepath("/home/cpii/storage_d1/RL1/olddata_sac_imgsize500");
const QString datagen("/home/cpii/storage_d1/RL1/SAC/data_generate_rectangle");
std::vector<double> markervecs_97 = {0.0, 0.0, 0.0}, markercoordinate_98 = {0.0, 0.0, 0.0}, markercoordinate_99 = {0.0, 0.0, 0.0}, markertrans(2), markerposition_98(2), markerposition_99(2), avgHeight;
std::vector<double> grasp_last(3), release_last(3);
std::vector<int> offset(2);
std::vector<cv::Point2f> roi_corners(4), midpoints(4), dst_corners(4);
std::vector<float> trans(3), markercenter(2);
float gripper_length = 0.244;
constexpr int
iBound[2] = { 1000, 9000 };
constexpr uchar uThres = 245, uThres_low = 10;
cv::RNG rng(12345);

constexpr double robotDLimit = 0.85;    //Maximum distance the robot can reach
const int LOG_SIG_MIN = -2, LOG_SIG_MAX = 20, STATE_DIM = 15379, ACT_DIM = 3, START_STEP = 1000, TRAINEVERY = 1, SAVEMODELEVERY = 50, TESTEVERY = 25;
int maxepisode = 10000, maxstep = 20, teststep = 20, batch_size = 16, total_steps = 0;
const float ALPHA = 0.2, GAMMA = 0.99, POLYAK = 0.995;
double lrp = 1e-3, lrc = 1e-3;
int datanum = 0;
torch::Device device(torch::kCPU);

std::shared_ptr<QProcess> LP_Plugin_Garment_Manipulation::gProc_RViz;   //Define the static variable

// open the first webcam plugged in the computer
//cv::VideoCapture camera1(4); // Grey: 2, 8. Color: 4, 10.

bool gStopFindWorkspace = false, gPlan = false, gQuit = false, mCollectAvgH = false;
QFuture<void> gFuture;
QImage gNullImage, gCurrentGLFrame, gEdgeImage, gWarpedImage, gInvWarpImage, gDetectImage;
QReadWriteLock gLock;

auto build_fc_layers (std::vector<int> dims) {
        torch::nn::Sequential layers;
        for(auto i=0; i<dims.size()-1; i++){
            if(i == dims.size()-2) {
                layers->push_back(torch::nn::LinearImpl(dims[i], dims[i+1]));
            } else {
                layers->push_back(torch::nn::LinearImpl(dims[i], dims[i+1]));
                layers->push_back(torch::nn::ReLUImpl());
            }
        }
        return layers;
}

// Memory
struct Data {
    torch::Tensor before_state;
    torch::Tensor before_pick_point;
    torch::Tensor place_point;
    torch::Tensor reward;
    torch::Tensor done;
    torch::Tensor after_state;
    torch::Tensor after_pick_point;
};
std::deque<Data> memory;

struct policy_output
{
    torch::Tensor action;
    torch::Tensor logp_pi;
};

struct PolicyImpl : torch::nn::Module {
    PolicyImpl(std::vector<int> fc_dims)
        : conv1(torch::nn::Conv2dOptions(3, 8, 7).stride(3).padding(3).bias(false)),
          conv2(torch::nn::Conv2dOptions(8, 16, 7).stride(3).padding(3).bias(false)),
          conv3(torch::nn::Conv2dOptions(16, 32, 5).stride(2).padding(2).bias(false)),
          conv4(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(1).bias(false)),
          maxpool(torch::nn::MaxPool2dOptions(3).stride({2, 2}))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("maxpool", maxpool);
        mlp = register_module("mlp", build_fc_layers(fc_dims));
        mlp->push_back(torch::nn::ReLUImpl());
        mean_linear = register_module("mean_linear", torch::nn::Linear(fc_dims[fc_dims.size()-1], ACT_DIM));
        log_std_linear = register_module("log_std_linear", torch::nn::Linear(fc_dims[fc_dims.size()-1], ACT_DIM));
    }

    policy_output forward(torch::Tensor state, bool deterministic, bool log_prob) {
        //torch::Tensor x = relu(conv1(state)); // 510*510
        torch::Tensor x = state;
        //std::cout << "x: " << x.sizes() << std::endl;

        //x = torch::relu(maxpool(x)); // 254*254
        //std::cout << "x: " << x.sizes() << " " << x.dtype() << std::endl;

//        torch::Tensor out_tensor1 = x;
//        out_tensor1 = out_tensor1.index({0, 14}).to(torch::kF32).clone().detach().to(torch::kCPU);
//        std::cout << "out_tensor1: " << out_tensor1.sizes() << " " << out_tensor1.dtype() << std::endl;
//        cv::Mat cv_mat1(254, 254, CV_32FC1, out_tensor1.data_ptr());
//        auto min1 = out_tensor1.min().item().toFloat();
//        auto max1 = out_tensor1.max().item().toFloat();
//        std::cout << "min1: " << min1 << "max1: " << max1 << std::endl;
//        cv_mat1.convertTo(cv_mat1, CV_8U, 255.0/(max1-min1));
//        std::cout << cv_mat1.type() << std::endl;
//        cv::cvtColor(cv_mat1, cv_mat1, CV_GRAY2BGR);

        //x = relu(conv2(x)); // 254*254
        //std::cout << "x: " << x.sizes() << std::endl;

        //x = torch::relu(maxpool(x)); // 126*126
        //std::cout << "x: " << x.sizes() << std::endl;

//        torch::Tensor out_tensor2 = x*255;
//        out_tensor2 = out_tensor2.index({0, 3}).to(torch::kF32).clone().detach().to(torch::kCPU);
//        std::cout << "out_tensor2: " << out_tensor2.sizes() << " " << out_tensor2.dtype() << std::endl;
//        cv::Mat cv_mat2(126, 126, CV_32FC1, out_tensor2.data_ptr());
//        auto min2 = out_tensor2.min().item().toFloat();
//        auto max2 = out_tensor2.max().item().toFloat();
//        std::cout << "min2: " << min2 << "max2: " << max2 << std::endl;
//        cv_mat2.convertTo(cv_mat2, CV_8U);
//        cv::cvtColor(cv_mat2, cv_mat2, CV_GRAY2BGR);

        //x = relu(conv3(x)); // 126*126
        //std::cout << "x: " << x.sizes() << std::endl;

        //x = torch::relu(maxpool(x)); // 62*62
        //std::cout << "x: " << x.sizes() << std::endl;

        //x = relu(conv4(x)); // 126*126

//        torch::Tensor out_tensor3 = x*255;
//        out_tensor3 = out_tensor3.index({0, 2}).to(torch::kF32).clone().detach().to(torch::kCPU);
//        std::cout << "out_tensor3: " << out_tensor3.sizes() << " " << out_tensor3.dtype() << std::endl;
//        cv::Mat cv_mat3(62, 62, CV_32FC1, out_tensor3.data_ptr());
//        auto min3 = out_tensor3.min().item().toFloat();
//        auto max3 = out_tensor3.max().item().toFloat();
//        std::cout << "min3: " << min3 << "max3: " << max3 << std::endl;
//        cv_mat3.convertTo(cv_mat3, CV_8U);
//        cv::cvtColor(cv_mat3, cv_mat3, CV_GRAY2BGR);

        //std::cout << cv_mat1.size() << " " << cv_mat2.size() << " " << cv_mat3.size() << std::endl;

//        gLock.lockForWrite();
//        gWarpedImage = QImage((uchar*) cv_mat1.data, cv_mat1.cols, cv_mat1.rows, cv_mat1.step, QImage::Format_BGR888).copy();
//        gInvWarpImage = QImage((uchar*) cv_mat2.data, cv_mat2.cols, cv_mat2.rows, cv_mat2.step, QImage::Format_BGR888).copy();
//        gEdgeImage = QImage((uchar*) cv_mat3.data, cv_mat3.cols, cv_mat3.rows, cv_mat3.step, QImage::Format_BGR888).copy();
//        gLock.unlock();

        x = x.view({x.size(0), -1});

        //std::cout << "x: " << x.sizes() << std::endl;
//        std::cout << "pick_point: " << pick_point.sizes() << std::endl;

        torch::Tensor netout = mlp->forward(x);

        torch::Tensor mean = mean_linear(netout);

        torch::Tensor log_std = log_std_linear(netout);

        log_std = torch::clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX);

        torch::Tensor std = log_std.exp();

        torch::Tensor action;
        if(deterministic){
            // Only used for evaluating policy at test time.
            action = mean;
        } else {
            auto shape = mean.sizes();
            auto eps = torch::randn(shape) * torch::ones(shape, mean.dtype()) + torch::zeros(shape, mean.dtype());
            action = mean + std * eps.to(device);  // for reparameterization trick (mean + std * N(0,1))

//            auto eps = at::normal(0, 1, mean.sizes()).to(mean.device());
//            eps.set_requires_grad(false);
//            action = mean + eps * std;// for reparameterization trick (mean + std * N(0,1))
        }

        //# action rescaling
//        torch::Tensor action_scale = torch::ones({1}).to(device) * 1.0;
//        torch::Tensor action_bias = torch::ones({1}).to(device) * 0.0;

//        static auto logSqrt2Pi = torch::zeros({1}).to(mean.device());
//        static std::once_flag flag;
//        std::call_once(flag, [](){
//            logSqrt2Pi[0] = 2*M_PI;
//            logSqrt2Pi = torch::log(torch::sqrt(logSqrt2Pi));
//        });
//        static auto log_prob_func = [](torch::Tensor value, torch::Tensor mean, torch::Tensor std){
//            auto var = std.pow(2);
//            auto log_scale = std.log();
//            return -(value - mean).pow(2) / (2 * var) - log_scale - logSqrt2Pi;
//        };

        torch::Tensor logp_pi;
        if(log_prob){
            // Calculate log_prob
            auto var = pow(std, 2);
            auto log_scale = log(std);
            logp_pi = -pow(action - mean, 2) / (2.0 * var) - log_scale - log(sqrt(2.0 * M_PI));

            // Enforcing Action Bound
            logp_pi = logp_pi.sum(-1);
            logp_pi -= torch::sum(2.0 * (log(2.0) - action - torch::nn::functional::softplus(-2.0 * action)), 1);
            logp_pi = torch::unsqueeze(logp_pi, -1);

//            logp_pi = log_prob_func(action, mean, std);
//            // Enforcing Action Bound
//            logp_pi -= torch::log(action_scale * (1 - torch::tanh(action).pow(2)) + 1e-6);
//            logp_pi = logp_pi.sum(1, true);
        } else {
            logp_pi = torch::zeros(1).to(device);
        }

        //action = torch::tanh(action);
        //action = torch::tanh(action) * action_scale + action_bias;

//        auto A_Slice_tensor = action.index({"...", torch::indexing::Slice({torch::indexing::None}, 2)});
//        auto B_Slice_tensor = action.index({"...", torch::indexing::Slice(2, 3)});

//        auto A_action = torch::tanh(A_Slice_tensor);
//        auto B_action = torch::sigmoid(B_Slice_tensor);
//        action = torch::cat({A_action, B_action}, 1);

        action = torch::sigmoid(action);

        policy_output output = {action, logp_pi};

        return output;
    }

    policy_output forward_pick_point(torch::Tensor state, torch::Tensor pick_point, bool deterministic, bool log_prob) {
        torch::Tensor x = conv1(state); // 510*510
        //std::cout << "x: " << x.sizes() << std::endl;

        x = torch::relu(maxpool(x)); // 254*254
        //std::cout << "x: " << x.sizes() << " " << x.dtype() << std::endl;

        x = conv2(x); // 254*254
        //std::cout << "x: " << x.sizes() << std::endl;

        x = torch::relu(maxpool(x)); // 126*126
        //std::cout << "x: " << x.sizes() << std::endl;

        x = conv3(x); // 126*126
        //std::cout << "x: " << x.sizes() << std::endl;

        x = torch::relu(maxpool(x)); // 62*62
        //std::cout << "x: " << x.sizes() << std::endl;

        x = x.view({x.size(0), -1});

//        std::cout << "x: " << x.sizes() << std::endl;
//        std::cout << "pick_point: " << pick_point.sizes() << std::endl;

        x = torch::cat({x, pick_point}, -1); // 62*62*4+3 = 15379
        //std::cout << "x: " << x.sizes() << std::endl;

        torch::Tensor netout = mlp->forward(x);

        torch::Tensor mean = mean_linear(netout);

        torch::Tensor log_std = log_std_linear(netout);

        log_std = torch::clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX);

        torch::Tensor std = log_std.exp();

        torch::Tensor action;
        if(deterministic){
            // Only used for evaluating policy at test time.
            action = mean;
        } else {
            auto shape = mean.sizes();
            auto eps = torch::randn(shape) * torch::ones(shape, mean.dtype()) + torch::zeros(shape, mean.dtype());
            action = mean + std * eps.to(device);  // for reparameterization trick (mean + std * N(0,1))

//            auto eps = at::normal(0, 1, mean.sizes()).to(mean.device());
//            eps.set_requires_grad(false);
//            action = mean + eps * std;// for reparameterization trick (mean + std * N(0,1))
        }

        //# action rescaling
//        torch::Tensor action_scale = torch::ones({1}).to(device) * 1.0;
//        torch::Tensor action_bias = torch::ones({1}).to(device) * 0.0;

//        static auto logSqrt2Pi = torch::zeros({1}).to(mean.device());
//        static std::once_flag flag;
//        std::call_once(flag, [](){
//            logSqrt2Pi[0] = 2*M_PI;
//            logSqrt2Pi = torch::log(torch::sqrt(logSqrt2Pi));
//        });
//        static auto log_prob_func = [](torch::Tensor value, torch::Tensor mean, torch::Tensor std){
//            auto var = std.pow(2);
//            auto log_scale = std.log();
//            return -(value - mean).pow(2) / (2 * var) - log_scale - logSqrt2Pi;
//        };

        torch::Tensor logp_pi;
        if(log_prob){
            // Calculate log_prob
            auto var = pow(std, 2);
            auto log_scale = log(std);
            logp_pi = -pow(action - mean, 2) / (2.0 * var) - log_scale - log(sqrt(2.0 * M_PI));

            // Enforcing Action Bound
            logp_pi = logp_pi.sum(-1);
            logp_pi -= torch::sum(2.0 * (log(2.0) - action - torch::nn::functional::softplus(-2.0 * action)), 1);
            logp_pi = torch::unsqueeze(logp_pi, -1);

//            logp_pi = log_prob_func(action, mean, std);
//            // Enforcing Action Bound
//            logp_pi -= torch::log(action_scale * (1 - torch::tanh(action).pow(2)) + 1e-6);
//            logp_pi = logp_pi.sum(1, true);
        } else {
            logp_pi = torch::zeros(1).to(device);
        }

        //action = torch::tanh(action);
        //action = torch::tanh(action) * action_scale + action_bias;

        action = torch::sigmoid(action);

        std::vector<float> action_limit{0.8, 0.8, 0.2};
        torch::Tensor action_limit_tensor = torch::from_blob(action_limit.data(), { 1, 3 }, at::kFloat);

        action = torch::mul(action, action_limit_tensor.to(device));

        std::vector<float> action_limit2{0.1, 0.1, 0.05};
        torch::Tensor action_limit_tensor2 = torch::from_blob(action_limit2.data(), { 1, 3 }, at::kFloat);

        action = torch::add(action, action_limit_tensor2.to(device));

        policy_output output = {action, logp_pi};

        return output;
    }

    torch::nn::Sequential mlp{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    torch::nn::MaxPool2d maxpool;
    torch::nn::Linear mean_linear{nullptr}, log_std_linear{nullptr};
};
TORCH_MODULE(Policy);


struct MLPQFunctionImpl : torch::nn::Module {
    MLPQFunctionImpl(std::vector<int> fc_dims)
        : conv1(torch::nn::Conv2dOptions(3, 8, 7).stride(3).padding(3).bias(false)),
          conv2(torch::nn::Conv2dOptions(8, 16, 7).stride(3).padding(3).bias(false)),
          conv3(torch::nn::Conv2dOptions(16, 32, 5).stride(2).padding(2).bias(false)),
          conv4(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(1).bias(false)),
          maxpool(torch::nn::MaxPool2dOptions(3).stride({2, 2}))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("maxpool", maxpool);
        fc_dims.push_back(1);
        q = register_module("q", build_fc_layers(fc_dims));
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action){
        //torch::Tensor x = relu(conv1(state)); // 510*510
        torch::Tensor x = state; // 510*510
        //std::cout << "x: " << x.sizes() << std::endl;

        //x = torch::relu(maxpool(x)); // 254*254
        //std::cout << "x: " << x.sizes() << std::endl;

        //x = relu(conv2(x)); // 254*254
        //std::cout << "x: " << x.sizes() << std::endl;

        //x = torch::relu(maxpool(x)); // 126*126
        //std::cout << "x: " << x.sizes() << std::endl;

        //x = relu(conv3(x)); // 126*126
        //std::cout << "x: " << x.sizes() << std::endl;

        //x = relu(conv4(x)); // 126*126
        //std::cout << "x: " << x.sizes() << std::endl;

        //x = torch::relu(maxpool(x)); // 62*62
        //std::cout << "x: " << x.sizes() << std::endl;

        x = x.view({x.size(0), -1});
        //std::cout << "x: " << x.sizes() << std::endl;

        x = q->forward(torch::cat({x, action}, -1));

        //std::cout << "x: " << x.sizes() << std::endl;

        return x;
    }

    torch::Tensor forward_pick_point(torch::Tensor state, torch::Tensor pick_point, torch::Tensor action){
        torch::Tensor x = conv1(state); // 510*510

        x = torch::relu(maxpool(x)); // 254*254

        x = conv2(x); // 254*254

        x = torch::relu(maxpool(x)); // 126*126

        x = conv3(x); // 126*126

        x = torch::relu(maxpool(x)); // 62*62

        x = x.view({x.size(0), -1});

        x = torch::cat({x, pick_point}, -1); // 62*62*4+3 = 15379

        x = q->forward(torch::cat({x, action}, -1));

        return x;
    }
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    torch::nn::MaxPool2d maxpool;
    torch::nn::Sequential q{nullptr};
};
TORCH_MODULE(MLPQFunction);


struct ActorCriticImpl : torch::nn::Module {
    ActorCriticImpl(std::vector<int> policy_fc_dims, std::vector<int> critic_fc_dims) {
          pi = Policy(policy_fc_dims);
          q1 = MLPQFunction(critic_fc_dims);
          q2 = MLPQFunction(critic_fc_dims);
    }

    torch::Tensor act(torch::Tensor state, bool deterministic) {
        torch::NoGradGuard disable;

        policy_output p = pi->forward(state, deterministic, false);

        torch::Tensor action = p.action;

        return action;
    }

    torch::Tensor act_pick_point(torch::Tensor state, torch::Tensor pick_point, bool deterministic) {
        torch::NoGradGuard disable;

        policy_output p = pi->forward_pick_point(state, pick_point, deterministic, false);

        torch::Tensor action = p.action;

        return action;
    }

    Policy pi{nullptr};
    MLPQFunction q1{nullptr}, q2{nullptr};
};
TORCH_MODULE(ActorCritic);


std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; file >> line;) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

struct filter_slider_ui
{
    std::string name;
    std::string label;
    std::string description;
    bool is_int;
    float value;
    rs2::option_range range;

    bool render(const float3& location, bool enabled);
    static bool is_all_integers(const rs2::option_range& range);
};

class filter_options
{
public:
    filter_options(const std::string name, rs2::filter& filter);
    filter_options(filter_options&& other);
    std::string filter_name;                                   //Friendly name of the filter
    rs2::filter& filter;                                       //The filter in use
    std::map<rs2_option, filter_slider_ui> supported_options;  //maps from an option supported by the filter, to the corresponding slider
    std::atomic_bool is_enabled;                               //A boolean controlled by the user that determines whether to apply the filter or not
};

void LP_Plugin_Garment_Manipulation::update_data(rs2::frame_queue& data, rs2::frame& colorized_depth, rs2::points& points, rs2::pointcloud& pc, rs2::colorizer& color_map)
{
    rs2::frame f;
    if (data.poll_for_frame(&f))  // Try to take the depth and points from the queue
    {
        points = pc.calculate(f); // Generate pointcloud from the depth data
        colorized_depth = color_map.process(f);     // Colorize the depth frame with a color map
        pc.map_to(colorized_depth);         // Map the colored depth to the point cloud
    }
}

QImage realsenseFrameToQImage(const rs2::frame &f)
{
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r = QImage((uchar*) f.get_data(), w, h, w*3, QImage::QImage::Format_RGB888);
        return r;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        auto df = vf.as<depth_frame>();

        auto r = QImage(w, h, QImage::QImage::Format_RGB888);

        static auto rainbow = [](int p, int np, float&r, float&g, float&b) {    //16,777,216
                float inc = 6.0 / np;
                float x = p * inc;
                r = 0.0f; g = 0.0f; b = 0.0f;
                if ((0 <= x && x <= 1) || (5 <= x && x <= 6)) r = 1.0f;
                else if (4 <= x && x <= 5) r = x - 4;
                else if (1 <= x && x <= 2) r = 1.0f - (x - 1);
                if (1 <= x && x <= 3) g = 1.0f;
                else if (0 <= x && x <= 1) g = x - 0;
                else if (3 <= x && x <= 4) g = 1.0f - (x - 3);
                if (3 <= x && x <= 5) b = 1.0f;
                else if (2 <= x && x <= 3) b = x - 2;
                else if (5 <= x && x <= 6) b = 1.0f - (x - 5);
            };

       // auto curPixel = r.bits();
        float maxDepth = 1.0 / 2.0;
        float R, G, B;
        for ( int i=0; i<w; ++i ){
            for ( int j=0; j<h; ++j/*, ++curPixel */){
                int tmp = 65535 * df.get_distance(i,j) * maxDepth;
                rainbow(tmp, 65535, R, G, B);
                r.setPixelColor(i, j, qRgb(R*255, G*255, B*255));
            }
        }

        return r;
    } else {
        qDebug() << "Unknown!";
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

static double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void LP_Plugin_Garment_Manipulation::textbox_draw(cv::Mat src, std::vector<cv::Rect>& groups, std::vector<float>& probs, std::vector<int>& indexes)
{
    static std::shared_ptr<tesseract::TessBaseAPI> api;
    static std::once_flag flag;
    std::call_once(flag, [](){
        api = std::make_shared<tesseract::TessBaseAPI>();
        if (api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY)) {
            std::cout << stderr << std::endl;
            exit(1);
        }
        api->SetPageSegMode(tesseract::PSM_SPARSE_TEXT);
    });

    auto ocrImg = src.clone();

    QString ad = QString("/home/cpii/Desktop/digit/img/img.jpg");
    QByteArray filename_Srcc = ad.toLocal8Bit();
    const char *filename_Srccc = filename_Srcc.data();
    cv::imwrite(filename_Srccc, src);

    std::string detectedChars = " ++ ";

    for (size_t i = 0; i < indexes.size(); i++)
    {
        if (src.type() == CV_8UC3)
        {
            auto rect = groups[indexes[i]];
            constexpr auto ratio = 0.5;
            auto rectW = rect.width;
            auto rectH = rect.height;
            rect.width *= 1.0 + ratio;
            rect.height *= 1.0 + ratio;
            rect.x = std::max(0.0, rect.x - rectW * 0.5 * ratio);
            rect.y = std::max(0.0, rect.y - rectH * 0.5 * ratio);
            rect.width = std::min(src.cols - rect.x, rect.width);
            rect.height = std::min(src.rows - rect.y, rect.height);

            cv::Mat tmp = ocrImg(rect).clone();

            QString filename_after_image = QString("/home/cpii/Desktop/digit/img/image%1.jpg").arg(i);
            QByteArray filename_after_imageqb = filename_after_image.toLocal8Bit();
            const char *filename_after_imagechar = filename_after_imageqb.data();
            cv::imwrite(filename_after_imagechar, tmp);

            api->SetImage(tmp.data, tmp.cols, tmp.rows, 3, tmp.step);
            auto outText = api->GetUTF8Text();
            if (outText != nullptr)
            {
                //std::cout << std::string(outText) << std::endl;
                detectedChars += std::string(outText) + std::string(", ");
            } else {
                std::cout << "GG" << std::endl;
            }
            delete[] outText;
        }
    }
    std::cout << "Done" << std::endl
              << detectedChars << std::endl;
}

// returns sequence of squares detected on the image.
void LP_Plugin_Garment_Manipulation::findSquares( const cv::Mat& image, std::vector<std::vector<cv::Point>>& squares )
{
    squares.clear();
    cv::Mat pyr, timg, gray0(image.size(), CV_8U), gray;
    // down-scale and upscale the image to filter out the noise
    cv::pyrDown(image, pyr, cv::Size(image.cols/2, image.rows/2));
    cv::pyrUp(pyr, timg, image.size());
    std::vector<std::vector<cv::Point> > contours;
    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        cv::mixChannels(&timg, 1, &gray0, 1, ch, 1);
        // try several threshold levels
        float N = 10;
        float thresh = 3;
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                cv::Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                cv::dilate(gray, gray, cv::Mat(), cv::Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }
            // find contours and store them all as a list
            cv::findContours(gray, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
            std::vector<cv::Point> approx;
            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true)*0.02, true);
                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(cv::contourArea(approx)) > 1000 &&
                    cv::isContourConvex(approx) )
                {
                    double maxCosine = 0;
                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }
                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 ){
                        bool save = true;
                        for(auto j = 0; j < approx.size(); j++){
                            if (   static_cast<double>(approx[j].x) / imageWidth < 0.02
                                || static_cast<double>(approx[j].x) / imageHeight > 0.98
                                || static_cast<double>(approx[j].y) / imageWidth < 0.02
                                || static_cast<double>(approx[j].y) / imageHeight > 0.98){
                                save = false;
                            }
                        }
                        if(save){
                            squares.push_back(approx);
                        }
                    }
                }
            }
        }
    }
}


void LP_Plugin_Garment_Manipulation::findLines( const cv::Mat image)
{
    // Declare the output variables
    cv::Mat dst, cdst, cdstP;
    // Loads an image
    cv::Mat grey_image;
    cv::cvtColor( image, grey_image, cv::COLOR_BGR2GRAY );
    cv::Mat src = grey_image;
    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
    }
    // Edge detection
    cv::Canny(src, dst, 50, 200, 3);
    // Copy edges to the images that will display the results in BGR
    cv::cvtColor(dst, cdst, cv::COLOR_GRAY2BGR);
    cdstP = cdst.clone();
    // Standard Hough Line Transform
//    std::vector<cv::Vec2f> lines; // will hold the results of the detection
//    cv::HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection
//    // Draw the lines
//    for( size_t i = 0; i < lines.size(); i++ )
//    {
//        float rho = lines[i][0], theta = lines[i][1];
//        cv::Point pt1, pt2;
//        double a = cos(theta), b = sin(theta);
//        double x0 = a*rho, y0 = b*rho;
//        pt1.x = cvRound(x0 + 1000*(-b));
//        pt1.y = cvRound(y0 + 1000*(a));
//        pt2.x = cvRound(x0 - 1000*(-b));
//        pt2.y = cvRound(y0 - 1000*(a));
//        cv::line( cdst, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA);
//    }
    // Probabilistic Line Transform
    std::vector<cv::Vec4i> linesP; // will hold the results of the detection
    cv::HoughLinesP(dst, linesP, 1, CV_PI/180, 50, 50, 10 ); // runs the actual detection
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        cv::Vec4i l = linesP[i];
        float dx = l[0] - l[2];
        float dy = l[1] - l[3];
        float angle = atan2(dy, dx);
        if(angle<0){
            angle = angle + PI;
        }
        if(angle>(0.5*PI)){
            angle = PI - angle;
        }
        //qDebug() << "line " << i << " angle : " << angle;
        cv::line( cdstP, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, cv::LINE_AA);
        //cv::arrowedLine(cdstP, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, cv::LINE_AA, 0, 0.25);
    }

    // Show results
    gLock.lockForWrite();
    //gInvWarpImage = QImage((uchar*) cdst.data, cdst.cols, cdst.rows, cdst.step, QImage::Format_BGR888).copy();
    gDetectImage = QImage((uchar*) cdstP.data, cdstP.cols, cdstP.rows, cdstP.step, QImage::Format_BGR888).copy();
    gLock.unlock();
    emit glUpdateRequest();
}


LP_Plugin_Garment_Manipulation::~LP_Plugin_Garment_Manipulation()
{
    if ( gProc_RViz ) {
        gProc_RViz->terminate();
        gProc_RViz->waitForFinished();
        gProc_RViz.reset();
    }
    gQuit = true;
    gFuture.waitForFinished();

    // Clean the data
    emit glContextRequest([this](){
        delete mProgram_L;
        mProgram_L = nullptr;
    }, "Shade");

    emit glContextRequest([this](){
        delete mProgram_R;
        mProgram_R = nullptr;
    }, "Normal");

    Q_ASSERT(!mProgram_L);
    Q_ASSERT(!mProgram_R);

    for(int i=0; i<4; i++){
        roi_corners[i].x = 0;
        roi_corners[i].y = 0;
        midpoints[i].x = 0;
        midpoints[i].y = 0;
        dst_corners[i].x = 0;
        dst_corners[i].y = 0;
    }
    for(int i=0; i<3; i++){
        trans[i] = 0;
    }
    gCurrentGLFrame = gNullImage;
    gEdgeImage = gNullImage;
    gWarpedImage = gNullImage;
    gInvWarpImage = gNullImage;

    mDetector.reset();
}

QWidget *LP_Plugin_Garment_Manipulation::DockUi()
{
    mWidget = std::make_shared<QWidget>();
    QVBoxLayout *layout = new QVBoxLayout(mWidget.get());

    mLabel = new QLabel("Right click to find the workspace");
    mLabel2 = new QLabel(" ");
    mButton0 = new QPushButton("Start");
    mButton1 = new QPushButton("Start Reinforcement_Learning_1");
    mButton2 = new QPushButton("Start Reinforcement_Learning_2");
    mButton3 = new QPushButton("Start Train_rod");

    layout->addWidget(mLabel);
    layout->addWidget(mLabel);
    layout->addWidget(mButton1);
    layout->addWidget(mButton2);
    layout->addWidget(mButton3);

    mWidget->setLayout(layout);

    connect(mButton1, &QPushButton::clicked, [this](bool checked){
        if(gPlan && !mRunCollectData && !mRunReinforcementLearning1 && !mRunReinforcementLearning2 && !mRunTrainRod && !mGenerateData){
            mRunReinforcementLearning1 = true;
            mLabel->setText("Start Reinforcement Learning_1, press SPACE to stop");
            Reinforcement_Learning_1();
        } else if(mRunReinforcementLearning1 || mRunReinforcementLearning2 || mRunTrainRod){
            qDebug() << "Training, press SPACE to stop";
        } else if(mRunCollectData){
            qDebug() << "Collecting data, can not start training";
        } else if(mGenerateData){
            qDebug() << "Generating data, can not start training";
        } else {
            qDebug() << "Finding workspace, can not start training";
        }
    });
    connect(mButton2, &QPushButton::clicked, [this](bool checked){
        if(gPlan && !mRunCollectData && !mRunReinforcementLearning1 && !mRunReinforcementLearning2 && !mRunTrainRod && !mGenerateData){
            mRunReinforcementLearning2 = true;
            mLabel->setText("Start Reinforcement Learning_2, press SPACE to stop");
            Reinforcement_Learning_2();
        } else if(mRunReinforcementLearning1 || mRunReinforcementLearning2 || mRunTrainRod){
            qDebug() << "Training, press SPACE to stop";
        } else if(mRunCollectData){
            qDebug() << "Collecting data, can not start training";
        } else if(mGenerateData){
            qDebug() << "Generating data, can not start training";
        } else {
            qDebug() << "Finding workspace, can not start training";
        }
    });
    connect(mButton3, &QPushButton::clicked, [this](bool checked){
        if(gPlan && !mRunCollectData && !mRunReinforcementLearning1 && !mRunReinforcementLearning2 && !mRunTrainRod && !mGenerateData){
            mRunTrainRod = true;
            mLabel->setText("Start Train Rod, press SPACE to stop");
            Train_rod();
        } else if(mRunReinforcementLearning1 || mRunReinforcementLearning2 || mRunTrainRod){
            qDebug() << "Training, press SPACE to stop";
        } else if(mRunCollectData){
            qDebug() << "Collecting data, can not start training";
        } else if(mGenerateData){
            qDebug() << "Generating data, can not start training";
        } else {
            qDebug() << "Finding workspace, can not start training";
        }
    });

    return mWidget.get();
}

class Sleeper : public QThread
{
public:
    static void usleep(unsigned long usecs){QThread::usleep(usecs);}
    static void msleep(unsigned long msecs){QThread::msleep(msecs);}
    static void sleep(unsigned long secs){QThread::sleep(secs);}
};

void LP_Plugin_Garment_Manipulation::getIndex(std::vector<float> v, float K, int& index)
{
    auto it = std::find(v.begin(), v.end(), K);

    // If element was found
    if (it != v.end())
    {

        // calculating the index
        // of K
        index = it - v.begin();
        //std::cout << index << std::endl;
    }
    else {
        // If the element is not
        // present in the vector
        std::cout << "Not found" << std::endl;
    }
}

void LP_Plugin_Garment_Manipulation::trans_old_data(int datasize, QString location, QString savepath)
{
    qDebug() << "Start to load data: " << location;

    for(int i=0; i<datasize; i++){
        std::vector<float> before_points;
        std::vector<float> before_tablepointIDs;
        std::vector<float> after_points;
        std::vector<float> after_tablepointIDs;
        std::vector<float> grasp_release_points_and_height;
        std::vector<float> trans_matrix;

        qDebug() << "Memory loading: [" << i+1 << "/" << datasize << "]" ;

        cv::Mat before_image, after_image;

        QString filename_after_image = QString(location + QString::number(i) + "/after_warped_image.jpg");
        after_image = cv::imread(filename_after_image.toStdString());

//        if(location == "/home/cpii/storage_d1/robot_garment/data_Sep9_1/" && i<1254){
//            continue;
//        }

        if(!after_image.data){
            continue;
        }

        QString filename_before_image = QString(location + QString::number(i) + "/before_warped_image.jpg");
        before_image = cv::imread(filename_before_image.toStdString());

        QString filename_before_points = QString(location + QString::number(i) + "/before_points.txt");
        loaddata(filename_before_points.toStdString(), before_points);

        QString filename_before_tablepointIDs = QString(location + QString::number(i) + "/before_tablepointIDs.txt");
        loaddata(filename_before_tablepointIDs.toStdString(), before_tablepointIDs);

        QString filename_after_points = QString(location + QString::number(i) + "/after_points.txt");
        loaddata(filename_after_points.toStdString(), after_points);

        QString filename_after_tablepointIDs = QString(location + QString::number(i) + "/after_tablepointIDs.txt");
        loaddata(filename_after_tablepointIDs.toStdString(), after_tablepointIDs);

        QString filename_grasp_release_points_and_height = QString(location + QString::number(i) + "/grasp_release_points_and_height.txt");
        loaddata(filename_grasp_release_points_and_height.toStdString(), grasp_release_points_and_height);

        QString filename_transmatrix = QString(location + "/transformation_matrix.txt");
        loaddata(filename_transmatrix.toStdString(), trans_matrix);

        cv::Matx44f trans_matrix_t2c;
        trans_matrix_t2c = cv::Matx44f(trans_matrix[0], trans_matrix[1],  trans_matrix[2],  trans_matrix[3],
                                       trans_matrix[4], trans_matrix[5],  trans_matrix[6],  trans_matrix[7],
                                       trans_matrix[8], trans_matrix[9], trans_matrix[10], trans_matrix[11],
                                                  0.0f,            0.0f,             0.0f,             1.0f);

        auto matrixinv = trans_matrix_t2c.inv();

        std::vector<float> pick_point, place_point;
        pick_point.push_back(grasp_release_points_and_height[0]/warped_image_resize.width);
        pick_point.push_back(grasp_release_points_and_height[1]/warped_image_resize.height);
        pick_point.push_back(grasp_release_points_and_height[2]-gripper_length);
        place_point.push_back(grasp_release_points_and_height[5]/warped_image_resize.width);
        place_point.push_back(grasp_release_points_and_height[6]/warped_image_resize.height);
        place_point.push_back(grasp_release_points_and_height[7]-gripper_length);

        std::vector<float> before_height;
        std::vector<float> after_height;
        for(auto j=0; j<before_tablepointIDs.size(); j++){
            cv::Mat ptMat = (cv::Mat_<float>(4,1) << before_points[before_tablepointIDs[j]*3], -before_points[before_tablepointIDs[j]*3+1], -before_points[before_tablepointIDs[j]*3+2], 1);
            cv::Mat_<float> dstMat(matrixinv * ptMat);
            float scale = dstMat(0,3);
            QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);

            cv::Mat ptMat2 = (cv::Mat_<float>(4,1) << after_points[after_tablepointIDs[j]*3], -after_points[after_tablepointIDs[j]*3+1], -after_points[after_tablepointIDs[j]*3+2], 1);
            cv::Mat_<float> dstMat2(matrixinv * ptMat2);
            float scale2 = dstMat2(0,3);
            QVector4D Pt2(dstMat2(0,0)/scale2, dstMat2(0,1)/scale2, dstMat2(0,2)/scale2, 1.0f);

            if(Pt.z() < 0.0 || Pt.z() > 0.2){
                before_height.push_back(0.0);
            } else {
                before_height.push_back(Pt.z());
            }
            if(Pt2.z() < 0.0 || Pt2.z() > 0.2){
                after_height.push_back(0.0);
            } else {
                after_height.push_back(Pt2.z());
            }
        }

        cv::Mat before_height_mat = cv::Mat(warped_image_resize.width, warped_image_resize.height, CV_32FC1);
        memcpy(before_height_mat.data, before_height.data(), before_height.size()*sizeof(float));

        cv::Mat after_height_mat = cv::Mat(warped_image_resize.width, warped_image_resize.height, CV_32FC1);
        memcpy(after_height_mat.data, after_height.data(), after_height.size()*sizeof(float));

        before_height.clear();
        after_height.clear();

        int before_area = 0, after_area = 0;

        for(int row = 0; row<warped_image_resize.width; row++){
            for(int col = 0; col<warped_image_resize.height; col++){
                auto PT_RGB = before_image.at<cv::Vec3b>(row, col);
                auto PT_RGB2 = after_image.at<cv::Vec3b>(row, col);
                if((row >= (0.6*(float)warped_image_resize.width) && col >= (0.6*(float)warped_image_resize.height))
                    || (PT_RGB[0] >= uThres && PT_RGB[1] >= uThres && PT_RGB[2] >= uThres)){
                    before_height_mat.at<float>(row, col) = 0.0;
                } else {
                    before_area++;
                }
                if((row >= (0.6*(float)warped_image_resize.width) && col >= (0.6*(float)warped_image_resize.height))
                    || (PT_RGB2[0] >= uThres && PT_RGB2[1] >= uThres && PT_RGB2[2] >= uThres)){
                    after_height_mat.at<float>(row, col) = 0.0;
                } else {
                    after_area++;
                }
            }
        }

        double m, before_max, after_max;
        cv::minMaxLoc(before_height_mat, &m, &before_max);
        cv::minMaxLoc(after_height_mat, &m, &after_max);

//        float angle = 0;
        float conf_before = 0, conf_after = 0;
//        cv::Mat rotatedImg, rotatedImg2;
//        QImage rotatedImgqt, rotatedImgqt2;
//        for(int a = 0; a < 36; a++){
//            rotatedImgqt = QImage((uchar*) before_image.data, before_image.cols, before_image.rows, before_image.step, QImage::Format_BGR888);
//            rotatedImgqt2 = QImage((uchar*) after_image.data, after_image.cols, after_image.rows, after_image.step, QImage::Format_BGR888);

//            QMatrix r;

//            r.rotate(angle*10.0);

//            rotatedImgqt = rotatedImgqt.transformed(r);
//            rotatedImgqt2 = rotatedImgqt2.transformed(r);

//            rotatedImg = cv::Mat(rotatedImgqt.height(), rotatedImgqt.width(), CV_8UC3, rotatedImgqt.bits());
//            rotatedImg2 = cv::Mat(rotatedImgqt2.height(), rotatedImgqt2.width(), CV_8UC3, rotatedImgqt2.bits());

//            std::vector<bbox_t> test_result = mDetector->detect(rotatedImg);

//            if(test_result.size()>0){
//                for(auto i =0; i<test_result.size(); i++){
//                    if(test_result[i].obj_id == 1 && conf_before < test_result[i].prob && test_result[i].prob > 0.5){
//                        conf_before = test_result[i].prob;
//                    }
//                }
//            }

//            std::vector<bbox_t> test_result2 = mDetector->detect(rotatedImg2);

//            if(test_result2.size()>0){
//                for(auto i =0; i<test_result2.size(); i++){
//                    if(test_result2[i].obj_id == 1 && conf_after < test_result2[i].prob && test_result2[i].prob > 0.5){
//                        conf_after = test_result2[i].prob;
//                    }
//                }
//            }
//            angle+=1;
//        }

        std::vector<float> stepreward(1), done(1);
        float height_reward, garment_area_reward, conf_reward;
        height_reward = 5000 * ((float)before_max - (float)after_max);
        float area_diff = (float)after_area/(float)before_area;
        if(area_diff >= 1){
            garment_area_reward = 400 * (area_diff - 1);
        } else {
            garment_area_reward = -400 * (1/area_diff - 1);
        }
        conf_reward = 1000 * (conf_after - conf_before);
        stepreward[0] = height_reward + garment_area_reward + conf_reward;

        //qDebug() << "height_reward: " << height_reward << "garment_area_reward: " << garment_area_reward << "conf_reward: " << conf_reward;
        //qDebug() << "reward: " << stepreward[0];

        if(stepreward[0] > 100000 || stepreward[0] < -100000){
            continue;
        } else {
            datanum++;

            //qDebug() << "before_max: " << before_max << " after_max: " << after_max;
            //qDebug() << "before_area: " << before_area << " after_area: " << after_area;
            qDebug() << "data: " << datanum;

            torch::Tensor src_tensor, src_height_tensor, pick_point_tensor, place_point_tensor, before_state;
            torch::Tensor after_tensor, after_height_tensor, pick_point_tensor2, after_state;

            pick_point_tensor = torch::from_blob(pick_point.data(), { 3 }, at::kFloat);
            place_point_tensor = torch::from_blob(place_point.data(), { 3 }, at::kFloat);

            src_tensor = torch::from_blob(before_image.data, { before_image.rows, before_image.cols, before_image.channels() }, at::kByte);
            src_tensor = src_tensor.permute({ 2, 0, 1 });
            src_tensor = src_tensor.to(torch::kF32).to(torch::kCPU)/255;

            src_height_tensor = torch::from_blob(before_height_mat.data, { 1, before_height_mat.rows, before_height_mat.cols}, at::kFloat);
            src_height_tensor = src_height_tensor.to(torch::kF32).to(torch::kCPU);

            //std::cout << src_tensor.sizes() << " " << src_height_tensor.sizes() << std::endl;

            src_tensor = src_tensor.flatten();
            src_height_tensor = src_height_tensor.flatten();
            //std::cout << src_tensor.sizes() << " " << src_height_tensor.sizes() << std::endl;
            before_state = torch::cat({ src_tensor, src_height_tensor });
            before_state = torch::reshape(before_state, {4, warped_image_resize.width, warped_image_resize.height});

            cv::Mat gray;
            cv::cvtColor( after_image, gray, cv::COLOR_BGR2GRAY );
            cv::blur( gray, gray, cv::Size(3,3) );

            cv::Mat canny_output;
            cv::Canny( gray, canny_output, thresh, thresh*1.2 );
            std::vector<cv::Vec4i> hierarchy;
            cv::Size sz = gray.size();
            imageWidth = sz.width;
            imageHeight = sz.height;

            cv::findContours( canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );

            cv::Point grasp_point;

            int size = 0;

            for( size_t i = 0; i< contours.size(); i++ ){
                for (size_t j = 0; j < contours[i].size(); j++){

                    if ((static_cast<double>(contours[i][j].x) / imageWidth > 0.6
                            && static_cast<double>(contours[i][j].y) / imageHeight > 0.6 )
                            || (sqrt(pow((static_cast<double>(contours[i][j].x) / imageWidth - 0.83), 2) + pow((static_cast<double>(contours[i][j].y) / imageHeight - 0.83), 2)) > 0.7745)
                            || (static_cast<double>(contours[i][j].x) / imageWidth > 0.90
                            && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)
                            || (static_cast<double>(contours[i][j].x) / imageWidth < 0.10
                            && static_cast<double>(contours[i][j].y) / imageHeight > 0.90)
                            || (static_cast<double>(contours[i][j].x) / imageWidth > 0.480
                            && static_cast<double>(contours[i][j].x) / imageWidth < 0.580
                            && static_cast<double>(contours[i][j].y) / imageHeight < 0.10))
                    { // Filter out the robot arm and markers
                            size = size - 1;
                    }
                }
                size += contours[i].size();
            }

            int randcontour = rand()%int(contours.size());
            int randp = rand()%int(contours[randcontour].size());

            while((static_cast<double>(contours[randcontour][randp].x) / imageWidth > 0.6
                   && static_cast<double>(contours[randcontour][randp].y) / imageHeight > 0.6 )
                   || (sqrt(pow((static_cast<double>(contours[randcontour][randp].x) / imageWidth - 0.83), 2) + pow((static_cast<double>(contours[randcontour][randp].y) / imageHeight - 0.83), 2)) > 0.7745)
                   || (static_cast<double>(contours[randcontour][randp].x) / imageWidth > 0.90
                   && static_cast<double>(contours[randcontour][randp].y) / imageHeight < 0.10)
                   || (static_cast<double>(contours[randcontour][randp].x) / imageWidth < 0.10
                   && static_cast<double>(contours[randcontour][randp].y) / imageHeight > 0.90)
                   || (static_cast<double>(contours[randcontour][randp].x) / imageWidth > 0.480
                   && static_cast<double>(contours[randcontour][randp].x) / imageWidth < 0.580
                   && static_cast<double>(contours[randcontour][randp].y) / imageHeight < 0.10))
            { // Filter out the robot arm and markers
                randcontour = rand()%int(contours.size());
                randp = rand()%int(contours[randcontour].size());
            }
            grasp_point.x = contours[randcontour][randp].x;
            grasp_point.y = contours[randcontour][randp].y;

            std::vector<float> pick_point2;
            pick_point2.push_back(float(grasp_point.x)/warped_image_resize.width);
            pick_point2.push_back(float(grasp_point.y)/warped_image_resize.height);
            pick_point2.push_back(float(rand()%100)/1000);

            pick_point_tensor2 = torch::from_blob(pick_point2.data(), { 3 }, at::kFloat);

            after_tensor = torch::from_blob(after_image.data, { after_image.rows, after_image.cols, after_image.channels() }, at::kByte);
            after_tensor = after_tensor.permute({ 2, 0, 1 });
            after_tensor = after_tensor.to(torch::kF32).to(torch::kCPU)/255;

            after_height_tensor = torch::from_blob(after_height_mat.data, { 1, after_height_mat.rows, after_height_mat.cols }, at::kFloat);
            after_height_tensor = after_height_tensor.to(torch::kF32).to(torch::kCPU);

            after_tensor = after_tensor.flatten();
            after_height_tensor = after_height_tensor.flatten();
            after_state = torch::cat({ after_tensor, after_height_tensor });
            after_state = torch::reshape(after_state, {4, warped_image_resize.width, warped_image_resize.height});

            if(after_max < 0.2 && conf_after > 0.7) { // Max height when garment unfolded is about 0.015m, conf level is about 0.75
                stepreward[0] += 5000;
                done[0] = 1;
            } else {
                done[0] = 0;
            }
            auto reward_tensor = torch::from_blob(stepreward.data(), { 1 }, at::kFloat);
            auto done_tensor = torch::from_blob(done.data(), { 1 }, at::kFloat);

            std::vector<float> before_state_vector(before_state.data_ptr<float>(), before_state.data_ptr<float>() + before_state.numel());
            std::vector<float> after_state_vector(after_state.data_ptr<float>(), after_state.data_ptr<float>() + after_state.numel());

            QString filename_id = QString(savepath + "/%1").arg(datanum-1);
            QDir().mkdir(filename_id);

            QString filename_before_state = QString(filename_id + "/before_state.txt");
            savedata(filename_before_state, before_state_vector);
            QString filename_before_state_tensor = QString(filename_id + "/before_state_tensor.pt");
            torch::save(before_state, filename_before_state_tensor.toStdString());

            QString filename_before_pick_point = QString(filename_id + "/before_pick_point.txt");
            savedata(filename_before_pick_point, pick_point);
            QString filename_before_pick_point_tensor = QString(filename_id + "/before_pick_point_tensor.pt");
            torch::save(pick_point_tensor, filename_before_pick_point_tensor.toStdString());

            QString filename_place_point = QString(filename_id + "/place_point.txt");
            savedata(filename_place_point, place_point);
            QString filename_place_point_tensor = QString(filename_id + "/place_point_tensor.pt");
            torch::save(place_point_tensor, filename_place_point_tensor.toStdString());

            QString filename_reward = QString(filename_id + "/reward.txt");
            savedata(filename_reward, stepreward);
            QString filename_reward_tensor = QString(filename_id + "/reward_tensor.pt");
            torch::save(reward_tensor, filename_reward_tensor.toStdString());

            QString filename_done = QString(filename_id + "/done.txt");
            savedata(filename_done, done);
            QString filename_done_tensor = QString(filename_id + "/done_tensor.pt");
            torch::save(done_tensor, filename_done_tensor.toStdString());

            QString filename_after_state = QString(filename_id + "/after_state.txt");
            savedata(filename_after_state, after_state_vector);
            QString filename_after_state_tensor = QString(filename_id + "/after_state_tensor.pt");
            torch::save(after_state, filename_after_state_tensor.toStdString());

            QString filename_after_pick_point = QString(filename_id + "/after_pick_point.txt");
            savedata(filename_after_pick_point, pick_point2);
            QString filename_after_pick_point_tensor = QString(filename_id + "/after_pick_point_tensor.pt");
            torch::save(pick_point_tensor2, filename_after_pick_point_tensor.toStdString());

        }
    }
}

bool LP_Plugin_Garment_Manipulation::Run()
{
    //Test network
//    torch::manual_seed(0);

//    device = torch::Device(torch::kCPU);
//    if (torch::cuda::is_available()) {
//        std::cout << "CUDA is available! Training on GPU." << std::endl;
//        device = torch::Device(torch::kCUDA);
//    }

//    torch::autograd::DetectAnomalyGuard detect_anomaly;

//    qDebug() << "Creating models";

//    std::vector<int> policy_mlp_dims{STATE_DIM, 2048, 1024};
//    std::vector<int> critic_mlp_dims{STATE_DIM + ACT_DIM, 2048, 1024};

//    auto actor_critic = ActorCritic(policy_mlp_dims, critic_mlp_dims);
//    auto actor_critic_target = ActorCritic(policy_mlp_dims, critic_mlp_dims);

//    qDebug() << "Creating optimizer";

//    torch::AutoGradMode copy_disable(false);

//    std::vector<torch::Tensor> q_params;
//    for(size_t i=0; i<actor_critic->q1->parameters().size(); i++){
//        q_params.push_back(actor_critic->q1->parameters()[i]);
//    }
//    for(size_t i=0; i<actor_critic->q2->parameters().size(); i++){
//        q_params.push_back(actor_critic->q2->parameters()[i]);
//    }
//    torch::AutoGradMode copy_enable(true);

//    torch::optim::Adam policy_optimizer(actor_critic->pi->parameters(), torch::optim::AdamOptions(lrp));
//    torch::optim::Adam critic_optimizer(q_params, torch::optim::AdamOptions(lrc));

//    actor_critic->pi->to(device);
//    actor_critic->q1->to(device);
//    actor_critic->q2->to(device);
//    actor_critic_target->pi->to(device);
//    actor_critic_target->q1->to(device);
//    actor_critic_target->q2->to(device);

//    qDebug() << "Copying parameters to target models";
//    torch::AutoGradMode hardcopy_disable(false);
//    for(size_t i=0; i < actor_critic_target->pi->parameters().size(); i++){
//        actor_critic_target->pi->parameters()[i].copy_(actor_critic->pi->parameters()[i]);
//        actor_critic_target->pi->parameters()[i].set_requires_grad(false);
//    }
//    for(size_t i=0; i < actor_critic_target->q1->parameters().size(); i++){
//        actor_critic_target->q1->parameters()[i].copy_(actor_critic->q1->parameters()[i]);
//        actor_critic_target->q1->parameters()[i].set_requires_grad(false);
//    }
//    for(size_t i=0; i < actor_critic_target->q2->parameters().size(); i++){
//        actor_critic_target->q2->parameters()[i].copy_(actor_critic->q2->parameters()[i]);
//        actor_critic_target->q2->parameters()[i].set_requires_grad(false);
//    }
//    torch::AutoGradMode hardcopy_enable(true);

//    auto s_batch = torch::ones({16, 3, 512, 512}).to(device);
//    auto a_batch = torch::zeros({16, 5}).to(device);

//    torch::Tensor q1 = actor_critic->q1->forward(s_batch, a_batch);
//    policy_output sample = actor_critic->pi->forward(s_batch, false, true);

//    std::cout << q1 << std::endl
//              << sample.action << std::endl
//              << sample.logp_pi << std::endl;

//    return 0;
// ------------------------------------------------------------------------------------------------
//    int datasize = 0;
//    //int count = 0;
//    //std::vector<float> rewards{0};
//    //QString path = "/home/cpii/storage_d1/RL1/SAC/saved_memory";

//    for (const auto & file : std::filesystem::directory_iterator(memoryPath.toStdString())){
//        datasize++;
//    }
//    qDebug() << datasize;
//    for(int i=10000; i<10000+datasize-2; i++){
//        QString filename_reward = QString(memoryPath + "/" + QString::number(i) + "/reward.txt");
//        std::vector<float> tmpreward;
//        loaddata(filename_reward.toStdString(), tmpreward);
//        //qDebug() << tmpreward[0];
//        //QString filename = QString(memoryPath + "/" + QString::number(i));
//        if(tmpreward[0] > -1000 && tmpreward[0] < -500){
//            //rewards.push_back(i);
//            qDebug() << i;
//        }
////        std::vector<float> tmpreward;
////        QDir file(filename);
////        if(tmpreward[0] == -10000){
////            if (!file.removeRecursively()) {
////                qCritical() << "[Warning] Useless data cannot be deleted : " << i;
////            }
////            continue;
////        }
////        QString rename_p(path + "/" + QString::number(count));
////        count++;
////        file.rename(filename, rename_p);
////        qDebug() << i;
//    }
//    //std::cout << rewards;
//    return 0;

//    datasize = 7773;
//    for (const auto & file : std::filesystem::directory_iterator(path.toStdString())){
//            datasize++;
//    }
//    datasize -= 2;
//    qDebug() << datasize;
//    for(int i=0; i<datasize; i++){
//        QString filename_reward = QString(memoryPath + "/" + QString::number(i) + "/reward.txt");
//        std::vector<float> tmpreward;
//        loaddata(filename_reward.toStdString(), tmpreward);
//        rewards.push_back(tmpreward[0]);
//        if(tmpreward[0]==0){
//            QDir dir(filename_reward);
//            if (!dir.removeRecursively()) {
//                qCritical() << "[Warning] Useless data cannot be deleted : " << i;
//            }
//            tmpreward[0] = -10000;
//            savedata(filename_reward, tmpreward);
//            count++;
//            qDebug() << i;
//        }
//    }
//    qDebug() << count;
//    qDebug() << datasize;

//    savedata("/home/cpii/storage_d1/RL1/SAC/model_rewards_160episode_datas.txt", rewards);

//    return 0;

//    qDebug() << "Loading memory";

//    int dataid;

//    QString filename_memorysize = QString(memoryPath + "/memorysize.txt");
//    std::vector<float> memorysize;
//    loaddata(filename_memorysize.toStdString(), memorysize);
//    float datasize = memorysize[0];

//    QString filename_dataid = QString(memoryPath + "/dataid.txt");
//    std::vector<float> saved_dataid;
//    loaddata(filename_dataid.toStdString(), saved_dataid);
//    dataid = saved_dataid[0]+1;


//    for(int i=0; i<datasize; i++){
//        qDebug() << "Memory loading: [" << i+1 << "/" << datasize << "]" ;
//        QString filename_id = memoryPath + QString("/%1").arg(dataid-datasize+i);

//        // Load string(.txt)
//        QString filename_before_state = QString(filename_id + "/before_state.txt");
//        std::vector<float> before_state_vector;
//        loaddata(filename_before_state.toStdString(), before_state_vector);
//        torch::Tensor before_state_tensor = torch::from_blob(before_state_vector.data(), { 262147 }, torch::kFloat);

//        QString filename_place_point = QString(filename_id + "/place_point.txt");
//        std::vector<float> place_point_vector;
//        loaddata(filename_place_point.toStdString(), place_point_vector);
//        torch::Tensor place_point_tensor = torch::from_blob(place_point_vector.data(), { 3 }, torch::kFloat);

//        QString filename_reward = QString(filename_id + "/reward.txt");
//        std::vector<float> reward_vector;
//        loaddata(filename_reward.toStdString(), reward_vector);
//        torch::Tensor reward_tensor = torch::from_blob(reward_vector.data(), { 1 }, torch::kFloat);

//        QString filename_done = QString(filename_id + "/done.txt");
//        std::vector<float> done_vector;
//        loaddata(filename_done.toStdString(), done_vector);
//        torch::Tensor done_tensor = torch::from_blob(done_vector.data(), { 1 }, torch::kFloat);

//        QString filename_after_state = QString(filename_id + "/after_state.txt");
//        std::vector<float> after_state_vector;
//        loaddata(filename_after_state.toStdString(), after_state_vector);
//        torch::Tensor after_state_tensor = torch::from_blob(after_state_vector.data(), { 262147 }, torch::kFloat);

//        // Trans to tensor and save
//        QString sfilename_before_state_tensor = QString(filename_id + "/before_state_tensor.pt");
//        torch::save(before_state_tensor, sfilename_before_state_tensor.toStdString());

//        QString sfilename_place_point_tensor = QString(filename_id + "/place_point_tensor.pt");
//        torch::save(place_point_tensor, sfilename_place_point_tensor.toStdString());

//        QString sfilename_reward_tensor = QString(filename_id + "/reward_tensor.pt");
//        torch::save(reward_tensor, sfilename_reward_tensor.toStdString());

//        QString sfilename_done_tensor = QString(filename_id + "/done_tensor.pt");
//        torch::save(done_tensor, sfilename_done_tensor.toStdString());

//        QString sfilename_after_state_tensor = QString(filename_id + "/after_state_tensor.pt");
//        torch::save(after_state_tensor, sfilename_after_state_tensor.toStdString());
//    }

//    return 0;

//    std::vector<float> tt{1, 2, 3, 4, 5, 6, 7, 8};
//    torch::Tensor tt_tensor = torch::from_blob(tt.data(), { 2, 4 }, at::kFloat);

//    std::vector<float> tt2{1, 1, 1, 2};
//    torch::Tensor tt2_tensor = torch::from_blob(tt2.data(), { 1, 4 }, at::kFloat);

//    tt_tensor = torch::add(tt_tensor, tt2_tensor);

//    std::cout << "tt_tensor: " << tt_tensor << std::endl;

//    return 0;

//    std::vector<float> tt2{5, 6, 7, 8};
//    torch::Tensor tt2_tensor = torch::from_blob(tt2.data(), { 1, 2, 2 }, at::kFloat);

//    torch::Tensor cat = torch::cat({tt_tensor, tt2_tensor}, 0);

//    torch::Tensor view = cat.view({2,4});

//    std::cout << "tt_tensor: " << tt_tensor << std::endl
//              << "tt2_tensor: " << tt2_tensor << std::endl
//              << "cat: " << cat << std::endl
//              << "view: " << view << std::endl;

//    return 0;

//    std::vector<float> t{1, 2, 3};
//    torch::Tensor pick_point_tensor = torch::from_blob(t.data(), { 3 }, at::kFloat);
//    pick_point_tensor = pick_point_tensor.to(torch::kCPU);

//    std::vector<float> t2{4, 5, 6};
//    torch::Tensor src_tensor = torch::from_blob(t2.data(), { 1, 3 }, at::kFloat);
//    auto src_tensor_flatten = torch::flatten(src_tensor);

//    std::vector<float> t3{7, 8, 9, 10, 11, 12};
//    torch::Tensor src_height_tensor = torch::from_blob(t3.data(), { 1, 2, 3 }, at::kFloat);
//    auto src_height_tensor_flatten = torch::flatten(src_height_tensor);

//    auto before_state = torch::cat({ pick_point_tensor, src_tensor_flatten, src_height_tensor_flatten });

//    std::cout << before_state << std::endl;

//    before_state = torch::reshape(before_state, {3,4});

//    std::cout << before_state << std::endl;

//    return 0;

//    if ( !mDetector ) {
//        mDetector = std::make_shared<Detector>("/home/cpii/darknet-master/yolo_models/yolov3-df2.cfg", "/home/cpii/darknet-master/yolo_models/yolov3-df2_15000.weights");
//    }

//    float angle = 0;
//    cv::Mat rotatedImg;
//    QImage rotatedImgqt;
//    warped_image = cv::imread("/home/cpii/projects/data/0/after_warped_image.jpg");
//    QString filename_Src = QString("/home/cpii/projects/data/detect.jpg");
//    QByteArray filename_Srcc = filename_Src.toLocal8Bit();
//    const char *filename_Srccc = filename_Srcc.data();
//    float conf_before;
//    for(int a = 0; a < 36; a++){
//        rotatedImgqt = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888);

//        QMatrix r;

//        r.rotate(angle*10.0);

//        rotatedImgqt = rotatedImgqt.transformed(r);

//        rotatedImg = cv::Mat(rotatedImgqt.height(), rotatedImgqt.width(), CV_8UC3, rotatedImgqt.bits());

//        std::vector<bbox_t> test_result = mDetector->detect(rotatedImg);

//        if(test_result.size()>0){
//            for(int i=0; i<test_result.size(); i++){
//                if(test_result[i].obj_id == 1 && conf_before < test_result[i].prob){
//                    conf_before = test_result[i].prob;
//                    cv::imwrite(filename_Srccc, rotatedImg);
//                }
//            }
//        }
//        angle+=1;
//    }
//    qDebug() << "Classcification confidence level before action: " << conf_before;

//    torch::Device device(torch::kCPU);
//    if (torch::cuda::is_available()) {
//        std::cout << "CUDA is available! Training on GPU." << std::endl;
//        device = torch::Device(torch::kCUDA);
//    }

//    qDebug() << "Creating models";

//    auto policy = Policy();
//    auto target_policy = Policy();
//    auto critic = Critic();
//    auto target_critic = Critic();

//    policy->to(device);
//    critic->to(device);
//    target_policy->to(device);
//    target_critic->to(device);

//    qDebug() << "Creating optimizer";

//    torch::optim::Adam policy_optimizer(policy->parameters(), torch::optim::AdamOptions(lrp));
//    torch::optim::Adam critic_optimizer(critic->parameters(), torch::optim::AdamOptions(lrc));

//    qDebug() << "Loading models";
//    QString Pmodelname = "policy_model_t.pt";
//    QString Poptimodelname = "policy_optimizer_t.pt";
//    QString Cmodelname = "critic_model_t.pt";
//    QString Coptimodelname = "critic_optimizer_t.pt";
//    torch::load(policy, Pmodelname.toStdString());
//    torch::load(policy_optimizer, Poptimodelname.toStdString());
//    torch::load(critic, Cmodelname.toStdString());
//    torch::load(critic_optimizer, Coptimodelname.toStdString());

//    torch::AutoGradMode disable_grad_hardcopy(false);

//    qDebug() << "Copying parameters to target models";
//    for(size_t i=0; i < target_policy->parameters().size(); i++){
//        target_policy->parameters()[i].copy_(policy->parameters()[i]);
//    //    std::cout << "Pt: \n" << target_policy->parameters()[i].sizes() << std::endl;
//    //    std::cout << "P: \n" << policy->parameters()[i].sizes() << std::endl;
//    }

//    for(size_t i=0; i < target_critic->parameters().size(); i++){
//        target_critic->parameters()[i].copy_(critic->parameters()[i]);
//    //    std::cout << "Ct: \n" << target_critic->parameters()[i].sizes() << std::endl;
//    //    std::cout << "C: \n" << critic->parameters()[i].sizes() << std::endl;
//    }
//    //torch::AutoGradMode enable_grad_hardcopy(true);


//    std::vector<float> s(262147);
//    std::vector<float> s2(262147);
//    std::vector<float> a{3.5, 0.53, -4.5};
//    std::vector<float> r{200};
//    std::vector<float> d{0};
//    for(int i=0;i<262147;i++){
//        s2[i] = (rand()%200)/10.0f;
//    }
//    for(int i=0;i<262147;i++){
//        s[i] = 0.354;
//    }
//    auto s1_batch = torch::from_blob(s.data(), { 262147 }, at::kFloat);
//    auto s21_batch = torch::from_blob(s2.data(), { 262147 }, at::kFloat);
//    auto a1_batch = torch::from_blob(a.data(), { 3 }, at::kFloat);
//    auto r1_batch = torch::from_blob(r.data(), { 1 }, at::kFloat);
//    auto d1_batch = torch::from_blob(d.data(), { 1 }, at::kFloat);

//    memory.push_back({
//        s1_batch.clone(),
//        a1_batch.clone(),
//        r1_batch.clone(),
//        d1_batch.clone(),
//        s21_batch.clone(),
//        });

//    a = std::vector<float> {1.2, 0.532, 1.33};
//    a1_batch = torch::from_blob(a.data(), { 3 }, at::kFloat);

//    for(int i=0;i<262147;i++){
//        s2[i] = (rand()%200)/10.0f;
//    }

//    for(int i=0;i<262147;i++){
//        s[i] = 0.123;
//    }

//    s1_batch = torch::from_blob(s.data(), { 262147 }, at::kFloat);
//    s21_batch = torch::from_blob(s2.data(), { 262147 }, at::kFloat);

//    memory.push_back({
//        s1_batch.clone(),
//        a1_batch.clone(),
//        r1_batch.clone(),
//        d1_batch.clone(),
//        s21_batch.clone(),
//        });

//    r = std::vector<float> {100};
//    r1_batch = torch::from_blob(r.data(), { 1 }, at::kFloat);

//    memory.push_back({
//        s1_batch.clone(),
//        a1_batch.clone(),
//        r1_batch.clone(),
//        d1_batch.clone(),
//        s21_batch.clone()
//        });


//    int randomdata = 0;
//    std::vector<torch::Tensor> s_data(3), a_data(3), r_data(3), d_data(3), s2_data(3);
//    torch::Tensor s_batch, a_batch, r_batch, d_batch, s2_batch;


//    for (int i = 0; i < 3; i++) {
//        s_data[i] = torch::unsqueeze(memory[i+randomdata].before_state, 0);
//        a_data[i] = torch::unsqueeze(memory[i+randomdata].place_point, 0);
//        r_data[i] = torch::unsqueeze(memory[i+randomdata].reward, 0);
//        d_data[i] = torch::unsqueeze(memory[i+randomdata].done, 0);
//        s2_data[i] = torch::unsqueeze(memory[i+randomdata].after_state, 0);
//    }


//    s_batch = s_data[0]; a_batch = a_data[0]; r_batch = r_data[0]; d_batch = d_data[0]; s2_batch = s2_data[0];
//    for (int i = 1; i < 3; i++) {
//        //std::cout << s_data[i].sizes() << std::endl << s_batch.sizes() << std::endl;
//        s_batch = torch::cat({ s_batch, s_data[i] }, 0);
//        //std::cout << a_data[i].sizes() << std::endl << a_batch.sizes() << std::endl;
//        a_batch = torch::cat({ a_batch, a_data[i] }, 0);
//        //std::cout << r_data[i].sizes() << std::endl << r_batch.sizes() << std::endl;
//        r_batch = torch::cat({ r_batch, r_data[i] }, 0);
//        //std::cout << d_data[i].sizes() << std::endl << d_batch.sizes() << std::endl;
//        d_batch = torch::cat({ d_batch, d_data[i] }, 0);
//        //std::cout << s2_data[i].sizes() << std::endl << s2_batch.sizes() << std::endl;
//        s2_batch = torch::cat({ s2_batch, s2_data[i] }, 0);
//    }

//    std::cout << s_batch.mean() << std::endl
//              << a_batch << std::endl
//              << a_batch.mean() << std::endl
//              << r_batch << std::endl
//              << r_batch.mean() << std::endl
//              << d_batch << std::endl
//              << d_batch.mean() << std::endl
//              << s2_batch.mean() << std::endl;

//    s_batch = s_batch.to(device);
//    auto pout1 = policy->forward(s_batch);
//    std::cout << "pout1: " << pout1;

//    s_batch = s_batch.to(device);
//    auto pout2 = policy->forward(s_batch);
//    std::cout << "pout2: " << pout2;

//    s_batch = s_batch.to(device);
//    auto pout3 = policy->forward(s_batch);
//    std::cout << "pout3: " << pout3;

//    return 0;

//    s_batch = s_batch.to(device);
//    a_batch = a_batch.to(device);
//    r_batch = r_batch.to(device);
//    d_batch = d_batch.to(device);
//    s2_batch = s2_batch.to(device);

//    qDebug() << "\033[1;33Training critic model\033[0m";

//    auto a2_batch = target_policy->forward(s2_batch).to(device);
//    std::cout << "a2_batch: " << a2_batch << std::endl;
//    auto target_q = target_critic->forward(a2_batch, s2_batch).to(device);
//    std::cout << "target_q: " << target_q << std::endl;
//    std::cout << r_batch.type() << " " << d_batch.type() << " "<< target_q.type() << std::endl;
//    auto y = r_batch + (1.0 - d_batch) * GAMMA * target_q;
//    std::cout << "y: " << y << std::endl;
//    auto q = critic->forward(a_batch, s_batch);
//    std::cout << "q: " << q << std::endl;

//    auto critic_loss = torch::mse_loss(q, y.detach());
//    std::cout << "Critic loss: " << critic_loss;
//    float critic_lossf = critic_loss.item().toFloat();
//    std::cout << critic_lossf << std::endl;
//    critic_optimizer.zero_grad();
//    std::cout << critic_loss << std::endl;
//    critic_loss.backward();
//    critic_optimizer.step();

//    qDebug() << "\033[1;33mCritic optimizer step\033[0m";

//    // Policy loss
//    qDebug() << "\033[1;33mTraining policy model\033[0m";
//    auto a_predict = policy->forward(s_batch).to(device);
//    std::cout << "a_predict: " << a_predict;
//    auto policy_loss = -critic->forward(a_predict, s_batch);
//    std::cout << "policyloss: " << policy_loss << std::endl;
//    policy_loss = policy_loss.mean();
//    std::cout << "policyloss mean: " << policy_loss << std::endl;
//    auto policy_lossf = policy_loss.item().toFloat();
//    std::cout << policy_lossf << std::endl;
//    policy_optimizer.zero_grad();
//    std::cout << policy_loss.type() << std::endl;
//    policy_loss.backward();
//    policy_optimizer.step();

//    qDebug() << "\033[1;33mPolicy optimizer step\033[0m";

//    QString Pmodelname = "policy_model_t.pt";
//    QString Poptimodelname = "policy_optimizer_t.pt";
//    QString Cmodelname = "critic_model_t.pt";
//    QString Coptimodelname = "critic_optimizer_t.pt";
//    torch::save(policy, Pmodelname.toStdString());
//    torch::save(policy_optimizer, Poptimodelname.toStdString());
//    torch::save(critic, Cmodelname.toStdString());
//    torch::save(critic_optimizer, Coptimodelname.toStdString());

//    auto pout = policy->forward(s_batch);
//    std::cout << "pout: " << pout;

//    return 0;


//    qDebug() << "train";

//    policy->train();

//    auto a2_batch = target_policy->forward(s2_batch).to(device);
//    std::cout << a2_batch << std::endl;
//    auto target_q = target_critic->forward(a2_batch, s2_batch).to(device);
//    std::cout << target_q << std::endl;
//    //std::cout << r_batch.type() << " " << << d_batch.type() << " "<< target_q.type() << std::endl;
//    auto y = r_batch + (1.0 - d_batch) * GAMMA * target_q;
//    std::cout << "y: " << y << std::endl;
//    auto q = critic->forward(a_batch, s_batch);
//    std::cout << "q: " << q << std::endl;

//    auto critic_loss = torch::mse_loss(q, y.detach());
//    std::cout << "Critic loss: " << critic_loss << std::endl;
//    critic_optimizer.zero_grad();
//    critic_loss.backward();
//    critic_optimizer.step();

//    qDebug() << "\033[1;33mCritic optimizer step\033[0m";

//    // Policy loss
//    qDebug() << "\033[1;33mTraining policy model\033[0m";
//    auto a_predict = policy->forward(s_batch).to(device);
//    std::cout << "a_predict: " << a_predict << std::endl;
//    auto policy_loss = -critic->forward(a_predict, s_batch);
//    std::cout << "policyloss: " << policy_loss << std::endl;
//    policy_loss = policy_loss.mean();
//    std::cout << "policyloss mean: " << policy_loss << std::endl;
//    std::cout << policy_loss.type() << std::endl;
//    policy_optimizer.zero_grad();
//    policy_loss.backward();
//    policy_optimizer.step();

//    return 0;


    return false;
}

bool LP_Plugin_Garment_Manipulation::eventFilter(QObject *watched, QEvent *event)
{
    static int timerID = 0;

    if ( QEvent::MouseButtonRelease == event->type()){
        auto e = static_cast<QMouseEvent*>(event);

        if ( e->button() == Qt::RightButton ){
            if (markercount==0){
                qDebug("No marker data!");
            } else if (!gFoundBackground){

                markervecs_97[0] /= static_cast<double>(frame);
                markervecs_97[1] /= static_cast<double>(frame);
                markervecs_97[2] /= static_cast<double>(frame);

                cv::Mat rotM = cv::Mat::zeros(3,3, CV_32F);
                cv::Rodrigues(markervecs_97, rotM);

                float tmp_depth_point_97[2] = {0}, tmp_color_point_97[2] = {markertrans[0], markertrans[1]}, tmp_depth_point_98[2] = {0}, tmp_color_point_98[2] = {markercoordinate_98[0], markercoordinate_98[1]}, tmp_depth_point_99[2] = {0}, tmp_color_point_99[2] = {markercoordinate_99[0], markercoordinate_99[1]};
                rs2::frame depth;
                if(use_filter){
                    depth = filtered_frame;
                } else {
                    depth = frames.get_depth_frame();
                }
                rs2_project_color_pixel_to_depth_pixel(tmp_depth_point_97, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_point_97);
                int P_97 = int(tmp_depth_point_97[0])*depthh + (depthh-int(tmp_depth_point_97[1]));
                cv::Point3f Po_97 = cv::Point3f{mPointCloud[P_97].x(), -mPointCloud[P_97].y(), -mPointCloud[P_97].z()};
                Transformationmatrix_T2C = cv::Matx44f(rotM.at<double>(0,0), rotM.at<double>(0,1), rotM.at<double>(0,2), Po_97.x,
                                                       rotM.at<double>(1,0), rotM.at<double>(1,1), rotM.at<double>(1,2), Po_97.y,
                                                       rotM.at<double>(2,0), rotM.at<double>(2,1), rotM.at<double>(2,2), Po_97.z,
                                                                       0.0f,                 0.0f,                 0.0f,    1.0f);

                rs2_project_color_pixel_to_depth_pixel(tmp_depth_point_98, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_point_98);
                int P_98 = int(tmp_depth_point_98[0])*depthh + (depthh-int(tmp_depth_point_98[1]));
                cv::Point3f Po_98 = cv::Point3f{mPointCloud[P_98].x(), -mPointCloud[P_98].y(), -mPointCloud[P_98].z()};
                cv::Mat ptMat_98 = (cv::Mat_<float>(4,1) << Po_98.x, Po_98.y, Po_98.z, 1);
                cv::Mat_<float> dstMat_98(Transformationmatrix_T2C.inv() * ptMat_98);
                float scale_98 = dstMat_98(0,3);
                QVector4D Pt_98(dstMat_98(0,0)/scale_98, dstMat_98(0,1)/scale_98, dstMat_98(0,2)/scale_98, 1.0f);
                double thetaX = asin(Pt_98.z()/0.903);
                cv::Matx44f Rotatex = cv::Matx44f(1.0f,            0.0f,             0.0f, 0.0f,
                                                  0.0f,     cos(thetaX),     -sin(thetaX), 0.0f,
                                                  0.0f,     sin(thetaX),      cos(thetaX), 0.0f,
                                                  0.0f,            0.0f,             0.0f, 1.0f);
                Transformationmatrix_T2C = Transformationmatrix_T2C * Rotatex;

                rs2_project_color_pixel_to_depth_pixel(tmp_depth_point_99, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_point_99);
                int P_99 = int(tmp_depth_point_99[0])*depthh + (depthh-int(tmp_depth_point_99[1]));
                cv::Point3f Po_99 = cv::Point3f{mPointCloud[P_99].x(), -mPointCloud[P_99].y(), -mPointCloud[P_99].z()};
                cv::Mat ptMat_99 = (cv::Mat_<float>(4,1) << Po_99.x, Po_99.y, Po_99.z, 1);
                cv::Mat_<float> dstMat_99(Transformationmatrix_T2C.inv() * ptMat_99);
                float scale_99 = dstMat_99(0,3);
                QVector4D Pt_99(dstMat_99(0,0)/scale_99, dstMat_99(0,1)/scale_99, dstMat_99(0,2)/scale_99, 1.0f);
                double thetaY = asin(Pt_99.z()/0.9);
                cv::Matx44f Rotatey = cv::Matx44f( cos(-thetaY), 0.0f, sin(-thetaY), 0.0f,
                                                           0.0f, 1.0f,         0.0f, 0.0f,
                                                  -sin(-thetaY), 0.0f, cos(-thetaY), 0.0f,
                                                           0.0f, 0.0f,         0.0f, 1.0f);
                Transformationmatrix_T2C = Transformationmatrix_T2C * Rotatey;

                roi_corners[0].x = round(roi_corners[0].x / markercount);
                roi_corners[0].y = round(roi_corners[0].y / markercount);
                roi_corners[1].x = round(roi_corners[1].x / markercount);
                roi_corners[1].y = round(roi_corners[1].y / markercount);
                roi_corners[2].x = round(roi_corners[2].x / markercount);
                roi_corners[2].y = round(roi_corners[2].y / markercount);
                roi_corners[3].x = round(roi_corners[3].x / markercount);
                roi_corners[3].y = round(roi_corners[3].y / markercount);
                Robot_Plan( 0, 0 );
                gFoundBackground = true;

                // Save transformation matrix
                QString t_matrix = QString("%1 %2 %3 %4 %5 %6 %7 %8 %9 %10 %11 %12").arg(Transformationmatrix_T2C(0,0)).arg(Transformationmatrix_T2C(0,1)).arg(Transformationmatrix_T2C(0,2)).arg(Transformationmatrix_T2C(0,3)).arg(Transformationmatrix_T2C(1,0)).arg(Transformationmatrix_T2C(1,1)).arg(Transformationmatrix_T2C(1,2)).arg(Transformationmatrix_T2C(1,3)).arg(Transformationmatrix_T2C(2,0)).arg(Transformationmatrix_T2C(2,1)).arg(Transformationmatrix_T2C(2,2)).arg(Transformationmatrix_T2C(2,3));
                QString filename_transformation_matrix = QString(dataPath + "/transformation_matrix.txt");
                QFile filep(filename_transformation_matrix);
                if(filep.open(QIODevice::ReadWrite)) {
                    QTextStream streamp(&filep);
                    streamp << t_matrix;
                }

                mLabel->setText("Right click to plan");
                mLabel2->setText("Press 'T' to show table coordinate frame");
            } else if (!gStopFindWorkspace) {
                gStopFindWorkspace = true;
                mLabel2->setText(" ");
                Robot_Plan( 0, 0 );
                mLabel->setText("Right click to get the garment");
            } else if (!gPlan) {
                QProcess plan;
                QStringList planarg;

                planarg << "/home/cpii/projects/scripts/move.sh";
                plan.start("xterm", planarg);

                if ( plan.waitForFinished(60000)) {
                    mLabel->setText("Right click to collect data\n"
                                    "Press buttons to start training\n"
                                    "Press 'G' to start generate data");
                    gPlan = true;
                } else {
                    qCritical() << "Initialize garment POSITION failed.";
                }
            } else if (gPlan && !mRunReinforcementLearning1 && !mGenerateData) {
                mLabel->setText("Press SPACE to quit");
                if ( !mRunCollectData ) {
                    mRunCollectData = true;
                    timerID = startTimer(60000);        //Report Every 60s
                    auto future = QtConcurrent::run([this](){
                        uint consecutiveUseless = 0;

                        while(mRunCollectData){
                            bool bResetRobot = consecutiveUseless >= 5;

                            Robot_Plan(0, 0);

                            QProcess unfold;
                            QStringList unfoldarg;

                            unfoldarg << "/home/cpii/projects/scripts/unfold.sh";

                            //unfold.startDetached("xterm", unfoldarg);
                            unfold.start("xterm", unfoldarg);

                            constexpr int timeout_count = 60000; //60000 mseconds
                            if ( unfold.waitForFinished(timeout_count)) {
                                qDebug() << QString("\033[1;36m[%2] Robot action epoch %1 finished\033[0m").arg(datanumber, 5, 10, QChar('.'))
                                            .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                            .toUtf8().data();
                            } else {
                                qWarning() << QString("Robot action not finished within %1s").arg(timeout_count*0.001);
                                qWarning() << unfold.errorString();
                                unfold.kill();
                                unfold.waitForFinished();

                                bResetRobot = true;
                            }

                            if ( bResetRobot ) {
                                QMetaObject::invokeMethod(this, "resetRViz",
                                                          Qt::BlockingQueuedConnection);

                                if (!resetRobotPosition()){
                                    mRunCollectData = false;
                                    continue;
                                }
                                int k;
                                for ( k=0; k<=consecutiveUseless && 0<datanumber; ++k ) {
                                    QDir dir(QString("%1/%2").arg(dataPath).arg(datanumber--));
                                    if (!dir.removeRecursively()) {
                                        qCritical() << "[Warning] Useless data cannot be deleted : " << datanumber;
                                    }
                                }
                                QThread::msleep(6000);  //Wait for the robot to reset
                                useless_data -= k;
                                if ( useless_data < 0 ) {
                                    useless_data = 0;
                                }
                                consecutiveUseless = 0;
                                qDebug() << QString("\n-----[ %1 ]-----\n")
                                            .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                            .toUtf8().data();
                                continue;
                            }

                            // Save data
                            QString filename_dir = QString(dataPath + "/%1").arg(datanumber);

                            // Save points
                            std::vector<std::string> points;
                            auto future_camPt = QtConcurrent::run([&](){
                                for (auto i=0; i<mPointCloud.size(); i++){
                                    points.push_back(QString("%1 %2 %3").arg(mPointCloud[i].x())
                                                                        .arg(mPointCloud[i].y())
                                                                        .arg(mPointCloud[i].z()).toStdString());
                                }

                                QString filename_points = QString(filename_dir + "/after_points.txt");
                                QByteArray filename_pointsc = filename_points.toLocal8Bit();
                                const char *filename_pointscc = filename_pointsc.data();
                                std::ofstream output_file(filename_pointscc);
                                std::ostream_iterator<std::string> output_iterator(output_file, "\n");
                                std::copy(points.begin(), points.end(), output_iterator);
                            });

                            //Extract the moved image
                            gCamimage.copyTo(Src);
                            cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                            cv::resize(warped_image, warped_image, warped_image_resize);
                            warped_image = background - warped_image;
                            warped_image = ~warped_image;

                            // Save table points
                            auto future_tablePt = QtConcurrent::run([&](){
                                std::vector<std::string> tablepointIDs(warped_image.cols * warped_image.rows);
                                rs2::frame depth;
                                if(use_filter){
                                    depth = filtered_frame;
                                } else {
                                    depth = frames.get_depth_frame();
                                }

                                const auto &&mat_inv = WarpMatrix.inv();
                                const auto &&WW = 1.f/static_cast<float>(imageWidth)*warped_image_size.width,
                                           &&HH = 1.f/static_cast<float>(imageHeight)*warped_image_size.height;
                                const auto *start_ = tablepointIDs.data();

                                auto _future = QtConcurrent::map(tablepointIDs, [&](std::string &v){
                                    const auto id = &v - start_;
                                    int &&i = id / warped_image.cols;
                                    int &&j = id % warped_image.cols;
                                    cv::Point2f warpedp = {i * WW, j * HH};
                                    cv::Point3f &&homogeneous = mat_inv * warpedp;

                                    float tmp_depth_point[2] = {0}, tmp_color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                                    rs2_project_color_pixel_to_depth_pixel(tmp_depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_point);
                                    int tmpTableP = int(tmp_depth_point[0])*depthh + (depthh-int(tmp_depth_point[1]));
                        //            mTestP[i] = mPointCloud[tmpTableP];
                                    tablepointIDs.at(id) = QString("%1").arg(tmpTableP).toStdString();
                                });
                                _future.waitForFinished();
//                                for(int i=0; i<warped_image.cols; i++){
//                                    const auto &&i_ = i*WW;
//                                    for(int j=0; j<warped_image.rows; j++){
//                                        cv::Point2f warpedp = {i_, j * HH};
//                                        cv::Point3f &&homogeneous = mat_inv * warpedp;
//                                        OritablepointIDs.emplace_back(homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z);
//                                    }
//                                }

//                                //@!!!!!!!!!!!! The following part is WRONG!!!! The after depth should be averaged!!
//                                std::vector<std::string> tablepointIDs;
//                                auto depth = frames.get_depth_frame();
//                        //        mTestP.resize(OritablepointIDs.size());
//                                for (int i=0; i<OritablepointIDs.size(); i++){
//                                    float tmp_depth_point[2] = {0}, tmp_color_point[2] = {OritablepointIDs[i].x, OritablepointIDs[i].y};
//                                    rs2_project_color_pixel_to_depth_pixel(tmp_depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_point);
//                                    int tmpTableP = int(tmp_depth_point[0])*depthh + (depthh-int(tmp_depth_point[1]));
//                        //            mTestP[i] = mPointCloud[tmpTableP];
//                                    tablepointIDs.push_back(QString("%1").arg(tmpTableP).toStdString());
//                                }

                                QString filename_tablepointIDs = QString(filename_dir + "/after_tablepointIDs.txt");
                                QByteArray filename_tablepointIDsc = filename_tablepointIDs.toLocal8Bit();
                                const char *filename_tablepointIDscc = filename_tablepointIDsc.data();
                                std::ofstream output_file2(filename_tablepointIDscc);
                                std::ostream_iterator<std::string> output_iterator2(output_file2, "\n");
                                std::copy(tablepointIDs.begin(), tablepointIDs.end(), output_iterator2);
                            });


                            auto future_images = QtConcurrent::run([&](){
                                // Save Src
                                QString filename_Src = QString(filename_dir + "/after_Src.jpg");
                                QByteArray filename_Srcc = filename_Src.toLocal8Bit();
                                const char *filename_Srccc = filename_Srcc.data();
                                cv::imwrite(filename_Srccc, Src);

                                // Save warped image
                                cv::Mat sub_image;
                                sub_image = saved_warped_image - warped_image;
                                auto mean = cv::mean(sub_image);
                                qDebug() << "Pixel color diff mean: "<< mean[0] << " "<<mean[1]<< " "<< mean[2];
                                if(mean[0]<1.0 && mean[1]<1.0 && mean[2]<1.0){
                                    ++consecutiveUseless;
                                    qDebug() << "\033[0;33mNothing Changed(mean<1.0) : " << ++useless_data << "\033[0m";
                                } else {
                                    QString filename_warped = QString(filename_dir + "/after_warped_image.jpg");
                                    QByteArray filename_warpedc = filename_warped.toLocal8Bit();
                                    const char *filename_warpedcc = filename_warpedc.data();
                                    cv::imwrite(filename_warpedcc, warped_image);
                                    consecutiveUseless = 0;
                                }
                            });

                            future_images.waitForFinished();
                            future_tablePt.waitForFinished();
                            future_camPt.waitForFinished();

                            qDebug() << QString("\033[1;32m[%2] Data %1 Saved \033[0m")
                                        .arg(datanumber, 5, 10, QChar('.'))
                                        .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                        .toUtf8().data();

                            datanumber++;
                        }
                        qDebug() << "Quit CollectData()";
                    });
                } else {
                    if ( timerID > 0 ) {
                        killTimer(timerID);
                        timerID = 0;
                    }
                    qDebug() << "Collecting Data, press SPACE to stop";
                }
            }
        } else if ( e->button() == Qt::LeftButton ){
            QWidget *glW = static_cast<QWidget*>(watched);

            if(mGenerateData && mGetPlacePoint){
                place_pointxy.setX(float(e->pos().x()) / glW->width());
                place_pointxy.setY(float(e->pos().y()) / glW->height());
                qDebug() << float(e->pos().x()) / glW->width()
                         << float(e->pos().y()) / glW->height();
                mGetPlacePoint = false;
            }
        }
    } else if ( QEvent::KeyRelease == event->type()){
        auto e = static_cast<QKeyEvent*>(event);

        if ( e->key() == Qt::Key_Space ){
            if (gStopFindWorkspace){
                mRunCollectData = false;
                mRunReinforcementLearning1 = false;
                mRunReinforcementLearning2 = false;
                mRunTrainRod = false;
                mGenerateData = false;
                QProcess exit;
                QStringList exitarg;
                exitarg << "/home/cpii/projects/scripts/exit.sh";
                exit.startDetached("xterm", exitarg);
                mLabel->setText("Right click to collect data\n"
                                "Press buttons to start training\n"
                                "Press 'G' to start generate data");
            }
        } else if ( e->key() == Qt::Key_T ) {
            mShowInTableCoordinateFrame = !mShowInTableCoordinateFrame;
        }
//          else if ( e->key() == Qt::Key_R && gPlan && !mRunCollectData && !mRunReinforcementLearning1 && !mGenerateData){
//            mRunReinforcementLearning1 = true;
//            mLabel->setText("Training model, press SPACE to stop");
//            Reinforcement_Learning_1();
//        }
          else if ( e->key() == Qt::Key_G && gPlan && !mRunCollectData && !mRunReinforcementLearning1 && !mRunReinforcementLearning2 && !mRunTrainRod && !mGenerateData){
            mGenerateData = true;
            mLabel->setText("Generating data, press SPACE to stop\n"
                            "Press 'D' to delete the last data");
            Generate_Data();
        } else if ( e->key() == Qt::Key_D && gPlan && !mRunCollectData && !mRunReinforcementLearning1 && !mRunReinforcementLearning2 && !mRunTrainRod && mGenerateData){
            QDir dir(QString("%1/%2").arg(datagen).arg(total_steps--));
            if (!dir.removeRecursively()) {
                qCritical() << "[Warning] Useless data cannot be deleted : " << total_steps+1;
            } else {
                qDebug() << "Data deleted : " << total_steps+1;
            }
        }
    }

    return QObject::eventFilter(watched, event);
}

bool LP_Plugin_Garment_Manipulation::saveCameraParams(const std::string &filename, cv::Size imageSize, float aspectRatio, int flags,
                             const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, double totalAvgErr) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if(!fs.isOpened())
        return false;

    time_t tt;
    time(&tt);
    struct tm *t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    if(flags & cv::CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;

    if(flags != 0) {
        sprintf(buf, "flags: %s%s%s%s",
                flags & cv::CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                flags & cv::CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                flags & cv::CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                flags & cv::CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;

    return true;
}

void LP_Plugin_Garment_Manipulation::calibrate()
{
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);

    // create charuco board object
    cv::Ptr<cv::aruco::CharucoBoard> charucoboard = cv::aruco::CharucoBoard::create(11, 8, 0.02, 0.015, dictionary); // create charuco board;

    // collect data from each frame
    std::vector< std::vector< std::vector< cv::Point2f > > > allCorners;
    std::vector< std::vector< int > > allIds;
    std::vector< cv::Mat > allImgs;
    cv::Size imgSize;


    // for ( int i=0; i<30; ++i ){
    cv::VideoCapture inputVideo(4); // Grey: 2, 8. Color: 4, 10.

    inputVideo.set(cv::CAP_PROP_FRAME_WIDTH, 1280); // valueX = your wanted width
    inputVideo.set(cv::CAP_PROP_FRAME_HEIGHT, 800); // valueY = your wanted heigth

    double aspectRatio = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH) / inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);

    if (!inputVideo.isOpened()) {
        std::cerr << "ERROR: Could not open camera "  << std::endl;
        return;
     }
   // }

    cv::Mat frame1;
    inputVideo >> frame1;
    qDebug() << frame1.cols << "x" << frame1.rows << " Aspect : " << aspectRatio;

    while(inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);

        std::vector< int > ids;
        std::vector< std::vector< cv::Point2f > > corners;

        // detect markers
        cv::aruco::detectMarkers(image, dictionary, corners, ids);


        // interpolate charuco corners
        cv::Mat currentCharucoCorners, currentCharucoIds;
        if(ids.size() > 0)
            cv::aruco::interpolateCornersCharuco(corners, ids, image, charucoboard, currentCharucoCorners,
                                             currentCharucoIds);

        // draw results
        image.copyTo(imageCopy);
        if(ids.size() > 0) cv::aruco::drawDetectedMarkers(imageCopy, corners);

        if(currentCharucoCorners.total() > 0)
            cv::aruco::drawDetectedCornersCharuco(imageCopy, currentCharucoCorners, currentCharucoIds);

        cv::putText(imageCopy, "Press 'c' to add current frame. 'ESC' to finish and calibrate",
                cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

//        cv::imshow("out", imageCopy);
        char key = (char)cv::waitKey(30);
        if(key == 27) break;
        if(key == 'c' && ids.size() > 0) {
            std::cout << "Frame captured" << "\n";
            allCorners.push_back(corners);
            allIds.push_back(ids);
            allImgs.push_back(image);
            imgSize = image.size();
        }
    }

    if(allIds.size() < 1) {
        std::cerr << "Not enough captures for calibration" << "\n";
        return;
    }

    cv::Mat cameraMatrix, distCoeffs;
    std::vector< cv::Mat > rvecs, tvecs;
    double repError;
    int calibrationFlags = 0;


    if(calibrationFlags & cv::CALIB_FIX_ASPECT_RATIO) {
        cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        cameraMatrix.at< double >(0, 0) = aspectRatio;
    }

    // prepare data for charuco calibration
    int nFrames = (int)allCorners.size();
    std::vector< cv::Mat > allCharucoCorners;
    std::vector< cv::Mat > allCharucoIds;
    std::vector< cv::Mat > filteredImages;
    allCharucoCorners.reserve(nFrames);
    allCharucoIds.reserve(nFrames);

    for(int i = 0; i < nFrames; i++) {
        // interpolate using camera parameters
        cv::Mat currentCharucoCorners, currentCharucoIds;
        cv::aruco::interpolateCornersCharuco(allCorners[i], allIds[i], allImgs[i], charucoboard,
                                         currentCharucoCorners, currentCharucoIds, cameraMatrix,
                                         distCoeffs);

        allCharucoCorners.push_back(currentCharucoCorners);
        allCharucoIds.push_back(currentCharucoIds);
        filteredImages.push_back(allImgs[i]);
    }

    if(allCharucoCorners.size() < 4) {
        std::cerr << "Not enough corners for calibration" << "\n";
        return;
    }

    // calibrate camera using charuco
    repError =
        cv::aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, charucoboard, imgSize,
                                      cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);

    bool saveOk =  saveCameraParams("cam_cal_165", imgSize, aspectRatio, calibrationFlags,
                                    cameraMatrix, distCoeffs, repError);

    if(!saveOk) {
        std::cerr << "Cannot save output file" << "\n";
        return;
    }

}

void LP_Plugin_Garment_Manipulation::resetRViz()
{
    if ( gProc_RViz ) {
        gProc_RViz->terminate();
        gProc_RViz->waitForFinished();
        gProc_RViz.reset();
    }
    gProc_RViz = std::make_shared<QProcess>(nullptr);
    QString openrvizarg{"/home/cpii/projects/scripts/openrviz.sh"};
    gProc_RViz->start("xterm", {openrvizarg});
    Sleeper::sleep(3);
}

bool LP_Plugin_Garment_Manipulation::resetRobotPosition()
{
    QString filename = "/home/cpii/projects/scripts/reset.sh";
    QFile file(filename);

    if (!file.open(QIODevice::ReadWrite)) {
        qWarning() << "Cannot set reset file.";
        return false;
    }
    file.setPermissions(QFileDevice::Permissions(1909));
    QTextStream stream(&file);
    stream << "#!/bin/bash" << "\n"
          << "\n"
          << "cd" << "\n"
          << "\n"
          << "source /opt/ros/foxy/setup.bash" << "\n"
          << "\n"
          << "source ~/ws_moveit2/install/setup.bash" << "\n"
          << "\n"
          << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
          << "\n"
          << "cd tm_robot_gripper/" << "\n"
          << "\n"
          << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
          << "\n"
          << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 1, positions: [-1.025, -0.29, 1.952, -0.1, 1.57, 0.545], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
          //<< "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
          << "\n";

    file.close();
    QProcess proc;
    QStringList args;
    args << filename;
    proc.start("xterm", args);
    constexpr int timeout_count = 60000; //60000 mseconds
    if ( proc.waitForFinished(timeout_count)) {
       qDebug() << "Robot reset finished";
       ++gRobotResetCount;
    } else {
       qWarning() << QString("Robot reset not finished within %1s").arg(timeout_count*0.001);
       qWarning() << proc.errorString();
       return false;
    }
    return true;
}

void LP_Plugin_Garment_Manipulation::savedata(QString fileName, std::vector<float> datas){
    std::vector<std::string> data_string;

    for (auto i=0; i<datas.size(); i++){
        data_string.push_back(QString("%1").arg(datas[i]).toStdString());
    }

    QByteArray filename = fileName.toLocal8Bit();
    const char *filenamecc = filename.data();
    std::ofstream output_file(filenamecc);
    std::ostream_iterator<std::string> output_iterator(output_file, "\n");
    std::copy(data_string.begin(), data_string.end(), output_iterator);
    output_file.close();
}

void LP_Plugin_Garment_Manipulation::loaddata(std::string fileName, std::vector<float> &datas){
    // Open the File
    std::ifstream in(fileName.c_str());
    std::string str;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Build an istream that holds the input string
        std::istringstream iss(str);

        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0){
            // Iterate over the istream, using >> to grab floats
            // and push_back to store them in the vector
            std::copy(std::istream_iterator<float>(iss),
                  std::istream_iterator<float>(),
                  std::back_inserter(datas));
        }
    }
    //Close The File
    in.close();
}

void LP_Plugin_Garment_Manipulation::findgrasp(std::vector<double> &grasp, cv::Point &grasp_point, float &Rz, std::vector<cv::Vec4i> hierarchy){
    int size = 0;

    for( size_t i = 0; i< contours.size(); i++ ){
        cv::Scalar color = cv::Scalar( rng.uniform(0, 180), rng.uniform(50,180), rng.uniform(0,180) );
        cv::drawContours( drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0 );

        for (size_t j = 0; j < contours[i].size(); j++){
            if ((static_cast<double>(contours[i][j].x) / imageWidth > 0.6
                    && static_cast<double>(contours[i][j].y) / imageHeight > 0.6 )
                    || (sqrt(pow((static_cast<double>(contours[i][j].x) / imageWidth - 0.83), 2) + pow((static_cast<double>(contours[i][j].y) / imageHeight - 0.83), 2)) > 0.7745)
                    || (static_cast<double>(contours[i][j].x) / imageWidth > 0.90
                    && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)
                    || (static_cast<double>(contours[i][j].x) / imageWidth < 0.10
                    && static_cast<double>(contours[i][j].y) / imageHeight > 0.90)
                    || (static_cast<double>(contours[i][j].x) / imageWidth > 0.480
                    && static_cast<double>(contours[i][j].x) / imageWidth < 0.580
                    && static_cast<double>(contours[i][j].y) / imageHeight < 0.10))
            { // Filter out the robot arm and markers
                    size = size - 1;
            }
        }
        size += contours[i].size();
    }

    if (size == 0){
        qDebug() << "No garment detected!";
        return;
    }

    int randcontour = rand()%int(contours.size());
    int randp = rand()%int(contours[randcontour].size());

    while((static_cast<double>(contours[randcontour][randp].x) / imageWidth > 0.6
           && static_cast<double>(contours[randcontour][randp].y) / imageHeight > 0.6 )
           || (sqrt(pow((static_cast<double>(contours[randcontour][randp].x) / imageWidth - 0.83), 2) + pow((static_cast<double>(contours[randcontour][randp].y) / imageHeight - 0.83), 2)) > 0.7745)
           || (static_cast<double>(contours[randcontour][randp].x) / imageWidth > 0.90
           && static_cast<double>(contours[randcontour][randp].y) / imageHeight < 0.10)
           || (static_cast<double>(contours[randcontour][randp].x) / imageWidth < 0.10
           && static_cast<double>(contours[randcontour][randp].y) / imageHeight > 0.90)
           || (static_cast<double>(contours[randcontour][randp].x) / imageWidth > 0.480
           && static_cast<double>(contours[randcontour][randp].x) / imageWidth < 0.580
           && static_cast<double>(contours[randcontour][randp].y) / imageHeight < 0.10))
    { // Filter out the robot arm and markers
        randcontour = rand()%int(contours.size());
        randp = rand()%int(contours[randcontour].size());
    }
    double tangent;
    cv::Point left, mid, right;
    if(contours[randcontour].size() < 3){

    } else if (randp == 0){
        if(contours[randcontour][randp].x < contours[randcontour][randp+1].x){
            left = contours[randcontour][randp];
            right = contours[randcontour][randp+1];
        } else {
            left = contours[randcontour][randp+1];
            right = contours[randcontour][randp];
        }
        mid.y = std::min(contours[randcontour][randp].y, contours[randcontour][randp+1].y);
        if(contours[randcontour][randp].y > contours[randcontour][randp+1].y){
            mid.x = contours[randcontour][randp].x;
        } else {
            mid.x = contours[randcontour][randp+1].x;
        }
    } else if (randp+1 == contours[randcontour].size()){
        if(contours[randcontour][randp].x < contours[randcontour][randp-1].x){
            left = contours[randcontour][randp];
            right = contours[randcontour][randp-1];
        } else {
            left = contours[randcontour][randp-1];
            right = contours[randcontour][randp];
        }
        mid.y = std::min(contours[randcontour][randp].y, contours[randcontour][randp-1].y);
        if(contours[randcontour][randp].y > contours[randcontour][randp-1].y){
            mid.x = contours[randcontour][randp].x;
        } else {
            mid.x = contours[randcontour][randp-1].x;
        }
    } else {
        if(contours[randcontour][randp-1].x < contours[randcontour][randp+1].x){
            left = contours[randcontour][randp-1];
            right = contours[randcontour][randp+1];
        } else {
            left = contours[randcontour][randp+1];
            right = contours[randcontour][randp-1];
        }
        mid.y = std::min(contours[randcontour][randp+1].y, contours[randcontour][randp-1].y);
        if(contours[randcontour][randp+1].y > contours[randcontour][randp-1].y){
            mid.x = contours[randcontour][randp+1].x;
        } else {
            mid.x = contours[randcontour][randp-1].x;
        }
    }
    float Pi2Angle = 1.0/PI*180.0;
    float Angle2Pi = 1.0/180.0*PI;
    if(contours[randcontour].size() < 3){
        Rz = 0.0;
    } else if(left.y < right.y){
        tangent = acos(angle(right, mid, left));
        Rz = float(tangent) - (45.0*Angle2Pi);
    } else {
        tangent = acos(angle(left, mid, right));
        if(tangent >= 0 && tangent <= 45.0*Angle2Pi){
            Rz = float(-tangent) - (45.0*Angle2Pi);
        } else {
            Rz = float(-tangent) + (135.0*Angle2Pi);
        }
    }
    grasp_point.x = contours[randcontour][randp].x;
    grasp_point.y = contours[randcontour][randp].y;

    cv::Point2f warpedpg = cv::Point2f(grasp_point.x/static_cast<float>(imageWidth)*warped_image_size.width, grasp_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
    cv::Point3f homogeneousg = WarpMatrix.inv() * warpedpg;
    float depth_pointg[2] = {0}, color_pointg[2] = {homogeneousg.x/homogeneousg.z, homogeneousg.y/homogeneousg.z};
    rs2::frame depth;
    if(use_filter){
        depth = filtered_frame;
    } else {
        depth = frames.get_depth_frame();
    }
    rs2_project_color_pixel_to_depth_pixel(depth_pointg, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointg);

    offset[0] = 0; offset[1] = 0;
    graspx = int(depth_pointg[0]);
    graspy = int(depth_pointg[1]);
    mCalAveragePoint = true;
    avgHeight.clear();
    acount = 0;
    while (acount<30){
        Sleeper::msleep(200);
    }
    mCalAveragePoint = false;
    grasp[0] = grasppt.x();
    grasp[1] = grasppt.y();
    grasp[2] = gripper_length + grasppt.z() - 0.005;
    if(grasp[2] < mTableHeight){
//        qDebug() << QString("\033[1;31m[Warning] Crashing table : (%1, %2, %3).  "
//                            "\033[1;33mChange h to : %4\033[0m")
//                      .arg(grasp[0],0,'f',4).arg(grasp[1],0,'f',4).arg(grasp[2],0,'f',4)
//                      .arg(mTableHeight).toUtf8().data();
        grasp[2] = mTableHeight;
    }
}

void LP_Plugin_Garment_Manipulation::findrelease(std::vector<double> &release, cv::Point &release_point, cv::Point grasp_point){
    double random_T[2] = { 0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1]),    //[0.1 < x < 0.9]
                           0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1])};

    release_point = {random_T[0]*imageWidth,
                     random_T[1]*imageHeight};

    auto releasePt_RGB = warped_image.at<cv::Vec3b>(release_point.y, release_point.x);

    while((random_T[0] > 0.6 && random_T[1] > 0.6 )                                         //At the robot root region
          || (sqrt(pow(random_T[0] - 0.83, 2) + pow( random_T[1] - 0.83, 2)) > 0.7745)      //Too far for robot arm (max. allowed : distance=850cm, height=350cm)
          || releasePt_RGB[0] < uThres
          || releasePt_RGB[1] < uThres
          || releasePt_RGB[2] < uThres                                                      //Only white region (empty Table)
          || 20.0 > cv::norm(release_point - grasp_point))                                  //Distance between 2 points less than 20 pixels (searching region 20x20)
    { // Limit the release point
        cv::drawMarker( drawing,
                        release_point,
                        cv::Scalar(0, 205, 205 ),
                        cv::MARKER_TILTED_CROSS,
                        20, 2, cv::LINE_AA);

//        qDebug() << QString("\033[0;37mRGB( %1, %2, %3 ), XY(%4, %5)\033[0m")
//                    .arg(releasePt_RGB[0])
//                    .arg(releasePt_RGB[1])
//                    .arg(releasePt_RGB[2])
//                    .arg(release_point.x)
//                    .arg(release_point.y).toUtf8().data();

        random_T[0] = 0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1]);
        random_T[1] = 0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1]);

        release_point = {random_T[0]*imageWidth,
                         random_T[1]*imageHeight};

        releasePt_RGB = warped_image.at<cv::Vec3b>(release_point.y, release_point.x);
    }
}

void LP_Plugin_Garment_Manipulation::Find_Rod_Angle(std::string type, cv::Point& point, float& angle){
    bool RedoFindPoint = true;
    while(RedoFindPoint){
        RedoFindPoint = false;

        gCamimage.copyTo(Src);
        cv::Mat inv_warp_imager;
        cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
        cv::resize(warped_image, inv_warp_imager, warped_image_size);
        cv::warpPerspective(inv_warp_imager, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
        cv::resize(warped_image, warped_image, warped_image_resize);
        warped_image = background - warped_image;
        warped_image = ~warped_image;

        cv::Point center_point = cv::Point(0, 0);
        cv::Mat3b hsv;
        cv::Mat1b center_point_mask;
        cv::cvtColor(warped_image, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(50, 25, 25), cv::Scalar(70, 255, 255), center_point_mask);
        std::vector<cv::Point> center_point_locations;   // output, locations of non-zero pixels
        cv::findNonZero(center_point_mask, center_point_locations);
        // access pixel coordinates
        for(int i=0; i<center_point_locations.size(); i++){
            center_point.x += center_point_locations[i].x;
            center_point.y += center_point_locations[i].y;
        }
        if(center_point_locations.size()==0){
            qDebug() << "No green point found! Redo";
            RedoFindPoint = true;
            continue;
        } else {
            center_point.x /= center_point_locations.size();
            center_point.y /= center_point_locations.size();
        }

        if(type == "red"){
            cv::Point red_point;
            cv::Mat1b red_point_mask;
            cv::inRange(hsv, cv::Scalar(165, 100, 100), cv::Scalar(180, 255, 255), red_point_mask);
            std::vector<cv::Point> red_point_locations;   // output, locations of non-zero pixels
            cv::findNonZero(red_point_mask, red_point_locations);
            // access pixel coordinates
            for(int i=0; i<red_point_locations.size(); i++){
                red_point.x += red_point_locations[i].x;
                red_point.y += red_point_locations[i].y;
            }
            if(red_point_locations.size()==0){
                qDebug() << "No red point found! Redo";
                RedoFindPoint = true;
                continue;
            } else {
                red_point.x /= red_point_locations.size();
                red_point.y /= red_point_locations.size();
            }

            float dx = red_point.x - center_point.x;
            float dy = red_point.y - center_point.y;
            angle = atan2(dy, dx);
            if(angle < 0.0){
                angle += 2*PI;
            }
            point = red_point;

//            gLock.lockForWrite();
//            gInvWarpImage = QImage((uchar*) red_point_mask.data, red_point_mask.cols, red_point_mask.rows, red_point_mask.step, QImage::Format_Grayscale8).copy();
//            gLock.unlock();
//            emit glUpdateRequest();
        } else if(type == "blue"){
            cv::Point blue_point;
            cv::Mat1b blue_point_mask;
            cv::inRange(hsv, cv::Scalar(100, 25, 25), cv::Scalar(140, 255, 255), blue_point_mask);
            std::vector<cv::Point> blue_point_locations;   // output, locations of non-zero pixels
            cv::findNonZero(blue_point_mask, blue_point_locations);
            // access pixel coordinates
            for(int i=0; i<blue_point_locations.size(); i++){
                blue_point.x += blue_point_locations[i].x;
                blue_point.y += blue_point_locations[i].y;
            }
            if(blue_point_locations.size()==0){
                qDebug() << "No blue point found! Redo";
                RedoFindPoint = true;
                continue;
            } else {
                blue_point.x /= blue_point_locations.size();
                blue_point.y /= blue_point_locations.size();
            }

            float dx = blue_point.x - center_point.x;
            float dy = blue_point.y - center_point.y;
            angle = atan2(dy, dx);
            if(angle < 0.0){
                angle += 2*PI;
            }
            point = blue_point;
        } else if(type == "marker0"){
            mMarkerPosi.x = 0;
            mMarkerPosi.y = 0;
            mTarget_marker = 0;
            mFindmarker = true;
            while (mFindmarker){
                Sleeper::msleep(200);
            }
            mFindmarker = false;

            cv::Point2f warpedpg = cv::Point2f(mMarkerPosi.x, mMarkerPosi.y);
            cv::Point3f homogeneousg = WarpMatrix * warpedpg;
            cv::Point marker0 = cv::Point((homogeneousg.x/homogeneousg.z)/float(warped_image_size.width)*warped_image_resize.width, (homogeneousg.y/homogeneousg.z)/float(warped_image_size.height)*warped_image_resize.height);

            float dx = marker0.x - center_point.x;
            float dy = marker0.y - center_point.y;
            angle = atan2(dy, dx);
            if(angle < 0.0){
                angle += 2*PI;
            }
            point = marker0;
        } else if(type == "marker17"){
            mMarkerPosi.x = 0;
            mMarkerPosi.y = 0;
            mTarget_marker = 17;
            mFindmarker = true;
            while (mFindmarker){
                Sleeper::msleep(200);
            }
            mFindmarker = false;

            cv::Point2f warpedpg = cv::Point2f(mMarkerPosi.x, mMarkerPosi.y);
            cv::Point3f homogeneousg = WarpMatrix * warpedpg;
            cv::Point marker17 = cv::Point((homogeneousg.x/homogeneousg.z)/float(warped_image_size.width)*warped_image_resize.width, (homogeneousg.y/homogeneousg.z)/float(warped_image_size.height)*warped_image_resize.height);

            float dx = marker17.x - center_point.x;
            float dy = marker17.y - center_point.y;
            angle = atan2(dy, dx);
            if(angle < 0.0){
                angle += 2*PI;
            }
            point = marker17;
        } else if(type == "marker20"){
            mMarkerPosi.x = 0;
            mMarkerPosi.y = 0;
            mTarget_marker = 20;
            mFindmarker = true;
            while (mFindmarker){
                Sleeper::msleep(200);
            }
            mFindmarker = false;

            cv::Point2f warpedpg = cv::Point2f(mMarkerPosi.x, mMarkerPosi.y);
            cv::Point3f homogeneousg = WarpMatrix * warpedpg;
            cv::Point marker20 = cv::Point((homogeneousg.x/homogeneousg.z)/float(warped_image_size.width)*warped_image_resize.width, (homogeneousg.y/homogeneousg.z)/float(warped_image_size.height)*warped_image_resize.height);

            float dx = marker20.x - center_point.x;
            float dy = marker20.y - center_point.y;
            angle = atan2(dy, dx);
            if(angle < 0.0){
                angle += 2*PI;
            }
            point = marker20;
        }
    }
}

void LP_Plugin_Garment_Manipulation::Find_Angle_Between_Rods(float& angle_between_rods, float& red_angle, float& blue_angle){
    cv::Point red_point, blue_point;
    Find_Rod_Angle("red", red_point, red_angle);
    Find_Rod_Angle("blue", blue_point, blue_angle);
    if(red_angle > blue_angle){
        angle_between_rods = red_angle - blue_angle;
    } else {
        angle_between_rods = blue_angle - red_angle;
    }
    if(angle_between_rods > PI){
        angle_between_rods = 2*PI - angle_between_rods;
    }

//    qDebug() << "red: " << red_point.x << " " << red_point.y << " " << red_angle << "\n"
//             << "blue: " << blue_point.x << " " << blue_point.y << " " << blue_angle << "\n"
//             << "angle: " << angle_between_rods;

//    warped_image.copyTo(drawing);

//    cv::circle( drawing,
//                red_point,
//                12,
//                cv::Scalar( 0, 255, 255 ),
//                3,//cv::FILLED,
//                cv::LINE_AA );
//    cv::circle( drawing,
//                blue_point,
//                12,
//                cv::Scalar( 255, 0, 0 ),
//                3,//cv::FILLED,
//                cv::LINE_AA );

//    gLock.lockForWrite();
//    gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
//    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
//    gLock.unlock();
//    emit glUpdateRequest();
}


void LP_Plugin_Garment_Manipulation::Find_Square_Angle(float& Square_Angle){
    cv::Point marker17, marker20;
    float marker17_angle, marker20_angle;
    Find_Rod_Angle("marker17", marker17, marker17_angle);
    Find_Rod_Angle("marker20", marker20, marker20_angle);
    if(marker17_angle > marker20_angle){
        Square_Angle = marker17_angle - marker20_angle;
    } else {
        Square_Angle = marker20_angle - marker17_angle;
    }
    if(Square_Angle > PI){
        Square_Angle = 2*PI - Square_Angle;
    }
}

void LP_Plugin_Garment_Manipulation::Push_Down(cv::Point push_point, float& rod_angle, float push_distance, std::vector<float>& start_point, std::vector<float>& end_point){
    float push_angle;
    if(PI*0.5<=rod_angle && rod_angle < PI*1.5){
        push_angle = rod_angle+0.5*PI;
    } else {
        push_angle = rod_angle-0.5*PI;
        if(push_angle < 0.0){
            push_angle+=2*PI;
        }
    }
    cv::Point2f warpedp = cv::Point2f(push_point.x/static_cast<float>(imageWidth)*warped_image_size.width,
                                      push_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
    cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
    float depth_point[2] = {0},
          color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
    rs2::frame depth;
    if(use_filter){
        depth = filtered_frame;
    } else {
        depth = frames.get_depth_frame();
    }
    rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
    int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
    cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
    cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
    cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
    float scale = dstMat(0,3);
    QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
    QVector4D push_pt = Transformationmatrix_T2R.inverted() * Pt;
    start_point[0] = push_pt.x()+0.04*cos(push_angle-0.25*PI);
    start_point[1] = push_pt.y()-0.04*sin(push_angle-0.25*PI);
    float push_angle_R = push_angle+PI;
    if(push_angle_R>2*PI){
        push_angle_R -= 2*PI;
    }
    end_point[0] = push_pt.x()+push_distance*cos(push_angle_R-0.25*PI);
    end_point[1] = push_pt.y()-push_distance*sin(push_angle_R-0.25*PI);

    auto draw_x = push_point.x+50*cos(push_angle);
    auto draw_y = push_point.y+50*sin(push_angle);
    cv::arrowedLine(drawing, cv::Point(draw_x, draw_y), push_point, cv::Scalar(0,0,255), 3, cv::LINE_AA, 0, 0.25);

    gLock.lockForWrite();
    gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
    gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
    gLock.unlock();
    emit glUpdateRequest();
}

void LP_Plugin_Garment_Manipulation::Push_Up(cv::Point push_point, float& rod_angle, float push_distance, std::vector<float>& start_point, std::vector<float>& end_point){
    float push_angle;
    if(PI*0.5<=rod_angle && rod_angle<PI*1.5){
        push_angle = rod_angle-0.5*PI;
    } else {
        push_angle = rod_angle+0.5*PI;
        if(push_angle > 2*PI){
            push_angle-=2*PI;
        }
    }
    cv::Point2f warpedp = cv::Point2f(push_point.x/static_cast<float>(imageWidth)*warped_image_size.width,
                                      push_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
    cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
    float depth_point[2] = {0},
          color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
    rs2::frame depth;
    if(use_filter){
        depth = filtered_frame;
    } else {
        depth = frames.get_depth_frame();
    }
    rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
    int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
    cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
    cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
    cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
    float scale = dstMat(0,3);
    QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
    QVector4D push_pt = Transformationmatrix_T2R.inverted() * Pt;
    start_point[0] = push_pt.x()+0.04*cos(push_angle-0.25*PI);
    start_point[1] = push_pt.y()-0.04*sin(push_angle-0.25*PI);
    float push_angle_R = push_angle+PI;
    if(push_angle_R>2*PI){
        push_angle_R -= 2*PI;
    }
    end_point[0] = push_pt.x()+push_distance*cos(push_angle_R-0.25*PI);
    end_point[1] = push_pt.y()-push_distance*sin(push_angle_R-0.25*PI);

    auto draw_x = push_point.x+50*cos(push_angle);
    auto draw_y = push_point.y+50*sin(push_angle);
    cv::arrowedLine(drawing, cv::Point(draw_x, draw_y), push_point, cv::Scalar(0,0,255), 3, cv::LINE_AA, 0, 0.25);

    gLock.lockForWrite();
    gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
    gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
    gLock.unlock();
    emit glUpdateRequest();
}

void LP_Plugin_Garment_Manipulation::Push_to_Center(cv::Point push_point, float& rod_angle, float push_distance, std::vector<float>& start_point, std::vector<float>& end_point){
    float push_angle = rod_angle;

    cv::Point2f warpedp = cv::Point2f(push_point.x/static_cast<float>(imageWidth)*warped_image_size.width,
                                      push_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
    cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
    float depth_point[2] = {0},
          color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
    rs2::frame depth;
    if(use_filter){
        depth = filtered_frame;
    } else {
        depth = frames.get_depth_frame();
    }
    rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
    int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
    cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
    cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
    cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
    float scale = dstMat(0,3);
    QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
    QVector4D push_pt = Transformationmatrix_T2R.inverted() * Pt;
    start_point[0] = push_pt.x()+0.03*cos(push_angle-0.25*PI);
    start_point[1] = push_pt.y()-0.03*sin(push_angle-0.25*PI);
    float push_angle_R = push_angle+PI;
    if(push_angle_R>2*PI){
        push_angle_R -= 2*PI;
    }
    end_point[0] = push_pt.x()+push_distance*cos(push_angle_R-0.25*PI);
    end_point[1] = push_pt.y()-push_distance*sin(push_angle_R-0.25*PI);

    auto draw_x = push_point.x+50*cos(push_angle);
    auto draw_y = push_point.y+50*sin(push_angle);
    cv::arrowedLine(drawing, cv::Point(draw_x, draw_y), push_point, cv::Scalar(0,0,255), 3, cv::LINE_AA, 0, 0.25);

    gLock.lockForWrite();
    gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
    gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
    gLock.unlock();
    emit glUpdateRequest();
}

void LP_Plugin_Garment_Manipulation::Push_Rod(int push_point, float push_distance, bool use_clock){
    bool bResetRobot = true;
    while(bResetRobot){
        // push_point = 0: push red point down, 1: push red point up, 2: push blue point down, 3: push blue point up
        // if use_clock: 0: push red point clockwise, 1: push red point counterclockwise, 2: push blue point clockwise, 3: push blue point counterclockwise
        std::cout << "\033[1;34mPush point: " << push_point << "\nDistance: " << push_distance*1000 << " mm" << "\033[0m" << std::endl;

        gCamimage.copyTo(Src);
        cv::Mat inv_warp_imager;
        cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
        cv::resize(warped_image, inv_warp_imager, warped_image_size);
        cv::warpPerspective(inv_warp_imager, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
        cv::resize(warped_image, warped_image, warped_image_resize);
        warped_image = background - warped_image;
        warped_image = ~warped_image;
        warped_image.copyTo(drawing);

        bResetRobot = false;
        cv::Point red_point, blue_point;
        float rod_angle;
        std::vector<float> start_point(2), end_point(2);
        if(use_clock){
            if(push_point == 0 || push_point == 1){
                Find_Rod_Angle("red", red_point, rod_angle);
                if(push_point == 0){
                    if(rod_angle < PI*0.5 || rod_angle > PI*1.5){
                        Push_Down(red_point, rod_angle, push_distance, start_point, end_point);
                    } else {
                        Push_Up(red_point, rod_angle, push_distance, start_point, end_point);
                    }
                } else {
                    if(rod_angle < PI*0.5 || rod_angle > PI*1.5){
                        Push_Up(red_point, rod_angle, push_distance, start_point, end_point);
                    } else {
                        Push_Down(red_point, rod_angle, push_distance, start_point, end_point);
                    }
                }
            } else {
                Find_Rod_Angle("blue", blue_point, rod_angle);
                if(push_point == 2){
                    if(rod_angle < PI*0.5 || rod_angle > PI*1.5){
                        Push_Down(blue_point, rod_angle, push_distance, start_point, end_point);
                    } else {
                        Push_Up(blue_point, rod_angle, push_distance, start_point, end_point);
                    }
                } else {
                    if(rod_angle < PI*0.5 || rod_angle > PI*1.5){
                        Push_Up(blue_point, rod_angle, push_distance, start_point, end_point);
                    } else {
                        Push_Down(blue_point, rod_angle, push_distance, start_point, end_point);
                    }
                }
            }
        } else {
            if(push_point == 0 || push_point == 1){
                Find_Rod_Angle("red", red_point, rod_angle);
                if(push_point == 0){
                    Push_Down(red_point, rod_angle, push_distance, start_point, end_point);
                } else {
                    Push_Up(red_point, rod_angle, push_distance, start_point, end_point);
                }
            } else {
                Find_Rod_Angle("blue", blue_point, rod_angle);
                if(push_point == 2){
                    Push_Down(blue_point, rod_angle, push_distance, start_point, end_point);
                } else {
                    Push_Up(blue_point, rod_angle, push_distance, start_point, end_point);
                }
            }
        }
        // Write the unfold plan file
        QString filename = "/home/cpii/projects/scripts/push_rod.sh";
        QFile file(filename);


        float Rz = 2*PI-(rod_angle+0.25*PI);
        if(Rz < 0){
            Rz += 2*PI;
        }
        if(Rz > PI){
            Rz -= PI;
        }
        if(Rz > 0.5*PI){
            Rz -= PI;
        }

//        std::cout << "start_point:" <<start_point << std::endl
//                  << "end_point:" <<end_point << std::endl
//                  << "Rz:" <<Rz << std::endl;

        if (file.open(QIODevice::ReadWrite)) {
           file.setPermissions(QFileDevice::Permissions(1909));
           QTextStream stream(&file);
           stream << "#!/bin/bash" << "\n"
                  << "\n"
                  << "cd" << "\n"
                  << "\n"
                  << "source /opt/ros/foxy/setup.bash" << "\n"
                  << "\n"
                  << "source ~/ws_moveit2/install/setup.bash" << "\n"
                  << "\n"
                  << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                  << "\n"
                  << "cd tm_robot_gripper/" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << start_point[0] <<", " << start_point[1] <<", " << mTableHeight+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 2" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << start_point[0] <<", " << start_point[1] <<", " << gripper_length+0.011 <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 4, positions: [" << end_point[0] <<", " << end_point[1] <<", " << gripper_length+0.011 <<", -3.14, 0, "<< Rz <<"], velocity: 0.3, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << end_point[0] <<", " << end_point[1] <<", " << mTableHeight+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 1, positions: [-1.025, -0.29, 1.952, -0.1, 1.57, 0.545], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  //<< "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 2" << "\n"
                  << "\n";
        } else {
           qDebug("file open error");
        }
        file.close();

        QProcess push_rod;
        QStringList push_rodarg;

        push_rodarg << "/home/cpii/projects/scripts/push_rod.sh";

        push_rod.start("xterm", push_rodarg);
        constexpr int timeout_count = 60000; //60000 mseconds
        if ( push_rod.waitForFinished(timeout_count)) {
            qDebug() << QString("\033[1;36m[%1] Robot action finished\033[0m\n")
                        .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                        .toUtf8().data();
        } else {
            qWarning() << QString("\033[1;31mRobot action not finished within %1s\033[0m").arg(timeout_count*0.001);
            qWarning() << push_rod.errorString();
            push_rod.kill();
            push_rod.waitForFinished();
            bResetRobot = true;
        }
        if(bResetRobot){
            QMetaObject::invokeMethod(this, "resetRViz",
                                      Qt::BlockingQueuedConnection);
            Sleeper::sleep(3);
            QProcess reset;
            QStringList resetarg;
            resetarg << "/home/cpii/projects/scripts/reset.sh";
            reset.start("xterm", resetarg);
            Sleeper::sleep(3);
            qDebug() << "\033[1;31mResetting Rviz.\033[0m";
        }
        Sleeper::sleep(2);
    }
}

void LP_Plugin_Garment_Manipulation::Push_Square(int push_point, float push_distance, bool exceeded_limit){
    bool bResetRobot = true;
    while(bResetRobot){
        // push_point = 0: push marker 0 towards center, 1: push marker 17 clockwise, 2: push marker 20 counterclockwise
        std::cout << "\033[1;34mPush point: " << push_point << "\nDistance: " << push_distance*1000 << " mm" << "\033[0m" << std::endl;

        gCamimage.copyTo(Src);
        cv::Mat inv_warp_imager;
        cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
        cv::resize(warped_image, inv_warp_imager, warped_image_size);
        cv::warpPerspective(inv_warp_imager, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
        cv::resize(warped_image, warped_image, warped_image_resize);
        warped_image = background - warped_image;
        warped_image = ~warped_image;
        warped_image.copyTo(drawing);

        bResetRobot = false;
        cv::Point marker0, marker17, marker20;
        float rod_angle;
        std::vector<float> start_point(2), end_point(2);

        if(push_point == 0){
            Find_Rod_Angle("marker0", marker0, rod_angle);
            Push_to_Center(marker0, rod_angle, push_distance, start_point, end_point);
        } else if(push_point == 1) {
            Find_Rod_Angle("marker17", marker17, rod_angle);
            if(rod_angle < PI*0.5 || rod_angle > PI*1.5){
                Push_Down(marker17, rod_angle, push_distance, start_point, end_point);
            } else {
                Push_Up(marker17, rod_angle, push_distance, start_point, end_point);
            }
        } else if(push_point == 2){
            Find_Rod_Angle("marker20", marker20, rod_angle);
            if(rod_angle < PI*0.5 || rod_angle > PI*1.5){
                Push_Up(marker20, rod_angle, push_distance, start_point, end_point);
            } else {
                Push_Down(marker20, rod_angle, push_distance, start_point, end_point);
            }
        }

        float Rz = 2*PI-(rod_angle+0.25*PI);
        if(Rz < 0){
            Rz += 2*PI;
        }
        if(Rz > PI){
            Rz -= PI;
        }
        if(Rz > 0.5*PI){
            Rz -= PI;
        }
        if(push_point==0){
            if(Rz > 0.0){
                Rz -= 0.5*PI;
            } else {
                Rz += 0.5*PI;
            }
        }

        if(exceeded_limit){
            break;
        }

//        std::cout << "start_point:" <<start_point << std::endl
//                  << "end_point:" <<end_point << std::endl
//                  << "Rz:" <<Rz << std::endl;

        // Write the unfold plan file
        QString filename = "/home/cpii/projects/scripts/push_rod.sh";
        QFile file(filename);

        if (file.open(QIODevice::ReadWrite)) {
           file.setPermissions(QFileDevice::Permissions(1909));
           QTextStream stream(&file);
           stream << "#!/bin/bash" << "\n"
                  << "\n"
                  << "cd" << "\n"
                  << "\n"
                  << "source /opt/ros/foxy/setup.bash" << "\n"
                  << "\n"
                  << "source ~/ws_moveit2/install/setup.bash" << "\n"
                  << "\n"
                  << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                  << "\n"
                  << "cd tm_robot_gripper/" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << start_point[0] <<", " << start_point[1] <<", " << mTableHeight+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 2" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << start_point[0] <<", " << start_point[1] <<", " << gripper_length+0.011 <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 4, positions: [" << end_point[0] <<", " << end_point[1] <<", " << gripper_length+0.011 <<", -3.14, 0, "<< Rz <<"], velocity: 0.3, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << end_point[0] <<", " << end_point[1] <<", " << mTableHeight+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 1, positions: [-1.025, -0.29, 1.952, -0.1, 1.57, 0.545], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  //<< "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 2" << "\n"
                  << "\n";
        } else {
           qDebug("file open error");
        }
        file.close();

        QProcess push_rod;
        QStringList push_rodarg;

        push_rodarg << "/home/cpii/projects/scripts/push_rod.sh";

        push_rod.start("xterm", push_rodarg);
        constexpr int timeout_count = 60000; //60000 mseconds
        if ( push_rod.waitForFinished(timeout_count)) {
            qDebug() << QString("\033[1;36m[%1] Robot action finished\033[0m\n")
                        .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                        .toUtf8().data();
        } else {
            qWarning() << QString("\033[1;31mRobot action not finished within %1s\033[0m").arg(timeout_count*0.001);
            qWarning() << push_rod.errorString();
            push_rod.kill();
            push_rod.waitForFinished();
            bResetRobot = true;
        }
        if(bResetRobot){
            QMetaObject::invokeMethod(this, "resetRViz",
                                      Qt::BlockingQueuedConnection);
            Sleeper::sleep(3);
            QProcess reset;
            QStringList resetarg;
            resetarg << "/home/cpii/projects/scripts/reset.sh";
            reset.start("xterm", resetarg);
            Sleeper::sleep(3);
            qDebug() << "\033[1;31mResetting Rviz.\033[0m";
        }
        Sleeper::sleep(2);
    }
}

void LP_Plugin_Garment_Manipulation::Robot_Plan(int, void* )
{
    if(!gFoundBackground){
        // Find the table
        midpoints[0] = 0.5 * (roi_corners[0] + roi_corners[1]);
        midpoints[1] = 0.5 * (roi_corners[1] + roi_corners[2]);
        midpoints[2] = 0.5 * (roi_corners[2] + roi_corners[3]);
        midpoints[3] = 0.5 * (roi_corners[3] + roi_corners[0]);
        dst_corners[0].x = 0;
        dst_corners[0].y = 0;
        dst_corners[1].x = (float)norm(midpoints[1] - midpoints[3]);
        dst_corners[1].y = 0;
        dst_corners[2].x = dst_corners[1].x;
        dst_corners[2].y = (float)norm(midpoints[0] - midpoints[2]);
        dst_corners[3].x = 0;
        dst_corners[3].y = dst_corners[2].y;
        warped_image_size = cv::Size(cvRound(dst_corners[2].x), cvRound(dst_corners[2].y));
        WarpMatrix = cv::getPerspectiveTransform(roi_corners, dst_corners);
    }

    cv::Mat inv_warp_image, OriginalCoordinates;

    gCamimage.copyTo(Src);

    cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation

    cv::resize(warped_image, inv_warp_image, warped_image_size);
    cv::warpPerspective(inv_warp_image, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation

    cv::resize(warped_image, warped_image, warped_image_resize);

    if(!gFoundBackground){
        background = warped_image;
        // Save background
        QString filename_Src = QString(dataPath + "/background.jpg");
        QByteArray filename_Srcc = filename_Src.toLocal8Bit();
        const char *filename_Srccc = filename_Srcc.data();
        cv::imwrite(filename_Srccc, background);
        return;
    }

    warped_image = background - warped_image;
    warped_image = ~warped_image;

    gLock.lockForWrite();
    gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
    gLock.unlock();
    emit glUpdateRequest();


//    qDebug() << "Done gWarpedImage";

    gLock.lockForWrite();
    gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
    gLock.unlock();
    emit glUpdateRequest();



//    qDebug() << "Done gInvWarpImage";

    // Find contours
    cv::Mat src_gray;
    cv::cvtColor( warped_image, src_gray, cv::COLOR_BGR2GRAY );
    cv::blur( src_gray, src_gray, cv::Size(3,3) );

    cv::Mat canny_output;
    cv::Canny( src_gray, canny_output, thresh, thresh*1.2 );
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Point center;
    int size = 0;
    cv::Size sz = src_gray.size();
    imageWidth = sz.width;
    imageHeight = sz.height;
    std::vector<double> grasp(3), release(2);
    cv::Point grasp_point, release_point;
    int close_point = 999;

    cv::findContours( canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
    cv::Mat drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );

    if (contours.size() == 0){
        qDebug() << "No garment detected!";
        return;
    }

//    qDebug() << "Done contours.size() : " << contours.size();


    for( size_t i = 0; i< contours.size(); i++ ){
        cv::Scalar color = cv::Scalar( rng.uniform(0, 180), rng.uniform(50,180), rng.uniform(0,180) );
        cv::drawContours( drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0 );

        for (size_t j = 0; j < contours[i].size(); j++){
//                std::cout << "\n" << i << " " << j << "Points with coordinates: x = " << contours[i][j].x << " y = " << contours[i][j].y;

            if ((static_cast<double>(contours[i][j].x) / imageWidth > 0.6
                    && static_cast<double>(contours[i][j].y) / imageHeight > 0.6 )
                    || (sqrt(pow((static_cast<double>(contours[i][j].x) / imageWidth - 0.83), 2) + pow((static_cast<double>(contours[i][j].y) / imageHeight - 0.83), 2)) > 0.7745)
                    || (static_cast<double>(contours[i][j].x) / imageWidth > 0.90
                    && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)
                    || (static_cast<double>(contours[i][j].x) / imageWidth < 0.10
                    && static_cast<double>(contours[i][j].y) / imageHeight > 0.90)
                    || (static_cast<double>(contours[i][j].x) / imageWidth > 0.480
                    && static_cast<double>(contours[i][j].x) / imageWidth < 0.580
                    && static_cast<double>(contours[i][j].y) / imageHeight < 0.10))
            { // Filter out the robot arm and markers
                    size = size - 1;
            } else {
                center.x += contours[i][j].x;
                center.y += contours[i][j].y;
            }
        }
        size += contours[i].size();
    }

    if (size == 0){
        qDebug() << "No garment detected!";
        return;
    }

    // Calculate the center of the cloth
    center.x = round(center.x/size);
    center.y = round(center.y/size);
    //std::cout << "\n" << "grasp_pointx: " << grasp_point.x << "grasp_pointy: " << grasp_point.y;


    if(!gPlan){
        for( size_t i = 0; i< contours.size(); i++ ){
            for (size_t j = 0; j < contours[i].size(); j++){
                if ((static_cast<double>(contours[i][j].x) / imageWidth > 0.6
                     && static_cast<double>(contours[i][j].y) / imageHeight > 0.6 )
                     || (sqrt(pow((static_cast<double>(contours[i][j].x) / imageWidth - 0.83), 2) + pow((static_cast<double>(contours[i][j].y) / imageHeight - 0.83), 2)) > 0.7745)
                     || (static_cast<double>(contours[i][j].x) / imageWidth > 0.90
                     && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)
                     || (static_cast<double>(contours[i][j].x) / imageWidth < 0.10
                     && static_cast<double>(contours[i][j].y) / imageHeight > 0.90)
                     || (static_cast<double>(contours[i][j].x) / imageWidth > 0.480
                     && static_cast<double>(contours[i][j].x) / imageWidth < 0.580
                     && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)){ // Filter out the robot arm and markers
                } else if (sqrt(pow((static_cast<double>(contours[i][j].x) - center.x), 2) + pow((static_cast<double>(contours[i][j].y) - center.y), 2)) < close_point){
                    close_point = sqrt(pow((static_cast<double>(contours[i][j].x) - center.x), 2) + pow((static_cast<double>(contours[i][j].y) - center.y), 2));
                    grasp_point.x = contours[i][j].x;
                    grasp_point.y = contours[i][j].y;
                }
            }
        }

        // Find the location of grasp point
        cv::Point2f warpedp = cv::Point2f(grasp_point.x/static_cast<float>(imageWidth)*warped_image_size.width, grasp_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
        rs2::frame depth;
        if(use_filter){
            depth = filtered_frame;
        } else {
            depth = frames.get_depth_frame();
        }
        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
        int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);


        offset[0] = 0; offset[1] = 0;
        graspx = int(depth_point[0]);
        graspy = int(depth_point[1]);
        mCalAveragePoint = true;
        avgHeight.clear();
        acount = 0;
        while (acount<30){
            Sleeper::msleep(200);
        }
        mCalAveragePoint = false;
        grasp[0] = grasppt.x();
        grasp[1] = grasppt.y();
        grasp[2] = gripper_length + grasppt.z() - 0.005;
        if(grasp[2] < mTableHeight ){
            qDebug() << QString("\033[1;31m[Warning] Crashing table : (%1, %2, %3).  "
                                "\033[1;33mChange height to : %4\033[0m")
                          .arg(grasp[0],0,'f',3).arg(grasp[1],0,'f',3).arg(grasp[2],0,'f',3)
                          .arg(mTableHeight,0,'f',3).toUtf8().data();
            grasp[2] = mTableHeight;
        }
        //    qDebug() << "tx: " << Pt.x() << "ty: "<< Pt.y() << "tz: "<< Pt.z();
        //    qDebug() << "rx: "<< grasp[0] << "ry: "<< grasp[1] << "rz: "<< grasp[2];

        if(P >= mPointCloud.size()){
            for(int i=0; i<4; i++){
                roi_corners[i].x = 0;
                roi_corners[i].y = 0;
            }
            for(int i=0; i<3; i++){
                trans[i] = 0;
            }
            frame = 0;
            gStopFindWorkspace = false;

            qDebug() << "Wrong grasp point!";

            return;
        }
    } else {
        int random_number = rand()%200;

        const double invImageW = 1.0 / imageWidth,
                     invImageH = 1.0 / imageHeight;
        constexpr int iBound[2] = { 1000, 9000 };
        double random_T[2] = { 0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1]),    //[0.1 < x < 0.9]
                               0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1])};

        release_point = {random_T[0]*imageWidth,
                         random_T[1]*imageHeight};

        auto releasePt_RGB = warped_image.at<cv::Vec3b>(release_point.y, release_point.x);

//        int c;

        for( size_t i = 0; i< contours.size(); i++ ){
            for (size_t j = 0; j < contours[i].size(); j++){
                if ((static_cast<double>(contours[i][j].x) / imageWidth > 0.6
                     && static_cast<double>(contours[i][j].y) / imageHeight > 0.6 )
                     || (sqrt(pow((static_cast<double>(contours[i][j].x) / imageWidth - 0.83), 2) + pow((static_cast<double>(contours[i][j].y) / imageHeight - 0.83), 2)) > 0.7745)
                     || (static_cast<double>(contours[i][j].x) / imageWidth > 0.90
                     && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)
                     || (static_cast<double>(contours[i][j].x) / imageWidth < 0.10
                     && static_cast<double>(contours[i][j].y) / imageHeight > 0.90)
                     || (static_cast<double>(contours[i][j].x) / imageWidth > 0.480
                     && static_cast<double>(contours[i][j].x) / imageWidth < 0.580
                     && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)){
                    // Filter out the robot arm and markers
                } else if (abs(sqrt(pow((static_cast<double>(contours[i][j].x) - center.x), 2) + pow((static_cast<double>(contours[i][j].y) - center.y), 2)) - random_number) < close_point){
                     close_point = abs(sqrt(pow((static_cast<double>(contours[i][j].x) - center.x), 2) + pow((static_cast<double>(contours[i][j].y) - center.y), 2)) - random_number);
                     grasp_point.x = contours[i][j].x;
                     grasp_point.y = contours[i][j].y;
                }
            }
        }
        while((random_T[0] > 0.6 && random_T[1] > 0.6 )                                         //At the robot root region
              || (sqrt(pow(random_T[0] - 0.83, 2) + pow( random_T[1] - 0.83, 2)) > 0.7745)      //Too far for robot arm (max. allowed : distance=850cm, height=350cm)
              || releasePt_RGB[0] < uThres
              || releasePt_RGB[1] < uThres
              || releasePt_RGB[2] < uThres                                                      //Only white region (empty Table)
              || 20.0 > cv::norm(release_point - grasp_point))                                  //Distance between 2 points less than 20 pixels (searching region 20x20)
        { // Limit the release point
//            c++;
            cv::drawMarker( drawing,
                            release_point,
                            cv::Scalar(0, 205, 205 ),
                            cv::MARKER_TILTED_CROSS,
                            20, 2, cv::LINE_AA);

            qDebug() << QString("\033[0;37mRGB( %1, %2, %3 ), XY(%4, %5)\033[0m")
                        .arg(releasePt_RGB[0])
                        .arg(releasePt_RGB[1])
                        .arg(releasePt_RGB[2])
                        .arg(release_point.x)
                        .arg(release_point.y).toUtf8().data();

            random_T[0] = 0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1]);
            random_T[1] = 0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1]);

            release_point = {random_T[0]*imageWidth,
                             random_T[1]*imageHeight};

            releasePt_RGB = warped_image.at<cv::Vec3b>(release_point.y, release_point.x);


//            qDebug()<<"x: "<< random_T[0]*imageWidth << "y: "<< random_T[1] * imageHeight;

            //To image scale

    //        qDebug() << c;
    //        qDebug() << warped_image.at<cv::Vec3b>(release_point.x, release_point.y)[0]
    //                 << warped_image.at<cv::Vec3b>(release_point.x, release_point.y)[1]
    //                 << warped_image.at<cv::Vec3b>(release_point.x, release_point.y)[2];
        }

        // Find the location of grasp and release point
        cv::Point2f warpedp = cv::Point2f(grasp_point.x/static_cast<float>(imageWidth)*warped_image_size.width, grasp_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
        rs2::frame depth;
        if(use_filter){
            depth = filtered_frame;
        } else {
            depth = frames.get_depth_frame();
        }
        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);

        offset[0] = 0; offset[1] = 0;
        graspx = int(depth_point[0]);
        graspy = int(depth_point[1]);
        mCalAveragePoint = true;
        avgHeight.clear();
        acount = 0;
        while (acount<30){
            Sleeper::msleep(200);
        }
        mCalAveragePoint = false;
        grasp[0] = grasppt.x();
        grasp[1] = grasppt.y();
        grasp[2] = gripper_length + grasppt.z() - 0.005;
        if(grasp[2] < mTableHeight){
            qDebug() << QString("\033[1;31m[Warning] Crashing table : (%1, %2, %3).  "
                                "\033[1;33mChange h to : %4\033[0m")
                          .arg(grasp[0],0,'f',4).arg(grasp[1],0,'f',4).arg(grasp[2],0,'f',4)
                          .arg(mTableHeight).toUtf8().data();
            grasp[2] = mTableHeight;
        }
        //    qDebug() << "tx: " << Pt.x() << "ty: "<< Pt.y() << "tz: "<< Pt.z();
        //    qDebug() << "rx: "<< grasp[0] << "ry: "<< grasp[1] << "rz: "<< grasp[2];

        cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                          release_point.y*invImageH*warped_image_size.height);
        cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
        float depth_pointr[2] = {0},
              color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
        rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
        int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
        cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
        cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
        cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
        float scale = dstMat(0,3);
        QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
        QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
        release[0] = releasept.x();
        release[1] = releasept.y();

        height = 0.01 * QRandomGenerator::global()->bounded(5,20);
        auto distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05)),
             distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
        constexpr double robotDLimit = 0.85;    //Maximum distance the robot can reach
        while ( distance1 > robotDLimit
              || distance2 > robotDLimit){
            qDebug() << QString("\033[1;31m[Warning] Robot can't reach (<%4): d1=%1, d2=%2, h=%3\033[0m")
                        .arg(distance1,0,'f',3).arg(distance2,0,'f',3).arg(height,0,'f',3).arg(robotDLimit,0,'f',3).toUtf8().data();

            gLock.lockForWrite();
            gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
            gLock.unlock();
            emit glUpdateRequest();

            if(   sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit
               || sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){

                if(sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){
                    qDebug() << "Grasp point exceeds limit, distance: " << sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05));
                }
                if(sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){
                    qDebug() << "Release point exceeds limit, distance: " << sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05));
                }

                qDebug() << "Recalculate grasp and release point";
                close_point = 999;
                random_number = rand()%200;

                for( size_t i = 0; i< contours.size(); i++ ){
                    for (size_t j = 0; j < contours[i].size(); j++){
                        if ((static_cast<double>(contours[i][j].x) / imageWidth > 0.6
                             && static_cast<double>(contours[i][j].y) / imageHeight > 0.6 )
                             || (sqrt(pow((static_cast<double>(contours[i][j].x) / imageWidth - 0.83), 2) + pow((static_cast<double>(contours[i][j].y) / imageHeight - 0.83), 2)) > 0.7745)
                             || (static_cast<double>(contours[i][j].x) / imageWidth > 0.90
                             && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)
                             || (static_cast<double>(contours[i][j].x) / imageWidth < 0.10
                             && static_cast<double>(contours[i][j].y) / imageHeight > 0.90)
                             || (static_cast<double>(contours[i][j].x) / imageWidth > 0.480
                             && static_cast<double>(contours[i][j].x) / imageWidth < 0.580
                             && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)){
                            // Filter out the robot arm and markers
                        } else if (abs(sqrt(pow((static_cast<double>(contours[i][j].x) - center.x), 2) + pow((static_cast<double>(contours[i][j].y) - center.y), 2)) - random_number) < close_point){
                             close_point = abs(sqrt(pow((static_cast<double>(contours[i][j].x) - center.x), 2) + pow((static_cast<double>(contours[i][j].y) - center.y), 2)) - random_number);
                             grasp_point.x = contours[i][j].x;
                             grasp_point.y = contours[i][j].y;
                        }
                    }
                }
                cv::drawMarker(drawing,
                               grasp_point,
                               cv::Scalar(0, 200, 220 ),
                               cv::MARKER_TRIANGLE_UP,
                               20, 2, cv::LINE_AA );


                double random_T[2] = { 0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1]),    //[0.1 < x < 0.9]
                                       0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1])};

                release_point = {random_T[0]*imageWidth,
                                 random_T[1]*imageHeight};

                releasePt_RGB = warped_image.at<cv::Vec3b>(release_point.y, release_point.x);


                while((random_T[0] > 0.6 && random_T[1] > 0.6 )                                         //At the robot root region
                      || (sqrt(pow(random_T[0] - 0.83, 2) + pow( random_T[1] - 0.83, 2)) > 0.7745)      //Too far for robot arm (max. allowed : distance=850cm, height=350cm)
                      || releasePt_RGB[0] < uThres
                      || releasePt_RGB[1] < uThres
                      || releasePt_RGB[2] < uThres                                                      //Only white region (empty Table)
                      || 20.0 > cv::norm(release_point - grasp_point))                                  //Distance between 2 points less than 20 pixels (searching region 20x20)
                { // Limit the release point
        //            c++;
                    cv::drawMarker(drawing,
                                   release_point,
                                   cv::Scalar(0, 255, 255 ),
                                   cv::MARKER_TILTED_CROSS,
                                   25, 2, cv::LINE_AA );


                    qDebug() << QString("\033[0;37mRGB( %1, %2, %3 ), XY(%4, %5)\033[0m")
                                .arg(releasePt_RGB[0])
                                .arg(releasePt_RGB[1])
                                .arg(releasePt_RGB[2])
                                .arg(release_point.x)
                                .arg(release_point.y).toUtf8().data();

                    random_T[0] = 0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1]);
                    random_T[1] = 0.0001 * QRandomGenerator::global()->bounded(iBound[0], iBound[1]);

                    release_point = {random_T[0]*imageWidth,
                                     random_T[1]*imageHeight};

                    releasePt_RGB = warped_image.at<cv::Vec3b>(release_point.y, release_point.x);

            //        qDebug()<<"x: "<< random_T[0]*imageWidth << "y: "<< random_T[1] * imageHeight;

                    //To image scale

            //        qDebug() << c;
            //        qDebug() << warped_image.at<cv::Vec3b>(release_point.x, release_point.y)[0]
            //                 << warped_image.at<cv::Vec3b>(release_point.x, release_point.y)[1]
            //                 << warped_image.at<cv::Vec3b>(release_point.x, release_point.y)[2];
                }

                // Find the location of grasp and release point
                cv::Point2f warpedp = cv::Point2f(grasp_point.x/static_cast<float>(imageWidth)*warped_image_size.width, grasp_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
                cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
                offset[0] = 0; offset[1] = 0;
                graspx = int(depth_point[0]);
                graspy = int(depth_point[1]);
                mCalAveragePoint = true;
                avgHeight.clear();
                acount = 0;
                while (acount<30){
                    Sleeper::msleep(200);
                }
                mCalAveragePoint = false;
                grasp[0] = grasppt.x();
                grasp[1] = grasppt.y();
                grasp[2] = gripper_length + grasppt.z() - 0.005;
                if(grasp[2] < mTableHeight){
                    qDebug() << QString("\033[1;31m[Warning] Crashing table : (%1, %2, %3).  "
                                        "\033[1;33mChange h to : %4\033[0m")
                                  .arg(grasp[0],0,'f',4).arg(grasp[1],0,'f',4).arg(grasp[2],0,'f',4)
                                  .arg(mTableHeight).toUtf8().data();
                    grasp[2] = mTableHeight;
                }
                //    qDebug() << "tx: " << Pt.x() << "ty: "<< Pt.y() << "tz: "<< Pt.z();
                //    qDebug() << "rx: "<< grasp[0] << "ry: "<< grasp[1] << "rz: "<< grasp[2];

                cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                  release_point.y*invImageH*warped_image_size.height);
                cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                float depth_pointr[2] = {0},
                      color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                rs2::frame depth;
                if(use_filter){
                    depth = filtered_frame;
                } else {
                    depth = frames.get_depth_frame();
                }
                rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                float scale = dstMat(0,3);
                QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                release[0] = releasept.x();
                release[1] = releasept.y();
            }

            height = 0.01 * QRandomGenerator::global()->bounded(5,20);

            distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
            distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
        }

        cv::circle( drawing,
                    release_point,
                    12,
                    cv::Scalar( 0, 255, 255 ),
                    3,//cv::FILLED,
                    cv::LINE_AA );

        cv::arrowedLine(drawing, grasp_point, release_point, cv::Scalar( 0, 255, 0 ), 3, cv::LINE_AA, 0, 0.25);

//        float h = 0;
//        mTestP.resize(1);
//        for(int i=0; i<warped_image.rows; i++){
//           for(int j=0; j<warped_image.cols; j++){
//               auto PT_RGB = warped_image.at<cv::Vec3b>(i, j);
//               if((i >= (0.6*warped_image.rows) && j >= (0.6*warped_image.cols))
//                       || (PT_RGB[0] >= uThres && PT_RGB[1] >= uThres && PT_RGB[2] >= uThres)){
//                   // do nothing
//               } else {
//                   cv::Point2f warpedp = cv::Point2f(j/static_cast<float>(imageWidth)*warped_image_size.width, i/static_cast<float>(imageHeight)*warped_image_size.height);
//                   cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
//                   float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
//                   rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
//                   int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
//                   cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
//                   cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
//                   cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
//                   float scale = dstMat(0,3);
//                   QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
//                   if(h < Pt.z()){
//                       h = Pt.z();
//                       mTestP[0] = mPointCloud[P];
//                   }
//               }
//           }
//        }
//        qDebug() <<h;
    }

    cv::circle( drawing,
                grasp_point,
                12,
                cv::Scalar( 0, 0, 255 ),
                3,//cv::FILLED,
                cv::LINE_AA );

    gLock.lockForWrite();
    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
    gLock.unlock();
    emit glUpdateRequest();

    cv::Point2f warpedp = cv::Point2f(grasp_point.x/static_cast<float>(imageWidth)*warped_image_size.width, grasp_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
    cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
    float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
    rs2::frame depth;
    if(use_filter){
        depth = filtered_frame;
    } else {
        depth = frames.get_depth_frame();
    }
    rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
    int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
    mGraspP.resize(1);
    mGraspP[0] = mPointCloud[P];

    //findLines(warped_image);

//    float angle_between_rods, red_angle, blue_angle;
//    Find_Angle_Between_Rods(angle_between_rods, red_angle, blue_angle);


    if(!gPlan){
        // Write the plan file
        QString filename = "/home/cpii/projects/scripts/move.sh";
        QFile file(filename);

        if (file.open(QIODevice::ReadWrite)) {
           file.setPermissions(QFileDevice::Permissions(1909));
           QTextStream stream(&file);
           stream << "#!/bin/bash" << "\n"
                  << "\n"
                  << "cd" << "\n"
                  << "\n"
                  << "source /opt/ros/foxy/setup.bash" << "\n"
                  << "\n"
                  << "source ~/ws_moveit2/install/setup.bash" << "\n"
                  << "\n"
                  << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                  << "\n"
                  << "cd tm_robot_gripper/" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
//                  << "\n"
//                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+0.1 <<", -3.14, 0, 0], velocity: 1.5, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 2" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2] <<", -3.14, 0, 0], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
//                  << "\n"
//                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
//                  << "\n"
//                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [-0.5, 0, 0.7, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\""<< "\n"
//                  << "\n"
//                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 1.5, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n";
        } else {
           qDebug("file open error");
        }
        file.close();
    } else {
        // Save data
        // Create Directory

        QString filename_dir = QString(dataPath + "/%1").arg(datanumber);
        QDir dir;
        dir.mkpath(filename_dir);

        // Save points
        auto future_camPt = QtConcurrent::run([&](){
            std::vector<std::string> points;
            for (auto i=0; i<mPointCloud.size(); i++){
                points.push_back(QString("%1 %2 %3").arg(mPointCloud[i].x())
                                                    .arg(mPointCloud[i].y())
                                                    .arg(mPointCloud[i].z()).toStdString());
            }

            QString filename_points = QString(filename_dir + "/before_points.txt");
            QByteArray filename_pointsc = filename_points.toLocal8Bit();
            const char *filename_pointscc = filename_pointsc.data();
            std::ofstream output_file(filename_pointscc);
            std::ostream_iterator<std::string> output_iterator(output_file, "\n");
            std::copy(points.begin(), points.end(), output_iterator);
        });


        // Save table points
        auto future_tablePt = QtConcurrent::run([&](){
            std::vector<std::string> tablepointIDs(warped_image.cols * warped_image.rows);
            rs2::frame depth;
            if(use_filter){
                depth = filtered_frame;
            } else {
                depth = frames.get_depth_frame();
            }

            const auto &&mat_inv = WarpMatrix.inv();
            const auto &&WW = 1.f/static_cast<float>(imageWidth)*warped_image_size.width,
                       &&HH = 1.f/static_cast<float>(imageHeight)*warped_image_size.height;
            const auto *start_ = tablepointIDs.data();

            auto _future = QtConcurrent::map(tablepointIDs, [&](std::string &v){
                const auto id = &v - start_;
                int &&i = id / warped_image.cols;
                int &&j = id % warped_image.cols;
                cv::Point2f warpedp = {i * WW, j * HH};
                cv::Point3f &&homogeneous = mat_inv * warpedp;

                float tmp_depth_point[2] = {0}, tmp_color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                rs2_project_color_pixel_to_depth_pixel(tmp_depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_point);
                int tmpTableP = int(tmp_depth_point[0])*depthh + (depthh-int(tmp_depth_point[1]));
    //            mTestP[i] = mPointCloud[tmpTableP];
                tablepointIDs.at(id) = QString("%1").arg(tmpTableP).toStdString();
            });
            _future.waitForFinished();

            // Save table points
            QString filename_tablepointIDs = QString(filename_dir + "/before_tablepointIDs.txt");
            QByteArray filename_tablepointIDsc = filename_tablepointIDs.toLocal8Bit();
            const char *filename_tablepointIDscc = filename_tablepointIDsc.data();
            std::ofstream output_file2(filename_tablepointIDscc);
            std::ostream_iterator<std::string> output_iterator2(output_file2, "\n");
            std::copy(tablepointIDs.begin(), tablepointIDs.end(), output_iterator2);
        });

        auto future_images = QtConcurrent::run([&](){
            // Save Src
            QString filename_Src = QString(filename_dir + "/before_Src.jpg");
            QByteArray filename_Srcc = filename_Src.toLocal8Bit();
            const char *filename_Srccc = filename_Srcc.data();
            cv::imwrite(filename_Srccc, Src);

            // Save warped image
            QString filename_warped = QString(filename_dir + "/before_warped_image.jpg");
            QByteArray filename_warpedc = filename_warped.toLocal8Bit();
            const char *filename_warpedcc = filename_warpedc.data();
            cv::imwrite(filename_warpedcc, warped_image);
        });

        future_images.waitForFinished();
        future_tablePt.waitForFinished();
        future_camPt.waitForFinished();


        cv::Point2f warpedp = cv::Point2f(release_point.x/static_cast<float>(imageWidth)*warped_image_size.width, release_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
        rs2::frame depth;
        if(use_filter){
            depth = filtered_frame;
        } else {
            depth = frames.get_depth_frame();
        }
        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
        int PP = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
        mReleaseP.resize(1);
        mReleaseP[0] = mPointCloud[PP];

        saved_warped_image = warped_image;

        // Save grasp, release points and height
        QString filename_grp = QString(filename_dir + "/grasp_release_points_and_height.txt");
        QFile filep(filename_grp);
        if(filep.open(QIODevice::ReadWrite)) {
            QTextStream streamp(&filep);
            streamp << grasp_point.x << " " << grasp_point.y << " " << grasp[2] << " " << offset[0] << " " << offset[1] << "\n"
                    << release_point.x << " " << release_point.y << " " << grasp[2]+height;
        }
        filep.close();

        qDebug() << QString("Grasp -> Release: (%1, %2 %3) -> (%4, %5, %6)")
                    .arg(grasp[0],0,'f',3).arg(grasp[1],0,'f',3).arg(grasp[2],0,'f',3)
                    .arg(release[0],0,'f',3).arg(release[1],0,'f',3).arg(grasp[2]+height,0,'f',3)
                    .toUtf8().data();


        // Write the unfold plan file
        QString filename = "/home/cpii/projects/scripts/unfold.sh";
        QFile file(filename);

        if (file.open(QIODevice::ReadWrite)) {
           file.setPermissions(QFileDevice::Permissions(1909));
           QTextStream stream(&file);
           stream << "#!/bin/bash" << "\n"
                  << "\n"
                  << "cd" << "\n"
                  << "\n"
                  << "source /opt/ros/foxy/setup.bash" << "\n"
                  << "\n"
                  << "source ~/ws_moveit2/install/setup.bash" << "\n"
                  << "\n"
                  << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                  << "\n"
                  << "cd tm_robot_gripper/" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
//                  << "\n"
//                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+0.1 <<", -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 2" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2] <<", -3.14, 0, 0], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
//                  << "\n"
//                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+height <<", -3.14, 0, 0], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
//                  << "\n"
//                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 4, positions: [" << release[0] <<", " << release[1] <<", " << grasp[2]+height <<", -3.14, 0, 0], velocity: 0.7, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 2" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\""<< "\n"
//                << "\n"
//                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << release[0] <<", " << release[1] <<", " << grasp[2]+height+0.05 <<", -3.14, 0, 0], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
//                  << "\n"
//                  << "sleep 1" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 3" << "\n"
                  << "\n";
        } else {
           qDebug("file open error");
        }
        file.close();
    }
}

void LP_Plugin_Garment_Manipulation::Generate_Data(){
//    if ( !mDetector ) {
//        mDetector = std::make_shared<Detector>("/home/cpii/darknet-master/yolo_models/yolov3-df2.cfg", "/home/cpii/darknet-master/yolo_models/yolov3-df2_15000.weights");
//    }

    auto future_get_place_point = QtConcurrent::run([this](){
        rs2::frame depth;
        if(use_filter){
            depth = filtered_frame;
        } else {
            depth = frames.get_depth_frame();
        }
        cv::Mat warped_image_copy;
        total_steps = 0;
        while(mGenerateData){
            qDebug() << "---------------------------------------------------------------------------";
            qDebug() << "\033[0;34mTotal steps: " << total_steps+1 << "\033[0m";
            cv::Mat inv_warp_image;
            torch::Tensor before_pick_point_tensor, after_pick_point_tensor, place_point_tensor;
            std::vector<float> pick_point(3), place_point(3), src_tableheight(warped_image.cols * warped_image.rows), after_tableheight(warped_image.cols * warped_image.rows);
            std::vector<double> grasp(3), release(3), grasp_before(3), release_before(3);
            cv::Point grasp_point, release_point, grasp_point_before, release_point_before;
            torch::Tensor src_tensor, before_state, src_height_tensor, after_state, after_height_tensor;
            std::vector<float> stepreward(1);
            float max_height_before = 0, max_height_after = 0, garment_area_before = 0, garment_area_after = 0, conf_before = 0, conf_after = 0;
            stepreward[0] = 0;
            std::vector<int> done(1);
            done[0] = 0;
            cv::Mat before_image, after_image;

            // Preprocess environment
            gCamimage.copyTo(Src);
            cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
            cv::resize(warped_image, inv_warp_image, warped_image_size);
            cv::warpPerspective(inv_warp_image, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
            cv::resize(warped_image, warped_image, warped_image_resize);
            warped_image = background - warped_image;
            warped_image = ~warped_image;

            before_image = warped_image;

            src_tensor = torch::from_blob(warped_image.data, { warped_image.rows, warped_image.cols, warped_image.channels() }, at::kByte);
            src_tensor = src_tensor.permute({ 2, 0, 1 });
            src_tensor = src_tensor.unsqueeze(0).to(torch::kF32)/255;

            // Find contours

            cv::Mat src_gray;
            cv::cvtColor( warped_image, src_gray, cv::COLOR_BGR2GRAY );
            cv::blur( src_gray, src_gray, cv::Size(3,3) );

            cv::Mat canny_output;
            cv::Canny( src_gray, canny_output, thresh, thresh*1.2 );
            std::vector<cv::Vec4i> hierarchy;
            cv::Size sz = src_gray.size();
            imageWidth = sz.width;
            imageHeight = sz.height;

            cv::findContours( canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
            drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );

            mCalAveragePoint = true;
            mCalAvgMat = true;
            avgHeight.clear();
            acount = 0;
            while (acount<30){
                Sleeper::msleep(200);
            }
            mCalAvgMat = false;
            mCalAveragePoint = false;

            float avg_garment_height = 0;
            for(int i=0; i<warped_image.rows; i++){
                for(int j=0; j<warped_image.cols; j++){
                    auto PT_RGB = warped_image.at<cv::Vec3b>(i, j);
                    int id = i*warped_image.cols+j;
                    if((i >= (0.6*(float)warped_image.rows) && j >= (0.6*(float)warped_image.cols))
                            || (PT_RGB[0] >= uThres && PT_RGB[1] >= uThres && PT_RGB[2] >= uThres)){
                        src_tableheight[id] = 0.0;
                    } else {
                        cv::Point2f warpedp = cv::Point2f(j/static_cast<float>(imageWidth)*warped_image_size.width, i/static_cast<float>(imageHeight)*warped_image_size.height);
                        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
                        int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
                        if(avgHeight.at(P) > 0.0 && avgHeight.at(P) < 0.20){
                            //src_tableheight[id] = Pt.z();
                            src_tableheight[id] = (float)avgHeight.at(P);
                            avg_garment_height += src_tableheight[id];
                            garment_area_before+=1;
                            //std::cout<<"H: "<<(float)avgHeight.at(P)<< '\n';
                        } else {
                            src_tableheight[id] = 0.0;
                        }
                    }
                }
            }
            avg_garment_height /= garment_area_before;
            for(int i=0; i<src_tableheight.size(); i++){
                if(src_tableheight[i] > max_height_before && src_tableheight[i]-0.05 < avg_garment_height){
                    max_height_before = src_tableheight[i];
                }
            }
            src_height_tensor = torch::from_blob(src_tableheight.data(), { 1, warped_image.rows, warped_image.cols }, at::kFloat);
            src_height_tensor = src_height_tensor.to(torch::kF32);

            // Check height mat
//            torch::Tensor out_tensor = src_height_tensor*255;
//            cv::Mat cv_mat1(512, 512, CV_32FC1, out_tensor.data_ptr());
//            auto min = out_tensor.min().item().toFloat();
//            auto max = out_tensor.max().item().toFloat();
//            cv_mat1.convertTo(cv_mat1, CV_8U, 255.0/(max-min));
//            cv::cvtColor(cv_mat1, cv_mat1, CV_GRAY2BGR);


//            float angle = 0;

//            cv::Mat rotatedImg;
//            QImage rotatedImgqt;
//            for(int a = 0; a < 36; a++){
//                rotatedImgqt = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888);

//                QMatrix r;

//                r.rotate(angle*10.0);

//                rotatedImgqt = rotatedImgqt.transformed(r);

//                rotatedImg = cv::Mat(rotatedImgqt.height(), rotatedImgqt.width(), CV_8UC3, rotatedImgqt.bits());

//                std::vector<bbox_t> test_result = mDetector->detect(rotatedImg);

//                if(test_result.size()>0){
//                    for(auto i=0; i<test_result.size(); i++){
//                        if(test_result[i].obj_id == 1 && conf_before < test_result[i].prob && test_result[i].prob > 0.5){
//                            conf_before = test_result[i].prob;
//                        }
//                    }
//                }
//                angle+=1;
//            }
            float Rz = 0;
            findgrasp(grasp, grasp_point, Rz, hierarchy);
            auto graspdis = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
            while (graspdis >= robotDLimit){
                findgrasp(grasp, grasp_point, Rz, hierarchy);
                graspdis = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
            }

            pick_point[0] = grasp_point.x/static_cast<float>(imageWidth); pick_point[1] = grasp_point.y/static_cast<float>(imageHeight); pick_point[2] = grasp[2]-gripper_length;
            before_pick_point_tensor = torch::from_blob(pick_point.data(), { 3 }, at::kFloat).clone().detach();
            src_tensor = src_tensor.flatten();
            src_height_tensor = src_height_tensor.flatten();
            before_state = torch::cat({ src_tensor, src_height_tensor});
            before_state = before_state.reshape({4, warped_image_resize.width, warped_image_resize.height});

            qDebug() << "\033[1;32mMax height before action: \033[0m" << max_height_before;
            qDebug() << "\033[1;32mGarment area before action: \033[0m" << garment_area_before;
            qDebug() << "\033[1;32mClasscification confidence level before action: \033[0m" << conf_before;


            warped_image_copy = warped_image;

            cv::circle( drawing,
                        grasp_point,
                        12,
                        cv::Scalar( 0, 0, 255 ),
                        3,//cv::FILLED,
                        cv::LINE_AA );

            cv::circle( warped_image,
                        grasp_point,
                        12,
                        cv::Scalar( 0, 0, 255 ),
                        3,//cv::FILLED,
                        cv::LINE_AA );

            gLock.lockForWrite();
            gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
            gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
            gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
            gLock.unlock();
            emit glUpdateRequest();

            cv::Point2f warpedp = cv::Point2f(grasp_point.x/static_cast<float>(imageWidth)*warped_image_size.width, grasp_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
            cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
            float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
            rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
            int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
            mGraspP.resize(1);
            mGraspP[0] = mPointCloud[P];

            // Action
            if(manual_mode){
                qDebug() << "\033[1;33mClick to select the place point on the table\033[0m";

                mGetPlacePoint = true;

                while(mGetPlacePoint){
                    QThread::msleep(500);
                }

                place_point[0] = place_pointxy.x();
                place_point[1] = place_pointxy.y();
                place_point[2] = float(rand()%15) * 0.01 + 0.05;

                release_point.x = place_point[0]*static_cast<float>(imageWidth); release_point.y = place_point[1]*static_cast<float>(imageHeight);

                cv::Point2f warpedpr = cv::Point2f(release_point.x/static_cast<float>(imageWidth)*warped_image_size.width,
                                                  release_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
                cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                float depth_pointr[2] = {0},
                      color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                rs2::frame depth;
                if(use_filter){
                    depth = filtered_frame;
                } else {
                    depth = frames.get_depth_frame();
                }
                rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                int Pid = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                cv::Point3f Pc = {mPointCloud[Pid].x(), -mPointCloud[Pid].y(), -mPointCloud[Pid].z()};
                cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                float scale = dstMat(0,3);
                QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                release[0] = releasept.x();
                release[1] = releasept.y();
                release[2] = place_point[2]+gripper_length;

                float distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
                float distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(release[2]+0.05)*(release[2]+0.05));
                qDebug() << "distance1: " << distance1 << "distance2: " << distance2;
                qDebug() << "place_point: " << place_point[0] << " " << place_point[1] << " " << place_point[2];
                if(distance1 >= robotDLimit
                   || place_point[2] < 0.05 || place_point[2] > 0.25
                   || (place_point[0] > 0.6 && place_point[1] > 0.6)
                   || place_point[0] < 0.1 || place_point[0] > 0.9
                   || place_point[1] < 0.1 || place_point[1] > 0.9){
                    qDebug() << "Exceeded limit, redo step";
                    qDebug() << "\033[1;31m";
                    if(distance1 >= robotDLimit){
                        qDebug() << "Error: distance1 >= robotDLimit";
                    }
                    if(place_point[2] < 0.05){
                        qDebug() << "Error: place_point[2] < 0.05";
                    }
                    if(place_point[2] > 0.25){
                        qDebug() << "Error: place_point[2] > 0.25";
                    }
                    if(place_point[0] > 0.6 && place_point[1] > 0.6){
                        qDebug() << "Error: place_point[0] > 0.6 && place_point[1] > 0.6";
                    }
                    if(place_point[0] < 0.1){
                        qDebug() << "Error: place_point[0] < 0.1";
                    }
                    if(place_point[0] > 0.9){
                        qDebug() << "Error: place_point[0] > 0.9";
                    }
                    if(place_point[1] < 0.1){
                        qDebug() << "Error: place_point[1] < 0.1";
                    }
                    if(place_point[1] > 0.9){
                        qDebug() << "Error: place_point[1] > 0.9";
                    }
                    qDebug() << "\033[0m";
                    continue;
                } else if (distance2 >= robotDLimit){
                    if(sqrt(release[0]*release[0]+release[1]*release[1]+((0.05+gripper_length)+0.05)*((0.05+gripper_length)+0.05)) >= robotDLimit){
                        qDebug() << "Exceeded limit2, redo step";
                        qDebug() << "Error: distance2 >= robotDLimit";
                        continue;
                    } else {
                        int lim = 15;
                        while(distance2 >= robotDLimit){
                            if(lim > 1){
                                lim -= 1;
                            }
                            place_point[2] = float(rand()%lim) * 0.01 + 0.05;
                            release[2] = place_point[2]+gripper_length;
                            distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(release[2]+0.05)*(release[2]+0.05));
//                            qDebug() << "lim: " << lim;
//                            qDebug() << "place_point2: " << place_point[2];
//                            qDebug() << "distance2: " << distance2;
                        }
                    }
                }
            } else {
                findrelease(release, release_point, grasp_point);

                // Find the location of grasp and release point
                const double invImageW = 1.0 / imageWidth,
                             invImageH = 1.0 / imageHeight;

                cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                  release_point.y*invImageH*warped_image_size.height);
                cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                float depth_pointr[2] = {0},
                      color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                float scale = dstMat(0,3);
                QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                release[0] = releasept.x();
                release[1] = releasept.y();

                height = 0.01 * QRandomGenerator::global()->bounded(5,20);
                auto distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05)),
                     distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                while ( distance1 > robotDLimit
                      || distance2 > robotDLimit){
//                            qDebug() << QString("\033[1;31m[Warning] Robot can't reach (<%4): d1=%1, d2=%2, h=%3\033[0m")
//                                        .arg(distance1,0,'f',3).arg(distance2,0,'f',3).arg(height,0,'f',3).arg(robotDLimit,0,'f',3).toUtf8().data();

                    gLock.lockForWrite();
                    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                    gLock.unlock();
                    emit glUpdateRequest();

                    if(   sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit
                       || sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){

                        findgrasp(grasp, grasp_point, Rz, hierarchy);
                        findrelease(release, release_point, grasp_point);

                        cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                          release_point.y*invImageH*warped_image_size.height);
                        cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                        float depth_pointr[2] = {0},
                              color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                        rs2::frame depth;
                        if(use_filter){
                            depth = filtered_frame;
                        } else {
                            depth = frames.get_depth_frame();
                        }
                        rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                        int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                        cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                        cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                        cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                        float scale = dstMat(0,3);
                        QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                        QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                        release[0] = releasept.x();
                        release[1] = releasept.y();
                    }
                    height = 0.01 * QRandomGenerator::global()->bounded(5,20);

                    distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                    distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                }
                release[2] = grasp[2] + height - 0.005;
                place_point[0] = release_point.x/static_cast<float>(imageWidth); place_point[1] = release_point.y/static_cast<float>(imageHeight); place_point[2] = release[2]-gripper_length;
            }

            place_point_tensor = torch::from_blob(place_point.data(), { 3 }, at::kFloat).clone().detach();

            grasp_before = grasp;
            release_before = release;
            grasp_point_before = grasp_point;
            release_point_before = release_point;

            //qDebug() << "width: " << imageWidth << "height: " << imageHeight;
            qDebug() << "\033[0;32mGrasp: " << grasp_point.x/static_cast<float>(imageWidth)  << " "<< grasp_point.y/static_cast<float>(imageHeight)  << " " << grasp[2]-gripper_length << "\033[0m";
            qDebug() << "\033[0;32mRelease: "<< place_point[0] << " " << place_point[1] << " " << place_point[2] << "\033[0m";

            cv::circle( drawing,
                        grasp_point,
                        12,
                        cv::Scalar( 0, 0, 255 ),
                        3,//cv::FILLED,
                        cv::LINE_AA );

            cv::circle( drawing,
                        release_point,
                        12,
                        cv::Scalar( 0, 255, 255 ),
                        3,//cv::FILLED,
                        cv::LINE_AA );
            cv::arrowedLine(drawing, grasp_point, release_point, cv::Scalar( 0, 255, 0 ), 3, cv::LINE_AA, 0, 0.25);

            cv::circle( warped_image,
                        release_point,
                        12,
                        cv::Scalar( 0, 255, 255 ),
                        3,//cv::FILLED,
                        cv::LINE_AA );
            cv::arrowedLine(warped_image, grasp_point, release_point, cv::Scalar( 0, 255, 0 ), 3, cv::LINE_AA, 0, 0.25);

            gLock.lockForWrite();
            gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
            gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
            gLock.unlock();

            Rz = 0;

            // Pick and place task
            // Write the unfold plan file
            QString filename = "/home/cpii/projects/scripts/unfold.sh";
            QFile file(filename);

            if (file.open(QIODevice::ReadWrite)) {
               file.setPermissions(QFileDevice::Permissions(1909));
               QTextStream stream(&file);
               stream << "#!/bin/bash" << "\n"
                      << "\n"
                      << "cd" << "\n"
                      << "\n"
                      << "source /opt/ros/foxy/setup.bash" << "\n"
                      << "\n"
                      << "source ~/ws_moveit2/install/setup.bash" << "\n"
                      << "\n"
                      << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                      << "\n"
                      << "cd tm_robot_gripper/" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
    //                  << "\n"
    //                  << "sleep 1" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                      << "\n"
                      << "sleep 2" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2] <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                      << "\n"
    //                << "sleep 1" << "\n"
    //                << "\n"
                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
    //                  << "\n"
    //                  << "sleep 1" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
    //                  << "\n"
    //                  << "sleep 1" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 4, positions: [" << release[0] <<", " << release[1] <<", " << release[2] <<", -3.14, 0, "<< Rz <<"], velocity: 0.7, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                      << "\n"
                      << "sleep 2" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\""<< "\n"
    //                << "\n"
    //                << "sleep 1" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << release[0] <<", " << release[1] <<", " << release[2]+0.05 <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
    //                  << "\n"
    //                  << "sleep 1" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                      << "\n"
                      << "sleep 2" << "\n"
                      << "\n";
            } else {
               qDebug("file open error");
            }
            file.close();

            QProcess unfold;
            QStringList unfoldarg;

            unfoldarg << "/home/cpii/projects/scripts/unfold.sh";

            unfold.start("xterm", unfoldarg);

            bool bResetRobot = false;
            constexpr int timeout_count = 60000; //60000 mseconds
            if ( unfold.waitForFinished(timeout_count)) {
                qDebug() << QString("\033[1;36m[%1] Robot action finished\033[0m")
                            .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                            .toUtf8().data();
            } else {
                qWarning() << QString("\033[1;31mRobot action not finished within %1s\033[0m").arg(timeout_count*0.001);
                qWarning() << unfold.errorString();
                unfold.kill();
                unfold.waitForFinished();

                bResetRobot = true;
            }
            if ( bResetRobot ) {
                QMetaObject::invokeMethod(this, "resetRViz",
                                          Qt::BlockingQueuedConnection);

                qDebug() << "\033[1;31mResetting Rviz.\033[0m";

                if (!resetRobotPosition()){
                    mGenerateData = false;
                    qDebug() << "\033[1;31mCan not reset Rviz, end.\033[0m";
                    break;
                }
                QThread::msleep(6000);  //Wait for the robot to reset

                qDebug() << QString("\n-----[ %1 ]-----\n")
                            .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                            .toUtf8().data();
                continue;
            }

            Sleeper::sleep(3);

            // Reward & State

            gCamimage.copyTo(Src);
            cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
            cv::resize(warped_image, inv_warp_image, warped_image_size);
            cv::warpPerspective(inv_warp_image, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
            cv::resize(warped_image, warped_image, warped_image_resize);
            warped_image = background - warped_image;
            warped_image = ~warped_image;
            after_image = warped_image;
            torch::Tensor after_image_tensor = torch::from_blob(warped_image.data, { warped_image.rows, warped_image.cols, warped_image.channels() }, at::kByte);
            after_image_tensor = after_image_tensor.permute({ 2, 0, 1 });
            after_image_tensor = after_image_tensor.unsqueeze(0).to(torch::kF32)/255;

            cv::Mat sub_image;
            sub_image = warped_image_copy - warped_image;
            auto mean = cv::mean(sub_image);
            qDebug() << "Pixel color diff mean: "<< mean[0] << " "<<mean[1]<< " "<< mean[2];
            if(mean[0]<1.0 && mean[1]<1.0 && mean[2]<1.0 && done[0]!=1.0){
                qDebug() << "\033[0;33mNothing Changed(mean<1.0), mean: " << mean[0] << " " << mean[1] << " " << mean[2] << " redo step\033[0m";
                continue;
            }

            mCalAveragePoint = true;
            mCalAvgMat = true;
            avgHeight.clear();
            acount = 0;
            while (acount<30){
                Sleeper::msleep(200);
            }
            mCalAvgMat = false;
            mCalAveragePoint = false;

            avg_garment_height = 0;
            for(int i=0; i<warped_image.rows; i++){
                for(int j=0; j<warped_image.cols; j++){
                    auto PT_RGB = warped_image.at<cv::Vec3b>(i, j);
                    int id = i*warped_image.cols+j;
                    if((i >= (0.6*(float)warped_image.rows) && j >= (0.6*(float)warped_image.cols))
                            || (PT_RGB[0] >= uThres && PT_RGB[1] >= uThres && PT_RGB[2] >= uThres)){
                        after_tableheight[id] = 0.0;
                    } else {
                        cv::Point2f warpedp = cv::Point2f(j/static_cast<float>(imageWidth)*warped_image_size.width, i/static_cast<float>(imageHeight)*warped_image_size.height);
                        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
                        int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
                        if(avgHeight.at(P) > 0.0 && avgHeight.at(P) < 0.20){
                            after_tableheight[id] = (float)avgHeight.at(P);
                            avg_garment_height += after_tableheight[id];
                            garment_area_after+=1;
                        } else {
                            after_tableheight[id] = 0.0;
                        }
                    }
                }
            }
            avg_garment_height /= garment_area_after;
            for(int i=0; i<after_tableheight.size(); i++){
                if(after_tableheight[i] > max_height_after && after_tableheight[i]-0.05 < avg_garment_height){
                    max_height_after = after_tableheight[i];
                }
            }
            qDebug() << "\033[1;32mMax height after action: \033[0m" << max_height_after;
            qDebug() << "\033[1;32mGarment area after action: \033[0m" << garment_area_after;
            after_height_tensor = torch::from_blob(after_tableheight.data(), { 1, warped_image.rows, warped_image.cols }, at::kFloat);
            after_height_tensor = after_height_tensor.to(torch::kF32);

//            float angle2 = 0;
//            cv::Mat rotatedImg2;
//            QImage rotatedImgqt2;
//            for(int a = 0; a < 36; a++){
//                rotatedImgqt2 = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888);

//                QMatrix r;

//                r.rotate(angle2*10.0);

//                rotatedImgqt2 = rotatedImgqt2.transformed(r);

//                rotatedImg2 = cv::Mat(rotatedImgqt2.height(), rotatedImgqt2.width(), CV_8UC3, rotatedImgqt2.bits());

//                std::vector<bbox_t> test_result = mDetector->detect(rotatedImg2);

//                if(test_result.size()>0){
//                    for(auto i =0; i<test_result.size(); i++){
//                        if(test_result[i].obj_id == 1 && conf_after < test_result[i].prob && test_result[i].prob > 0.5){
//                            conf_after = test_result[i].prob;
//                        }
//                    }
//                }
//                angle2+=1;
//            }
//            qDebug() << "\033[1;32mClasscification confidence level after action: \033[0m" << conf_after;

            // Get after state

            cv::Mat src_gray2;
            cv::cvtColor( warped_image, src_gray2, cv::COLOR_BGR2GRAY );
            cv::blur( src_gray2, src_gray2, cv::Size(3,3) );

            cv::Mat canny_output2;
            cv::Canny( src_gray2, canny_output2, thresh, thresh*1.2 );
            std::vector<cv::Vec4i> hierarchy2;
            cv::Size sz2 = src_gray2.size();
            imageWidth = sz2.width;
            imageHeight = sz2.height;

            cv::findContours( canny_output2, contours, hierarchy2, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
            drawing = cv::Mat::zeros( canny_output2.size(), CV_8UC3 );

            findgrasp(grasp, grasp_point, Rz, hierarchy2);
            auto graspdis2 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
            while (graspdis2 >= robotDLimit){
                findgrasp(grasp, grasp_point, Rz, hierarchy2);
                graspdis2 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
            }

            std::vector<std::vector<cv::Point>> squares;
            cv::Mat sqr_img;
            warped_image.copyTo(sqr_img);
            findSquares(sqr_img, squares);

            cv::polylines(sqr_img, squares, true, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);

            qDebug() << "sqr size: " << squares.size();

            cv::arrowedLine(drawing, grasp_point, release_point, cv::Scalar( 0, 255, 0 ), 3, cv::LINE_AA, 0, 0.25);

            gLock.lockForWrite();
            gWarpedImage = QImage((uchar*) sqr_img.data, sqr_img.cols, sqr_img.rows, sqr_img.step, QImage::Format_BGR888).copy();
            gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
            gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
            gLock.unlock();

            emit glUpdateRequest();

            std::vector<float> pick_point2(3);
            pick_point2[0] = grasp_point.x/static_cast<float>(imageWidth); pick_point2[1] = grasp_point.y/static_cast<float>(imageHeight); pick_point2[2] = grasp[2]-gripper_length;
            after_pick_point_tensor = torch::from_blob(pick_point2.data(), { 3 }, at::kFloat).clone().detach();

            //std::cout << "grasp point: " << grasp_point << std::endl << "pick_point2: " << pick_point2 << std::endl << "after_p_tensor: " << after_pick_point_tensor << std::endl;

            after_image_tensor = after_image_tensor.flatten();
            after_height_tensor = after_height_tensor.flatten();
            after_state = torch::cat({ after_image_tensor, after_height_tensor});
            after_state = after_state.reshape({4, warped_image_resize.width, warped_image_resize.height});

            float height_reward, garment_area_reward, conf_reward;
            height_reward = 5000 * (max_height_before - max_height_after);
            float area_diff = garment_area_after/garment_area_before;

            if(area_diff >= 1){
                garment_area_reward = 400 * (area_diff - 1);
            } else {
                garment_area_reward = -400 * (1/area_diff - 1);
            }
            conf_reward = 1000 * (conf_after - conf_before);
            stepreward[0] = height_reward + garment_area_reward + conf_reward;
            qDebug() << "\033[1;31mReward from height: " << height_reward << " Reward from area: " << garment_area_reward << " Reward from classifier: " << conf_reward << "\033[0m";

//            if (max_height_after < 0.02 && conf_after > 0.7) { // Max height when garment unfolded is about 0.015m, conf level is about 0.75
//                done[0] = 1.0;
//                stepreward[0] += 5000;
//                qDebug() << "\033[1;31mGarment is unfolded\033[0m";
//            }

            if (max_height_after < 0.02 && squares.size() > 0) { // Max height when garment unfolded is about 0.015m and sqr size > 0
                done[0] = 1.0;
                stepreward[0] += 5000;
                qDebug() << "\033[1;31mGarment is unfolded\033[0m";
            }

            qDebug() << "\033[1;31mStep reward: " << stepreward[0] << "\033[0m";

            auto reward_tensor = torch::from_blob(stepreward.data(), { 1 }, torch::kFloat);
            //std::cout << "reward_tensor: " << reward_tensor << std::endl;
            auto done_tensor = torch::from_blob(done.data(), { 1 }, torch::kFloat);
            //std::cout << "done_tensor: " << done_tensor << std::endl;

            // Generate only data with reward > 100
//            if(stepreward[0] > -100 && stepreward[0] < 100){
//                qDebug() << "Reward too low, continue";
//                continue;
//            }
/*
            auto future_savedata = QtConcurrent::run([before_image = before_image,
                                                     before_state_CPU = before_state.clone().detach(),
                                                     before_pick_point_CPU = before_pick_point_tensor.clone().detach(),
                                                     place_point_CPU = place_point_tensor.clone().detach(),
                                                     reward_CPU = reward_tensor.clone().detach(),
                                                     done_CPU = done_tensor.clone().detach(),
                                                     after_image = after_image,
                                                     after_state_CPU = after_state.clone().detach(),
                                                     after_pick_point_CPU = after_pick_point_tensor.clone().detach(),
                                                     datasize = total_steps,
                                                     garment_area_before = garment_area_before,
                                                     garment_area_after = garment_area_after,
                                                     max_height_before = max_height_before,
                                                     max_height_after = max_height_after,
                                                     this
                                                     ](){

                qDebug() << "Saving data";

                std::vector<float> before_state_vector(before_state_CPU.data_ptr<float>(), before_state_CPU.data_ptr<float>() + before_state_CPU.numel());
                std::vector<float> before_pick_point_vector(before_pick_point_CPU.data_ptr<float>(), before_pick_point_CPU.data_ptr<float>() + before_pick_point_CPU.numel());
                std::vector<float> place_point_vector(place_point_CPU.data_ptr<float>(), place_point_CPU.data_ptr<float>() + place_point_CPU.numel());
                std::vector<float> reward_vector(reward_CPU.data_ptr<float>(), reward_CPU.data_ptr<float>() + reward_CPU.numel());
                std::vector<float> done_vector(done_CPU.data_ptr<float>(), done_CPU.data_ptr<float>() + done_CPU.numel());
                std::vector<float> after_state_vector(after_state_CPU.data_ptr<float>(), after_state_CPU.data_ptr<float>() + after_state_CPU.numel());
                std::vector<float> after_pick_point_vector(after_pick_point_CPU.data_ptr<float>(), after_pick_point_CPU.data_ptr<float>() + after_pick_point_CPU.numel());
                std::vector<float> garment_area_before_vec{garment_area_before};
                std::vector<float> garment_area_after_vec{garment_area_after};
                std::vector<float> max_height_before_vec{max_height_before};
                std::vector<float> max_height_after_vec{max_height_after};

                QString filename_id = QString(datagen + "/%1").arg(datasize);
                QDir().mkdir(filename_id);

                QString filename_before_image = QString(filename_id + "/before_image.jpg");
                QByteArray filename_before_imageqb = filename_before_image.toLocal8Bit();
                const char *filename_before_imagechar = filename_before_imageqb.data();
                cv::imwrite(filename_before_imagechar, before_image);

                QString filename_before_state = QString(filename_id + "/before_state.txt");
                savedata(filename_before_state, before_state_vector);
                QString filename_before_state_tensor = QString(filename_id + "/before_state_tensor.pt");
                torch::save(before_state_CPU, filename_before_state_tensor.toStdString());

                QString filename_before_pick_point = QString(filename_id + "/before_pick_point.txt");
                savedata(filename_before_pick_point, before_pick_point_vector);
                QString filename_before_pick_point_tensor = QString(filename_id + "/before_pick_point_tensor.pt");
                torch::save(before_pick_point_CPU, filename_before_pick_point_tensor.toStdString());

                QString filename_place_point = QString(filename_id + "/place_point.txt");
                savedata(filename_place_point, place_point_vector);
                QString filename_place_point_tensor = QString(filename_id + "/place_point_tensor.pt");
                torch::save(place_point_CPU, filename_place_point_tensor.toStdString());

                QString filename_reward = QString(filename_id + "/reward.txt");
                savedata(filename_reward, reward_vector);
                QString filename_reward_tensor = QString(filename_id + "/reward_tensor.pt");
                torch::save(reward_CPU, filename_reward_tensor.toStdString());

                QString filename_done = QString(filename_id + "/done.txt");
                savedata(filename_done, done_vector);
                QString filename_done_tensor = QString(filename_id + "/done_tensor.pt");
                torch::save(done_CPU, filename_done_tensor.toStdString());

                QString filename_after_image = QString(filename_id + "/after_image.jpg");
                QByteArray filename_after_imageqb = filename_after_image.toLocal8Bit();
                const char *filename_after_imagechar = filename_after_imageqb.data();
                cv::imwrite(filename_after_imagechar, after_image);

                QString filename_after_state = QString(filename_id + "/after_state.txt");
                savedata(filename_after_state, after_state_vector);
                QString filename_after_state_tensor = QString(filename_id + "/after_state_tensor.pt");
                torch::save(after_state_CPU, filename_after_state_tensor.toStdString());

                QString filename_after_pick_point = QString(filename_id + "/after_pick_point.txt");
                savedata(filename_after_pick_point, after_pick_point_vector);
                QString filename_after_pick_point_tensor = QString(filename_id + "/after_pick_point_tensor.pt");
                torch::save(after_pick_point_CPU, filename_after_pick_point_tensor.toStdString());

                QString filename_garment_area_before = QString(filename_id + "/garment_area_before.txt");
                savedata(filename_garment_area_before, garment_area_before_vec);

                QString filename_garment_area_after = QString(filename_id + "/garment_area_after.txt");
                savedata(filename_garment_area_after, garment_area_after_vec);

                QString filename_max_height_before= QString(filename_id + "/max_height_before.txt");
                savedata(filename_max_height_before, max_height_before_vec);

                QString filename_max_height_after = QString(filename_id + "/max_height_after.txt");
                savedata(filename_max_height_after, max_height_after_vec);
            });
        future_savedata.waitForFinished();
*/
        total_steps++;
        if(done[0] == 1){
            // Reset garment
            gCamimage.copyTo(Src);
            std::vector<double> graspr(3);
            cv::Point grasp_pointr;
            cv::Mat inv_warp_imager;
            cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
            cv::resize(warped_image, inv_warp_imager, warped_image_size);
            cv::warpPerspective(inv_warp_imager, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
            cv::resize(warped_image, warped_image, warped_image_resize);
            warped_image = background - warped_image;
            warped_image = ~warped_image;

            // Find contours
            cv::Mat src_gray;
            cv::cvtColor( warped_image, src_gray, cv::COLOR_BGR2GRAY );
            cv::blur( src_gray, src_gray, cv::Size(3,3) );

            cv::Mat canny_output;
            cv::Canny( src_gray, canny_output, thresh, thresh*1.2 );
            std::vector<cv::Vec4i> hierarchyr;
            cv::Size sz = src_gray.size();
            imageWidth = sz.width;
            imageHeight = sz.height;

            cv::findContours( canny_output, contours, hierarchyr, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
            drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );

            findgrasp(graspr, grasp_pointr, Rz, hierarchyr);

            gLock.lockForWrite();
            gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
            gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
            gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
            gLock.unlock();
            emit glUpdateRequest();

            // Write the plan file
            QString filename = "/home/cpii/projects/scripts/move.sh";
            QFile file(filename);

            if (file.open(QIODevice::ReadWrite)) {
               file.setPermissions(QFileDevice::Permissions(1909));
               QTextStream stream(&file);
               stream << "#!/bin/bash" << "\n"
                      << "\n"
                      << "cd" << "\n"
                      << "\n"
                      << "source /opt/ros/foxy/setup.bash" << "\n"
                      << "\n"
                      << "source ~/ws_moveit2/install/setup.bash" << "\n"
                      << "\n"
                      << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                      << "\n"
                      << "cd tm_robot_gripper/" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
    //                  << "\n"
    //                  << "sleep 1" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << graspr[0] <<", " << graspr[1] <<", " << graspr[2]+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 1.5, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                      << "\n"
                      << "sleep 2" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << graspr[0] <<", " << graspr[1] <<", " << graspr[2] <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                      << "\n"
                      << "sleep 1" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
    //                  << "\n"
    //                  << "sleep 1" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [-0.5, 0, 0.7, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                      << "\n"
                      << "sleep 1" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\""<< "\n"
    //                  << "\n"
    //                  << "sleep 1" << "\n"
                      << "\n"
                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 1.5, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                      << "\n"
                      << "sleep 2" << "\n"
                      << "\n";
            } else {
               qDebug("file open error");
            }
            file.close();

            QProcess plan;
            QStringList planarg;

            planarg << "/home/cpii/projects/scripts/move.sh";
            plan.start("xterm", planarg);

            constexpr int timeout_count = 60000; //60000 mseconds
            if ( plan.waitForFinished(timeout_count)) {
                qDebug() << QString("\033[1;36m[%1] Env reset finished\033[0m")
                            .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                            .toUtf8().data();
            } else {
                qWarning() << QString("\033[1;31mRobot action not finished within %1s\033[0m").arg(timeout_count*0.001);
                qWarning() << plan.errorString();
                plan.kill();
                plan.waitForFinished();
            }
            Sleeper::sleep(3);
        }
        }
    });
}


void LP_Plugin_Garment_Manipulation::Env_reset(float &Rz, bool &bResetRobot, int &datasize, bool &end_training){
    // Reset garment
    gCamimage.copyTo(Src);
    std::vector<double> graspr(3);
    cv::Point grasp_pointr;
    cv::Mat inv_warp_imager;
    cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
    cv::resize(warped_image, inv_warp_imager, warped_image_size);
    cv::warpPerspective(inv_warp_imager, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
    cv::resize(warped_image, warped_image, warped_image_resize);
    warped_image = background - warped_image;
    warped_image = ~warped_image;

    // Find contours
    cv::Mat src_gray;
    cv::cvtColor( warped_image, src_gray, cv::COLOR_BGR2GRAY );
    cv::blur( src_gray, src_gray, cv::Size(3,3) );

    cv::Mat canny_output;
    cv::Canny( src_gray, canny_output, thresh, thresh*1.2 );
    std::vector<cv::Vec4i> hierarchyr;
    cv::Size sz = src_gray.size();
    imageWidth = sz.width;
    imageHeight = sz.height;

    cv::findContours( canny_output, contours, hierarchyr, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
    drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );

    findgrasp(graspr, grasp_pointr, Rz, hierarchyr);

    gLock.lockForWrite();
    gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
    gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
    gLock.unlock();
    emit glUpdateRequest();

    // Write the plan file
    QString filename = "/home/cpii/projects/scripts/move.sh";
    QFile file(filename);

    Rz = 0;

    if (file.open(QIODevice::ReadWrite)) {
       file.setPermissions(QFileDevice::Permissions(1909));
       QTextStream stream(&file);
       stream << "#!/bin/bash" << "\n"
              << "\n"
              << "cd" << "\n"
              << "\n"
              << "source /opt/ros/foxy/setup.bash" << "\n"
              << "\n"
              << "source ~/ws_moveit2/install/setup.bash" << "\n"
              << "\n"
              << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
              << "\n"
              << "cd tm_robot_gripper/" << "\n"
              << "\n"
              << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
//                  << "\n"
//                  << "sleep 1" << "\n"
              << "\n"
              << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << graspr[0] <<", " << graspr[1] <<", " << graspr[2]+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 1.5, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
              << "\n"
              << "sleep 2" << "\n"
              << "\n"
              << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << graspr[0] <<", " << graspr[1] <<", " << graspr[2] <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
              << "\n"
              << "sleep 1" << "\n"
              << "\n"
              << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
//                  << "\n"
//                  << "sleep 1" << "\n"
              << "\n"
              << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [-0.5, 0, 0.7, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
              << "\n"
              << "sleep 1" << "\n"
              << "\n"
              << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\""<< "\n"
//                  << "\n"
//                  << "sleep 1" << "\n"
              << "\n"
              << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 1.5, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
              << "\n"
              << "sleep 2" << "\n"
              << "\n";
    } else {
       qDebug("file open error");
    }
    file.close();

    QProcess plan;
    QStringList planarg;

    planarg << "/home/cpii/projects/scripts/move.sh";
    plan.start("xterm", planarg);

    constexpr int timeout_count = 60000; //60000 mseconds
    if ( plan.waitForFinished(timeout_count)) {
        qDebug() << QString("\033[1;36m[%1] Env reset finished\033[0m")
                    .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                    .toUtf8().data();
    } else {
        qWarning() << QString("\033[1;31mRobot action not finished within %1s\033[0m").arg(timeout_count*0.001);
        qWarning() << plan.errorString();
        plan.kill();
        plan.waitForFinished();

        bResetRobot = true;
    }
    if ( bResetRobot ) {
        QMetaObject::invokeMethod(this, "resetRViz",
                                  Qt::BlockingQueuedConnection);

        qDebug() << "\033[1;31mResetting Rviz.\033[0m";

        QDir dir(QString("%1/%2").arg(memoryPath).arg(datasize--));
        if (!dir.removeRecursively()) {
            qCritical() << "[Warning] Useless data cannot be deleted : " << datasize;
        }
        total_steps--;

        if (!resetRobotPosition()){
            mRunReinforcementLearning1 = false;
            qDebug() << "\033[1;31mCan not reset Rviz, end training.\033[0m";
            end_training = true;
        }
        QThread::msleep(6000);  //Wait for the robot to reset

        qDebug() << QString("\n-----[ %1 ]-----\n")
                    .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                    .toUtf8().data();
        end_training = true;
    }
    Sleeper::sleep(3);
}

void LP_Plugin_Garment_Manipulation::RobotReset_RunCamera(){
    srand((unsigned)time(NULL));

    useless_data = 0;
    datanumber = 0;
    frame = 0;
    markercount = 0;
    mTableHeight = gripper_length + 0.006;
    warped_image_resize = cv::Size(512, 512);
    gStopFindWorkspace = false;
    gPlan = false;
    gQuit = false;
    mCalAveragePoint = false;
    gFoundBackground = false;
    Transformationmatrix_T2R.setToIdentity();
    Transformationmatrix_T2R.rotate(-45.0, QVector3D(0.f, 0.f, 1.0f));
    Transformationmatrix_T2R.translate(0.46777, 0.63396, 0.f);


    bool transolddata = false;
    if(transolddata){
        //auto future = QtConcurrent::run([this](){
            mLabel->setText("Loading old data");
        //        if ( !mDetector ) {
        //            mDetector = std::make_shared<Detector>("/home/cpii/darknet-master/yolo_models/yolov3-df2.cfg", "/home/cpii/darknet-master/yolo_models/yolov3-df2_15000.weights");
        //        }

            qDebug() << "Loading old data";

            QString data = "/home/cpii/storage_d1/robot_garment";

            std::vector<std::string> transed;

            for (const auto & file : std::filesystem::directory_iterator(data.toStdString())){
                int datacount = 0;
//                if(file.path() == "/home/cpii/storage_d1/robot_garment/data_Sep2_1"
//                   || file.path() == "/home/cpii/storage_d1/robot_garment/data_Sep10_2"
//                   || file.path() == "/home/cpii/storage_d1/robot_garment/data_Sep3_3"
//                   || file.path() == "/home/cpii/storage_d1/robot_garment/data_Sep6_4"
//                   || file.path() == "/home/cpii/storage_d1/robot_garment/data_Sep6_1"
//                   || file.path() == "/home/cpii/storage_d1/robot_garment/data_Sep8_1"
//                   || file.path() == "/home/cpii/storage_d1/robot_garment/data_Sep3_2"){
//                    continue;
//                }
                for(const auto & tmpfile : std::filesystem::directory_iterator(file.path())){
                    datacount++;
                }
                QString datapath = QString(QString::fromStdString(file.path()) + "/");
                trans_old_data(datacount - 5, datapath, olddatasavepath);
                transed.push_back(file.path());
                std::cout << "Done folders: " << std::endl << transed << std::endl;
            }
            qDebug() << "Trans old data done";
        //});
        //future.waitForFinished();
        return;
    }

    //Reset RViz
    resetRViz();
    //Start the RViz

    //Reset robot position
    resetRobotPosition();

    //calibrate();
    //return false;

    auto config = rs2::config();
    config.enable_device(cam_num1);

    rs2::pipeline_profile profile = pipe.start(config);
    dev = profile.get_device();

    // Data for camera 105
    cameraMatrix = (cv::Mat_<double>(3, 3) <<
                    6.3613879282253527e+02, 0.0,                    6.2234190978343929e+02,
                    0.0,                    6.3812811500350199e+02, 3.9467355577736072e+02,
                    0.0,                    0.0,                    1.0);

    distCoeffs = (cv::Mat_<double>(1, 5) <<
                  -4.9608290899185239e-02,
                  5.5765107471082952e-02,
                  -4.1332161311619011e-04,
                  -2.9084475830604890e-03,
                  -8.1804097972212695e-03);

    // Data for camera 165
//    cameraMatrix = (cv::Mat_<double>(3, 3) <<
//                    6.3571421284896633e+02, 0.0,                    6.3524956160971124e+02,
//                    0.0,                    6.3750269218122367e+02, 4.0193458977992344e+02,
//                    0.0,                    0.0,                    1.0);

//    distCoeffs = (cv::Mat_<double>(1, 5) <<
//                  -4.8277907059162739e-02,
//                  5.3985456400810893e-02,
//                  -2.2871626654868312e-04,
//                  -6.3558631226346730e-04,
//                  -1.1703243048952400e-02);

    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
    dictionary2 = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);

    gFuture = QtConcurrent::run([this](){

//        int countx = 0;
//        int county = 360;

        // Declare filters
        rs2::decimation_filter dec_filter;  // Decimation - reduces depth frame density
        rs2::threshold_filter thr_filter;   // Threshold  - removes values outside recommended range
        rs2::spatial_filter spat_filter;    // Spatial    - edge-preserving spatial smoothing
        rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise

        // Declare disparity transform from depth to disparity and vice versa
        const std::string disparity_filter_name = "Disparity";
        rs2::disparity_transform depth_to_disparity(true);
        rs2::disparity_transform disparity_to_depth(false);

        // Initialize a vector that holds filters and their options
        std::vector<filter_options> filters;

        // The following order of emplacement will dictate the orders in which filters are applied
        //filters.emplace_back("Decimate", dec_filter);
        filters.emplace_back("Threshold", thr_filter);
        filters.emplace_back(disparity_filter_name, depth_to_disparity);
        filters.emplace_back("Spatial", spat_filter);
        filters.emplace_back("Temporal", temp_filter);

        //int frame_num = 0;

        //std::string prototxt = "/home/cpii/opencv/opencv_contrib-4.5.2/modules/text/samples/textbox.prototxt";
        //std::string weights = "/home/cpii/projects/TextBoxes_icdar13.caffemodel";
        //cv::Ptr<cv::text::TextDetectorCNN> detector = cv::text::TextDetectorCNN::create(prototxt, weights);

        while(!gQuit)
        {
            //qDebug() << frame_num++;
            // Wait for frames and get them as soon as they are ready
            frames = pipe.wait_for_frames(); // Wait for next set of frames from the camera
            rs2::depth_frame depth_frame = frames.get_depth_frame(); //Take the depth frame from the frameset
            if (!depth_frame) // Should not happen but if the pipeline is configured differently
                return;       //  it might not provide depth and we don't want to crash

            rs2::depth_frame filtered = depth_frame; // Does not copy the frame, only adds a reference
            /* Apply filters.
            The implemented flow of the filters pipeline is in the following order:
            1. apply decimation filter
            2. apply threshold filter
            3. transform the scene into disparity domain
            4. apply spatial filter
            5. apply temporal filter
            6. revert the results back (if step Disparity filter was applied
            to depth domain (each post processing block is optional and can be applied independantly).
            */
            bool revert_disparity = false;
            for (auto&& filter : filters)
            {
                if (filter.is_enabled)
                {
                    filtered = filter.filter.process(filtered);
                    if (filter.filter_name == disparity_filter_name)
                    {
                        revert_disparity = true;
                    }
                }
            }
            if (revert_disparity)
            {
                filtered = disparity_to_depth.process(filtered);
            }
            if(use_filter){
                depthw = filtered.get_width();
                depthh = filtered.get_height();
            } else {
                depthw = depth_frame.get_width();
                depthh = depth_frame.get_height();
            }
//            qDebug() << "filtered:" << filtered.get_width() << "x" << filtered.get_height();
//            qDebug() << "depth:" << depth_frame.get_width() << "x" << depth_frame.get_height();

            // Push filtered & original data to their respective queues
            // Note, pushing to two different queues might cause the application to display
            //  original and filtered pointclouds from different depth frames
            //  To make sure they are synchronized you need to push them together or add some
            //  synchronization mechanisms
//            original_data.enqueue(depth_frame);
//            filtered_data.enqueue(filtered);

//            qDebug() << "filtered size:"
//                     << filtered.get_data_size();

            // Our rgb frame
            rs2::frame rgb = frames.get_color_frame();
            pc.map_to(rgb);

            // Try to get new data from the queues and update the view with new texture
            //update_data(original_data, colored_depth, original_points, original_pc, color_map);
            //update_data(filtered_data, filtered_frame, filtered_points, filtered_pc, color_map);

            rs2::frame depth;
            if(use_filter){
                //depth = filtered_frame;
                filtered_frame = filtered;
                depth = filtered_frame;
            } else {
                depth = depth_frame;
            }
//            std::cout << "depth size:\n"
//                      << depth_frame.get_data_size()
//                      << "filter size:\n"
//                      << filtered_frame.get_data_size()
//                      << "\n";

            // Device information
            depth_i = depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            color_i = rgb.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            d2c_e = depth.get_profile().as<rs2::video_stream_profile>().get_extrinsics_to(rgb.get_profile());
            c2d_e = rgb.get_profile().as<rs2::video_stream_profile>().get_extrinsics_to(depth.get_profile());
            rs2::depth_sensor ds = dev.query_sensors().front().as<rs2::depth_sensor>();
            depth_scale = ds.get_depth_scale();
            //std::cout << "depth_scale: " << depth_scale;
        //                float fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy, distC1 = j.coeffs[0], distC2 = j.coeffs[1], distC3 = j.coeffs[2], distC4 = j.coeffs[3], distC5 = j.coeffs[4];
        //                qDebug()<< "fx: "<< fx << "fy: "<< fy << "cx: "<< cx << "cy: "<< cy << "coeffs: "<< distC1 << " "<< distC2 << " "<< distC3 << " "<< distC4 << " "<< distC5;
        //                QMatrix4x4 K = {fx,   0.0f,   cx, 0.0f,
        //                                0.0f,   fy,   cy, 0.0f,
        //                                0.0f, 0.0f, 1.0f, 0.0f,
        //                                0.0f, 0.0f, 0.0f, 0.0f};

            // Generate the pointcloud and texture mappings
            const rs2::vertex* vertices;
            if(use_filter){
                //vertices = filtered_points.get_vertices();
                points = pc.calculate(depth);
                vertices = points.get_vertices();
            } else {
                points = pc.calculate(depth);
                vertices = points.get_vertices();
            }

//            qDebug() << "points size:"
//                     << points.size();

            // Let's convert them to QImage
            auto q_rgb = realsenseFrameToQImage(rgb);

            cv::Mat camimage = cv::Mat(q_rgb.height(),q_rgb.width(), CV_8UC3, q_rgb.bits());
            cv::cvtColor(camimage, camimage, cv::COLOR_BGR2RGB);

//            qDebug()<< "depthw: "<< depthw <<"depthh: " << depthh<< "q_rgbh: "<<q_rgb.height()<<"q_rgbw: "<<q_rgb.width();

            srcw = camimage.cols;
            srch = camimage.rows;

            camimage.copyTo(gCamimage);

//            //////////////////////////////////////
//            cv::Mat img = gCamimage;//cv::imread("/home/cpii/Desktop/digit/img.png", cv::IMREAD_COLOR);

//            std::vector<cv::Rect> box;
//            std::vector<float> conf;

//            detector->detect(img, box, conf);

//            if(!box.empty()){
//                std::vector<int> indexes;
//                cv::dnn::NMSBoxes(box, conf, 0.3f, 0.4f, indexes);
//                textbox_draw(img, box, conf, indexes);

//                //cv::imshow("TextBox Demo", img);
//            } else {
//                qDebug() << "nothing!";
//            }

//            //std::cout << "Done!" << std::endl << std::endl;
//            //std::cout << "Press any key to exit." << std::endl << std::endl;

//            //////////////////////////////////////

            // if at least one marker detected
            if ( !gFoundBackground ) {
                std::vector<int> ids;
                std::vector<std::vector<cv::Point2f>> corners;
                cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
                params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_CONTOUR;
                cv::aruco::detectMarkers(camimage, dictionary, corners, ids, params);

                if (ids.size() > 0) {

                    cv::aruco::drawDetectedMarkers(camimage, corners, ids);
                    cv::aruco::estimatePoseSingleMarkers(corners, 0.0844, cameraMatrix, distCoeffs, rvecs, tvecs);

                    // Get location of the table
                    std::vector<int> detected_markers(3);

                    // draw axis for each marker
                    for(auto i=0; i<ids.size(); i++){
                        cv::aruco::drawAxis(camimage, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);

                        constexpr int nIteration = 400;

                        if(markercount<=nIteration){
                            if(ids[i] == 97){
                                detected_markers[0] = 97;
                                std::vector< cv::Point3f> table_corners_3d, tmpO;
                                std::vector< cv::Point2f> table_corners_2d, tmpP;
                                table_corners_3d.push_back(cv::Point3f(-0.05, 0.95,  0.0));
                                table_corners_3d.push_back(cv::Point3f( 0.95, 0.95,  0.0));
                                table_corners_3d.push_back(cv::Point3f( 0.95,-0.05,  0.0));
                                table_corners_3d.push_back(cv::Point3f(-0.05,-0.05,  0.0));
                                tmpO.push_back(cv::Point3f(0.0, 0.0, 0.0));
                                cv::projectPoints(table_corners_3d, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, table_corners_2d);
                                cv::projectPoints(tmpO, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, tmpP);
                                for(int j=0; j<4; j++){
                                    roi_corners[j].x = roi_corners[j].x + table_corners_2d[j].x;
                                    roi_corners[j].y = roi_corners[j].y + table_corners_2d[j].y;
                                }

                                const auto &_rvec = rvecs[i];
                                if ( 0 == frame ) {
                                    markervecs_97[0] = _rvec[0];
                                    markervecs_97[1] = _rvec[1];
                                    markervecs_97[2] = _rvec[2];
                                } else {
                                    markervecs_97[0] += markervecs_97[0] > 0.0 ^ _rvec[0] > 0.0 ? -_rvec[0] : _rvec[0];
                                    markervecs_97[1] += markervecs_97[1] > 0.0 ^ _rvec[1] > 0.0 ? -_rvec[1] : _rvec[1];
                                    markervecs_97[2] += markervecs_97[2] > 0.0 ^ _rvec[2] > 0.0 ? -_rvec[2] : _rvec[2];
                                }

                                markertrans = std::vector<double>{tmpP[0].x, tmpP[0].y};
                                frame = frame + 1;
                                markercount = markercount + 1;
                            } else if (ids[i] == 98){
                                detected_markers[1] = 98;
                                std::vector< cv::Point3f> table_corners_3d, tmpO;
                                std::vector< cv::Point2f> table_corners_2d, tmpP;
                                table_corners_3d.push_back(cv::Point3f(-0.05, 0.04,  0.0));
                                table_corners_3d.push_back(cv::Point3f( 0.95, 0.04,  0.0));
                                table_corners_3d.push_back(cv::Point3f( 0.95,-0.96,  0.0));
                                table_corners_3d.push_back(cv::Point3f(-0.05,-0.96,  0.0));
                                tmpO.push_back(cv::Point3f(0.0, 0.0, 0.0));
                                cv::projectPoints(table_corners_3d, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, table_corners_2d);
                                cv::projectPoints(tmpO, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, tmpP);
                                for(int j=0; j<4; j++){
                                    roi_corners[j].x = roi_corners[j].x + table_corners_2d[j].x;
                                    roi_corners[j].y = roi_corners[j].y + table_corners_2d[j].y;
                                }
                                markercoordinate_98 = std::vector<double>{tmpP[0].x, tmpP[0].y};
                                markercount = markercount + 1;
                            } else if (ids[i] == 99){
                                detected_markers[2] = 99;
                                std::vector< cv::Point3f> table_corners_3d, tmpO;
                                std::vector< cv::Point2f> table_corners_2d, tmpP;
                                table_corners_3d.push_back(cv::Point3f(-0.95, 0.04, 0.0));
                                table_corners_3d.push_back(cv::Point3f( 0.05, 0.04, 0.0));
                                table_corners_3d.push_back(cv::Point3f( 0.05,-0.96, 0.0));
                                table_corners_3d.push_back(cv::Point3f(-0.95,-0.96, 0.0));
                                tmpO.push_back(cv::Point3f(0.0, 0.0, 0.0));
                                cv::projectPoints(table_corners_3d, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, table_corners_2d);
                                cv::projectPoints(tmpO, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, tmpP);
                                for(int j=0; j<4; j++){
                                    roi_corners[j].x = roi_corners[j].x + table_corners_2d[j].x;
                                    roi_corners[j].y = roi_corners[j].y + table_corners_2d[j].y;
                                }
                                markercoordinate_99 = std::vector<double>{tmpP[0].x, tmpP[0].y};
                                markercount = markercount + 1;
                            } else if (ids[i] == 0){
                                std::vector< cv::Point3f> table_corners_3d;
                                std::vector< cv::Point2f> table_corners_2d;
                                table_corners_3d.push_back(cv::Point3f(-0.53, 0.035, 0.0));
                                table_corners_3d.push_back(cv::Point3f( 0.47, 0.035, 0.0));
                                table_corners_3d.push_back(cv::Point3f( 0.47,-0.965, 0.0));
                                table_corners_3d.push_back(cv::Point3f(-0.53,-0.965, 0.0));
                                cv::projectPoints(table_corners_3d, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, table_corners_2d);
                                for(int j=0; j<4; j++){
                                    roi_corners[j].x = roi_corners[j].x + table_corners_2d[j].x;
                                    roi_corners[j].y = roi_corners[j].y + table_corners_2d[j].y;
                                }
                                markercount = markercount + 1;
                            }
                            if(markercount >= nIteration){
                                qDebug()<< "Alignment Done";
                            }
                        }
                    }
                }
            }


            // Draw PointCloud
            mPointCloud.resize(depthw * depthh);
            mPointCloudTex.resize(depthw * depthh);

            // and texture coordinates
            const rs2::texture_coordinate* tex_coords;
            if(use_filter){
                //tex_coords = filtered_points.get_texture_coordinates();
                tex_coords = points.get_texture_coordinates();
            } else {
                tex_coords = points.get_texture_coordinates();
            }
            if ( mShowInTableCoordinateFrame && gFoundBackground && !gStopFindWorkspace ) {
                const auto &&invM_T2C = Transformationmatrix_T2C.inv();
                const auto start = mPointCloud.data();
                auto future = QtConcurrent::map(mPointCloud, [&](QVector3D &v){
                    auto &&dIndex = &v - start;
                    float data[] = {vertices[dIndex].x, vertices[dIndex].y, vertices[dIndex].z, 1.f};
                    cv::Mat ptMato = cv::Mat(4, 1, CV_32F, data);
                    cv::Mat dstMato(invM_T2C * ptMato);
                    v = QVector3D(dstMato.at<float>(0,0), dstMato.at<float>(0,1), dstMato.at<float>(0,2));
                });
//                std::memcpy(mPointCloud.data(), vertices, depthw * depthh * sizeof(rs2::vertex));
                std::memcpy(mPointCloudTex.data(), tex_coords, depthw * depthh * sizeof(rs2::texture_coordinate));

                future.waitForFinished();
                //////////////////////////////////////////////////////////
//                    for ( int i=0; i<depthw; ++i ){
//                        for ( int j=0; j<depthh; ++j ){
//                            const auto &&dIndex = (depthh-j)*depthw-(depthw-i);
//                            const auto &&index_I = i*depthh + j;
//                            //cv::Point3f Pco = cv::Point3f{mPointCloud[index_I].x(), -mPointCloud[index_I].y(), -mPointCloud[index_I].z()};
//                            float data[] = {vertices[dIndex].x, vertices[dIndex].y, vertices[dIndex].z, 1.f};
//                            cv::Mat ptMato = cv::Mat(4, 1, CV_32F, data);
//                            cv::Mat dstMato(invM_T2C * ptMato);
//                            const float &scaleo = dstMato.at<float>(0,3);

//                            mPointCloud[index_I].setX(dstMato.at<float>(0,0)/scaleo);
//                            mPointCloud[index_I].setY(dstMato.at<float>(0,1)/scaleo);
//                            mPointCloud[index_I].setZ(dstMato.at<float>(0,2)/scaleo);

//                            mPointCloudTex[index_I] = QVector2D(tex_coords[dIndex].u, tex_coords[dIndex].v);
//                        }
//                    }
                //////////////////////////////////////////////////////////
            } else {
                for ( int i=0; i<depthw; ++i ){
                    for ( int j=0; j<depthh; ++j ){
                        const auto &&dIndex = (depthh-j)*depthw-(depthw-i);
                        if (vertices[dIndex].z){
                            const auto &&index_I = i*depthh + j;
                            mPointCloud[index_I] = QVector3D(vertices[dIndex].x, -vertices[dIndex].y, -vertices[dIndex].z);
                            mPointCloudTex[index_I] = QVector2D(tex_coords[dIndex].u, tex_coords[dIndex].v);
                        }
                    }
                }
            }

            if(mFindmarker){
                std::vector<int> ids2;
                std::vector<std::vector<cv::Point2f>> corners2;
                cv::Ptr<cv::aruco::DetectorParameters> params2 = cv::aruco::DetectorParameters::create();
                cv::aruco::detectMarkers(camimage, dictionary2, corners2, ids2, params2);

                if(!ids2.empty()){
                    cv::aruco::drawDetectedMarkers(camimage, corners2, ids2, cv::Scalar(0, 0, 255));
                    for(auto i=0; i<ids2.size(); i++){
                        if(ids2[i]==mTarget_marker){
                            for(int j=0; j<4; j++){
                                mMarkerPosi.x += corners2[i][j].x;
                                mMarkerPosi.y += corners2[i][j].y;
                            }
                            mMarkerPosi.x *= 0.25;
                            mMarkerPosi.y *= 0.25;

                            mFindmarker = false;
                        }
                    }
                }
            }

            if(mCalAveragePoint && acount<30){
                avgHeight.resize( depthh * depthw );

                auto *_start = avgHeight.data();
                auto mat_Inv = Transformationmatrix_T2C.inv();

                auto _future = QtConcurrent::map(avgHeight, [&](double &h ){
                    auto id = &h - _start;
//                    auto i = id / depthh;
//                    auto j = id % depthh;
//                    auto dIndex = (depthh-j)*depthw-(depthw-i);
                    cv::Point3f Pc = cv::Point3f{mPointCloud[id].x(), -mPointCloud[id].y(), -mPointCloud[id].z()};
                    cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                    cv::Mat_<float> dstMat(mat_Inv * ptMat);
                    float scale = dstMat(0,3);
                    QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                    avgHeight.at(id) += Pt.z();
                });
                _future.waitForFinished();

                acount++;

                if(acount>=30){
                    for(int i=0; i< avgHeight.size(); i++){
                        avgHeight.at(i) /= double(acount);
                    }
                    if(!mCalAvgMat && graspx!=0 && graspy!=0){
                        auto pickPtHeight = avgHeight.at(depthh*graspx + depthh-graspy);

                        cv::Point3f Pco = cv::Point3f{mPointCloud[depthh*graspx + depthh-graspy].x(), -mPointCloud[depthh*graspx + depthh-graspy].y(), -mPointCloud[depthh*graspx + depthh-graspy].z()};
        //                qDebug() << "Pco: "<< Pco.x << " "<<Pco.y <<" "<<Pco.z;
                        cv::Mat ptMato = (cv::Mat_<float>(4,1) << Pco.x, Pco.y, Pco.z, 1);
                        cv::Mat_<float> dstMato(mat_Inv * ptMato);
        //                qDebug() << "dstMato: "<< dstMato(0,0)<<" "<<dstMato(0,1)<<" "<< dstMato(0,2)<< " "<< dstMato(0,3);
                        float scaleo = dstMato(0,3);
                        QVector4D Pto(dstMato(0,0)/scaleo, dstMato(0,1)/scaleo, dstMato(0,2)/scaleo, 1.0f);
        //                qDebug() << "Pto: "<< Pto.x() <<" "<<Pto.y()<<" "<<Pto.z();
                        QVector4D Pro = Transformationmatrix_T2R.inverted() * Pto;
                        std::vector<double> tmph = {Pro.x(), Pro.y(), pickPtHeight};
        //                qDebug()<< "Pro: "<< Pro.x()<<" "<<Pro.y()<<" "<< Pro.z();

                        int range = 10;
                        for(int i=0; i<range; i++){
                            for(int j=0; j<range; j++){
                                auto id = depthh*(graspx-(range/2)+i) + depthh-(graspy-(range/2)+j);
                                cv::Point3f Pc = cv::Point3f{mPointCloud[id].x(), -mPointCloud[id].y(), -mPointCloud[id].z()};
                                cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                                cv::Mat_<float> dstMat(mat_Inv * ptMat);
                                float scale = dstMat(0,3);
                                QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                                QVector4D Pr = Transformationmatrix_T2R.inverted() * Pt;
                                if(avgHeight.at(id)>tmph[2]
                                   && sqrt(Pr.x()*Pr.x() - Pro.x()*Pro.x()) <= 0.1
                                   && sqrt(Pr.y()*Pr.y() - Pro.y()*Pro.y()) <= 0.1
                                   && sqrt(avgHeight.at(id)*avgHeight.at(id) - tmph[2]*tmph[2]) <= 0.1){
                                    tmph[0] = Pr.x();
                                    tmph[1] = Pr.y();
                                    tmph[2] = avgHeight.at(id);
                                    offset[0] = -range+i;
                                    offset[1] = -range+j;
                                }
                                    //qDebug()<< "P: "<< id << "H: "<< avgHeight.at(id);
                            }
                        }
                        grasppt = {tmph[0], tmph[1], tmph[2]};
                    }
    //               qDebug() << "gx: " << grasppt[0] << "gy: "<< grasppt[1]<< "gz: " << grasppt[2];
    //                    qDebug() << "x: "<< graspx << "y: "<< graspy;
    //                    qDebug() << "tmph: " << tmph;
                    }
            }

            // And finally we'll emit our signal
            gLock.lockForWrite();
            gCurrentGLFrame = QImage((uchar*) camimage.data, camimage.cols, camimage.rows, camimage.step, QImage::Format_BGR888);
            gLock.unlock();
            emit glUpdateRequest();
        }
    });
}

void LP_Plugin_Garment_Manipulation::Reinforcement_Learning_1(){
    auto rl1current = QtConcurrent::run([this](){
        try {
            torch::manual_seed(0);

//            if ( !mDetector ) {
//                mDetector = std::make_shared<Detector>("/home/cpii/darknet-master/yolo_models/yolov3-df2.cfg", "/home/cpii/darknet-master/yolo_models/yolov3-df2_15000.weights");
//            }

            device = torch::Device(torch::kCPU);
            if (torch::cuda::is_available()) {
                std::cout << "CUDA is available! Training on GPU." << std::endl;
                device = torch::Device(torch::kCUDA);
            }

            torch::autograd::DetectAnomalyGuard detect_anomaly;

            qDebug() << "Creating models";

            std::vector<int> policy_mlp_dims{STATE_DIM, 4096, 1024, 128, 32};
            std::vector<int> critic_mlp_dims{STATE_DIM + ACT_DIM, 4096, 512, 64, 16};

            auto actor_critic = ActorCritic(policy_mlp_dims, critic_mlp_dims);
            auto actor_critic_target = ActorCritic(policy_mlp_dims, critic_mlp_dims);

            qDebug() << "Creating optimizer";

            torch::AutoGradMode copy_disable(false);

            std::vector<torch::Tensor> q_params;
            for(size_t i=0; i<actor_critic->q1->parameters().size(); i++){
                q_params.push_back(actor_critic->q1->parameters()[i]);
            }
            for(size_t i=0; i<actor_critic->q2->parameters().size(); i++){
                q_params.push_back(actor_critic->q2->parameters()[i]);
            }
            torch::AutoGradMode copy_enable(true);

            torch::optim::Adam policy_optimizer(actor_critic->pi->parameters(), torch::optim::AdamOptions(lrp));
            torch::optim::Adam critic_optimizer(q_params, torch::optim::AdamOptions(lrc));

            actor_critic->pi->to(device);
            actor_critic->q1->to(device);
            actor_critic->q2->to(device);
            actor_critic_target->pi->to(device);
            actor_critic_target->q1->to(device);
            actor_critic_target->q2->to(device);

            //-----------------------------------------------------------------------------
//            //CV MAT TO TENSOR
//            gCamimage.copyTo(Src);
//            //cv::resize(Src, Src, cv::Size(4, 4));

//            auto tensor_image = torch::from_blob(Src.data, { Src.rows, Src.cols, Src.channels() }, at::kByte);
//            tensor_image = tensor_image.permute({ 2, 0, 1 });
//            //tensor_image = tensor_image.unsqueeze(0);
//            tensor_image = tensor_image.to(torch::kF32)/255;
//            tensor_image.to(torch::kCPU);

//            //std::cout << "Src: " << std::endl << Src << std::endl;
//            //std::cout << "Src tensor: " << std::endl << tensor_image*255 << std::endl;

//            //TENSOR TO CV MAT
//            torch::Tensor out_tensor = tensor_image*255;
//            out_tensor = out_tensor.permute({1, 2, 0}).to(torch::kF32);
//            cv::Mat cv_mat(Src.rows, Src.cols, CV_32FC3, out_tensor.data_ptr());
//            cv_mat.convertTo(cv_mat, CV_8UC3);

//            //std::cout << "cv_mat: " << std::endl << cv_mat << std::endl;

//            gLock.lockForWrite();
//            gWarpedImage = QImage((uchar*) Src.data, Src.cols, Src.rows, Src.step, QImage::Format_BGR888).copy();
//            //gInvWarpImage = QImage((uchar*) cv_mat.data, cv_mat.cols, cv_mat.rows, cv_mat.step, QImage::Format_BGR888).copy();
//            gEdgeImage = QImage((uchar*) cv_mat.data, cv_mat.cols, cv_mat.rows, cv_mat.step, QImage::Format_BGR888).copy();
//            gLock.unlock();
//            emit glUpdateRequest();

//            return 0;
            //-----------------------------------------------------------------------------

            //-----------------------------------------------------------------------------
            // Test
//            QString filename_before_state_tensor = QString(memoryPath + "/0/before_state_tensor.pt");
//            torch::Tensor before_state_tensor;
//            torch::load(before_state_tensor, filename_before_state_tensor.toStdString());

//            QString filename_before_pick = QString(memoryPath + "/0/before_pick_point_tensor.pt");
//            torch::Tensor before_pick;
//            torch::load(before_pick, filename_before_pick.toStdString());

//            before_state_tensor = before_state_tensor.unsqueeze(0).detach().to(device);
//            before_pick = before_pick.unsqueeze(0).detach().to(device);
//            auto out = actor_critic->pi->forward(before_state_tensor, before_pick, true, false);

//            std::cout << "action: " << out.action << std::endl;

//            //TENSOR TO CV MAT
//            torch::Tensor out_tensor = before_state_tensor*255;
//            auto out_tensor1 = out_tensor.index({3}).to(torch::kF32);
//            cv::Mat cv_mat1(512, 512, CV_32FC1, out_tensor1.data_ptr());
//            auto min = out_tensor1.min().item().toFloat();
//            auto max = out_tensor1.max().item().toFloat();
//            cv_mat1.convertTo(cv_mat1, CV_8U, 255.0/(max-min));
//            cv::cvtColor(cv_mat1, cv_mat1, CV_GRAY2BGR);

//            auto out_tensor2 = out_tensor.index({1}).to(torch::kF32);
//            cv::Mat cv_mat2(512, 512, CV_32FC1, out_tensor2.data_ptr());
//            cv_mat2.convertTo(cv_mat2, CV_8U);
//            cv::cvtColor(cv_mat2, cv_mat2, CV_GRAY2BGR);

//            auto out_tensor3= out_tensor.index({2}).to(torch::kF32);
//            cv::Mat cv_mat3(512, 512, CV_32FC1, out_tensor3.data_ptr());
//            cv_mat3.convertTo(cv_mat3, CV_8U);
//            cv::cvtColor(cv_mat3, cv_mat3, CV_GRAY2BGR);

//            //std::cout << "cv_mat: " << std::endl << cv_mat << std::endl;

//            gLock.lockForWrite();
//            gWarpedImage = QImage((uchar*) cv_mat1.data, cv_mat1.cols, cv_mat1.rows, cv_mat1.step, QImage::Format_BGR888).copy();
//            gInvWarpImage = QImage((uchar*) cv_mat2.data, cv_mat2.cols, cv_mat2.rows, cv_mat2.step, QImage::Format_BGR888).copy();
//            gEdgeImage = QImage((uchar*) cv_mat2.data, cv_mat3.cols, cv_mat3.rows, cv_mat3.step, QImage::Format_BGR888).copy();
//            gLock.unlock();
//            emit glUpdateRequest();

//            return 0;

            //-----------------------------------------------------------------------------

            GOOGLE_PROTOBUF_VERIFY_VERSION;
            TensorBoardLogger logger(kLogFile.c_str());

            int episode = 0;
            int datasize = 0;
            bool LoadOldData = false;
            bool RestoreFromCheckpoint = false;
            if(RestoreFromCheckpoint || LoadOldData){
//                int olddata_size = 0;

                if(LoadOldData){
                    qDebug() << "Load old data";
//                    for (const auto & file : std::filesystem::directory_iterator(memoryPath.toStdString())){
//                        olddata_size++;
//                    }
//                    datasize = olddata_size;
                    datasize = 10000;
                    qDebug() << "Data size: " << datasize;

                } else {
                    qDebug() << "Restore from check point";

                    QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
                    std::vector<float> saved_episode_num;
                    loaddata(filename_episode_num.toStdString(), saved_episode_num);
                    episode = saved_episode_num[0]-1;
                    maxepisode += episode;

                    QString filename_totalsteps = QString(memoryPath + "/totalsteps.txt");
                    std::vector<float> totalsteps;
                    loaddata(filename_totalsteps.toStdString(), totalsteps);
                    total_steps = int(totalsteps[0]);
                    qDebug() << "Total steps: " << total_steps;

                    for (const auto & file : std::filesystem::directory_iterator(memoryPath.toStdString())){
                        datasize++;
                    }
                    datasize -= 2;
                    qDebug() << "Data size: " << datasize;

                    qDebug() << "Loading models";

                    QString pi_para_path = QString(modelPath + "/pi_para/pi_para_" + QString::number(episode) + ".pt");
                    QString q1_para_path = QString(modelPath + "/q1_para/q1_para_" + QString::number(episode) + ".pt");
                    QString q2_para_path = QString(modelPath + "/q2_para/q2_para_" + QString::number(episode) + ".pt");
                    QString target_pi_para_path = QString(modelPath + "/target_pi_para/target_pi_para_" + QString::number(episode) + ".pt");
                    QString target_q1_para_path = QString(modelPath + "/target_q1_para/target_q1_para_" + QString::number(episode) + ".pt");
                    QString target_q2_para_path = QString(modelPath + "/target_q2_para/target_q2_para_" + QString::number(episode) + ".pt");
                    QString policy_opti_path = QString(modelPath + "/policy_optimizer/policy_optimizer_" + QString::number(episode) + ".pt");
                    QString critic_opti_path = QString(modelPath + "/critic_optimizer/critic_optimizer_" + QString::number(episode) + ".pt");

                    std::vector<torch::Tensor> pi_para, q1_para, q2_para, target_pi_para, target_q1_para, target_q2_para;

                    torch::load(pi_para, pi_para_path.toStdString());
                    torch::load(q1_para, q1_para_path.toStdString());
                    torch::load(q2_para, q2_para_path.toStdString());
                    torch::load(target_pi_para, target_pi_para_path.toStdString());
                    torch::load(target_q1_para, target_q1_para_path.toStdString());
                    torch::load(target_q2_para, target_q2_para_path.toStdString());
                    torch::load(policy_optimizer, policy_opti_path.toStdString());
                    torch::load(critic_optimizer, critic_opti_path.toStdString());

                    torch::AutoGradMode data_copy_disable(false);
                    for(size_t i=0; i < actor_critic->pi->parameters().size(); i++){
                        actor_critic->pi->parameters()[i].copy_(pi_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
                        actor_critic->q1->parameters()[i].copy_(q1_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
                        actor_critic->q2->parameters()[i].copy_(q2_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic_target->pi->parameters().size(); i++){
                        actor_critic_target->pi->parameters()[i].copy_(target_pi_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic_target->q1->parameters().size(); i++){
                        actor_critic_target->q1->parameters()[i].copy_(target_q1_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic_target->q2->parameters().size(); i++){
                        actor_critic_target->q2->parameters()[i].copy_(target_q2_para[i].clone().detach().to(device));
                    }
                    torch::AutoGradMode data_copy_enable(true);
                }

    //            qDebug() << "Loading memory";

    //            for(int i=0; i<datasize; i++){
    //                qDebug() << "Memory loading: [" << i+1 << "/" << datasize << "]";
    //                QString filename_id;
    //                if(RestoreFromCheckpoint){
    //                    filename_id = memoryPath + QString("/%1").arg(i);
    //                } else {
    //                    filename_id = olddatasavepath + QString("/%1").arg(i);
    //                }


    //                // Load tensor(.pt)
    //                QString filename_before_state_tensor = QString(filename_id + "/before_state_tensor.pt");
    //                torch::Tensor before_state_tensor;
    //                torch::load(before_state_tensor, filename_before_state_tensor.toStdString());

    //                QString filename_before_pick_point_tensor = QString(filename_id + "/before_pick_point_tensor.pt");
    //                torch::Tensor before_pick_point_tensor;
    //                torch::load(before_pick_point_tensor, filename_before_pick_point_tensor.toStdString());

    //                QString filename_place_point_tensor = QString(filename_id + "/place_point_tensor.pt");
    //                torch::Tensor place_point_tensor;
    //                torch::load(place_point_tensor, filename_place_point_tensor.toStdString());

    //                QString filename_reward_tensor = QString(filename_id + "/reward_tensor.pt");
    //                torch::Tensor reward_tensor;
    //                torch::load(reward_tensor, filename_reward_tensor.toStdString());

    //                QString filename_done_tensor = QString(filename_id + "/done_tensor.pt");
    //                torch::Tensor done_tensor;
    //                torch::load(done_tensor, filename_done_tensor.toStdString());

    //                QString filename_after_state_tensor = QString(filename_id + "/after_state_tensor.pt");
    //                torch::Tensor after_state_tensor;
    //                torch::load(after_state_tensor, filename_after_state_tensor.toStdString());

    //                QString filename_after_pick_point_tensor = QString(filename_id + "/after_pick_point_tensor.pt");
    //                torch::Tensor after_pick_point_tensor;
    //                torch::load(after_pick_point_tensor, filename_after_pick_point_tensor.toStdString());

                    // Load string(.txt)
    //                QString filename_before_state = QString(filename_id + "/before_state.txt");
    //                std::vector<float> before_state_vector;
    //                loaddata(filename_before_state.toStdString(), before_state_vector);
    //                torch::Tensor before_state_tensor = torch::from_blob(before_state_vector.data(), { 262147 }, torch::kFloat);

    //                QString filename_place_point = QString(filename_id + "/place_point.txt");
    //                std::vector<float> place_point_vector;
    //                loaddata(filename_place_point.toStdString(), place_point_vector);
    //                torch::Tensor place_point_tensor = torch::from_blob(place_point_vector.data(), { 3 }, torch::kFloat);

    //                QString filename_reward = QString(filename_id + "/reward.txt");
    //                std::vector<float> reward_vector;
    //                loaddata(filename_reward.toStdString(), reward_vector);
    //                torch::Tensor reward_tensor = torch::from_blob(reward_vector.data(), { 1 }, torch::kFloat);

    //                QString filename_done = QString(filename_id + "/done.txt");
    //                std::vector<float> done_vector;
    //                loaddata(filename_done.toStdString(), done_vector);
    //                torch::Tensor done_tensor = torch::from_blob(done_vector.data(), { 1 }, torch::kFloat);

    //                QString filename_after_state = QString(filename_id + "/after_state.txt");
    //                std::vector<float> after_state_vector;
    //                loaddata(filename_after_state.toStdString(), after_state_vector);
    //                torch::Tensor after_state_tensor = torch::from_blob(after_state_vector.data(), { 262147 }, torch::kFloat);

    //                // Trans to tensor and save
    //                QString sfilename_before_state_tensor = QString(filename_id + "/before_state_tensor.pt");
    //                torch::save(before_state_tensor, sfilename_before_state_tensor.toStdString());

    //                QString sfilename_place_point_tensor = QString(filename_id + "/place_point_tensor.pt");
    //                torch::save(place_point_tensor, sfilename_place_point_tensor.toStdString());

    //                QString sfilename_reward_tensor = QString(filename_id + "/reward_tensor.pt");
    //                torch::save(reward_tensor, sfilename_reward_tensor.toStdString());

    //                QString sfilename_done_tensor = QString(filename_id + "/done_tensor.pt");
    //                torch::save(done_tensor, sfilename_done_tensor.toStdString());

    //                QString sfilename_after_state_tensor = QString(filename_id + "/after_state_tensor.pt");
    //                torch::save(after_state_tensor, sfilename_after_state_tensor.toStdString());

    //                memory.push_back({before_state_tensor.clone().detach(),
    //                                  before_pick_point_tensor.clone().detach(),
    //                                  place_point_tensor.clone().detach(),
    //                                  reward_tensor.clone().detach(),
    //                                  done_tensor.clone().detach(),
    //                                  after_state_tensor.clone().detach(),
    //                                  after_pick_point_tensor.clone().detach()});

    //                std::cout << "before lowest: " << memory[i].before_state.min() << std::endl;
    //                std::cout << "before highest: " << memory[i].before_state.max() << std::endl;
    //                std::cout << "place_point: " << memory[i].place_point << std::endl;
    //                std::cout << "reward: " << memory[i].reward << std::endl;
    //                std::cout << "done: " << memory[i].done << std::endl;
    //                std::cout << "after lowest: " << memory[i].after_state.min() << std::endl;
    //                std::cout << "after highest: " << memory[i].after_state.max() << std::endl;
    //            }
            }

            if(!RestoreFromCheckpoint){
                qDebug() << "Copying parameters to target models";
                torch::AutoGradMode hardcopy_disable(false);
                for(size_t i=0; i < actor_critic_target->pi->parameters().size(); i++){
                    actor_critic_target->pi->parameters()[i].copy_(actor_critic->pi->parameters()[i]);
                    actor_critic_target->pi->parameters()[i].set_requires_grad(false);
                }
                for(size_t i=0; i < actor_critic_target->q1->parameters().size(); i++){
                    actor_critic_target->q1->parameters()[i].copy_(actor_critic->q1->parameters()[i]);
                    actor_critic_target->q1->parameters()[i].set_requires_grad(false);
                }
                for(size_t i=0; i < actor_critic_target->q2->parameters().size(); i++){
                    actor_critic_target->q2->parameters()[i].copy_(actor_critic->q2->parameters()[i]);
                    actor_critic_target->q2->parameters()[i].set_requires_grad(false);
                }
                torch::AutoGradMode hardcopy_enable(true);
            }

            int step = 0, train_number = 0, failtimes;
            float episode_reward = 0, episode_critic1_loss = 0, episode_critic2_loss = 0, episode_policy_loss = 0;
            float Rz = 0;
            std::vector<float> done(1);
            torch::Tensor done_tensor;
            rs2::frame depth;
            if(use_filter){
                depth = filtered_frame;
            } else {
                depth = frames.get_depth_frame();
            }
            cv::Mat warped_image_copy;
            total_reward = 0; total_critic_loss = 0; total_policy_loss = 0;
            bool bResetRobot = false, done_old_data = true, garment_unfolded = false, test_model = false, reset_env = false;
            std::vector<int> unfolded;
            if(LoadOldData){
                done_old_data = false;
            }

            while (episode < maxepisode) {
                qDebug() << "--------------------------------------------";
                if(test_model){
                    qDebug() << "\033[0;34mTest model\033[0m";
                } else {
                    qDebug() << "\033[0;34mEpisode " << episode+1 << " started\033[0m";
                }

                // Initialize environment
                episode_reward = 0;
                episode_critic1_loss = 0;
                episode_critic2_loss = 0;
                episode_policy_loss = 0;
                done[0] = 0;
                step = 0;
                train_number = 0;
                failtimes = 0;
                Rz = 0;
                bResetRobot = false;
                garment_unfolded = false;
                cv::Mat before_image, after_image;

                while (step < maxstep && mRunReinforcementLearning1) {
                    //std::cout << "p fc3: \n" << policy->fc3->parameters() << std::endl;
                    //std::cout << "p fc4: \n" << policy->fc4->parameters() << std::endl;
                    qDebug() << "--------------------------------------------";
                    qDebug() << QString("\n-----[ %1 ]-----\n")
                                .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                .toUtf8().data();
                    if(test_model){
                        qDebug() << "\033[0;34mTesting model : Step [" << step+1 << "/" << teststep << "] started\033[0m";
                    } else {
                        qDebug() << "\033[0;34mEpisode " << episode+1 << ": Step [" << step+1 << "/" << maxstep << "] started\033[0m";
    //                    if(done_old_data){
    //                        qDebug() << "\033[0;34mTotal steps: " << total_steps << "\033[0m";
    //                    } else {
    //                        qDebug() << "\033[0;34mTotal steps / Old data size: [" << total_steps << "/" << datasize << "]\033[0m";
    //                    }

                        qDebug() << "\033[0;34mTotal steps: " << total_steps+1;
                        qDebug() << "Unfolded step: ";
                        if(unfolded.size()<1){
                            qDebug() << "None";
                        } else {
                            for(int i=0; i<unfolded.size(); i++){
                                qDebug() << unfolded[i];
                            }
                        }
                        qDebug() << "\033[0m";
                    }

                    if(step == 0){
                        reset_env = true;
                    }

                    if(done_old_data && reset_env){
                        // Reset garment
/*
                        gCamimage.copyTo(Src);
                        std::vector<double> graspr(3);
                        cv::Point grasp_pointr;
                        cv::Mat inv_warp_imager;
                        cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                        cv::resize(warped_image, inv_warp_imager, warped_image_size);
                        cv::warpPerspective(inv_warp_imager, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
                        cv::resize(warped_image, warped_image, warped_image_resize);
                        warped_image = background - warped_image;
                        warped_image = ~warped_image;

                        // Find contours
                        cv::Mat src_gray;
                        cv::cvtColor( warped_image, src_gray, cv::COLOR_BGR2GRAY );
                        cv::blur( src_gray, src_gray, cv::Size(3,3) );

                        cv::Mat canny_output;
                        cv::Canny( src_gray, canny_output, thresh, thresh*1.2 );
                        std::vector<cv::Vec4i> hierarchyr;
                        cv::Size sz = src_gray.size();
                        imageWidth = sz.width;
                        imageHeight = sz.height;

                        cv::findContours( canny_output, contours, hierarchyr, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
                        drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );

                        findgrasp(graspr, grasp_pointr, Rz, hierarchyr);

                        gLock.lockForWrite();
                        gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
                        gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
                        gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                        gLock.unlock();
                        emit glUpdateRequest();

                        // Write the plan file
                        QString filename = "/home/cpii/projects/scripts/move.sh";
                        QFile file(filename);

                        Rz = 0;

                        if (file.open(QIODevice::ReadWrite)) {
                           file.setPermissions(QFileDevice::Permissions(1909));
                           QTextStream stream(&file);
                           stream << "#!/bin/bash" << "\n"
                                  << "\n"
                                  << "cd" << "\n"
                                  << "\n"
                                  << "source /opt/ros/foxy/setup.bash" << "\n"
                                  << "\n"
                                  << "source ~/ws_moveit2/install/setup.bash" << "\n"
                                  << "\n"
                                  << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                                  << "\n"
                                  << "cd tm_robot_gripper/" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
                //                  << "\n"
                //                  << "sleep 1" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << graspr[0] <<", " << graspr[1] <<", " << graspr[2]+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 1.5, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                  << "\n"
                                  << "sleep 2" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << graspr[0] <<", " << graspr[1] <<", " << graspr[2] <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                  << "\n"
                                  << "sleep 1" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
                //                  << "\n"
                //                  << "sleep 1" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [-0.5, 0, 0.7, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                  << "\n"
                                  << "sleep 1" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\""<< "\n"
                //                  << "\n"
                //                  << "sleep 1" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 1.5, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                  << "\n"
                                  << "sleep 2" << "\n"
                                  << "\n";
                        } else {
                           qDebug("file open error");
                        }
                        file.close();

                        QProcess plan;
                        QStringList planarg;

                        planarg << "/home/cpii/projects/scripts/move.sh";
                        plan.start("xterm", planarg);

                        constexpr int timeout_count = 60000; //60000 mseconds
                        if ( plan.waitForFinished(timeout_count)) {
                            qDebug() << QString("\033[1;36m[%1] Env reset finished\033[0m")
                                        .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                        .toUtf8().data();
                        } else {
                            qWarning() << QString("\033[1;31mRobot action not finished within %1s\033[0m").arg(timeout_count*0.001);
                            qWarning() << plan.errorString();
                            plan.kill();
                            plan.waitForFinished();

                            bResetRobot = true;
                        }
                        if ( bResetRobot ) {
                            QMetaObject::invokeMethod(this, "resetRViz",
                                                      Qt::BlockingQueuedConnection);

                            qDebug() << "\033[1;31mResetting Rviz.\033[0m";

                            QDir dir(QString("%1/%2").arg(memoryPath).arg(datasize--));
                            if (!dir.removeRecursively()) {
                                qCritical() << "[Warning] Useless data cannot be deleted : " << datasize;
                            }
                            total_steps--;

                            if (!resetRobotPosition()){
                                mRunReinforcementLearning1 = false;
                                qDebug() << "\033[1;31mCan not reset Rviz, end training.\033[0m";
                                break;
                            }
                            QThread::msleep(6000);  //Wait for the robot to reset

                            qDebug() << QString("\n-----[ %1 ]-----\n")
                                        .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                        .toUtf8().data();
                            break;
                        }
                        Sleeper::sleep(3);
*/

                        // Reset Environment
                        bool end_training = false;
                        Env_reset(Rz, bResetRobot, datasize, end_training);
                        if(end_training){
                            break;
                        }
                    }

                    if(done_old_data){
                        garment_unfolded = false;
                        bool exceed_limit = false;
                        cv::Mat inv_warp_image;
                        torch::Tensor before_pick_point_tensor, after_pick_point_tensor, place_point_tensor;
                        std::vector<float> pick_point(3), place_point(3), src_tableheight(warped_image.cols * warped_image.rows), after_tableheight(warped_image.cols * warped_image.rows);
                        std::vector<double> grasp(3), release(3), grasp_before(3), release_before(3);
                        cv::Point grasp_point, release_point, grasp_point_before, release_point_before;
                        torch::Tensor src_tensor, before_state, src_height_tensor, after_state, after_height_tensor;
                        std::vector<std::vector<cv::Point>> squares;
                        std::vector<float> stepreward(1);
                        float max_height_before = 0, max_height_after = 0, garment_area_before = 0, garment_area_after = 0, conf_before = 0, conf_after = 0;
                        stepreward[0] = 0;
                        done[0] = 0;
                        Rz = 0;
                        float Rz_before = 0;

                        if(reset_env){
                            // Preprocess environment
                            gCamimage.copyTo(Src);
                            cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                            cv::resize(warped_image, inv_warp_image, warped_image_size);
                            cv::warpPerspective(inv_warp_image, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
                            cv::resize(warped_image, warped_image, warped_image_resize);
                            warped_image = background - warped_image;
                            warped_image = ~warped_image;

                            before_image = warped_image;

                            src_tensor = torch::from_blob(warped_image.data, { warped_image.rows, warped_image.cols, warped_image.channels() }, at::kByte);
                            src_tensor = src_tensor.permute({ 2, 0, 1 });
                            src_tensor = src_tensor.unsqueeze(0).to(torch::kF32)/255;

    //                        src_tensor = torch::randn({3, 256, 256});

                            // Find contours

                            cv::Mat src_gray;
                            cv::cvtColor( warped_image, src_gray, cv::COLOR_BGR2GRAY );
                            cv::blur( src_gray, src_gray, cv::Size(3,3) );

                            cv::Mat canny_output;
                            cv::Canny( src_gray, canny_output, thresh, thresh*1.2 );
                            std::vector<cv::Vec4i> hierarchy;
                            cv::Size sz = src_gray.size();
                            imageWidth = sz.width;
                            imageHeight = sz.height;

                            cv::findContours( canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
                            drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );

                            mCalAveragePoint = true;
                            mCalAvgMat = true;
                            avgHeight.clear();
                            acount = 0;
                            while (acount<30){
                                Sleeper::msleep(200);
                            }
                            mCalAvgMat = false;
                            mCalAveragePoint = false;

                            float avg_garment_height = 0;
                            for(int i=0; i<warped_image.rows; i++){
                                for(int j=0; j<warped_image.cols; j++){
                                    auto PT_RGB = warped_image.at<cv::Vec3b>(i, j);
                                    int id = i*warped_image.cols+j;
                                    if((i >= (0.6*(float)warped_image.rows) && j >= (0.6*(float)warped_image.cols))
                                            || (PT_RGB[0] >= uThres && PT_RGB[1] >= uThres && PT_RGB[2] >= uThres)){
                                        src_tableheight[id] = 0.0;
                                    } else {
                                        cv::Point2f warpedp = cv::Point2f(j/static_cast<float>(imageWidth)*warped_image_size.width, i/static_cast<float>(imageHeight)*warped_image_size.height);
                                        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                                        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                                        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
                                        int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
                                        if(avgHeight.at(P) > 0.0 && avgHeight.at(P) < 0.20){
                                            //src_tableheight[id] = Pt.z();
                                            src_tableheight[id] = (float)avgHeight.at(P);
                                            avg_garment_height += src_tableheight[id];
                                            garment_area_before+=1;
                                            //std::cout<<"H: "<<(float)avgHeight.at(P)<< '\n';
                                        } else {
                                            src_tableheight[id] = 0.0;
                                        }
                                    }
                                }
                            }
                            avg_garment_height /= garment_area_before;
                            for(int i=0; i<src_tableheight.size(); i++){
                                if(src_tableheight[i] > max_height_before && src_tableheight[i]-0.05 < avg_garment_height){
                                    max_height_before = src_tableheight[i];
                                }
                            }

//                            auto maxm = *std::max_element(avgHeight.begin(), avgHeight.end());
//                            auto minm = *std::min_element(avgHeight.begin(), avgHeight.end());
//                            auto max = *std::max_element(src_tableheight.begin(), src_tableheight.end());
//                            auto min = *std::min_element(src_tableheight.begin(), src_tableheight.end());
//                            std::cout<<"Max value: "<<max<< std::endl << "min: " << min << '\n';
//                            std::cout<<"Max value m: "<<maxm<< std::endl << "min m: " << minm << '\n';

                            src_height_tensor = torch::from_blob(src_tableheight.data(), { 1, warped_image.rows, warped_image.cols }, at::kFloat);
                            src_height_tensor = src_height_tensor.to(torch::kF32);

//                            auto min1 = src_height_tensor.min().item().toFloat();
//                            auto max1 = src_height_tensor.max().item().toFloat();
//                            std::cout << "out_tensor1: " << src_height_tensor.sizes() << std::endl << min1 << " " << max1 << std::endl;
//                            src_height_tensor = src_height_tensor.squeeze();
//                            cv::Mat cv_mat1(512, 512, CV_32FC1, src_height_tensor.data_ptr());
//                            cv_mat1.convertTo(cv_mat1, CV_8U, 255.0/(0.005-min1));

//                            gLock.lockForWrite();
//                            gWarpedImage = QImage((uchar*) cv_mat1.data, cv_mat1.cols, cv_mat1.rows, cv_mat1.step, QImage::Format_Grayscale8).copy();
//                            gLock.unlock();
//                            emit glUpdateRequest();

//                            return 0;

    //                        src_height_tensor = torch::randn({256, 256});


//                            float angle = 0;
//                            cv::Mat rotatedImg;
//                            QImage rotatedImgqt;
//                            for(int a = 0; a < 36; a++){
//                                rotatedImgqt = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888);

//                                QMatrix r;

//                                r.rotate(angle*10.0);

//                                rotatedImgqt = rotatedImgqt.transformed(r);

//                                rotatedImg = cv::Mat(rotatedImgqt.height(), rotatedImgqt.width(), CV_8UC3, rotatedImgqt.bits());

//                                std::vector<bbox_t> test_result = mDetector->detect(rotatedImg);

//                                if(test_result.size()>0){
//                                    for(auto i=0; i<test_result.size(); i++){
//                                        if(test_result[i].obj_id == 1 && conf_before < test_result[i].prob && test_result[i].prob > 0.5){
//                                            conf_before = test_result[i].prob;
//                                        }
//                                    }
//                                }
//                                angle+=1;
//                            }

                            findgrasp(grasp, grasp_point, Rz, hierarchy);
                            auto graspdis = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
                            while (graspdis >= robotDLimit){
                                findgrasp(grasp, grasp_point, Rz, hierarchy);
                                graspdis = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
                            }

                            if (total_steps < START_STEP){
                                findrelease(release, release_point, grasp_point);

                                // Find the location of grasp and release point
                                const double invImageW = 1.0 / imageWidth,
                                             invImageH = 1.0 / imageHeight;

                                cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                                  release_point.y*invImageH*warped_image_size.height);
                                cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                                float depth_pointr[2] = {0},
                                      color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                                rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                                int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                                cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                                cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                                cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                                float scale = dstMat(0,3);
                                QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                                QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                                release[0] = releasept.x();
                                release[1] = releasept.y();

                                height = 0.01 * QRandomGenerator::global()->bounded(5,20);
                                auto distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05)),
                                     distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                while ( distance1 > robotDLimit
                                      || distance2 > robotDLimit){
        //                            qDebug() << QString("\033[1;31m[Warning] Robot can't reach (<%4): d1=%1, d2=%2, h=%3\033[0m")
        //                                        .arg(distance1,0,'f',3).arg(distance2,0,'f',3).arg(height,0,'f',3).arg(robotDLimit,0,'f',3).toUtf8().data();

                                    gLock.lockForWrite();
                                    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                                    gLock.unlock();
                                    emit glUpdateRequest();

                                    if(   sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit
                                       || sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){

        //                                if(sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){
        //                                    qDebug() << "Grasp point exceeds limit, distance: " << sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05));
        //                                }
        //                                if(sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){
        //                                    qDebug() << "Release point exceeds limit, distance: " << sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05));
        //                                }
        //                                qDebug() << "Recalculate grasp and release point";

                                        findgrasp(grasp, grasp_point, Rz, hierarchy);
                                        findrelease(release, release_point, grasp_point);

                                        cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                                          release_point.y*invImageH*warped_image_size.height);
                                        cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                                        float depth_pointr[2] = {0},
                                              color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                                        rs2::frame depth;
                                        if(use_filter){
                                            depth = filtered_frame;
                                        } else {
                                            depth = frames.get_depth_frame();
                                        }
                                        rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                                        int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                                        cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                                        cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                                        cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                                        float scale = dstMat(0,3);
                                        QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                                        QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                                        release[0] = releasept.x();
                                        release[1] = releasept.y();
                                    }
                                    height = 0.01 * QRandomGenerator::global()->bounded(5,20);

                                    distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                    distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                }
                                release[2] = grasp[2]+height;

                                cv::arrowedLine(drawing, grasp_point, release_point, cv::Scalar( 0, 255, 0 ), 3, cv::LINE_AA, 0, 0.25);
                            }
                            pick_point[0] = grasp_point.x/static_cast<float>(imageWidth); pick_point[1] = grasp_point.y/static_cast<float>(imageHeight); pick_point[2] = grasp[2]-gripper_length;
                            before_pick_point_tensor = torch::from_blob(pick_point.data(), { 3 }, at::kFloat).clone().detach();
                            src_tensor = src_tensor.flatten();
                            src_height_tensor = src_height_tensor.flatten();
                            before_state = torch::cat({ src_tensor, src_height_tensor});
                            before_state = before_state.reshape({4, warped_image_resize.width, warped_image_resize.height});
    //                        }
    //                        torch::Tensor pick_point_tensor = torch::randn({3});
    //                        auto src_tensor_flatten = torch::flatten(src_tensor);
    //                        auto src_height_tensor_flatten = torch::flatten(src_height_tensor);
    //                        before_state = torch::cat({ src_tensor_flatten, src_height_tensor_flatten, pick_point_tensor });
                        } else {
                            before_image = after_image_last;
                            before_state = after_state_last.clone().detach();
                            before_pick_point_tensor = after_pick_point_last.clone().detach();
                            for(int i=0; i<3; i++){
                                grasp[i] = grasp_last[i];
                                release[i] = release_last[i];
                            }
                            grasp_point = grasp_point_last;
                            release_point = release_point_last;
                            max_height_before = max_height_last;
                            garment_area_before = garment_area_last;
                            conf_before = conf_last;
                        }

                        reset_env = false;

                        qDebug() << "\033[1;32mMax height before action: \033[0m" << max_height_before;
                        qDebug() << "\033[1;32mGarment area before action: \033[0m" << garment_area_before;
                        //qDebug() << "\033[1;32mClasscification confidence level before action: \033[0m" << conf_before;

                        warped_image_copy = warped_image;

                        cv::circle( drawing,
                                    grasp_point,
                                    12,
                                    cv::Scalar( 0, 0, 255 ),
                                    3,//cv::FILLED,
                                    cv::LINE_AA );

                        gLock.lockForWrite();
                        gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
                        gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
                        gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                        gLock.unlock();
                        emit glUpdateRequest();

                        cv::Point2f warpedp = cv::Point2f(grasp_point.x/static_cast<float>(imageWidth)*warped_image_size.width, grasp_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
                        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
                        int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
                        mGraspP.resize(1);
                        mGraspP[0] = mPointCloud[P];

                        // Action
                        torch::AutoGradMode enable(true);
                        actor_critic->pi->train();
                        if (total_steps < START_STEP) {
                            qDebug() << "\033[1;33mStart exploration\033[0m";
                            place_point[0] = release_point.x/static_cast<float>(imageWidth); place_point[1] = release_point.y/static_cast<float>(imageHeight); place_point[2] = release[2]-gripper_length;
                            place_point_tensor = torch::from_blob(place_point.data(), { 3 }, at::kFloat);
    //                        place_point_tensor = torch::randn({3});
                            //std::cout << "place_point_tensor: " << place_point_tensor << std::endl;
                        } else {
                            qDebug() << "\033[1;33mStart exploitation\033[0m";
                            auto state = before_state.clone().detach().to(device);
                            auto p = before_pick_point_tensor.clone().detach().to(device);
                            if(test_model){
                                torch::AutoGradMode disable(false);
                                actor_critic->pi->eval();
                                place_point_tensor = actor_critic->act_pick_point(torch::unsqueeze(state, 0), torch::unsqueeze(p, 0), true);
                            } else {
                                place_point_tensor = actor_critic->act_pick_point(torch::unsqueeze(state, 0), torch::unsqueeze(p, 0), false);
                            }
                            place_point_tensor = place_point_tensor.squeeze().to(torch::kCPU);
                            std::cout << "\033[1;34mAction predict: \n" << place_point_tensor << "\033[0m" << std::endl;
                            place_point = std::vector(place_point_tensor.data_ptr<float>(), place_point_tensor.data_ptr<float>() + place_point_tensor.numel());
                            release_point.x = place_point[0]*static_cast<float>(imageWidth); release_point.y = place_point[1]*static_cast<float>(imageHeight);

                            cv::Point2f warpedpr = cv::Point2f(release_point.x/static_cast<float>(imageWidth)*warped_image_size.width,
                                                              release_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
                            cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                            float depth_pointr[2] = {0},
                                  color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                            rs2::frame depth;
                            if(use_filter){
                                depth = filtered_frame;
                            } else {
                                depth = frames.get_depth_frame();
                            }
                            rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                            int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                            cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                            cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                            cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                            float scale = dstMat(0,3);
                            QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                            QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                            release[0] = releasept.x();
                            release[1] = releasept.y();
                            release[2] = place_point[2]+gripper_length;

                            float distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
                            float distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(release[2]+0.05)*(release[2]+0.05));
                            qDebug() << "distance1: " << distance1 << "distance2: " << distance2;
                            //qDebug() << "place_point: " << place_point[0] << " " << place_point[1] << " " << place_point[2];
                            if(distance1 >= robotDLimit || distance2 >= robotDLimit
                               || place_point[2] < 0.05 || place_point[2] > 0.25
                               || (place_point[0] > 0.6 && place_point[1] > 0.6)
                               || place_point[0] < 0.1 || place_point[0] > 0.9
                               || place_point[1] < 0.1 || place_point[1] > 0.9){
                                qDebug() << "\033[1;31m";
                                if(distance1 >= robotDLimit){
                                    qDebug() << "Error: distance1 >= robotDLimit";
                                }
                                if(distance2 >= robotDLimit){
                                    qDebug() << "Error: distance2 >= robotDLimit";
                                }
                                if(place_point[2] < 0.05){
                                    qDebug() << "Error: place_point[2] < 0.05";
                                }
                                if(place_point[2] > 0.25){
                                    qDebug() << "Error: place_point[2] > 0.25";
                                }
                                if(place_point[0] > 0.6 && place_point[1] > 0.6){
                                    qDebug() << "Error: place_point[0] > 0.6 && place_point[1] > 0.6";
                                }
                                if(place_point[0] < 0.1){
                                    qDebug() << "Error: place_point[0] < 0.1";
                                }
                                if(place_point[0] > 0.9){
                                    qDebug() << "Error: place_point[0] > 0.9";
                                }
                                if(place_point[1] < 0.1){
                                    qDebug() << "Error: place_point[1] < 0.1";
                                }
                                if(place_point[1] > 0.9){
                                    qDebug() << "Error: place_point[1] > 0.9";
                                }
                                qDebug() << "\033[0m";
                                exceed_limit = true;
                            }
                            //qDebug() << "memory size: " << memory.size();
                        }

                        grasp_before = grasp;
                        release_before = release;
                        grasp_point_before = grasp_point;
                        release_point_before = release_point;
                        Rz_before = Rz;

                        //qDebug() << "width: " << imageWidth << "height: " << imageHeight;
                        qDebug() << "\033[0;32mGrasp: " << grasp_point.x/static_cast<float>(imageWidth)  << " "<< grasp_point.y/static_cast<float>(imageHeight)  << " " << grasp[2]-gripper_length << "\033[0m";
                        qDebug() << "\033[0;32mRelease: "<< place_point[0] << " " << place_point[1] << " " << place_point[2] << "\033[0m";

                        cv::circle( drawing,
                                    release_point,
                                    12,
                                    cv::Scalar( 0, 255, 255 ),
                                    3,//cv::FILLED,
                                    cv::LINE_AA );
                        cv::arrowedLine(drawing, grasp_point, release_point, cv::Scalar( 0, 255, 0 ), 3, cv::LINE_AA, 0, 0.25);

                        cv::Mat drawing_copy;
                        drawing.copyTo(drawing_copy);

                        gLock.lockForWrite();
                        gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                        gLock.unlock();

                        // Pick and place task

                        if (exceed_limit == true){
                            // Do nothing
                        } else {
                            // Write the unfold plan file
                            QString filename = "/home/cpii/projects/scripts/unfold.sh";
                            QFile file(filename);

                            Rz = 0;

                            if (file.open(QIODevice::ReadWrite)) {
                               file.setPermissions(QFileDevice::Permissions(1909));
                               QTextStream stream(&file);
                               stream << "#!/bin/bash" << "\n"
                                      << "\n"
                                      << "cd" << "\n"
                                      << "\n"
                                      << "source /opt/ros/foxy/setup.bash" << "\n"
                                      << "\n"
                                      << "source ~/ws_moveit2/install/setup.bash" << "\n"
                                      << "\n"
                                      << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                                      << "\n"
                                      << "cd tm_robot_gripper/" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
                    //                  << "\n"
                    //                  << "sleep 1" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                      << "\n"
                                      << "sleep 2" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2] <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                      << "\n"
                    //                << "sleep 1" << "\n"
                    //                << "\n"
                                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
                    //                  << "\n"
                    //                  << "sleep 1" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                    //                  << "\n"
                    //                  << "sleep 1" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 4, positions: [" << release[0] <<", " << release[1] <<", " << release[2] <<", -3.14, 0, "<< Rz <<"], velocity: 0.7, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                      << "\n"
                                      << "sleep 2" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\""<< "\n"
                    //                << "\n"
                    //                << "sleep 1" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << release[0] <<", " << release[1] <<", " << release[2]+0.05 <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                    //                  << "\n"
                    //                  << "sleep 1" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                      << "\n"
                                      << "sleep 2" << "\n"
                                      << "\n";
                            } else {
                               qDebug("file open error");
                            }
                            file.close();

                            QProcess unfold;
                            QStringList unfoldarg;

                            unfoldarg << "/home/cpii/projects/scripts/unfold.sh";

                            unfold.start("xterm", unfoldarg);
                            constexpr int timeout_count = 60000; //60000 mseconds
                            if ( unfold.waitForFinished(timeout_count)) {
                                qDebug() << QString("\033[1;36m[%1] Robot action finished\033[0m")
                                            .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                            .toUtf8().data();
                            } else {
                                qWarning() << QString("\033[1;31mRobot action not finished within %1s\033[0m").arg(timeout_count*0.001);
                                qWarning() << unfold.errorString();
                                unfold.kill();
                                unfold.waitForFinished();

                                bResetRobot = true;
                            }
                            if ( bResetRobot ) {
                                QMetaObject::invokeMethod(this, "resetRViz",
                                                          Qt::BlockingQueuedConnection);

                                qDebug() << "\033[1;31mResetting Rviz.\033[0m";

                                QDir dir(QString("%1/%2").arg(memoryPath).arg(datasize--));
                                if (!dir.removeRecursively()) {
                                    qCritical() << "[Warning] Useless data cannot be deleted : " << datasize;
                                }
                                total_steps--;

                                if (!resetRobotPosition()){
                                    mRunReinforcementLearning1 = false;
                                    qDebug() << "\033[1;31mCan not reset Rviz, end training.\033[0m";
                                    break;
                                }
                                QThread::msleep(6000);  //Wait for the robot to reset

                                qDebug() << QString("\n-----[ %1 ]-----\n")
                                            .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                            .toUtf8().data();
                                break;
                            }
                        }
                        Sleeper::sleep(3);


                        // Reward & State

                        if(!exceed_limit){
                            gCamimage.copyTo(Src);
                            cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                            cv::resize(warped_image, inv_warp_image, warped_image_size);
                            cv::warpPerspective(inv_warp_image, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
                            cv::resize(warped_image, warped_image, warped_image_resize);
                            warped_image = background - warped_image;
                            warped_image = ~warped_image;
                            after_image = warped_image;
                            torch::Tensor after_image_tensor = torch::from_blob(warped_image.data, { warped_image.rows, warped_image.cols, warped_image.channels() }, at::kByte);
                            after_image_tensor = after_image_tensor.permute({ 2, 0, 1 });
                            after_image_tensor = after_image_tensor.unsqueeze(0).to(torch::kF32)/255;

    //                      torch::Tensor after_image_tensor = torch::randn({3, 256, 256});

                            mCalAveragePoint = true;
                            mCalAvgMat = true;
                            avgHeight.clear();
                            acount = 0;
                            while (acount<30){
                                Sleeper::msleep(200);
                            }
                            mCalAvgMat = false;
                            mCalAveragePoint = false;

                            float avg_garment_height = 0;
                            for(int i=0; i<warped_image.rows; i++){
                                for(int j=0; j<warped_image.cols; j++){
                                    auto PT_RGB = warped_image.at<cv::Vec3b>(i, j);
                                    int id = i*warped_image.cols+j;
                                    if((i >= (0.6*(float)warped_image.rows) && j >= (0.6*(float)warped_image.cols))
                                            || (PT_RGB[0] >= uThres && PT_RGB[1] >= uThres && PT_RGB[2] >= uThres)){
                                        after_tableheight[id] = 0.0;
                                    } else {
                                        cv::Point2f warpedp = cv::Point2f(j/static_cast<float>(imageWidth)*warped_image_size.width, i/static_cast<float>(imageHeight)*warped_image_size.height);
                                        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                                        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                                        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
                                        int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
                                        if(avgHeight.at(P) > 0.0 && avgHeight.at(P) < 0.20){
                                            after_tableheight[id] = (float)avgHeight.at(P);
                                            avg_garment_height += after_tableheight[id];
                                            garment_area_after+=1;
                                        } else {
                                            after_tableheight[id] = 0.0;
                                        }
                                    }
                                }
                            }
                            avg_garment_height /= garment_area_after;
                            for(int i=0; i<after_tableheight.size(); i++){
                                if(after_tableheight[i] > max_height_after && after_tableheight[i]-0.05 < avg_garment_height){
                                    max_height_after = after_tableheight[i];
                                }
                            }
                            qDebug() << "\033[1;32mMax height after action: \033[0m" << max_height_after;
                            qDebug() << "\033[1;32mGarment area after action: \033[0m" << garment_area_after;
                            after_height_tensor = torch::from_blob(after_tableheight.data(), { 1, warped_image.rows, warped_image.cols }, at::kFloat);
                            after_height_tensor = after_height_tensor.to(torch::kF32);

    //                      after_height_tensor = torch::randn({256, 256});

//                            float angle = 0;
//                            cv::Mat rotatedImg;
//                            QImage rotatedImgqt;
//                            for(int a = 0; a < 36; a++){
//                                rotatedImgqt = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888);

//                                QMatrix r;

//                                r.rotate(angle*10.0);

//                                rotatedImgqt = rotatedImgqt.transformed(r);

//                                rotatedImg = cv::Mat(rotatedImgqt.height(), rotatedImgqt.width(), CV_8UC3, rotatedImgqt.bits());

//                                std::vector<bbox_t> test_result = mDetector->detect(rotatedImg);

//                                if(test_result.size()>0){
//                                    for(auto i =0; i<test_result.size(); i++){
//                                        if(test_result[i].obj_id == 1 && conf_after < test_result[i].prob && test_result[i].prob > 0.5){
//                                            conf_after = test_result[i].prob;
//                                        }
//                                    }
//                                }
//                                angle+=1;
//                            }
//                            qDebug() << "\033[1;32mClasscification confidence level after action: \033[0m" << conf_after;

                            // Get after state

                            cv::Mat src_gray;
                            cv::cvtColor( warped_image, src_gray, cv::COLOR_BGR2GRAY );
                            cv::blur( src_gray, src_gray, cv::Size(3,3) );

                            cv::Mat canny_output;
                            cv::Canny( src_gray, canny_output, thresh, thresh*1.2 );
                            std::vector<cv::Vec4i> hierarchy;
                            cv::Size sz = src_gray.size();
                            imageWidth = sz.width;
                            imageHeight = sz.height;

                            cv::findContours( canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
                            drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );

                            findgrasp(grasp, grasp_point, Rz, hierarchy);
                            auto graspdis = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
                            while (graspdis >= robotDLimit){
                                findgrasp(grasp, grasp_point, Rz, hierarchy);
                                graspdis = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
                            }

                            if (total_steps < START_STEP){
                                findrelease(release, release_point, grasp_point);

                                // Find the location of grasp and release point
                                const double invImageW = 1.0 / imageWidth,
                                             invImageH = 1.0 / imageHeight;

                                cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                                  release_point.y*invImageH*warped_image_size.height);
                                cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                                float depth_pointr[2] = {0},
                                      color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                                rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                                int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                                cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                                cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                                cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                                float scale = dstMat(0,3);
                                QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                                QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                                release[0] = releasept.x();
                                release[1] = releasept.y();

                                height = 0.01 * QRandomGenerator::global()->bounded(5,20);
                                auto distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05)),
                                     distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                while ( distance1 > robotDLimit
                                      || distance2 > robotDLimit){
            //                        qDebug() << QString("\033[1;31m[Warning] Robot can't reach (<%4): d1=%1, d2=%2, h=%3\033[0m")
            //                                    .arg(distance1,0,'f',3).arg(distance2,0,'f',3).arg(height,0,'f',3).arg(robotDLimit,0,'f',3).toUtf8().data();

                                    gLock.lockForWrite();
                                    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                                    gLock.unlock();
                                    emit glUpdateRequest();

                                    if(   sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit
                                       || sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){

            //                            if(sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){
            //                                qDebug() << "Grasp point exceeds limit, distance: " << sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05));
            //                            }
            //                            if(sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){
            //                                qDebug() << "Release point exceeds limit, distance: " << sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05));
            //                            }

            //                            qDebug() << "Recalculate grasp and release point";

                                        findgrasp(grasp, grasp_point, Rz, hierarchy);
                                        findrelease(release, release_point, grasp_point);

                                        cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                                          release_point.y*invImageH*warped_image_size.height);
                                        cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                                        float depth_pointr[2] = {0},
                                              color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                                        rs2::frame depth;
                                        if(use_filter){
                                            depth = filtered_frame;
                                        } else {
                                            depth = frames.get_depth_frame();
                                        }
                                        rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                                        int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                                        cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                                        cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                                        cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                                        float scale = dstMat(0,3);
                                        QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                                        QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                                        release[0] = releasept.x();
                                        release[1] = releasept.y();
                                    }
                                    height = 0.01 * QRandomGenerator::global()->bounded(5,20);

                                    distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                    distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                }
                                release[2] = grasp[2] + height - 0.005;

                                cv::arrowedLine(drawing, grasp_point, release_point, cv::Scalar( 0, 255, 0 ), 3, cv::LINE_AA, 0, 0.25);
                            }

                            cv::Mat sqr_img;
                            warped_image.copyTo(sqr_img);
                            findSquares(sqr_img, squares);
                            cv::polylines(sqr_img, squares, true, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);

                            qDebug() << "\033[1;31mSqr size: " << squares.size() << "\033[0m";

                            gLock.lockForWrite();
                            gWarpedImage = QImage((uchar*) sqr_img.data, sqr_img.cols, sqr_img.rows, sqr_img.step, QImage::Format_BGR888).copy();
                            gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
                            gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                            gLock.unlock();

                            emit glUpdateRequest();

                            std::vector<float> pick_point2(3);
                            pick_point2[0] = grasp_point.x/static_cast<float>(imageWidth); pick_point2[1] = grasp_point.y/static_cast<float>(imageHeight); pick_point2[2] = grasp[2]-gripper_length;
                            after_pick_point_tensor = torch::from_blob(pick_point2.data(), { 3 }, at::kFloat).clone().detach();

                            //std::cout << "grasp point: " << grasp_point << std::endl << "pick_point2: " << pick_point2 << std::endl << "after_p_tensor: " << after_pick_point_tensor << std::endl;

                            after_image_tensor = after_image_tensor.flatten();
                            after_height_tensor = after_height_tensor.flatten();
                            after_state = torch::cat({ after_image_tensor, after_height_tensor});
                            after_state = after_state.reshape({4, warped_image_resize.width, warped_image_resize.height});

    //                      max_height_before = rand()%200/10.0f;
    //                      max_height_after = rand()%200/10.0f;
    //                      garment_area_before = rand()%10000;
    //                      garment_area_after = rand()%10000;
    //                      conf_before = 0;
    //                      conf_after = 0;
    //                      auto after_image_flatten = torch::flatten(after_image_tensor);
    //                      auto after_height_tensor_flatten = torch::flatten(after_height_tensor);
    //                      torch::Tensor pick_point2_tensor = torch::randn({3});
    //                      auto after_state = torch::cat({ after_image_flatten, after_height_tensor_flatten, pick_point2_tensor });

                            float height_reward, garment_area_reward, conf_reward;
                            height_reward = 5000 * (max_height_before - max_height_after);
                            float area_diff = garment_area_after/garment_area_before;

                            if(area_diff >= 1){
                                garment_area_reward = 400 * (area_diff - 1);
                            } else {
                                garment_area_reward = -400 * (1/area_diff - 1);
                            }
                            conf_reward = 1000 * (conf_after - conf_before);
                            stepreward[0] = height_reward + garment_area_reward + conf_reward;
                            qDebug() << "\033[1;31mReward from height: " << height_reward << " Reward from area: " << garment_area_reward << "\033[0m";
                            //qDebug() << "\033[1;31mReward from height: " << height_reward << " Reward from area: " << garment_area_reward << " Reward from classifier: " << conf_reward << "\033[0m";
                        } else {
                            after_image = before_image;
                            after_state = before_state.clone().detach();
                            after_pick_point_tensor = before_pick_point_tensor.clone().detach();
                        }

                        if (max_height_after < 0.02 && squares.size() > 0) { // Max height when garment unfolded is about 0.015m and sqr size > 0
                            done[0] = 1.0;
                            stepreward[0] += 5000;
                            garment_unfolded = true;
                            qDebug() << "\033[1;31mGarment is unfolded, end episode\033[0m";
                        }

//                        if (max_height_after < 0.017 && conf_after > 0.7) { // Max height when garment unfolded is about 0.015m, conf level is about 0.75
//                            done[0] = 1.0;
//                            stepreward[0] += 5000;
//                            garment_unfolded = true;
//                            qDebug() << "\033[1;31mGarment is unfolded, end episode\033[0m";
//                        }

                        if (exceed_limit){
                            if(test_model){
                                reset_env = true;
                            }
                            done[0] = 1.0;
                            stepreward[0] = -10000;
                            after_image_last = after_image;
                            after_state_last = before_state.clone().detach();
                            after_pick_point_last = before_pick_point_tensor.clone().detach();
                            for(int i=0; i<3; i++){
                                grasp_last[i] = grasp_before[i];
                                release_last[i] = release_before[i];
                            }
                            Rz_last = Rz_before;
                            grasp_point_last = grasp_point_before;
                            release_point_last = release_point_before;
                            max_height_last = max_height_before;
                            garment_area_last = garment_area_before;
                            conf_last = conf_before;
                            qDebug() << "\033[1;31mExceeds limit\033[0m";
                        }

                        if (garment_unfolded == false && !exceed_limit){
                            after_image_last = after_image;
                            after_state_last = after_state.clone().detach();
                            after_pick_point_last = after_pick_point_tensor.clone().detach();
                            for(int i=0; i<3; i++){
                                grasp_last[i] = grasp[i];
                                release_last[i] = release[i];
                            }
                            Rz_last = Rz;
                            grasp_point_last = grasp_point;
                            release_point_last = release_point;
                            max_height_last = max_height_after;
                            garment_area_last = garment_area_after;
                            conf_last = conf_after;
                        }

                        cv::Mat sub_image;
                        sub_image = warped_image_copy - warped_image;
                        auto mean = cv::mean(sub_image);
                        //qDebug() << "Pixel color diff mean: "<< mean[0] << " "<<mean[1]<< " "<< mean[2];
                        if(mean[0]<1.0 && mean[1]<1.0 && mean[2]<1.0 && done[0]!=1.0){
                            qDebug() << "\033[0;33mNothing Changed(mean<1.0), mean: " << mean[0] << " " << mean[1] << " " << mean[2] << " redo step\033[0m";
                            torch::AutoGradMode reset_disable(false);
                            if(test_model){
                                reset_env = true;
                                continue;
                            }
                            failtimes ++;
                            if(failtimes > 10){
                                bResetRobot = true;
                            }
                            if ( bResetRobot ) {
                                QMetaObject::invokeMethod(this, "resetRViz",
                                                          Qt::BlockingQueuedConnection);

                                qDebug() << "\033[1;31mResetting Rviz.\033[0m";

                                QDir dir(QString("%1/%2").arg(memoryPath).arg(datasize--));
                                if (!dir.removeRecursively()) {
                                    qCritical() << "[Warning] Useless data cannot be deleted : " << datasize;
                                }
                                total_steps--;

                                if (!resetRobotPosition()){
                                    mRunReinforcementLearning1 = false;
                                    qDebug() << "\033[1;31mCan not reset Rviz, end training.\033[0m";
                                    break;
                                }
                                QThread::msleep(6000);  //Wait for the robot to reset

                                qDebug() << QString("\n-----[ %1 ]-----\n")
                                            .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                            .toUtf8().data();
                                break;
                            }
                            continue;
                        } else {
                            failtimes = 0;
                        }

                        qDebug() << "\033[1;31mStep reward: " << stepreward[0] << "\033[0m";
                        episode_reward += stepreward[0];

                        // Only save data with reward <=-100 and >200
//                        if (total_steps < START_STEP && stepreward[0] > -100 && stepreward[0] < 200){
//                            qDebug() << "\033[1;31mReward between -100 to 200, redo.\033[0m";
//                            continue;
//                        }

                        if(test_model){
                            auto future_savedata = QtConcurrent::run([before_image = before_image,
                                                                     before_pick_point_CPU = before_pick_point_tensor.clone().detach(),
                                                                     place_point_CPU = place_point_tensor.clone().detach(),
                                                                     stepreward = stepreward,
                                                                     after_image = after_image,
                                                                     drawing = drawing_copy,
                                                                     garment_area_before = garment_area_before,
                                                                     garment_area_after = garment_area_after,
                                                                     max_height_before = max_height_before,
                                                                     max_height_after = max_height_after,
                                                                     episode = episode,
                                                                     step = step,
                                                                     this
                                                                     ](){

                                qDebug() << "Saving data";

                                std::vector<float> before_pick_point_vector(before_pick_point_CPU.data_ptr<float>(), before_pick_point_CPU.data_ptr<float>() + before_pick_point_CPU.numel());
                                std::vector<float> place_point_vector(place_point_CPU.data_ptr<float>(), place_point_CPU.data_ptr<float>() + place_point_CPU.numel());
                                std::vector<float> garment_area_before_vec{garment_area_before};
                                std::vector<float> garment_area_after_vec{garment_area_after};
                                std::vector<float> max_height_before_vec{max_height_before};
                                std::vector<float> max_height_after_vec{max_height_after};


                                QString filename_id = QString(testPath + "/%1").arg(episode);
                                QDir().mkdir(filename_id);
                                QString filename_id2 = QString(filename_id + "/%1").arg(step);
                                QDir().mkdir(filename_id2);

                                QString filename_before_image = QString(filename_id2 + "/before_image.jpg");
                                QByteArray filename_before_imageqb = filename_before_image.toLocal8Bit();
                                const char *filename_before_imagechar = filename_before_imageqb.data();
                                cv::imwrite(filename_before_imagechar, before_image);

                                QString filename_before_pick_point = QString(filename_id2 + "/before_pick_point.txt");
                                savedata(filename_before_pick_point, before_pick_point_vector);

                                QString filename_place_point = QString(filename_id2 + "/place_point.txt");
                                savedata(filename_place_point, place_point_vector);

                                QString filename_reward = QString(filename_id2 + "/reward.txt");
                                savedata(filename_reward, stepreward);

                                QString filename_after_image = QString(filename_id2 + "/after_image.jpg");
                                QByteArray filename_after_imageqb = filename_after_image.toLocal8Bit();
                                const char *filename_after_imagechar = filename_after_imageqb.data();
                                cv::imwrite(filename_after_imagechar, after_image);

                                QString filename_drawing = QString(filename_id2 + "/drawing.jpg");
                                QByteArray filename_drawingqb = filename_drawing.toLocal8Bit();
                                const char *filename_drawingchar = filename_drawingqb.data();
                                cv::imwrite(filename_drawingchar, drawing);

                                QString filename_garment_area_before = QString(filename_id2 + "/garment_area_before.txt");
                                savedata(filename_garment_area_before, garment_area_before_vec);

                                QString filename_garment_area_after = QString(filename_id2 + "/garment_area_after.txt");
                                savedata(filename_garment_area_after, garment_area_after_vec);

                                QString filename_max_height_before= QString(filename_id2 + "/max_height_before.txt");
                                savedata(filename_max_height_before, max_height_before_vec);

                                QString filename_max_height_after = QString(filename_id2 + "/max_height_after.txt");
                                savedata(filename_max_height_after, max_height_after_vec);
                            });
                            future_savedata.waitForFinished();
                        } else {
                            auto reward_tensor = torch::from_blob(stepreward.data(), { 1 }, torch::kFloat);
                            //std::cout << "reward_tensor: " << reward_tensor << std::endl;
                            done_tensor = torch::from_blob(done.data(), { 1 }, torch::kFloat);
                            //std::cout << "done_tensor: " << done_tensor << std::endl;

        //                    if (memory.size() >= 10000) {
        //                        memory.pop_front();
        //                    }

    //                        memory.push_back({
    //                            before_state.clone().detach(),
    //                            before_pick_point_tensor.clone().detach(),
    //                            place_point_tensor.clone().detach(),
    //                            reward_tensor.clone().detach(),
    //                            done_tensor.clone().detach(),
    //                            after_state.clone().detach(),
    //                            after_pick_point_tensor.clone().detach()
    //                            });

            //                    std::cout << "before_state: " << memory[step].before_state.mean() << std::endl
            //                              << "place_point_tensor: " << memory[step].place_point_tensor << std::endl
            //                              << "reward_tensor: " << memory[step].reward_tensor << std::endl
            //                              << "done_tensor: " << memory[step].done_tensor << std::endl
            //                              << "after_state: " << memory[step].after_state.mean() << std::endl;

    //                        std::cout << "before_state: " << before_state.sizes() << std::endl;
    //                        std::cout << "before_pick_point_tensor: " << before_pick_point_tensor.sizes() << std::endl;
    //                        std::cout << "place_point_tensor: " << place_point_tensor.sizes() << std::endl;
    //                        std::cout << "reward_tensor: " << reward_tensor.sizes() << std::endl;
    //                        std::cout << "done_tensor: " << done_tensor.sizes() << std::endl;
    //                        std::cout << "after_state: " << after_state.sizes() << std::endl;
    //                        std::cout << "after_pick_point_tensor: " << after_pick_point_tensor.sizes() << std::endl;

                            auto future_savedata = QtConcurrent::run([before_image = before_image,
                                                                     before_state_CPU = before_state.clone().detach(),
                                                                     before_pick_point_CPU = before_pick_point_tensor.clone().detach(),
                                                                     place_point_CPU = place_point_tensor.clone().detach(),
                                                                     reward_CPU = reward_tensor.clone().detach(),
                                                                     done_CPU = done_tensor.clone().detach(),
                                                                     after_image = after_image,
                                                                     after_state_CPU = after_state.clone().detach(),
                                                                     after_pick_point_CPU = after_pick_point_tensor.clone().detach(),
                                                                     drawing = drawing_copy,
                                                                     garment_area_before = garment_area_before,
                                                                     garment_area_after = garment_area_after,
                                                                     max_height_before = max_height_before,
                                                                     max_height_after = max_height_after,
                                                                     datasize = datasize,
                                                                     this
                                                                     ](){

                                qDebug() << "Saving data";

                                std::vector<float> before_state_vector(before_state_CPU.data_ptr<float>(), before_state_CPU.data_ptr<float>() + before_state_CPU.numel());
                                std::vector<float> before_pick_point_vector(before_pick_point_CPU.data_ptr<float>(), before_pick_point_CPU.data_ptr<float>() + before_pick_point_CPU.numel());
                                std::vector<float> place_point_vector(place_point_CPU.data_ptr<float>(), place_point_CPU.data_ptr<float>() + place_point_CPU.numel());
                                std::vector<float> reward_vector(reward_CPU.data_ptr<float>(), reward_CPU.data_ptr<float>() + reward_CPU.numel());
                                std::vector<float> done_vector(done_CPU.data_ptr<float>(), done_CPU.data_ptr<float>() + done_CPU.numel());
                                std::vector<float> after_state_vector(after_state_CPU.data_ptr<float>(), after_state_CPU.data_ptr<float>() + after_state_CPU.numel());
                                std::vector<float> after_pick_point_vector(after_pick_point_CPU.data_ptr<float>(), after_pick_point_CPU.data_ptr<float>() + after_pick_point_CPU.numel());
                                std::vector<float> garment_area_before_vec{garment_area_before};
                                std::vector<float> garment_area_after_vec{garment_area_after};
                                std::vector<float> max_height_before_vec{max_height_before};
                                std::vector<float> max_height_after_vec{max_height_after};

                                QString filename_id = QString(memoryPath + "/%1").arg(datasize);
                                QDir().mkdir(filename_id);

                                QString filename_before_image = QString(filename_id + "/before_image.jpg");
                                QByteArray filename_before_imageqb = filename_before_image.toLocal8Bit();
                                const char *filename_before_imagechar = filename_before_imageqb.data();
                                cv::imwrite(filename_before_imagechar, before_image);

                                QString filename_before_state = QString(filename_id + "/before_state.txt");
                                savedata(filename_before_state, before_state_vector);
                                QString filename_before_state_tensor = QString(filename_id + "/before_state_tensor.pt");
                                torch::save(before_state_CPU, filename_before_state_tensor.toStdString());

                                QString filename_before_pick_point = QString(filename_id + "/before_pick_point.txt");
                                savedata(filename_before_pick_point, before_pick_point_vector);
                                QString filename_before_pick_point_tensor = QString(filename_id + "/before_pick_point_tensor.pt");
                                torch::save(before_pick_point_CPU, filename_before_pick_point_tensor.toStdString());

                                QString filename_place_point = QString(filename_id + "/place_point.txt");
                                savedata(filename_place_point, place_point_vector);
                                QString filename_place_point_tensor = QString(filename_id + "/place_point_tensor.pt");
                                torch::save(place_point_CPU, filename_place_point_tensor.toStdString());

                                QString filename_reward = QString(filename_id + "/reward.txt");
                                savedata(filename_reward, reward_vector);
                                QString filename_reward_tensor = QString(filename_id + "/reward_tensor.pt");
                                torch::save(reward_CPU, filename_reward_tensor.toStdString());

                                QString filename_done = QString(filename_id + "/done.txt");
                                savedata(filename_done, done_vector);
                                QString filename_done_tensor = QString(filename_id + "/done_tensor.pt");
                                torch::save(done_CPU, filename_done_tensor.toStdString());

                                QString filename_after_image = QString(filename_id + "/after_image.jpg");
                                QByteArray filename_after_imageqb = filename_after_image.toLocal8Bit();
                                const char *filename_after_imagechar = filename_after_imageqb.data();
                                cv::imwrite(filename_after_imagechar, after_image);

                                QString filename_after_state = QString(filename_id + "/after_state.txt");
                                savedata(filename_after_state, after_state_vector);
                                QString filename_after_state_tensor = QString(filename_id + "/after_state_tensor.pt");
                                torch::save(after_state_CPU, filename_after_state_tensor.toStdString());

                                QString filename_after_pick_point = QString(filename_id + "/after_pick_point.txt");
                                savedata(filename_after_pick_point, after_pick_point_vector);
                                QString filename_after_pick_point_tensor = QString(filename_id + "/after_pick_point_tensor.pt");
                                torch::save(after_pick_point_CPU, filename_after_pick_point_tensor.toStdString());

                                QString filename_drawing = QString(filename_id + "/drawing.jpg");
                                QByteArray filename_drawingqb = filename_drawing.toLocal8Bit();
                                const char *filename_drawingchar = filename_drawingqb.data();
                                cv::imwrite(filename_drawingchar, drawing);

                                QString filename_garment_area_before = QString(filename_id + "/garment_area_before.txt");
                                savedata(filename_garment_area_before, garment_area_before_vec);

                                QString filename_garment_area_after = QString(filename_id + "/garment_area_after.txt");
                                savedata(filename_garment_area_after, garment_area_after_vec);

                                QString filename_max_height_before= QString(filename_id + "/max_height_before.txt");
                                savedata(filename_max_height_before, max_height_before_vec);

                                QString filename_max_height_after = QString(filename_id + "/max_height_after.txt");
                                savedata(filename_max_height_after, max_height_after_vec);
                            });
                            future_savedata.waitForFinished();
                            datasize++;
                        }
                    }

                    step++;
                    if(test_model){
                        qDebug() << "\033[0;34mTesting model : Step [" << step << "/" << teststep << "] finished\033[0m";
                    } else {
                        total_steps++;

                        std::vector<float> totalsteps;
                        totalsteps.push_back(total_steps);
                        QString filename_totalsteps = QString(memoryPath + "/totalsteps.txt");
                        savedata(filename_totalsteps, totalsteps);

                        if (total_steps > batch_size) {
                            torch::AutoGradMode enable(true);
                            for(int train=0; train<TRAINEVERY; train++){
                                train_number++;
                                if(done_old_data){
                                    qDebug() << "\033[1;33mTraining model: [" << train+1 << "/" << TRAINEVERY << "]\033[0m";
                                }
                                int randomdata = rand()%(10000-batch_size+1);
                                randomdata = total_steps - 10000 + randomdata;
                                //qDebug() << "randomdata: " << randomdata;
                                //qDebug() << "memory size: " << memory.size();
                                std::vector<torch::Tensor> s_data(batch_size), p_data(batch_size), a_data(batch_size), r_data(batch_size), d_data(batch_size), s2_data(batch_size), p2_data(batch_size);
                                torch::Tensor s_batch, p_batch, a_batch, r_batch, d_batch, s2_batch, p2_batch;

                                for (int i = 0; i < batch_size; i++) {
                                    QString filename_id;
                                    filename_id = QString(memoryPath + "/%1").arg(i+randomdata);

                                    QString filename_s = QString(filename_id + "/before_state_tensor.pt");
                                    torch::Tensor tmp_s_data;
                                    torch::load(tmp_s_data, filename_s.toStdString());

                                    QString filename_p = QString(filename_id + "/before_pick_point_tensor.pt");
                                    torch::Tensor tmp_p_data;
                                    torch::load(tmp_p_data, filename_p.toStdString());

                                    QString filename_a = QString(filename_id + "/place_point_tensor.pt");
                                    torch::Tensor tmp_a_data;
                                    torch::load(tmp_a_data, filename_a.toStdString());

                                    QString filename_r = QString(filename_id + "/reward_tensor.pt");
                                    torch::Tensor tmp_r_data;
                                    torch::load(tmp_r_data, filename_r.toStdString());

                                    QString filename_d = QString(filename_id + "/done_tensor.pt");
                                    torch::Tensor tmp_d_data;
                                    torch::load(tmp_d_data, filename_d.toStdString());

                                    QString filename_s2 = QString(filename_id + "/after_state_tensor.pt");
                                    torch::Tensor tmp_s2_data;
                                    torch::load(tmp_s2_data, filename_s2.toStdString());

                                    QString filename_p2 = QString(filename_id + "/after_pick_point_tensor.pt");
                                    torch::Tensor tmp_p2_data;
                                    torch::load(tmp_p2_data, filename_p2.toStdString());

                                    s_data[i] = torch::unsqueeze(tmp_s_data.clone().detach(), 0);
                                    p_data[i] = torch::unsqueeze(tmp_p_data.clone().detach(), 0);
                                    a_data[i] = torch::unsqueeze(tmp_a_data.clone().detach(), 0);
                                    r_data[i] = torch::unsqueeze(tmp_r_data.clone().detach(), 0);
                                    d_data[i] = torch::unsqueeze(tmp_d_data.clone().detach(), 0);
                                    s2_data[i] = torch::unsqueeze(tmp_s2_data.clone().detach(), 0);
                                    p2_data[i] = torch::unsqueeze(tmp_p2_data.clone().detach(), 0);
                                }

                                s_batch = s_data[0]; p_batch = p_data[0]; a_batch = a_data[0]; r_batch = r_data[0]; d_batch = d_data[0]; s2_batch = s2_data[0]; p2_batch = p2_data[0];
                                for (int i = 1; i < batch_size; i++) {
                                    s_batch = torch::cat({ s_batch, s_data[i] }, 0);
                                    p_batch = torch::cat({ p_batch, p_data[i] }, 0);
                                    a_batch = torch::cat({ a_batch, a_data[i] }, 0);
                                    r_batch = torch::cat({ r_batch, r_data[i] }, 0);
                                    d_batch = torch::cat({ d_batch, d_data[i] }, 0);
                                    s2_batch = torch::cat({ s2_batch, s2_data[i] }, 0);
                                    p2_batch = torch::cat({ p2_batch, p2_data[i] }, 0);
                                }
                                s_batch = s_batch.clone().detach().to(device);
                                p_batch = p_batch.clone().detach().to(device);
                                a_batch = a_batch.clone().detach().to(device);
                                r_batch = r_batch.clone().detach().to(device);
                                d_batch = d_batch.clone().detach().to(device);
                                s2_batch = s2_batch.clone().detach().to(device);
                                p2_batch = p2_batch.clone().detach().to(device);

                                // Q-value networks training
                                if(done_old_data){
                                    qDebug() << "\033[1;33mTraining Q-value networks\033[0m";
                                }

                                torch::AutoGradMode q_enable(true);

                                torch::Tensor q1 = actor_critic->q1->forward_pick_point(s_batch, p_batch, a_batch);
                                torch::Tensor q2 = actor_critic->q2->forward_pick_point(s_batch, p_batch, a_batch);

                                torch::AutoGradMode disable(false);
                                // Target actions come from *current* policy
                                policy_output next_state_sample = actor_critic->pi->forward_pick_point(s2_batch, p2_batch, false, true);
                                torch::Tensor a2_batch = next_state_sample.action;
                                torch::Tensor logp_a2 = next_state_sample.logp_pi;
                                // Target Q-values
                                torch::Tensor q1_pi_target = actor_critic_target->q1->forward_pick_point(s2_batch, p2_batch, a2_batch);
                                torch::Tensor q2_pi_target = actor_critic_target->q2->forward_pick_point(s2_batch, p2_batch, a2_batch);
                                torch::Tensor backup = r_batch + GAMMA * (1.0 - d_batch) * (torch::min(q1_pi_target, q2_pi_target) - ALPHA * logp_a2);

                                // MSE loss against Bellman backup
                                torch::AutoGradMode loss_enable(true);
//                                torch::Tensor loss_q1 = torch::mean(pow(q1 - backup, 2)); // JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
//                                torch::Tensor loss_q2 = torch::mean(pow(q2 - backup, 2));
                                torch::Tensor loss_q1 = torch::nn::functional::mse_loss(q1, backup);
                                torch::Tensor loss_q2 = torch::nn::functional::mse_loss(q2, backup);
                                torch::Tensor loss_q = loss_q1 + loss_q2;

                                float loss_q1f = loss_q1.detach().item().toFloat();
                                float loss_q2f = loss_q2.detach().item().toFloat();
                                episode_critic1_loss += loss_q1f;
                                episode_critic2_loss += loss_q2f;

                                critic_optimizer.zero_grad();
                                loss_q.backward();
                                critic_optimizer.step();

                                if(done_old_data){
                                    qDebug() << "\033[1;33mCritic optimizer step\033[0m";
                                    qDebug() << "\033[1;33mTraining policy network\033[0m";
                                }

                                // Policy network training
                                for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
                                    actor_critic->q1->parameters()[i].set_requires_grad(false);
                                }
                                for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
                                    actor_critic->q2->parameters()[i].set_requires_grad(false);
                                }

                                policy_output sample = actor_critic->pi->forward_pick_point(s_batch, p_batch, false, true);
                                torch::Tensor pi = sample.action;
                                torch::Tensor log_pi = sample.logp_pi;
                                torch::Tensor q1_pi = actor_critic->q1->forward_pick_point(s_batch, p_batch, pi);
                                torch::Tensor q2_pi = actor_critic->q2->forward_pick_point(s_batch, p_batch, pi);
                                torch::Tensor q_pi = torch::min(q1_pi, q2_pi);

                                // Entropy-regularized policy loss
                                torch::Tensor loss_pi = torch::mean(ALPHA * log_pi - q_pi); // Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

                                float loss_pif = loss_pi.detach().item().toFloat();
                                episode_policy_loss += loss_pif;

                                policy_optimizer.zero_grad();
                                loss_pi.backward();
                                policy_optimizer.step();

                                for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
                                    actor_critic->q1->parameters()[i].set_requires_grad(true);
                                }
                                for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
                                    actor_critic->q2->parameters()[i].set_requires_grad(true);
                                }

                                if(done_old_data){
                                    qDebug() << "\033[1;33mPolicy optimizer step\033[0m";
                                    qDebug() << "\033[1;33mUpdating target models\033[0m";
                                }

                                // Update target networks
                                torch::AutoGradMode softcopy_disable(false);
                                for (size_t i = 0; i < actor_critic_target->pi->parameters().size(); i++) {
                                    actor_critic_target->pi->parameters()[i].mul_(POLYAK);
                                    actor_critic_target->pi->parameters()[i].add_((1.0 - POLYAK) * actor_critic->pi->parameters()[i]);
                                }
                                for (size_t i = 0; i < actor_critic_target->q1->parameters().size(); i++) {
                                    actor_critic_target->q1->parameters()[i].mul_(POLYAK);
                                    actor_critic_target->q1->parameters()[i].add_((1.0 - POLYAK) * actor_critic->q1->parameters()[i]);
                                }
                                for (size_t i = 0; i < actor_critic_target->q2->parameters().size(); i++) {
                                    actor_critic_target->q2->parameters()[i].mul_(POLYAK);
                                    actor_critic_target->q2->parameters()[i].add_((1.0 - POLYAK) * actor_critic->q2->parameters()[i]);
                                }
                                torch::AutoGradMode softcopy_enable(true);

                                if(done_old_data){
                                    qDebug() << "\033[1;31mCritic 1 loss: " << loss_q1f << "\n"
                                             << "Critic 2 loss: " << loss_q2f << "\n"
                                             << "Policy loss: " << loss_pif << "\n"
                                             << "Episode reward: " << episode_reward << "\033[0m";
                                }
                            }
                        }
                        qDebug() << "\033[0;34mEpisode " << episode+1 << ": Step [" << step << "/" << maxstep << "] finished\033[0m";
                    }

                    if(!done_old_data && total_steps+1 > datasize){
                        done_old_data = true;
                        qDebug() << "\033[1;31mDone training old data, start plan\033[0m";
                        break;
                    }
                    if(garment_unfolded == true){
                        break;
                    }
                    if(test_model && step == teststep){
                        break;
                    }
                }

                // Save
                if(test_model){
                    qDebug() << "\033[0;35m--------------------------------------------" << "\n"
                             << "Reward: " << episode_reward << "\n"
                             << "--------------------------------------------\033[0m";
                    logger.add_scalar("Test_Reward", episode, episode_reward);
                    qDebug() << "\033[0;34mTest model finished\033[0m\n"
                             << "--------------------------------------------";
                    test_model = false;
                } else if (bResetRobot){
                    qDebug() << "\033[0;35m--------------------------------------------" << "\n"
                             << "Restart episode" << episode+1 << "\n"
                             << "--------------------------------------------\033[0m";
                } else {
                    episode++;

                    episode_critic1_loss = episode_critic1_loss / (float)train_number;
                    episode_critic2_loss = episode_critic2_loss / (float)train_number;
                    episode_policy_loss = episode_policy_loss / (float)train_number;

                    qDebug() << "\033[0;35m--------------------------------------------" << "\n"
                        << "Episode: " << episode << "\n"
                        << "Reward: " << episode_reward << "\n"
                        << "Critic 1 Loss: " << episode_critic1_loss << "\n"
                        << "Critic 2 Loss: " << episode_critic2_loss << "\n"
                        << "Policy Loss: " << episode_policy_loss << "\n"
                        << "--------------------------------------------\033[0m";
                    logger.add_scalar("Episode_Reward", episode, episode_reward);
                    logger.add_scalar("Episode_Critic_1_Loss", episode, episode_critic1_loss);
                    logger.add_scalar("Episode_Critic_2_Loss", episode, episode_critic2_loss);
                    logger.add_scalar("Episode_Policy_Loss", episode, episode_policy_loss);

                    int save = SAVEMODELEVERY;
                    if(!done_old_data){
                        save = 50;
                    }
                    if (episode % save == 0) {
                        qDebug() << "Saving models";

                        QString pi_para_path = QString(modelPath + "/pi_para/pi_para_" + QString::number(episode) + ".pt");
                        QString q1_para_path = QString(modelPath + "/q1_para/q1_para_" + QString::number(episode) + ".pt");
                        QString q2_para_path = QString(modelPath + "/q2_para/q2_para_" + QString::number(episode) + ".pt");
                        QString target_pi_para_path = QString(modelPath + "/target_pi_para/target_pi_para_" + QString::number(episode) + ".pt");
                        QString target_q1_para_path = QString(modelPath + "/target_q1_para/target_q1_para_" + QString::number(episode) + ".pt");
                        QString target_q2_para_path = QString(modelPath + "/target_q2_para/target_q2_para_" + QString::number(episode) + ".pt");
                        QString policy_opti_path = QString(modelPath + "/policy_optimizer/policy_optimizer_" + QString::number(episode) + ".pt");
                        QString critic_opti_path = QString(modelPath + "/critic_optimizer/critic_optimizer_" + QString::number(episode) + ".pt");

                        torch::save(actor_critic->pi->parameters(), pi_para_path.toStdString());
                        torch::save(actor_critic->q1->parameters(), q1_para_path.toStdString());
                        torch::save(actor_critic->q2->parameters(), q2_para_path.toStdString());
                        torch::save(actor_critic_target->pi->parameters(), target_pi_para_path.toStdString());
                        torch::save(actor_critic_target->q1->parameters(), target_q1_para_path.toStdString());
                        torch::save(actor_critic_target->q2->parameters(), target_q2_para_path.toStdString());
                        torch::save(policy_optimizer, policy_opti_path.toStdString());
                        torch::save(critic_optimizer, critic_opti_path.toStdString());

                        std::vector<float> save_episode_num;
                        save_episode_num.push_back(episode+1);
                        QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
                        savedata(filename_episode_num, save_episode_num);

                        qDebug() << "Models saved";
                    }

                    qDebug() << "\033[0;34mEpisode " << episode << "finished\033[0m\n"
                             << "--------------------------------------------";
                    if(episode % 10 == 0){
                        test_model = true;
                    }
                }

                if(mRunReinforcementLearning1 == false){
                    qDebug() << "Quit Reinforcement Learning 1" ;
                    qDebug() << "Saving models";

                    QString pi_para_path = QString(modelPath + "/pi_para/pi_para_" + QString::number(episode) + ".pt");
                    QString q1_para_path = QString(modelPath + "/q1_para/q1_para_" + QString::number(episode) + ".pt");
                    QString q2_para_path = QString(modelPath + "/q2_para/q2_para_" + QString::number(episode) + ".pt");
                    QString target_pi_para_path = QString(modelPath + "/target_pi_para/target_pi_para_" + QString::number(episode) + ".pt");
                    QString target_q1_para_path = QString(modelPath + "/target_q1_para/target_q1_para_" + QString::number(episode) + ".pt");
                    QString target_q2_para_path = QString(modelPath + "/target_q2_para/target_q2_para_" + QString::number(episode) + ".pt");
                    QString policy_opti_path = QString(modelPath + "/policy_optimizer/policy_optimizer_" + QString::number(episode) + ".pt");
                    QString critic_opti_path = QString(modelPath + "/critic_optimizer/critic_optimizer_" + QString::number(episode) + ".pt");

                    torch::save(actor_critic->pi->parameters(), pi_para_path.toStdString());
                    torch::save(actor_critic->q1->parameters(), q1_para_path.toStdString());
                    torch::save(actor_critic->q2->parameters(), q2_para_path.toStdString());
                    torch::save(actor_critic_target->pi->parameters(), target_pi_para_path.toStdString());
                    torch::save(actor_critic_target->q1->parameters(), target_q1_para_path.toStdString());
                    torch::save(actor_critic_target->q2->parameters(), target_q2_para_path.toStdString());
                    torch::save(policy_optimizer, policy_opti_path.toStdString());
                    torch::save(critic_optimizer, critic_opti_path.toStdString());

                    std::vector<float> save_episode_num;
                    save_episode_num.push_back(episode+1);
                    QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
                    savedata(filename_episode_num, save_episode_num);

                    qDebug() << "Models saved";
                    break;
                }
            }
        } catch (const std::exception &e) {
            auto &&msg = torch::GetExceptionString(e);
            qWarning() << msg.c_str();
        } catch (...) {
            qCritical() << "GG";
        }
    });
}


void LP_Plugin_Garment_Manipulation::Reinforcement_Learning_2(){
    auto rl1current = QtConcurrent::run([this](){
        try {
            torch::manual_seed(0);

//            if ( !mDetector ) {
//                mDetector = std::make_shared<Detector>("/home/cpii/darknet-master/yolo_models/yolov3-df2.cfg", "/home/cpii/darknet-master/yolo_models/yolov3-df2_15000.weights");
//            }

            device = torch::Device(torch::kCPU);
            if (torch::cuda::is_available()) {
                std::cout << "CUDA is available! Training on GPU." << std::endl;
                device = torch::Device(torch::kCUDA);
            }

            torch::autograd::DetectAnomalyGuard detect_anomaly;

            qDebug() << "Creating models";

            std::vector<int> policy_mlp_dims{STATE_DIM, 4096, 1024, 128, 32};
            std::vector<int> critic_mlp_dims{STATE_DIM + ACT_DIM, 4096, 512, 64, 16};

            auto actor_critic = ActorCritic(policy_mlp_dims, critic_mlp_dims);
            auto actor_critic_target = ActorCritic(policy_mlp_dims, critic_mlp_dims);

            qDebug() << "Creating optimizer";

            torch::AutoGradMode copy_disable(false);

            std::vector<torch::Tensor> q_params;
            for(size_t i=0; i<actor_critic->q1->parameters().size(); i++){
                q_params.push_back(actor_critic->q1->parameters()[i]);
            }
            for(size_t i=0; i<actor_critic->q2->parameters().size(); i++){
                q_params.push_back(actor_critic->q2->parameters()[i]);
            }
            torch::AutoGradMode copy_enable(true);

            torch::optim::Adam policy_optimizer(actor_critic->pi->parameters(), torch::optim::AdamOptions(lrp));
            torch::optim::Adam critic_optimizer(q_params, torch::optim::AdamOptions(lrc));

            actor_critic->pi->to(device);
            actor_critic->q1->to(device);
            actor_critic->q2->to(device);
            actor_critic_target->pi->to(device);
            actor_critic_target->q1->to(device);
            actor_critic_target->q2->to(device);

            //-----------------------------------------------------------------------------
//            //CV MAT TO TENSOR
//            gCamimage.copyTo(Src);
//            //cv::resize(Src, Src, cv::Size(4, 4));

//            auto tensor_image = torch::from_blob(Src.data, { Src.rows, Src.cols, Src.channels() }, at::kByte);
//            tensor_image = tensor_image.permute({ 2, 0, 1 });
//            //tensor_image = tensor_image.unsqueeze(0);
//            tensor_image = tensor_image.to(torch::kF32)/255;
//            tensor_image.to(torch::kCPU);

//            //std::cout << "Src: " << std::endl << Src << std::endl;
//            //std::cout << "Src tensor: " << std::endl << tensor_image*255 << std::endl;

//            //TENSOR TO CV MAT
//            torch::Tensor out_tensor = tensor_image*255;
//            out_tensor = out_tensor.permute({1, 2, 0}).to(torch::kF32);
//            cv::Mat cv_mat(Src.rows, Src.cols, CV_32FC3, out_tensor.data_ptr());
//            cv_mat.convertTo(cv_mat, CV_8UC3);

//            //std::cout << "cv_mat: " << std::endl << cv_mat << std::endl;

//            gLock.lockForWrite();
//            gWarpedImage = QImage((uchar*) Src.data, Src.cols, Src.rows, Src.step, QImage::Format_BGR888).copy();
//            //gInvWarpImage = QImage((uchar*) cv_mat.data, cv_mat.cols, cv_mat.rows, cv_mat.step, QImage::Format_BGR888).copy();
//            gEdgeImage = QImage((uchar*) cv_mat.data, cv_mat.cols, cv_mat.rows, cv_mat.step, QImage::Format_BGR888).copy();
//            gLock.unlock();
//            emit glUpdateRequest();

//            return 0;
            //-----------------------------------------------------------------------------

            //-----------------------------------------------------------------------------
            // Test
//            QString filename_before_state_tensor = QString(memoryPath + "/0/before_state_tensor.pt");
//            torch::Tensor before_state_tensor;
//            torch::load(before_state_tensor, filename_before_state_tensor.toStdString());

//            QString filename_before_pick = QString(memoryPath + "/0/before_pick_point_tensor.pt");
//            torch::Tensor before_pick;
//            torch::load(before_pick, filename_before_pick.toStdString());

//            before_state_tensor = before_state_tensor.unsqueeze(0).detach().to(device);
//            before_pick = before_pick.unsqueeze(0).detach().to(device);
//            auto out = actor_critic->pi->forward(before_state_tensor, before_pick, true, false);

//            std::cout << "action: " << out.action << std::endl;

//            //TENSOR TO CV MAT
//            torch::Tensor out_tensor = before_state_tensor*255;
//            auto out_tensor1 = out_tensor.index({3}).to(torch::kF32);
//            cv::Mat cv_mat1(512, 512, CV_32FC1, out_tensor1.data_ptr());
//            auto min = out_tensor1.min().item().toFloat();
//            auto max = out_tensor1.max().item().toFloat();
//            cv_mat1.convertTo(cv_mat1, CV_8U, 255.0/(max-min));
//            cv::cvtColor(cv_mat1, cv_mat1, CV_GRAY2BGR);

//            auto out_tensor2 = out_tensor.index({1}).to(torch::kF32);
//            cv::Mat cv_mat2(512, 512, CV_32FC1, out_tensor2.data_ptr());
//            cv_mat2.convertTo(cv_mat2, CV_8U);
//            cv::cvtColor(cv_mat2, cv_mat2, CV_GRAY2BGR);

//            auto out_tensor3= out_tensor.index({2}).to(torch::kF32);
//            cv::Mat cv_mat3(512, 512, CV_32FC1, out_tensor3.data_ptr());
//            cv_mat3.convertTo(cv_mat3, CV_8U);
//            cv::cvtColor(cv_mat3, cv_mat3, CV_GRAY2BGR);

//            //std::cout << "cv_mat: " << std::endl << cv_mat << std::endl;

//            gLock.lockForWrite();
//            gWarpedImage = QImage((uchar*) cv_mat1.data, cv_mat1.cols, cv_mat1.rows, cv_mat1.step, QImage::Format_BGR888).copy();
//            gInvWarpImage = QImage((uchar*) cv_mat2.data, cv_mat2.cols, cv_mat2.rows, cv_mat2.step, QImage::Format_BGR888).copy();
//            gEdgeImage = QImage((uchar*) cv_mat2.data, cv_mat3.cols, cv_mat3.rows, cv_mat3.step, QImage::Format_BGR888).copy();
//            gLock.unlock();
//            emit glUpdateRequest();

//            return 0;

            //-----------------------------------------------------------------------------

            GOOGLE_PROTOBUF_VERIFY_VERSION;
            TensorBoardLogger logger(kLogFile2.c_str());

            int episode = 0;
            int datasize = 0;
            bool LoadOldData = false;
            bool RestoreFromCheckpoint = false;
            if(RestoreFromCheckpoint || LoadOldData){
//                int olddata_size = 0;

                if(LoadOldData){
                    qDebug() << "Load old data";
//                    for (const auto & file : std::filesystem::directory_iterator(memoryPath.toStdString())){
//                        olddata_size++;
//                    }
//                    datasize = olddata_size;
                    datasize = 10000;
                    qDebug() << "Data size: " << datasize;

                } else {
                    qDebug() << "Restore from check point";

                    QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
                    std::vector<float> saved_episode_num;
                    loaddata(filename_episode_num.toStdString(), saved_episode_num);
                    episode = saved_episode_num[0]-1;
                    maxepisode += episode;

                    QString filename_totalsteps = QString(memoryPath + "/totalsteps.txt");
                    std::vector<float> totalsteps;
                    loaddata(filename_totalsteps.toStdString(), totalsteps);
                    total_steps = int(totalsteps[0]);
                    qDebug() << "Total steps: " << total_steps;

                    for (const auto & file : std::filesystem::directory_iterator(memoryPath.toStdString())){
                        datasize++;
                    }
                    datasize -= 2;
                    qDebug() << "Data size: " << datasize;

                    qDebug() << "Loading models";

                    QString pi_para_path = QString(modelPath + "/pi_para/pi_para_" + QString::number(episode) + ".pt");
                    QString q1_para_path = QString(modelPath + "/q1_para/q1_para_" + QString::number(episode) + ".pt");
                    QString q2_para_path = QString(modelPath + "/q2_para/q2_para_" + QString::number(episode) + ".pt");
                    QString target_pi_para_path = QString(modelPath + "/target_pi_para/target_pi_para_" + QString::number(episode) + ".pt");
                    QString target_q1_para_path = QString(modelPath + "/target_q1_para/target_q1_para_" + QString::number(episode) + ".pt");
                    QString target_q2_para_path = QString(modelPath + "/target_q2_para/target_q2_para_" + QString::number(episode) + ".pt");
                    QString policy_opti_path = QString(modelPath + "/policy_optimizer/policy_optimizer_" + QString::number(episode) + ".pt");
                    QString critic_opti_path = QString(modelPath + "/critic_optimizer/critic_optimizer_" + QString::number(episode) + ".pt");

                    std::vector<torch::Tensor> pi_para, q1_para, q2_para, target_pi_para, target_q1_para, target_q2_para;

                    torch::load(pi_para, pi_para_path.toStdString());
                    torch::load(q1_para, q1_para_path.toStdString());
                    torch::load(q2_para, q2_para_path.toStdString());
                    torch::load(target_pi_para, target_pi_para_path.toStdString());
                    torch::load(target_q1_para, target_q1_para_path.toStdString());
                    torch::load(target_q2_para, target_q2_para_path.toStdString());
                    torch::load(policy_optimizer, policy_opti_path.toStdString());
                    torch::load(critic_optimizer, critic_opti_path.toStdString());

                    torch::AutoGradMode data_copy_disable(false);
                    for(size_t i=0; i < actor_critic->pi->parameters().size(); i++){
                        actor_critic->pi->parameters()[i].copy_(pi_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
                        actor_critic->q1->parameters()[i].copy_(q1_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
                        actor_critic->q2->parameters()[i].copy_(q2_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic_target->pi->parameters().size(); i++){
                        actor_critic_target->pi->parameters()[i].copy_(target_pi_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic_target->q1->parameters().size(); i++){
                        actor_critic_target->q1->parameters()[i].copy_(target_q1_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic_target->q2->parameters().size(); i++){
                        actor_critic_target->q2->parameters()[i].copy_(target_q2_para[i].clone().detach().to(device));
                    }
                    torch::AutoGradMode data_copy_enable(true);
                }

    //            qDebug() << "Loading memory";

    //            for(int i=0; i<datasize; i++){
    //                qDebug() << "Memory loading: [" << i+1 << "/" << datasize << "]";
    //                QString filename_id;
    //                if(RestoreFromCheckpoint){
    //                    filename_id = memoryPath + QString("/%1").arg(i);
    //                } else {
    //                    filename_id = olddatasavepath + QString("/%1").arg(i);
    //                }


    //                // Load tensor(.pt)
    //                QString filename_before_state_tensor = QString(filename_id + "/before_state_tensor.pt");
    //                torch::Tensor before_state_tensor;
    //                torch::load(before_state_tensor, filename_before_state_tensor.toStdString());

    //                QString filename_before_pick_point_tensor = QString(filename_id + "/before_pick_point_tensor.pt");
    //                torch::Tensor before_pick_point_tensor;
    //                torch::load(before_pick_point_tensor, filename_before_pick_point_tensor.toStdString());

    //                QString filename_place_point_tensor = QString(filename_id + "/place_point_tensor.pt");
    //                torch::Tensor place_point_tensor;
    //                torch::load(place_point_tensor, filename_place_point_tensor.toStdString());

    //                QString filename_reward_tensor = QString(filename_id + "/reward_tensor.pt");
    //                torch::Tensor reward_tensor;
    //                torch::load(reward_tensor, filename_reward_tensor.toStdString());

    //                QString filename_done_tensor = QString(filename_id + "/done_tensor.pt");
    //                torch::Tensor done_tensor;
    //                torch::load(done_tensor, filename_done_tensor.toStdString());

    //                QString filename_after_state_tensor = QString(filename_id + "/after_state_tensor.pt");
    //                torch::Tensor after_state_tensor;
    //                torch::load(after_state_tensor, filename_after_state_tensor.toStdString());

    //                QString filename_after_pick_point_tensor = QString(filename_id + "/after_pick_point_tensor.pt");
    //                torch::Tensor after_pick_point_tensor;
    //                torch::load(after_pick_point_tensor, filename_after_pick_point_tensor.toStdString());

                    // Load string(.txt)
    //                QString filename_before_state = QString(filename_id + "/before_state.txt");
    //                std::vector<float> before_state_vector;
    //                loaddata(filename_before_state.toStdString(), before_state_vector);
    //                torch::Tensor before_state_tensor = torch::from_blob(before_state_vector.data(), { 262147 }, torch::kFloat);

    //                QString filename_place_point = QString(filename_id + "/place_point.txt");
    //                std::vector<float> place_point_vector;
    //                loaddata(filename_place_point.toStdString(), place_point_vector);
    //                torch::Tensor place_point_tensor = torch::from_blob(place_point_vector.data(), { 3 }, torch::kFloat);

    //                QString filename_reward = QString(filename_id + "/reward.txt");
    //                std::vector<float> reward_vector;
    //                loaddata(filename_reward.toStdString(), reward_vector);
    //                torch::Tensor reward_tensor = torch::from_blob(reward_vector.data(), { 1 }, torch::kFloat);

    //                QString filename_done = QString(filename_id + "/done.txt");
    //                std::vector<float> done_vector;
    //                loaddata(filename_done.toStdString(), done_vector);
    //                torch::Tensor done_tensor = torch::from_blob(done_vector.data(), { 1 }, torch::kFloat);

    //                QString filename_after_state = QString(filename_id + "/after_state.txt");
    //                std::vector<float> after_state_vector;
    //                loaddata(filename_after_state.toStdString(), after_state_vector);
    //                torch::Tensor after_state_tensor = torch::from_blob(after_state_vector.data(), { 262147 }, torch::kFloat);

    //                // Trans to tensor and save
    //                QString sfilename_before_state_tensor = QString(filename_id + "/before_state_tensor.pt");
    //                torch::save(before_state_tensor, sfilename_before_state_tensor.toStdString());

    //                QString sfilename_place_point_tensor = QString(filename_id + "/place_point_tensor.pt");
    //                torch::save(place_point_tensor, sfilename_place_point_tensor.toStdString());

    //                QString sfilename_reward_tensor = QString(filename_id + "/reward_tensor.pt");
    //                torch::save(reward_tensor, sfilename_reward_tensor.toStdString());

    //                QString sfilename_done_tensor = QString(filename_id + "/done_tensor.pt");
    //                torch::save(done_tensor, sfilename_done_tensor.toStdString());

    //                QString sfilename_after_state_tensor = QString(filename_id + "/after_state_tensor.pt");
    //                torch::save(after_state_tensor, sfilename_after_state_tensor.toStdString());

    //                memory.push_back({before_state_tensor.clone().detach(),
    //                                  before_pick_point_tensor.clone().detach(),
    //                                  place_point_tensor.clone().detach(),
    //                                  reward_tensor.clone().detach(),
    //                                  done_tensor.clone().detach(),
    //                                  after_state_tensor.clone().detach(),
    //                                  after_pick_point_tensor.clone().detach()});

    //                std::cout << "before lowest: " << memory[i].before_state.min() << std::endl;
    //                std::cout << "before highest: " << memory[i].before_state.max() << std::endl;
    //                std::cout << "place_point: " << memory[i].place_point << std::endl;
    //                std::cout << "reward: " << memory[i].reward << std::endl;
    //                std::cout << "done: " << memory[i].done << std::endl;
    //                std::cout << "after lowest: " << memory[i].after_state.min() << std::endl;
    //                std::cout << "after highest: " << memory[i].after_state.max() << std::endl;
    //            }
            }

            if(!RestoreFromCheckpoint){
                qDebug() << "Copying parameters to target models";
                torch::AutoGradMode hardcopy_disable(false);
                for(size_t i=0; i < actor_critic_target->pi->parameters().size(); i++){
                    actor_critic_target->pi->parameters()[i].copy_(actor_critic->pi->parameters()[i]);
                    actor_critic_target->pi->parameters()[i].set_requires_grad(false);
                }
                for(size_t i=0; i < actor_critic_target->q1->parameters().size(); i++){
                    actor_critic_target->q1->parameters()[i].copy_(actor_critic->q1->parameters()[i]);
                    actor_critic_target->q1->parameters()[i].set_requires_grad(false);
                }
                for(size_t i=0; i < actor_critic_target->q2->parameters().size(); i++){
                    actor_critic_target->q2->parameters()[i].copy_(actor_critic->q2->parameters()[i]);
                    actor_critic_target->q2->parameters()[i].set_requires_grad(false);
                }
                torch::AutoGradMode hardcopy_enable(true);
            }

            int step = 0, train_number = 0, failtimes;
            float episode_reward = 0, episode_critic1_loss = 0, episode_critic2_loss = 0, episode_policy_loss = 0;
            float Rz = 0;
            std::vector<float> done(1);
            torch::Tensor done_tensor;
            rs2::frame depth;
            if(use_filter){
                depth = filtered_frame;
            } else {
                depth = frames.get_depth_frame();
            }
            cv::Mat warped_image_copy;
            total_reward = 0; total_critic_loss = 0; total_policy_loss = 0;
            bool bResetRobot = false, done_old_data = true, garment_unfolded = false, test_model = false, reset_env = false;
            std::vector<int> unfolded;
            if(LoadOldData){
                done_old_data = false;
            }

            while (episode < maxepisode) {
                qDebug() << "--------------------------------------------";
                if(test_model){
                    qDebug() << "\033[0;34mTest model\033[0m";
                } else {
                    qDebug() << "\033[0;34mEpisode " << episode+1 << " started\033[0m";
                }

                // Initialize environment
                episode_reward = 0;
                episode_critic1_loss = 0;
                episode_critic2_loss = 0;
                episode_policy_loss = 0;
                done[0] = 0;
                step = 0;
                train_number = 0;
                failtimes = 0;
                Rz = 0;
                bResetRobot = false;
                garment_unfolded = false;
                cv::Mat before_image, after_image;

                while (step < maxstep && mRunReinforcementLearning1) {
                    //std::cout << "p fc3: \n" << policy->fc3->parameters() << std::endl;
                    //std::cout << "p fc4: \n" << policy->fc4->parameters() << std::endl;
                    qDebug() << "--------------------------------------------";
                    qDebug() << QString("\n-----[ %1 ]-----\n")
                                .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                .toUtf8().data();
                    if(test_model){
                        qDebug() << "\033[0;34mTesting model : Step [" << step+1 << "/" << teststep << "] started\033[0m";
                    } else {
                        qDebug() << "\033[0;34mEpisode " << episode+1 << ": Step [" << step+1 << "/" << maxstep << "] started\033[0m";
    //                    if(done_old_data){
    //                        qDebug() << "\033[0;34mTotal steps: " << total_steps << "\033[0m";
    //                    } else {
    //                        qDebug() << "\033[0;34mTotal steps / Old data size: [" << total_steps << "/" << datasize << "]\033[0m";
    //                    }

                        qDebug() << "\033[0;34mTotal steps: " << total_steps+1;
                        qDebug() << "Unfolded step: ";
                        if(unfolded.size()<1){
                            qDebug() << "None";
                        } else {
                            for(int i=0; i<unfolded.size(); i++){
                                qDebug() << unfolded[i];
                            }
                        }
                        qDebug() << "\033[0m";
                    }

                    if(step == 0){
                        reset_env = true;
                    }

                    if(done_old_data && reset_env){
                        // Reset garment
                        gCamimage.copyTo(Src);
                        std::vector<double> graspr(3);
                        cv::Point grasp_pointr;
                        cv::Mat inv_warp_imager;
                        cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                        cv::resize(warped_image, inv_warp_imager, warped_image_size);
                        cv::warpPerspective(inv_warp_imager, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
                        cv::resize(warped_image, warped_image, warped_image_resize);
                        warped_image = background - warped_image;
                        warped_image = ~warped_image;

                        // Find contours
                        cv::Mat src_gray;
                        cv::cvtColor( warped_image, src_gray, cv::COLOR_BGR2GRAY );
                        cv::blur( src_gray, src_gray, cv::Size(3,3) );

                        cv::Mat canny_output;
                        cv::Canny( src_gray, canny_output, thresh, thresh*1.2 );
                        std::vector<cv::Vec4i> hierarchyr;
                        cv::Size sz = src_gray.size();
                        imageWidth = sz.width;
                        imageHeight = sz.height;

                        cv::findContours( canny_output, contours, hierarchyr, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
                        drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );

                        findgrasp(graspr, grasp_pointr, Rz, hierarchyr);

                        gLock.lockForWrite();
                        gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
                        gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
                        gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                        gLock.unlock();
                        emit glUpdateRequest();

                        // Write the plan file
                        QString filename = "/home/cpii/projects/scripts/move.sh";
                        QFile file(filename);

                        Rz = 0;

                        if (file.open(QIODevice::ReadWrite)) {
                           file.setPermissions(QFileDevice::Permissions(1909));
                           QTextStream stream(&file);
                           stream << "#!/bin/bash" << "\n"
                                  << "\n"
                                  << "cd" << "\n"
                                  << "\n"
                                  << "source /opt/ros/foxy/setup.bash" << "\n"
                                  << "\n"
                                  << "source ~/ws_moveit2/install/setup.bash" << "\n"
                                  << "\n"
                                  << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                                  << "\n"
                                  << "cd tm_robot_gripper/" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
                //                  << "\n"
                //                  << "sleep 1" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << graspr[0] <<", " << graspr[1] <<", " << graspr[2]+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 1.5, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                  << "\n"
                                  << "sleep 2" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << graspr[0] <<", " << graspr[1] <<", " << graspr[2] <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                  << "\n"
                                  << "sleep 1" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
                //                  << "\n"
                //                  << "sleep 1" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [-0.5, 0, 0.7, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                  << "\n"
                                  << "sleep 1" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\""<< "\n"
                //                  << "\n"
                //                  << "sleep 1" << "\n"
                                  << "\n"
                                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 1.5, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                  << "\n"
                                  << "sleep 2" << "\n"
                                  << "\n";
                        } else {
                           qDebug("file open error");
                        }
                        file.close();

                        QProcess plan;
                        QStringList planarg;

                        planarg << "/home/cpii/projects/scripts/move.sh";
                        plan.start("xterm", planarg);

                        constexpr int timeout_count = 60000; //60000 mseconds
                        if ( plan.waitForFinished(timeout_count)) {
                            qDebug() << QString("\033[1;36m[%1] Env reset finished\033[0m")
                                        .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                        .toUtf8().data();
                        } else {
                            qWarning() << QString("\033[1;31mRobot action not finished within %1s\033[0m").arg(timeout_count*0.001);
                            qWarning() << plan.errorString();
                            plan.kill();
                            plan.waitForFinished();

                            bResetRobot = true;
                        }
                        if ( bResetRobot ) {
                            QMetaObject::invokeMethod(this, "resetRViz",
                                                      Qt::BlockingQueuedConnection);

                            qDebug() << "\033[1;31mResetting Rviz.\033[0m";

                            QDir dir(QString("%1/%2").arg(memoryPath).arg(datasize--));
                            if (!dir.removeRecursively()) {
                                qCritical() << "[Warning] Useless data cannot be deleted : " << datasize;
                            }
                            total_steps--;

                            if (!resetRobotPosition()){
                                mRunReinforcementLearning1 = false;
                                qDebug() << "\033[1;31mCan not reset Rviz, end training.\033[0m";
                                break;
                            }
                            QThread::msleep(6000);  //Wait for the robot to reset

                            qDebug() << QString("\n-----[ %1 ]-----\n")
                                        .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                        .toUtf8().data();
                            break;
                        }
                        Sleeper::sleep(3);

                        // Rreset Environment
//                        bool end_training = false;
//                        Env_reset(Rz, bResetRobot, datasize, end_training);
//                        if(end_training){
//                            break;
//                        }
                    }

                    if(done_old_data){
                        garment_unfolded = false;
                        bool exceed_limit = false;
                        cv::Mat inv_warp_image;
                        torch::Tensor before_pick_point_tensor, after_pick_point_tensor, place_point_tensor;
                        std::vector<float> pick_point(3), place_point(3), src_tableheight(warped_image.cols * warped_image.rows), after_tableheight(warped_image.cols * warped_image.rows);
                        std::vector<double> grasp(3), release(3), grasp_before(3), release_before(3);
                        cv::Point grasp_point, release_point, grasp_point_before, release_point_before;
                        torch::Tensor src_tensor, before_state, src_height_tensor, after_state, after_height_tensor;
                        std::vector<std::vector<cv::Point>> squares;
                        std::vector<float> stepreward(1);
                        float max_height_before = 0, max_height_after = 0, garment_area_before = 0, garment_area_after = 0, conf_before = 0, conf_after = 0;
                        stepreward[0] = 0;
                        done[0] = 0;
                        Rz = 0;
                        float Rz_before = 0;

                        if(reset_env){
                            // Preprocess environment
                            gCamimage.copyTo(Src);
                            cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                            cv::resize(warped_image, inv_warp_image, warped_image_size);
                            cv::warpPerspective(inv_warp_image, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
                            cv::resize(warped_image, warped_image, warped_image_resize);
                            warped_image = background - warped_image;
                            warped_image = ~warped_image;

                            before_image = warped_image;

                            src_tensor = torch::from_blob(warped_image.data, { warped_image.rows, warped_image.cols, warped_image.channels() }, at::kByte);
                            src_tensor = src_tensor.permute({ 2, 0, 1 });
                            src_tensor = src_tensor.unsqueeze(0).to(torch::kF32)/255;

    //                        src_tensor = torch::randn({3, 256, 256});

                            // Find contours

                            cv::Mat src_gray;
                            cv::cvtColor( warped_image, src_gray, cv::COLOR_BGR2GRAY );
                            cv::blur( src_gray, src_gray, cv::Size(3,3) );

                            cv::Mat canny_output;
                            cv::Canny( src_gray, canny_output, thresh, thresh*1.2 );
                            std::vector<cv::Vec4i> hierarchy;
                            cv::Size sz = src_gray.size();
                            imageWidth = sz.width;
                            imageHeight = sz.height;

                            cv::findContours( canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
                            drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );

                            mCalAveragePoint = true;
                            mCalAvgMat = true;
                            avgHeight.clear();
                            acount = 0;
                            while (acount<30){
                                Sleeper::msleep(200);
                            }
                            mCalAvgMat = false;
                            mCalAveragePoint = false;

                            float avg_garment_height = 0;
                            for(int i=0; i<warped_image.rows; i++){
                                for(int j=0; j<warped_image.cols; j++){
                                    auto PT_RGB = warped_image.at<cv::Vec3b>(i, j);
                                    int id = i*warped_image.cols+j;
                                    if((i >= (0.6*(float)warped_image.rows) && j >= (0.6*(float)warped_image.cols))
                                            || (PT_RGB[0] >= uThres && PT_RGB[1] >= uThres && PT_RGB[2] >= uThres)){
                                        src_tableheight[id] = 0.0;
                                    } else {
                                        cv::Point2f warpedp = cv::Point2f(j/static_cast<float>(imageWidth)*warped_image_size.width, i/static_cast<float>(imageHeight)*warped_image_size.height);
                                        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                                        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                                        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
                                        int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
                                        if(avgHeight.at(P) > 0.0 && avgHeight.at(P) < 0.20){
                                            //src_tableheight[id] = Pt.z();
                                            src_tableheight[id] = (float)avgHeight.at(P);
                                            avg_garment_height += src_tableheight[id];
                                            garment_area_before+=1;
                                            //std::cout<<"H: "<<(float)avgHeight.at(P)<< '\n';
                                        } else {
                                            src_tableheight[id] = 0.0;
                                        }
                                    }
                                }
                            }
                            avg_garment_height /= garment_area_before;
                            for(int i=0; i<src_tableheight.size(); i++){
                                if(src_tableheight[i] > max_height_before && src_tableheight[i]-0.05 < avg_garment_height){
                                    max_height_before = src_tableheight[i];
                                }
                            }

//                            auto maxm = *std::max_element(avgHeight.begin(), avgHeight.end());
//                            auto minm = *std::min_element(avgHeight.begin(), avgHeight.end());
//                            auto max = *std::max_element(src_tableheight.begin(), src_tableheight.end());
//                            auto min = *std::min_element(src_tableheight.begin(), src_tableheight.end());
//                            std::cout<<"Max value: "<<max<< std::endl << "min: " << min << '\n';
//                            std::cout<<"Max value m: "<<maxm<< std::endl << "min m: " << minm << '\n';

                            src_height_tensor = torch::from_blob(src_tableheight.data(), { 1, warped_image.rows, warped_image.cols }, at::kFloat);
                            src_height_tensor = src_height_tensor.to(torch::kF32);

//                            auto min1 = src_height_tensor.min().item().toFloat();
//                            auto max1 = src_height_tensor.max().item().toFloat();
//                            std::cout << "out_tensor1: " << src_height_tensor.sizes() << std::endl << min1 << " " << max1 << std::endl;
//                            src_height_tensor = src_height_tensor.squeeze();
//                            cv::Mat cv_mat1(512, 512, CV_32FC1, src_height_tensor.data_ptr());
//                            cv_mat1.convertTo(cv_mat1, CV_8U, 255.0/(0.005-min1));

//                            gLock.lockForWrite();
//                            gWarpedImage = QImage((uchar*) cv_mat1.data, cv_mat1.cols, cv_mat1.rows, cv_mat1.step, QImage::Format_Grayscale8).copy();
//                            gLock.unlock();
//                            emit glUpdateRequest();

//                            return 0;

    //                        src_height_tensor = torch::randn({256, 256});


//                            float angle = 0;
//                            cv::Mat rotatedImg;
//                            QImage rotatedImgqt;
//                            for(int a = 0; a < 36; a++){
//                                rotatedImgqt = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888);

//                                QMatrix r;

//                                r.rotate(angle*10.0);

//                                rotatedImgqt = rotatedImgqt.transformed(r);

//                                rotatedImg = cv::Mat(rotatedImgqt.height(), rotatedImgqt.width(), CV_8UC3, rotatedImgqt.bits());

//                                std::vector<bbox_t> test_result = mDetector->detect(rotatedImg);

//                                if(test_result.size()>0){
//                                    for(auto i=0; i<test_result.size(); i++){
//                                        if(test_result[i].obj_id == 1 && conf_before < test_result[i].prob && test_result[i].prob > 0.5){
//                                            conf_before = test_result[i].prob;
//                                        }
//                                    }
//                                }
//                                angle+=1;
//                            }

                            findgrasp(grasp, grasp_point, Rz, hierarchy);
                            auto graspdis = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
                            while (graspdis >= robotDLimit){
                                findgrasp(grasp, grasp_point, Rz, hierarchy);
                                graspdis = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
                            }

                            if (total_steps < START_STEP){
                                findrelease(release, release_point, grasp_point);

                                // Find the location of grasp and release point
                                const double invImageW = 1.0 / imageWidth,
                                             invImageH = 1.0 / imageHeight;

                                cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                                  release_point.y*invImageH*warped_image_size.height);
                                cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                                float depth_pointr[2] = {0},
                                      color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                                rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                                int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                                cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                                cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                                cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                                float scale = dstMat(0,3);
                                QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                                QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                                release[0] = releasept.x();
                                release[1] = releasept.y();

                                height = 0.01 * QRandomGenerator::global()->bounded(5,20);
                                auto distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05)),
                                     distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                while ( distance1 > robotDLimit
                                      || distance2 > robotDLimit){
        //                            qDebug() << QString("\033[1;31m[Warning] Robot can't reach (<%4): d1=%1, d2=%2, h=%3\033[0m")
        //                                        .arg(distance1,0,'f',3).arg(distance2,0,'f',3).arg(height,0,'f',3).arg(robotDLimit,0,'f',3).toUtf8().data();

                                    gLock.lockForWrite();
                                    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                                    gLock.unlock();
                                    emit glUpdateRequest();

                                    if(   sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit
                                       || sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){

        //                                if(sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){
        //                                    qDebug() << "Grasp point exceeds limit, distance: " << sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05));
        //                                }
        //                                if(sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){
        //                                    qDebug() << "Release point exceeds limit, distance: " << sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05));
        //                                }
        //                                qDebug() << "Recalculate grasp and release point";

                                        findgrasp(grasp, grasp_point, Rz, hierarchy);
                                        findrelease(release, release_point, grasp_point);

                                        cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                                          release_point.y*invImageH*warped_image_size.height);
                                        cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                                        float depth_pointr[2] = {0},
                                              color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                                        rs2::frame depth;
                                        if(use_filter){
                                            depth = filtered_frame;
                                        } else {
                                            depth = frames.get_depth_frame();
                                        }
                                        rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                                        int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                                        cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                                        cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                                        cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                                        float scale = dstMat(0,3);
                                        QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                                        QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                                        release[0] = releasept.x();
                                        release[1] = releasept.y();
                                    }
                                    height = 0.01 * QRandomGenerator::global()->bounded(5,20);

                                    distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                    distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                }
                                release[2] = grasp[2]+height;

                                cv::arrowedLine(drawing, grasp_point, release_point, cv::Scalar( 0, 255, 0 ), 3, cv::LINE_AA, 0, 0.25);
                            }
                            pick_point[0] = grasp_point.x/static_cast<float>(imageWidth); pick_point[1] = grasp_point.y/static_cast<float>(imageHeight); pick_point[2] = grasp[2]-gripper_length;
                            before_pick_point_tensor = torch::from_blob(pick_point.data(), { 3 }, at::kFloat).clone().detach();
                            src_tensor = src_tensor.flatten();
                            src_height_tensor = src_height_tensor.flatten();
                            before_state = torch::cat({ src_tensor, src_height_tensor});
                            before_state = before_state.reshape({4, warped_image_resize.width, warped_image_resize.height});
    //                        }
    //                        torch::Tensor pick_point_tensor = torch::randn({3});
    //                        auto src_tensor_flatten = torch::flatten(src_tensor);
    //                        auto src_height_tensor_flatten = torch::flatten(src_height_tensor);
    //                        before_state = torch::cat({ src_tensor_flatten, src_height_tensor_flatten, pick_point_tensor });
                        } else {
                            before_image = after_image_last;
                            before_state = after_state_last.clone().detach();
                            before_pick_point_tensor = after_pick_point_last.clone().detach();
                            for(int i=0; i<3; i++){
                                grasp[i] = grasp_last[i];
                                release[i] = release_last[i];
                            }
                            grasp_point = grasp_point_last;
                            release_point = release_point_last;
                            max_height_before = max_height_last;
                            garment_area_before = garment_area_last;
                            conf_before = conf_last;
                        }

                        reset_env = false;

                        qDebug() << "\033[1;32mMax height before action: \033[0m" << max_height_before;
                        qDebug() << "\033[1;32mGarment area before action: \033[0m" << garment_area_before;
                        //qDebug() << "\033[1;32mClasscification confidence level before action: \033[0m" << conf_before;

                        warped_image_copy = warped_image;

                        cv::circle( drawing,
                                    grasp_point,
                                    12,
                                    cv::Scalar( 0, 0, 255 ),
                                    3,//cv::FILLED,
                                    cv::LINE_AA );

                        gLock.lockForWrite();
                        gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
                        gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
                        gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                        gLock.unlock();
                        emit glUpdateRequest();

                        cv::Point2f warpedp = cv::Point2f(grasp_point.x/static_cast<float>(imageWidth)*warped_image_size.width, grasp_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
                        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
                        int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
                        mGraspP.resize(1);
                        mGraspP[0] = mPointCloud[P];

                        // Action
                        torch::AutoGradMode enable(true);
                        actor_critic->pi->train();
                        if (total_steps < START_STEP) {
                            qDebug() << "\033[1;33mStart exploration\033[0m";
                            place_point[0] = release_point.x/static_cast<float>(imageWidth); place_point[1] = release_point.y/static_cast<float>(imageHeight); place_point[2] = release[2]-gripper_length;
                            place_point_tensor = torch::from_blob(place_point.data(), { 3 }, at::kFloat);
    //                        place_point_tensor = torch::randn({3});
                            //std::cout << "place_point_tensor: " << place_point_tensor << std::endl;
                        } else {
                            qDebug() << "\033[1;33mStart exploitation\033[0m";
                            auto state = before_state.clone().detach().to(device);
                            auto p = before_pick_point_tensor.clone().detach().to(device);
                            if(test_model){
                                torch::AutoGradMode disable(false);
                                actor_critic->pi->eval();
                                place_point_tensor = actor_critic->act_pick_point(torch::unsqueeze(state, 0), torch::unsqueeze(p, 0), true);
                            } else {
                                place_point_tensor = actor_critic->act_pick_point(torch::unsqueeze(state, 0), torch::unsqueeze(p, 0), false);
                            }
                            place_point_tensor = place_point_tensor.squeeze().to(torch::kCPU);
                            std::cout << "\033[1;34mAction predict: \n" << place_point_tensor << "\033[0m" << std::endl;
                            place_point = std::vector(place_point_tensor.data_ptr<float>(), place_point_tensor.data_ptr<float>() + place_point_tensor.numel());
                            release_point.x = place_point[0]*static_cast<float>(imageWidth); release_point.y = place_point[1]*static_cast<float>(imageHeight);

                            cv::Point2f warpedpr = cv::Point2f(release_point.x/static_cast<float>(imageWidth)*warped_image_size.width,
                                                              release_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
                            cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                            float depth_pointr[2] = {0},
                                  color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                            rs2::frame depth;
                            if(use_filter){
                                depth = filtered_frame;
                            } else {
                                depth = frames.get_depth_frame();
                            }
                            rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                            int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                            cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                            cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                            cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                            float scale = dstMat(0,3);
                            QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                            QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                            release[0] = releasept.x();
                            release[1] = releasept.y();
                            release[2] = place_point[2]+gripper_length;

                            float distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
                            float distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(release[2]+0.05)*(release[2]+0.05));
                            qDebug() << "distance1: " << distance1 << "distance2: " << distance2;
                            //qDebug() << "place_point: " << place_point[0] << " " << place_point[1] << " " << place_point[2];
                            if(distance1 >= robotDLimit || distance2 >= robotDLimit
                               || place_point[2] < 0.05 || place_point[2] > 0.25
                               || (place_point[0] > 0.6 && place_point[1] > 0.6)
                               || place_point[0] < 0.1 || place_point[0] > 0.9
                               || place_point[1] < 0.1 || place_point[1] > 0.9){
                                qDebug() << "\033[1;31m";
                                if(distance1 >= robotDLimit){
                                    qDebug() << "Error: distance1 >= robotDLimit";
                                }
                                if(distance2 >= robotDLimit){
                                    qDebug() << "Error: distance2 >= robotDLimit";
                                }
                                if(place_point[2] < 0.05){
                                    qDebug() << "Error: place_point[2] < 0.05";
                                }
                                if(place_point[2] > 0.25){
                                    qDebug() << "Error: place_point[2] > 0.25";
                                }
                                if(place_point[0] > 0.6 && place_point[1] > 0.6){
                                    qDebug() << "Error: place_point[0] > 0.6 && place_point[1] > 0.6";
                                }
                                if(place_point[0] < 0.1){
                                    qDebug() << "Error: place_point[0] < 0.1";
                                }
                                if(place_point[0] > 0.9){
                                    qDebug() << "Error: place_point[0] > 0.9";
                                }
                                if(place_point[1] < 0.1){
                                    qDebug() << "Error: place_point[1] < 0.1";
                                }
                                if(place_point[1] > 0.9){
                                    qDebug() << "Error: place_point[1] > 0.9";
                                }
                                qDebug() << "\033[0m";
                                exceed_limit = true;
                            }
                            //qDebug() << "memory size: " << memory.size();
                        }

                        grasp_before = grasp;
                        release_before = release;
                        grasp_point_before = grasp_point;
                        release_point_before = release_point;
                        Rz_before = Rz;

                        //qDebug() << "width: " << imageWidth << "height: " << imageHeight;
                        qDebug() << "\033[0;32mGrasp: " << grasp_point.x/static_cast<float>(imageWidth)  << " "<< grasp_point.y/static_cast<float>(imageHeight)  << " " << grasp[2]-gripper_length << "\033[0m";
                        qDebug() << "\033[0;32mRelease: "<< place_point[0] << " " << place_point[1] << " " << place_point[2] << "\033[0m";

                        cv::circle( drawing,
                                    release_point,
                                    12,
                                    cv::Scalar( 0, 255, 255 ),
                                    3,//cv::FILLED,
                                    cv::LINE_AA );
                        cv::arrowedLine(drawing, grasp_point, release_point, cv::Scalar( 0, 255, 0 ), 3, cv::LINE_AA, 0, 0.25);

                        cv::Mat drawing_copy;
                        drawing.copyTo(drawing_copy);

                        gLock.lockForWrite();
                        gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                        gLock.unlock();

                        // Pick and place task

                        if (exceed_limit == true){
                            // Do nothing
                        } else {
                            // Write the unfold plan file
                            QString filename = "/home/cpii/projects/scripts/unfold.sh";
                            QFile file(filename);

                            Rz = 0;

                            if (file.open(QIODevice::ReadWrite)) {
                               file.setPermissions(QFileDevice::Permissions(1909));
                               QTextStream stream(&file);
                               stream << "#!/bin/bash" << "\n"
                                      << "\n"
                                      << "cd" << "\n"
                                      << "\n"
                                      << "source /opt/ros/foxy/setup.bash" << "\n"
                                      << "\n"
                                      << "source ~/ws_moveit2/install/setup.bash" << "\n"
                                      << "\n"
                                      << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                                      << "\n"
                                      << "cd tm_robot_gripper/" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
                    //                  << "\n"
                    //                  << "sleep 1" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                      << "\n"
                                      << "sleep 2" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2] <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                      << "\n"
                    //                << "sleep 1" << "\n"
                    //                << "\n"
                                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
                    //                  << "\n"
                    //                  << "sleep 1" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+0.1 <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                    //                  << "\n"
                    //                  << "sleep 1" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 4, positions: [" << release[0] <<", " << release[1] <<", " << release[2] <<", -3.14, 0, "<< Rz <<"], velocity: 0.7, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                      << "\n"
                                      << "sleep 2" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\""<< "\n"
                    //                << "\n"
                    //                << "sleep 1" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << release[0] <<", " << release[1] <<", " << release[2]+0.05 <<", -3.14, 0, "<< Rz <<"], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                    //                  << "\n"
                    //                  << "sleep 1" << "\n"
                                      << "\n"
                                      << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 2, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                                      << "\n"
                                      << "sleep 2" << "\n"
                                      << "\n";
                            } else {
                               qDebug("file open error");
                            }
                            file.close();

                            QProcess unfold;
                            QStringList unfoldarg;

                            unfoldarg << "/home/cpii/projects/scripts/unfold.sh";

                            unfold.start("xterm", unfoldarg);
                            constexpr int timeout_count = 60000; //60000 mseconds
                            if ( unfold.waitForFinished(timeout_count)) {
                                qDebug() << QString("\033[1;36m[%1] Robot action finished\033[0m")
                                            .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                            .toUtf8().data();
                            } else {
                                qWarning() << QString("\033[1;31mRobot action not finished within %1s\033[0m").arg(timeout_count*0.001);
                                qWarning() << unfold.errorString();
                                unfold.kill();
                                unfold.waitForFinished();

                                bResetRobot = true;
                            }
                            if ( bResetRobot ) {
                                QMetaObject::invokeMethod(this, "resetRViz",
                                                          Qt::BlockingQueuedConnection);

                                qDebug() << "\033[1;31mResetting Rviz.\033[0m";

                                QDir dir(QString("%1/%2").arg(memoryPath).arg(datasize--));
                                if (!dir.removeRecursively()) {
                                    qCritical() << "[Warning] Useless data cannot be deleted : " << datasize;
                                }
                                total_steps--;

                                if (!resetRobotPosition()){
                                    mRunReinforcementLearning1 = false;
                                    qDebug() << "\033[1;31mCan not reset Rviz, end training.\033[0m";
                                    break;
                                }
                                QThread::msleep(6000);  //Wait for the robot to reset

                                qDebug() << QString("\n-----[ %1 ]-----\n")
                                            .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                            .toUtf8().data();
                                break;
                            }
                        }
                        Sleeper::sleep(3);


                        // Reward & State

                        if(!exceed_limit){
                            gCamimage.copyTo(Src);
                            cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                            cv::resize(warped_image, inv_warp_image, warped_image_size);
                            cv::warpPerspective(inv_warp_image, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
                            cv::resize(warped_image, warped_image, warped_image_resize);
                            warped_image = background - warped_image;
                            warped_image = ~warped_image;
                            after_image = warped_image;
                            torch::Tensor after_image_tensor = torch::from_blob(warped_image.data, { warped_image.rows, warped_image.cols, warped_image.channels() }, at::kByte);
                            after_image_tensor = after_image_tensor.permute({ 2, 0, 1 });
                            after_image_tensor = after_image_tensor.unsqueeze(0).to(torch::kF32)/255;

    //                      torch::Tensor after_image_tensor = torch::randn({3, 256, 256});

                            mCalAveragePoint = true;
                            mCalAvgMat = true;
                            avgHeight.clear();
                            acount = 0;
                            while (acount<30){
                                Sleeper::msleep(200);
                            }
                            mCalAvgMat = false;
                            mCalAveragePoint = false;

                            float avg_garment_height = 0;
                            for(int i=0; i<warped_image.rows; i++){
                                for(int j=0; j<warped_image.cols; j++){
                                    auto PT_RGB = warped_image.at<cv::Vec3b>(i, j);
                                    int id = i*warped_image.cols+j;
                                    if((i >= (0.6*(float)warped_image.rows) && j >= (0.6*(float)warped_image.cols))
                                            || (PT_RGB[0] >= uThres && PT_RGB[1] >= uThres && PT_RGB[2] >= uThres)){
                                        after_tableheight[id] = 0.0;
                                    } else {
                                        cv::Point2f warpedp = cv::Point2f(j/static_cast<float>(imageWidth)*warped_image_size.width, i/static_cast<float>(imageHeight)*warped_image_size.height);
                                        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                                        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                                        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
                                        int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
                                        if(avgHeight.at(P) > 0.0 && avgHeight.at(P) < 0.20){
                                            after_tableheight[id] = (float)avgHeight.at(P);
                                            avg_garment_height += after_tableheight[id];
                                            garment_area_after+=1;
                                        } else {
                                            after_tableheight[id] = 0.0;
                                        }
                                    }
                                }
                            }
                            avg_garment_height /= garment_area_after;
                            for(int i=0; i<after_tableheight.size(); i++){
                                if(after_tableheight[i] > max_height_after && after_tableheight[i]-0.05 < avg_garment_height){
                                    max_height_after = after_tableheight[i];
                                }
                            }
                            qDebug() << "\033[1;32mMax height after action: \033[0m" << max_height_after;
                            qDebug() << "\033[1;32mGarment area after action: \033[0m" << garment_area_after;
                            after_height_tensor = torch::from_blob(after_tableheight.data(), { 1, warped_image.rows, warped_image.cols }, at::kFloat);
                            after_height_tensor = after_height_tensor.to(torch::kF32);

    //                      after_height_tensor = torch::randn({256, 256});

//                            float angle = 0;
//                            cv::Mat rotatedImg;
//                            QImage rotatedImgqt;
//                            for(int a = 0; a < 36; a++){
//                                rotatedImgqt = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888);

//                                QMatrix r;

//                                r.rotate(angle*10.0);

//                                rotatedImgqt = rotatedImgqt.transformed(r);

//                                rotatedImg = cv::Mat(rotatedImgqt.height(), rotatedImgqt.width(), CV_8UC3, rotatedImgqt.bits());

//                                std::vector<bbox_t> test_result = mDetector->detect(rotatedImg);

//                                if(test_result.size()>0){
//                                    for(auto i =0; i<test_result.size(); i++){
//                                        if(test_result[i].obj_id == 1 && conf_after < test_result[i].prob && test_result[i].prob > 0.5){
//                                            conf_after = test_result[i].prob;
//                                        }
//                                    }
//                                }
//                                angle+=1;
//                            }
//                            qDebug() << "\033[1;32mClasscification confidence level after action: \033[0m" << conf_after;

                            // Get after state

                            cv::Mat src_gray;
                            cv::cvtColor( warped_image, src_gray, cv::COLOR_BGR2GRAY );
                            cv::blur( src_gray, src_gray, cv::Size(3,3) );

                            cv::Mat canny_output;
                            cv::Canny( src_gray, canny_output, thresh, thresh*1.2 );
                            std::vector<cv::Vec4i> hierarchy;
                            cv::Size sz = src_gray.size();
                            imageWidth = sz.width;
                            imageHeight = sz.height;

                            cv::findContours( canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
                            drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );

                            findgrasp(grasp, grasp_point, Rz, hierarchy);
                            auto graspdis = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
                            while (graspdis >= robotDLimit){
                                findgrasp(grasp, grasp_point, Rz, hierarchy);
                                graspdis = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.1)*(grasp[2]+0.1));
                            }

                            if (total_steps < START_STEP){
                                findrelease(release, release_point, grasp_point);

                                // Find the location of grasp and release point
                                const double invImageW = 1.0 / imageWidth,
                                             invImageH = 1.0 / imageHeight;

                                cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                                  release_point.y*invImageH*warped_image_size.height);
                                cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                                float depth_pointr[2] = {0},
                                      color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                                rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                                int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                                cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                                cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                                cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                                float scale = dstMat(0,3);
                                QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                                QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                                release[0] = releasept.x();
                                release[1] = releasept.y();

                                height = 0.01 * QRandomGenerator::global()->bounded(5,20);
                                auto distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05)),
                                     distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                while ( distance1 > robotDLimit
                                      || distance2 > robotDLimit){
            //                        qDebug() << QString("\033[1;31m[Warning] Robot can't reach (<%4): d1=%1, d2=%2, h=%3\033[0m")
            //                                    .arg(distance1,0,'f',3).arg(distance2,0,'f',3).arg(height,0,'f',3).arg(robotDLimit,0,'f',3).toUtf8().data();

                                    gLock.lockForWrite();
                                    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                                    gLock.unlock();
                                    emit glUpdateRequest();

                                    if(   sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit
                                       || sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){

            //                            if(sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){
            //                                qDebug() << "Grasp point exceeds limit, distance: " << sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05));
            //                            }
            //                            if(sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05)) > robotDLimit){
            //                                qDebug() << "Release point exceeds limit, distance: " << sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+0.05+0.05)*(grasp[2]+0.05+0.05));
            //                            }

            //                            qDebug() << "Recalculate grasp and release point";

                                        findgrasp(grasp, grasp_point, Rz, hierarchy);
                                        findrelease(release, release_point, grasp_point);

                                        cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                                          release_point.y*invImageH*warped_image_size.height);
                                        cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                                        float depth_pointr[2] = {0},
                                              color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                                        rs2::frame depth;
                                        if(use_filter){
                                            depth = filtered_frame;
                                        } else {
                                            depth = frames.get_depth_frame();
                                        }
                                        rs2_project_color_pixel_to_depth_pixel(depth_pointr, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointr);
                                        int P = int(depth_pointr[0])*depthh + depthh-int(depth_pointr[1]);
                                        cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                                        cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                                        cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                                        float scale = dstMat(0,3);
                                        QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                                        QVector4D releasept = Transformationmatrix_T2R.inverted() * Pt;
                                        release[0] = releasept.x();
                                        release[1] = releasept.y();
                                    }
                                    height = 0.01 * QRandomGenerator::global()->bounded(5,20);

                                    distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                    distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(grasp[2]+height+0.05)*(grasp[2]+height+0.05));
                                }
                                release[2] = grasp[2] + height - 0.005;

                                cv::arrowedLine(drawing, grasp_point, release_point, cv::Scalar( 0, 255, 0 ), 3, cv::LINE_AA, 0, 0.25);
                            }

                            cv::Mat sqr_img;
                            warped_image.copyTo(sqr_img);
                            findSquares(sqr_img, squares);
                            cv::polylines(sqr_img, squares, true, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);

                            qDebug() << "\033[1;31mSqr size: " << squares.size() << "\033[0m";

                            gLock.lockForWrite();
                            gWarpedImage = QImage((uchar*) sqr_img.data, sqr_img.cols, sqr_img.rows, sqr_img.step, QImage::Format_BGR888).copy();
                            gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
                            gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                            gLock.unlock();

                            emit glUpdateRequest();

                            std::vector<float> pick_point2(3);
                            pick_point2[0] = grasp_point.x/static_cast<float>(imageWidth); pick_point2[1] = grasp_point.y/static_cast<float>(imageHeight); pick_point2[2] = grasp[2]-gripper_length;
                            after_pick_point_tensor = torch::from_blob(pick_point2.data(), { 3 }, at::kFloat).clone().detach();

                            //std::cout << "grasp point: " << grasp_point << std::endl << "pick_point2: " << pick_point2 << std::endl << "after_p_tensor: " << after_pick_point_tensor << std::endl;

                            after_image_tensor = after_image_tensor.flatten();
                            after_height_tensor = after_height_tensor.flatten();
                            after_state = torch::cat({ after_image_tensor, after_height_tensor});
                            after_state = after_state.reshape({4, warped_image_resize.width, warped_image_resize.height});

    //                      max_height_before = rand()%200/10.0f;
    //                      max_height_after = rand()%200/10.0f;
    //                      garment_area_before = rand()%10000;
    //                      garment_area_after = rand()%10000;
    //                      conf_before = 0;
    //                      conf_after = 0;
    //                      auto after_image_flatten = torch::flatten(after_image_tensor);
    //                      auto after_height_tensor_flatten = torch::flatten(after_height_tensor);
    //                      torch::Tensor pick_point2_tensor = torch::randn({3});
    //                      auto after_state = torch::cat({ after_image_flatten, after_height_tensor_flatten, pick_point2_tensor });

                            float height_reward, garment_area_reward, conf_reward;
                            height_reward = 5000 * (max_height_before - max_height_after);
                            float area_diff = garment_area_after/garment_area_before;

                            if(area_diff >= 1){
                                garment_area_reward = 400 * (area_diff - 1);
                            } else {
                                garment_area_reward = -400 * (1/area_diff - 1);
                            }
                            conf_reward = 1000 * (conf_after - conf_before);
                            stepreward[0] = height_reward + garment_area_reward + conf_reward;
                            qDebug() << "\033[1;31mReward from height: " << height_reward << " Reward from area: " << garment_area_reward << "\033[0m";
                            //qDebug() << "\033[1;31mReward from height: " << height_reward << " Reward from area: " << garment_area_reward << " Reward from classifier: " << conf_reward << "\033[0m";
                        } else {
                            after_image = before_image;
                            after_state = before_state.clone().detach();
                            after_pick_point_tensor = before_pick_point_tensor.clone().detach();
                        }

                        if (max_height_after < 0.02 && squares.size() > 0) { // Max height when garment unfolded is about 0.015m and sqr size > 0
                            done[0] = 1.0;
                            stepreward[0] += 5000;
                            garment_unfolded = true;
                            qDebug() << "\033[1;31mGarment is unfolded, end episode\033[0m";
                        }

//                        if (max_height_after < 0.017 && conf_after > 0.7) { // Max height when garment unfolded is about 0.015m, conf level is about 0.75
//                            done[0] = 1.0;
//                            stepreward[0] += 5000;
//                            garment_unfolded = true;
//                            qDebug() << "\033[1;31mGarment is unfolded, end episode\033[0m";
//                        }

                        if (exceed_limit){
                            if(test_model){
                                reset_env = true;
                            }
                            done[0] = 1.0;
                            stepreward[0] = -10000;
                            after_image_last = after_image;
                            after_state_last = before_state.clone().detach();
                            after_pick_point_last = before_pick_point_tensor.clone().detach();
                            for(int i=0; i<3; i++){
                                grasp_last[i] = grasp_before[i];
                                release_last[i] = release_before[i];
                            }
                            Rz_last = Rz_before;
                            grasp_point_last = grasp_point_before;
                            release_point_last = release_point_before;
                            max_height_last = max_height_before;
                            garment_area_last = garment_area_before;
                            conf_last = conf_before;
                            qDebug() << "\033[1;31mExceeds limit\033[0m";
                        }

                        if (garment_unfolded == false && !exceed_limit){
                            after_image_last = after_image;
                            after_state_last = after_state.clone().detach();
                            after_pick_point_last = after_pick_point_tensor.clone().detach();
                            for(int i=0; i<3; i++){
                                grasp_last[i] = grasp[i];
                                release_last[i] = release[i];
                            }
                            Rz_last = Rz;
                            grasp_point_last = grasp_point;
                            release_point_last = release_point;
                            max_height_last = max_height_after;
                            garment_area_last = garment_area_after;
                            conf_last = conf_after;
                        }

                        cv::Mat sub_image;
                        sub_image = warped_image_copy - warped_image;
                        auto mean = cv::mean(sub_image);
                        //qDebug() << "Pixel color diff mean: "<< mean[0] << " "<<mean[1]<< " "<< mean[2];
                        if(mean[0]<1.0 && mean[1]<1.0 && mean[2]<1.0 && done[0]!=1.0){
                            qDebug() << "\033[0;33mNothing Changed(mean<1.0), mean: " << mean[0] << " " << mean[1] << " " << mean[2] << " redo step\033[0m";
                            torch::AutoGradMode reset_disable(false);
                            if(test_model){
                                reset_env = true;
                                continue;
                            }
                            failtimes ++;
                            if(failtimes > 10){
                                bResetRobot = true;
                            }
                            if ( bResetRobot ) {
                                QMetaObject::invokeMethod(this, "resetRViz",
                                                          Qt::BlockingQueuedConnection);

                                qDebug() << "\033[1;31mResetting Rviz.\033[0m";

                                QDir dir(QString("%1/%2").arg(memoryPath).arg(datasize--));
                                if (!dir.removeRecursively()) {
                                    qCritical() << "[Warning] Useless data cannot be deleted : " << datasize;
                                }
                                total_steps--;

                                if (!resetRobotPosition()){
                                    mRunReinforcementLearning1 = false;
                                    qDebug() << "\033[1;31mCan not reset Rviz, end training.\033[0m";
                                    break;
                                }
                                QThread::msleep(6000);  //Wait for the robot to reset

                                qDebug() << QString("\n-----[ %1 ]-----\n")
                                            .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                            .toUtf8().data();
                                break;
                            }
                            continue;
                        } else {
                            failtimes = 0;
                        }

                        qDebug() << "\033[1;31mStep reward: " << stepreward[0] << "\033[0m";
                        episode_reward += stepreward[0];

                        // Only save data with reward <=-100 and >200
//                        if (total_steps < START_STEP && stepreward[0] > -100 && stepreward[0] < 200){
//                            qDebug() << "\033[1;31mReward between -100 to 200, redo.\033[0m";
//                            continue;
//                        }

                        if(test_model){
                            auto future_savedata = QtConcurrent::run([before_image = before_image,
                                                                     before_pick_point_CPU = before_pick_point_tensor.clone().detach(),
                                                                     place_point_CPU = place_point_tensor.clone().detach(),
                                                                     stepreward = stepreward,
                                                                     after_image = after_image,
                                                                     drawing = drawing_copy,
                                                                     garment_area_before = garment_area_before,
                                                                     garment_area_after = garment_area_after,
                                                                     max_height_before = max_height_before,
                                                                     max_height_after = max_height_after,
                                                                     episode = episode,
                                                                     step = step,
                                                                     this
                                                                     ](){

                                qDebug() << "Saving data";

                                std::vector<float> before_pick_point_vector(before_pick_point_CPU.data_ptr<float>(), before_pick_point_CPU.data_ptr<float>() + before_pick_point_CPU.numel());
                                std::vector<float> place_point_vector(place_point_CPU.data_ptr<float>(), place_point_CPU.data_ptr<float>() + place_point_CPU.numel());
                                std::vector<float> garment_area_before_vec{garment_area_before};
                                std::vector<float> garment_area_after_vec{garment_area_after};
                                std::vector<float> max_height_before_vec{max_height_before};
                                std::vector<float> max_height_after_vec{max_height_after};


                                QString filename_id = QString(testPath + "/%1").arg(episode);
                                QDir().mkdir(filename_id);
                                QString filename_id2 = QString(filename_id + "/%1").arg(step);
                                QDir().mkdir(filename_id2);

                                QString filename_before_image = QString(filename_id2 + "/before_image.jpg");
                                QByteArray filename_before_imageqb = filename_before_image.toLocal8Bit();
                                const char *filename_before_imagechar = filename_before_imageqb.data();
                                cv::imwrite(filename_before_imagechar, before_image);

                                QString filename_before_pick_point = QString(filename_id2 + "/before_pick_point.txt");
                                savedata(filename_before_pick_point, before_pick_point_vector);

                                QString filename_place_point = QString(filename_id2 + "/place_point.txt");
                                savedata(filename_place_point, place_point_vector);

                                QString filename_reward = QString(filename_id2 + "/reward.txt");
                                savedata(filename_reward, stepreward);

                                QString filename_after_image = QString(filename_id2 + "/after_image.jpg");
                                QByteArray filename_after_imageqb = filename_after_image.toLocal8Bit();
                                const char *filename_after_imagechar = filename_after_imageqb.data();
                                cv::imwrite(filename_after_imagechar, after_image);

                                QString filename_drawing = QString(filename_id2 + "/drawing.jpg");
                                QByteArray filename_drawingqb = filename_drawing.toLocal8Bit();
                                const char *filename_drawingchar = filename_drawingqb.data();
                                cv::imwrite(filename_drawingchar, drawing);

                                QString filename_garment_area_before = QString(filename_id2 + "/garment_area_before.txt");
                                savedata(filename_garment_area_before, garment_area_before_vec);

                                QString filename_garment_area_after = QString(filename_id2 + "/garment_area_after.txt");
                                savedata(filename_garment_area_after, garment_area_after_vec);

                                QString filename_max_height_before= QString(filename_id2 + "/max_height_before.txt");
                                savedata(filename_max_height_before, max_height_before_vec);

                                QString filename_max_height_after = QString(filename_id2 + "/max_height_after.txt");
                                savedata(filename_max_height_after, max_height_after_vec);
                            });
                            future_savedata.waitForFinished();
                        } else {
                            auto reward_tensor = torch::from_blob(stepreward.data(), { 1 }, torch::kFloat);
                            //std::cout << "reward_tensor: " << reward_tensor << std::endl;
                            done_tensor = torch::from_blob(done.data(), { 1 }, torch::kFloat);
                            //std::cout << "done_tensor: " << done_tensor << std::endl;

        //                    if (memory.size() >= 10000) {
        //                        memory.pop_front();
        //                    }

    //                        memory.push_back({
    //                            before_state.clone().detach(),
    //                            before_pick_point_tensor.clone().detach(),
    //                            place_point_tensor.clone().detach(),
    //                            reward_tensor.clone().detach(),
    //                            done_tensor.clone().detach(),
    //                            after_state.clone().detach(),
    //                            after_pick_point_tensor.clone().detach()
    //                            });

            //                    std::cout << "before_state: " << memory[step].before_state.mean() << std::endl
            //                              << "place_point_tensor: " << memory[step].place_point_tensor << std::endl
            //                              << "reward_tensor: " << memory[step].reward_tensor << std::endl
            //                              << "done_tensor: " << memory[step].done_tensor << std::endl
            //                              << "after_state: " << memory[step].after_state.mean() << std::endl;

    //                        std::cout << "before_state: " << before_state.sizes() << std::endl;
    //                        std::cout << "before_pick_point_tensor: " << before_pick_point_tensor.sizes() << std::endl;
    //                        std::cout << "place_point_tensor: " << place_point_tensor.sizes() << std::endl;
    //                        std::cout << "reward_tensor: " << reward_tensor.sizes() << std::endl;
    //                        std::cout << "done_tensor: " << done_tensor.sizes() << std::endl;
    //                        std::cout << "after_state: " << after_state.sizes() << std::endl;
    //                        std::cout << "after_pick_point_tensor: " << after_pick_point_tensor.sizes() << std::endl;

                            auto future_savedata = QtConcurrent::run([before_image = before_image,
                                                                     before_state_CPU = before_state.clone().detach(),
                                                                     before_pick_point_CPU = before_pick_point_tensor.clone().detach(),
                                                                     place_point_CPU = place_point_tensor.clone().detach(),
                                                                     reward_CPU = reward_tensor.clone().detach(),
                                                                     done_CPU = done_tensor.clone().detach(),
                                                                     after_image = after_image,
                                                                     after_state_CPU = after_state.clone().detach(),
                                                                     after_pick_point_CPU = after_pick_point_tensor.clone().detach(),
                                                                     drawing = drawing_copy,
                                                                     garment_area_before = garment_area_before,
                                                                     garment_area_after = garment_area_after,
                                                                     max_height_before = max_height_before,
                                                                     max_height_after = max_height_after,
                                                                     datasize = datasize,
                                                                     this
                                                                     ](){

                                qDebug() << "Saving data";

                                std::vector<float> before_state_vector(before_state_CPU.data_ptr<float>(), before_state_CPU.data_ptr<float>() + before_state_CPU.numel());
                                std::vector<float> before_pick_point_vector(before_pick_point_CPU.data_ptr<float>(), before_pick_point_CPU.data_ptr<float>() + before_pick_point_CPU.numel());
                                std::vector<float> place_point_vector(place_point_CPU.data_ptr<float>(), place_point_CPU.data_ptr<float>() + place_point_CPU.numel());
                                std::vector<float> reward_vector(reward_CPU.data_ptr<float>(), reward_CPU.data_ptr<float>() + reward_CPU.numel());
                                std::vector<float> done_vector(done_CPU.data_ptr<float>(), done_CPU.data_ptr<float>() + done_CPU.numel());
                                std::vector<float> after_state_vector(after_state_CPU.data_ptr<float>(), after_state_CPU.data_ptr<float>() + after_state_CPU.numel());
                                std::vector<float> after_pick_point_vector(after_pick_point_CPU.data_ptr<float>(), after_pick_point_CPU.data_ptr<float>() + after_pick_point_CPU.numel());
                                std::vector<float> garment_area_before_vec{garment_area_before};
                                std::vector<float> garment_area_after_vec{garment_area_after};
                                std::vector<float> max_height_before_vec{max_height_before};
                                std::vector<float> max_height_after_vec{max_height_after};

                                QString filename_id = QString(memoryPath + "/%1").arg(datasize);
                                QDir().mkdir(filename_id);

                                QString filename_before_image = QString(filename_id + "/before_image.jpg");
                                QByteArray filename_before_imageqb = filename_before_image.toLocal8Bit();
                                const char *filename_before_imagechar = filename_before_imageqb.data();
                                cv::imwrite(filename_before_imagechar, before_image);

                                QString filename_before_state = QString(filename_id + "/before_state.txt");
                                savedata(filename_before_state, before_state_vector);
                                QString filename_before_state_tensor = QString(filename_id + "/before_state_tensor.pt");
                                torch::save(before_state_CPU, filename_before_state_tensor.toStdString());

                                QString filename_before_pick_point = QString(filename_id + "/before_pick_point.txt");
                                savedata(filename_before_pick_point, before_pick_point_vector);
                                QString filename_before_pick_point_tensor = QString(filename_id + "/before_pick_point_tensor.pt");
                                torch::save(before_pick_point_CPU, filename_before_pick_point_tensor.toStdString());

                                QString filename_place_point = QString(filename_id + "/place_point.txt");
                                savedata(filename_place_point, place_point_vector);
                                QString filename_place_point_tensor = QString(filename_id + "/place_point_tensor.pt");
                                torch::save(place_point_CPU, filename_place_point_tensor.toStdString());

                                QString filename_reward = QString(filename_id + "/reward.txt");
                                savedata(filename_reward, reward_vector);
                                QString filename_reward_tensor = QString(filename_id + "/reward_tensor.pt");
                                torch::save(reward_CPU, filename_reward_tensor.toStdString());

                                QString filename_done = QString(filename_id + "/done.txt");
                                savedata(filename_done, done_vector);
                                QString filename_done_tensor = QString(filename_id + "/done_tensor.pt");
                                torch::save(done_CPU, filename_done_tensor.toStdString());

                                QString filename_after_image = QString(filename_id + "/after_image.jpg");
                                QByteArray filename_after_imageqb = filename_after_image.toLocal8Bit();
                                const char *filename_after_imagechar = filename_after_imageqb.data();
                                cv::imwrite(filename_after_imagechar, after_image);

                                QString filename_after_state = QString(filename_id + "/after_state.txt");
                                savedata(filename_after_state, after_state_vector);
                                QString filename_after_state_tensor = QString(filename_id + "/after_state_tensor.pt");
                                torch::save(after_state_CPU, filename_after_state_tensor.toStdString());

                                QString filename_after_pick_point = QString(filename_id + "/after_pick_point.txt");
                                savedata(filename_after_pick_point, after_pick_point_vector);
                                QString filename_after_pick_point_tensor = QString(filename_id + "/after_pick_point_tensor.pt");
                                torch::save(after_pick_point_CPU, filename_after_pick_point_tensor.toStdString());

                                QString filename_drawing = QString(filename_id + "/drawing.jpg");
                                QByteArray filename_drawingqb = filename_drawing.toLocal8Bit();
                                const char *filename_drawingchar = filename_drawingqb.data();
                                cv::imwrite(filename_drawingchar, drawing);

                                QString filename_garment_area_before = QString(filename_id + "/garment_area_before.txt");
                                savedata(filename_garment_area_before, garment_area_before_vec);

                                QString filename_garment_area_after = QString(filename_id + "/garment_area_after.txt");
                                savedata(filename_garment_area_after, garment_area_after_vec);

                                QString filename_max_height_before= QString(filename_id + "/max_height_before.txt");
                                savedata(filename_max_height_before, max_height_before_vec);

                                QString filename_max_height_after = QString(filename_id + "/max_height_after.txt");
                                savedata(filename_max_height_after, max_height_after_vec);
                            });
                            future_savedata.waitForFinished();
                            datasize++;
                        }
                    }

                    step++;
                    if(test_model){
                        qDebug() << "\033[0;34mTesting model : Step [" << step << "/" << teststep << "] finished\033[0m";
                    } else {
                        total_steps++;

                        std::vector<float> totalsteps;
                        totalsteps.push_back(total_steps);
                        QString filename_totalsteps = QString(memoryPath + "/totalsteps.txt");
                        savedata(filename_totalsteps, totalsteps);

                        if (total_steps > batch_size) {
                            torch::AutoGradMode enable(true);
                            for(int train=0; train<TRAINEVERY; train++){
                                train_number++;
                                if(done_old_data){
                                    qDebug() << "\033[1;33mTraining model: [" << train+1 << "/" << TRAINEVERY << "]\033[0m";
                                }
                                int randomdata = rand()%(10000-batch_size+1);
                                randomdata = total_steps - 10000 + randomdata;
                                //qDebug() << "randomdata: " << randomdata;
                                //qDebug() << "memory size: " << memory.size();
                                std::vector<torch::Tensor> s_data(batch_size), p_data(batch_size), a_data(batch_size), r_data(batch_size), d_data(batch_size), s2_data(batch_size), p2_data(batch_size);
                                torch::Tensor s_batch, p_batch, a_batch, r_batch, d_batch, s2_batch, p2_batch;

                                for (int i = 0; i < batch_size; i++) {
                                    QString filename_id;
                                    filename_id = QString(memoryPath + "/%1").arg(i+randomdata);

                                    QString filename_s = QString(filename_id + "/before_state_tensor.pt");
                                    torch::Tensor tmp_s_data;
                                    torch::load(tmp_s_data, filename_s.toStdString());

                                    QString filename_p = QString(filename_id + "/before_pick_point_tensor.pt");
                                    torch::Tensor tmp_p_data;
                                    torch::load(tmp_p_data, filename_p.toStdString());

                                    QString filename_a = QString(filename_id + "/place_point_tensor.pt");
                                    torch::Tensor tmp_a_data;
                                    torch::load(tmp_a_data, filename_a.toStdString());

                                    QString filename_r = QString(filename_id + "/reward_tensor.pt");
                                    torch::Tensor tmp_r_data;
                                    torch::load(tmp_r_data, filename_r.toStdString());

                                    QString filename_d = QString(filename_id + "/done_tensor.pt");
                                    torch::Tensor tmp_d_data;
                                    torch::load(tmp_d_data, filename_d.toStdString());

                                    QString filename_s2 = QString(filename_id + "/after_state_tensor.pt");
                                    torch::Tensor tmp_s2_data;
                                    torch::load(tmp_s2_data, filename_s2.toStdString());

                                    QString filename_p2 = QString(filename_id + "/after_pick_point_tensor.pt");
                                    torch::Tensor tmp_p2_data;
                                    torch::load(tmp_p2_data, filename_p2.toStdString());

                                    s_data[i] = torch::unsqueeze(tmp_s_data.clone().detach(), 0);
                                    p_data[i] = torch::unsqueeze(tmp_p_data.clone().detach(), 0);
                                    a_data[i] = torch::unsqueeze(tmp_a_data.clone().detach(), 0);
                                    r_data[i] = torch::unsqueeze(tmp_r_data.clone().detach(), 0);
                                    d_data[i] = torch::unsqueeze(tmp_d_data.clone().detach(), 0);
                                    s2_data[i] = torch::unsqueeze(tmp_s2_data.clone().detach(), 0);
                                    p2_data[i] = torch::unsqueeze(tmp_p2_data.clone().detach(), 0);
                                }

                                s_batch = s_data[0]; p_batch = p_data[0]; a_batch = a_data[0]; r_batch = r_data[0]; d_batch = d_data[0]; s2_batch = s2_data[0]; p2_batch = p2_data[0];
                                for (int i = 1; i < batch_size; i++) {
                                    s_batch = torch::cat({ s_batch, s_data[i] }, 0);
                                    p_batch = torch::cat({ p_batch, p_data[i] }, 0);
                                    a_batch = torch::cat({ a_batch, a_data[i] }, 0);
                                    r_batch = torch::cat({ r_batch, r_data[i] }, 0);
                                    d_batch = torch::cat({ d_batch, d_data[i] }, 0);
                                    s2_batch = torch::cat({ s2_batch, s2_data[i] }, 0);
                                    p2_batch = torch::cat({ p2_batch, p2_data[i] }, 0);
                                }
                                s_batch = s_batch.clone().detach().to(device);
                                p_batch = p_batch.clone().detach().to(device);
                                a_batch = a_batch.clone().detach().to(device);
                                r_batch = r_batch.clone().detach().to(device);
                                d_batch = d_batch.clone().detach().to(device);
                                s2_batch = s2_batch.clone().detach().to(device);
                                p2_batch = p2_batch.clone().detach().to(device);

                                // Q-value networks training
                                if(done_old_data){
                                    qDebug() << "\033[1;33mTraining Q-value networks\033[0m";
                                }

                                torch::AutoGradMode q_enable(true);

                                torch::Tensor q1 = actor_critic->q1->forward_pick_point(s_batch, p_batch, a_batch);
                                torch::Tensor q2 = actor_critic->q2->forward_pick_point(s_batch, p_batch, a_batch);

                                torch::AutoGradMode disable(false);
                                // Target actions come from *current* policy
                                policy_output next_state_sample = actor_critic->pi->forward_pick_point(s2_batch, p2_batch, false, true);
                                torch::Tensor a2_batch = next_state_sample.action;
                                torch::Tensor logp_a2 = next_state_sample.logp_pi;
                                // Target Q-values
                                torch::Tensor q1_pi_target = actor_critic_target->q1->forward_pick_point(s2_batch, p2_batch, a2_batch);
                                torch::Tensor q2_pi_target = actor_critic_target->q2->forward_pick_point(s2_batch, p2_batch, a2_batch);
                                torch::Tensor backup = r_batch + GAMMA * (1.0 - d_batch) * (torch::min(q1_pi_target, q2_pi_target) - ALPHA * logp_a2);

                                // MSE loss against Bellman backup
                                torch::AutoGradMode loss_enable(true);
                                torch::Tensor loss_q1 = torch::mean(pow(q1 - backup, 2));
                                torch::Tensor loss_q2 = torch::mean(pow(q2 - backup, 2));
                                torch::Tensor loss_q = loss_q1 + loss_q2;

                                float loss_q1f = loss_q1.detach().item().toFloat();
                                float loss_q2f = loss_q2.detach().item().toFloat();
                                episode_critic1_loss += loss_q1f;
                                episode_critic2_loss += loss_q2f;

                                critic_optimizer.zero_grad();
                                loss_q.backward();
                                critic_optimizer.step();

                                if(done_old_data){
                                    qDebug() << "\033[1;33mCritic optimizer step\033[0m";
                                    qDebug() << "\033[1;33mTraining policy network\033[0m";
                                }

                                // Policy network training
                                for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
                                    actor_critic->q1->parameters()[i].set_requires_grad(false);
                                }
                                for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
                                    actor_critic->q2->parameters()[i].set_requires_grad(false);
                                }

                                policy_output sample = actor_critic->pi->forward_pick_point(s_batch, p_batch, false, true);
                                torch::Tensor pi = sample.action;
                                torch::Tensor log_pi = sample.logp_pi;
                                torch::Tensor q1_pi = actor_critic->q1->forward_pick_point(s_batch, p_batch, pi);
                                torch::Tensor q2_pi = actor_critic->q2->forward_pick_point(s_batch, p_batch, pi);
                                torch::Tensor q_pi = torch::min(q1_pi, q2_pi);

                                // Entropy-regularized policy loss
                                torch::Tensor loss_pi = torch::mean(ALPHA * log_pi - q_pi); // Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

                                float loss_pif = loss_pi.detach().item().toFloat();
                                episode_policy_loss += loss_pif;

                                policy_optimizer.zero_grad();
                                loss_pi.backward();
                                policy_optimizer.step();

                                for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
                                    actor_critic->q1->parameters()[i].set_requires_grad(true);
                                }
                                for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
                                    actor_critic->q2->parameters()[i].set_requires_grad(true);
                                }

                                if(done_old_data){
                                    qDebug() << "\033[1;33mPolicy optimizer step\033[0m";
                                    qDebug() << "\033[1;33mUpdating target models\033[0m";
                                }

                                // Update target networks
                                torch::AutoGradMode softcopy_disable(false);
                                for (size_t i = 0; i < actor_critic_target->pi->parameters().size(); i++) {
                                    actor_critic_target->pi->parameters()[i].mul_(POLYAK);
                                    actor_critic_target->pi->parameters()[i].add_((1.0 - POLYAK) * actor_critic->pi->parameters()[i]);
                                }
                                for (size_t i = 0; i < actor_critic_target->q1->parameters().size(); i++) {
                                    actor_critic_target->q1->parameters()[i].mul_(POLYAK);
                                    actor_critic_target->q1->parameters()[i].add_((1.0 - POLYAK) * actor_critic->q1->parameters()[i]);
                                }
                                for (size_t i = 0; i < actor_critic_target->q2->parameters().size(); i++) {
                                    actor_critic_target->q2->parameters()[i].mul_(POLYAK);
                                    actor_critic_target->q2->parameters()[i].add_((1.0 - POLYAK) * actor_critic->q2->parameters()[i]);
                                }
                                torch::AutoGradMode softcopy_enable(true);

                                if(done_old_data){
                                    qDebug() << "\033[1;31mCritic 1 loss: " << loss_q1f << "\n"
                                             << "Critic 2 loss: " << loss_q2f << "\n"
                                             << "Policy loss: " << loss_pif << "\n"
                                             << "Episode reward: " << episode_reward << "\033[0m";
                                }
                            }
                        }
                        qDebug() << "\033[0;34mEpisode " << episode+1 << ": Step [" << step << "/" << maxstep << "] finished\033[0m";
                    }

                    if(!done_old_data && total_steps+1 > datasize){
                        done_old_data = true;
                        qDebug() << "\033[1;31mDone training old data, start plan\033[0m";
                        break;
                    }
                    if(garment_unfolded == true){
                        break;
                    }
                    if(test_model && step == teststep){
                        break;
                    }
                }

                // Save
                if(test_model){
                    qDebug() << "\033[0;35m--------------------------------------------" << "\n"
                             << "Reward: " << episode_reward << "\n"
                             << "--------------------------------------------\033[0m";
                    logger.add_scalar("Test_Reward", episode, episode_reward);
                    qDebug() << "\033[0;34mTest model finished\033[0m\n"
                             << "--------------------------------------------";
                    test_model = false;
                } else if (bResetRobot){
                    qDebug() << "\033[0;35m--------------------------------------------" << "\n"
                             << "Restart episode" << episode+1 << "\n"
                             << "--------------------------------------------\033[0m";
                } else {
                    episode++;

                    episode_critic1_loss = episode_critic1_loss / (float)train_number;
                    episode_critic2_loss = episode_critic2_loss / (float)train_number;
                    episode_policy_loss = episode_policy_loss / (float)train_number;

                    qDebug() << "\033[0;35m--------------------------------------------" << "\n"
                        << "Episode: " << episode << "\n"
                        << "Reward: " << episode_reward << "\n"
                        << "Critic 1 Loss: " << episode_critic1_loss << "\n"
                        << "Critic 2 Loss: " << episode_critic2_loss << "\n"
                        << "Policy Loss: " << episode_policy_loss << "\n"
                        << "--------------------------------------------\033[0m";
                    logger.add_scalar("Episode_Reward", episode, episode_reward);
                    logger.add_scalar("Episode_Critic_1_Loss", episode, episode_critic1_loss);
                    logger.add_scalar("Episode_Critic_2_Loss", episode, episode_critic2_loss);
                    logger.add_scalar("Episode_Policy_Loss", episode, episode_policy_loss);

                    int save = SAVEMODELEVERY;
                    if(!done_old_data){
                        save = 50;
                    }
                    if (episode % save == 0) {
                        qDebug() << "Saving models";

                        QString pi_para_path = QString(modelPath + "/pi_para/pi_para_" + QString::number(episode) + ".pt");
                        QString q1_para_path = QString(modelPath + "/q1_para/q1_para_" + QString::number(episode) + ".pt");
                        QString q2_para_path = QString(modelPath + "/q2_para/q2_para_" + QString::number(episode) + ".pt");
                        QString target_pi_para_path = QString(modelPath + "/target_pi_para/target_pi_para_" + QString::number(episode) + ".pt");
                        QString target_q1_para_path = QString(modelPath + "/target_q1_para/target_q1_para_" + QString::number(episode) + ".pt");
                        QString target_q2_para_path = QString(modelPath + "/target_q2_para/target_q2_para_" + QString::number(episode) + ".pt");
                        QString policy_opti_path = QString(modelPath + "/policy_optimizer/policy_optimizer_" + QString::number(episode) + ".pt");
                        QString critic_opti_path = QString(modelPath + "/critic_optimizer/critic_optimizer_" + QString::number(episode) + ".pt");

                        torch::save(actor_critic->pi->parameters(), pi_para_path.toStdString());
                        torch::save(actor_critic->q1->parameters(), q1_para_path.toStdString());
                        torch::save(actor_critic->q2->parameters(), q2_para_path.toStdString());
                        torch::save(actor_critic_target->pi->parameters(), target_pi_para_path.toStdString());
                        torch::save(actor_critic_target->q1->parameters(), target_q1_para_path.toStdString());
                        torch::save(actor_critic_target->q2->parameters(), target_q2_para_path.toStdString());
                        torch::save(policy_optimizer, policy_opti_path.toStdString());
                        torch::save(critic_optimizer, critic_opti_path.toStdString());

                        std::vector<float> save_episode_num;
                        save_episode_num.push_back(episode+1);
                        QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
                        savedata(filename_episode_num, save_episode_num);

                        qDebug() << "Models saved";
                    }

                    qDebug() << "\033[0;34mEpisode " << episode << "finished\033[0m\n"
                             << "--------------------------------------------";
                    if(episode % 10 == 0){
                        test_model = true;
                    }
                }

                if(mRunReinforcementLearning1 == false){
                    qDebug() << "Quit Reinforcement Learning 1" ;
                    qDebug() << "Saving models";

                    QString pi_para_path = QString(modelPath + "/pi_para/pi_para_" + QString::number(episode) + ".pt");
                    QString q1_para_path = QString(modelPath + "/q1_para/q1_para_" + QString::number(episode) + ".pt");
                    QString q2_para_path = QString(modelPath + "/q2_para/q2_para_" + QString::number(episode) + ".pt");
                    QString target_pi_para_path = QString(modelPath + "/target_pi_para/target_pi_para_" + QString::number(episode) + ".pt");
                    QString target_q1_para_path = QString(modelPath + "/target_q1_para/target_q1_para_" + QString::number(episode) + ".pt");
                    QString target_q2_para_path = QString(modelPath + "/target_q2_para/target_q2_para_" + QString::number(episode) + ".pt");
                    QString policy_opti_path = QString(modelPath + "/policy_optimizer/policy_optimizer_" + QString::number(episode) + ".pt");
                    QString critic_opti_path = QString(modelPath + "/critic_optimizer/critic_optimizer_" + QString::number(episode) + ".pt");

                    torch::save(actor_critic->pi->parameters(), pi_para_path.toStdString());
                    torch::save(actor_critic->q1->parameters(), q1_para_path.toStdString());
                    torch::save(actor_critic->q2->parameters(), q2_para_path.toStdString());
                    torch::save(actor_critic_target->pi->parameters(), target_pi_para_path.toStdString());
                    torch::save(actor_critic_target->q1->parameters(), target_q1_para_path.toStdString());
                    torch::save(actor_critic_target->q2->parameters(), target_q2_para_path.toStdString());
                    torch::save(policy_optimizer, policy_opti_path.toStdString());
                    torch::save(critic_optimizer, critic_opti_path.toStdString());

                    std::vector<float> save_episode_num;
                    save_episode_num.push_back(episode+1);
                    QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
                    savedata(filename_episode_num, save_episode_num);

                    qDebug() << "Models saved";
                    break;
                }
            }
        } catch (const std::exception &e) {
            auto &&msg = torch::GetExceptionString(e);
            qWarning() << msg.c_str();
        } catch (...) {
            qCritical() << "GG";
        }
    });
}


void LP_Plugin_Garment_Manipulation::Train_rod(){
    auto rl1current = QtConcurrent::run([this](){
        try {
            torch::manual_seed(0);

            device = torch::Device(torch::kCPU);
            if (torch::cuda::is_available()) {
                std::cout << "CUDA is available! Training on GPU." << std::endl;
                device = torch::Device(torch::kCUDA);
            }

            torch::autograd::DetectAnomalyGuard detect_anomaly;

            qDebug() << "Creating models";

            std::vector<int> policy_mlp_dims{STATE_DIM, 64, 64};
            std::vector<int> critic_mlp_dims{STATE_DIM + ACT_DIM, 64, 64};

            auto actor_critic = ActorCritic(policy_mlp_dims, critic_mlp_dims);
            auto actor_critic_target = ActorCritic(policy_mlp_dims, critic_mlp_dims);

            qDebug() << "Creating optimizer";

            torch::AutoGradMode copy_disable(false);

            std::vector<torch::Tensor> q_params;
            for(size_t i=0; i<actor_critic->q1->parameters().size(); i++){
                q_params.push_back(actor_critic->q1->parameters()[i]);
            }
            for(size_t i=0; i<actor_critic->q2->parameters().size(); i++){
                q_params.push_back(actor_critic->q2->parameters()[i]);
            }
            torch::AutoGradMode copy_enable(true);

            torch::optim::Adam policy_optimizer(actor_critic->pi->parameters(), torch::optim::AdamOptions(lrp));
            torch::optim::Adam critic_optimizer(q_params, torch::optim::AdamOptions(lrc));

            actor_critic->pi->to(device);
            actor_critic->q1->to(device);
            actor_critic->q2->to(device);
            actor_critic_target->pi->to(device);
            actor_critic_target->q1->to(device);
            actor_critic_target->q2->to(device);

            GOOGLE_PROTOBUF_VERIFY_VERSION;
            TensorBoardLogger logger(kLogFile.c_str());

            // Mode
            bool train_two_rod = false;
            bool test_mode = true;
            bool LoadOldData = false;
            bool RestoreFromCheckpoint = true;

            int load_model = 0;
            int episode = 0;
            int datasize = 0;

            if(test_mode){
                RestoreFromCheckpoint = true;
                load_model = 5000;
                qDebug() << "TEST MODE: episode " << load_model;
            }
            if(RestoreFromCheckpoint || LoadOldData){
//                int olddata_size = 0;

                if(LoadOldData){
                    qDebug() << "Load old data";
                    for (const auto & file : std::filesystem::directory_iterator(memoryPath.toStdString())){
                        datasize++;
                    }
                    //datasize -= 2;
                    qDebug() << "Data size: " << datasize;

                } else {
                    qDebug() << "Restore from check point";

                    QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
                    std::vector<float> saved_episode_num;
                    loaddata(filename_episode_num.toStdString(), saved_episode_num);
                    episode = saved_episode_num[0]-1;
                    maxepisode += episode;

                    QString filename_totalsteps = QString(memoryPath + "/totalsteps.txt");
                    std::vector<float> totalsteps;
                    loaddata(filename_totalsteps.toStdString(), totalsteps);
                    total_steps = int(totalsteps[0]);
                    qDebug() << "Total steps: " << total_steps;

                    for (const auto & file : std::filesystem::directory_iterator(memoryPath.toStdString())){
                        datasize++;
                    }
                    datasize -= 2;
                    qDebug() << "Data size: " << datasize;

                    if(test_mode){
                        episode = load_model;
                    }
                    qDebug() << "Loading models " << episode;
                    QString pi_para_path = QString(modelPath + "/pi_para/pi_para_" + QString::number(episode) + ".pt");
                    QString q1_para_path = QString(modelPath + "/q1_para/q1_para_" + QString::number(episode) + ".pt");
                    QString q2_para_path = QString(modelPath + "/q2_para/q2_para_" + QString::number(episode) + ".pt");
                    QString target_pi_para_path = QString(modelPath + "/target_pi_para/target_pi_para_" + QString::number(episode) + ".pt");
                    QString target_q1_para_path = QString(modelPath + "/target_q1_para/target_q1_para_" + QString::number(episode) + ".pt");
                    QString target_q2_para_path = QString(modelPath + "/target_q2_para/target_q2_para_" + QString::number(episode) + ".pt");
                    QString policy_opti_path = QString(modelPath + "/policy_optimizer/policy_optimizer_" + QString::number(episode) + ".pt");
                    QString critic_opti_path = QString(modelPath + "/critic_optimizer/critic_optimizer_" + QString::number(episode) + ".pt");

                    std::vector<torch::Tensor> pi_para, q1_para, q2_para, target_pi_para, target_q1_para, target_q2_para;

                    torch::load(pi_para, pi_para_path.toStdString());
                    torch::load(q1_para, q1_para_path.toStdString());
                    torch::load(q2_para, q2_para_path.toStdString());
                    torch::load(target_pi_para, target_pi_para_path.toStdString());
                    torch::load(target_q1_para, target_q1_para_path.toStdString());
                    torch::load(target_q2_para, target_q2_para_path.toStdString());
                    torch::load(policy_optimizer, policy_opti_path.toStdString());
                    torch::load(critic_optimizer, critic_opti_path.toStdString());

                    torch::AutoGradMode data_copy_disable(false);

                    for(size_t i=0; i < actor_critic->pi->parameters().size(); i++){
                        actor_critic->pi->parameters()[i].copy_(pi_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
                        actor_critic->q1->parameters()[i].copy_(q1_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
                        actor_critic->q2->parameters()[i].copy_(q2_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic_target->pi->parameters().size(); i++){
                        actor_critic_target->pi->parameters()[i].copy_(target_pi_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic_target->q1->parameters().size(); i++){
                        actor_critic_target->q1->parameters()[i].copy_(target_q1_para[i].clone().detach().to(device));
                    }
                    for(size_t i=0; i < actor_critic_target->q2->parameters().size(); i++){
                        actor_critic_target->q2->parameters()[i].copy_(target_q2_para[i].clone().detach().to(device));
                    }
                    torch::AutoGradMode data_copy_enable(true);
                }
            }

            if(!RestoreFromCheckpoint){
                qDebug() << "Copying parameters to target models";
                torch::AutoGradMode hardcopy_disable(false);
                for(size_t i=0; i < actor_critic_target->pi->parameters().size(); i++){
                    actor_critic_target->pi->parameters()[i].copy_(actor_critic->pi->parameters()[i]);
                    actor_critic_target->pi->parameters()[i].set_requires_grad(false);
                }
                for(size_t i=0; i < actor_critic_target->q1->parameters().size(); i++){
                    actor_critic_target->q1->parameters()[i].copy_(actor_critic->q1->parameters()[i]);
                    actor_critic_target->q1->parameters()[i].set_requires_grad(false);
                }
                for(size_t i=0; i < actor_critic_target->q2->parameters().size(); i++){
                    actor_critic_target->q2->parameters()[i].copy_(actor_critic->q2->parameters()[i]);
                    actor_critic_target->q2->parameters()[i].set_requires_grad(false);
                }
                torch::AutoGradMode hardcopy_enable(true);
            }

            int step = 0, train_number = 0, failtimes;
            float episode_reward = 0, episode_critic1_loss = 0, episode_critic2_loss = 0, episode_policy_loss = 0;
            std::vector<float> done(1);
            torch::Tensor done_tensor;
            rs2::frame depth;
            if(use_filter){
                depth = filtered_frame;
            } else {
                depth = frames.get_depth_frame();
            }
            total_reward = 0; total_critic_loss = 0; total_policy_loss = 0;
            bool bResetRobot = false, task_done = false, test_model = false, use_clock = true;
            std::vector<int> done_count, test_done_count;

            imageWidth = warped_image.cols;
            imageHeight = warped_image.rows;

            while (episode < maxepisode) {
                qDebug() << "--------------------------------------------";
                if(test_mode){
                    test_model = true;
                    episode++;
                }
                if(test_model){
                    qDebug() << "\033[0;34mTest model\033[0m";
                } else {
                    qDebug() << "\033[0;34mEpisode " << episode+1 << " started\033[0m";
                }

                // Initialize environment
                episode_reward = 0;
                episode_critic1_loss = 0;
                episode_critic2_loss = 0;
                episode_policy_loss = 0;
                done[0] = 0;
                step = 0;
                train_number = 0;
                failtimes = 0;
                bResetRobot = false;
                task_done = false;
                cv::Mat before_image, after_image;
                torch::Tensor before_state, action_tensor, after_state;


                while (step < maxstep && mRunTrainRod && done[0]==0) {
                    //std::cout << "p fc3: \n" << policy->fc3->parameters() << std::endl;
                    //std::cout << "p fc4: \n" << policy->fc4->parameters() << std::endl;
                    qDebug() << "--------------------------------------------";
                    qDebug() << QString("\n-----[ %1 ]-----\n")
                                .arg(QDateTime::currentDateTime().toString("hh:mm:ss dd-MM-yyyy"))
                                .toUtf8().data();

                    if(LoadOldData && !test_model){
                        qDebug() << "\033[0;34mTraining old data: " << total_steps+1 << "/" << datasize << "\033[0m";
                    } else {
                        if(test_model){
                            qDebug() << "\033[0;34mTesting model : Step [" << step+1 << "/" << teststep << "] started\033[0m";
                        } else {
                            qDebug() << "\033[0;34mEpisode " << episode+1 << ": Step [" << step+1 << "/" << maxstep << "] started\033[0m";
        //                    if(done_old_data){
        //                        qDebug() << "\033[0;34mTotal steps: " << total_steps << "\033[0m";
        //                    } else {
        //                        qDebug() << "\033[0;34mTotal steps / Old data size: [" << total_steps << "/" << datasize << "]\033[0m";
        //                    }

                            qDebug() << "\033[0;34mTotal steps: " << total_steps+1;
                            qDebug() << "Episodes that finished task: ";
                            if(done_count.size()<1 && test_done_count.size()<1){
                                qDebug() << "None";
                            } else {
                                qDebug() << "total: " << done_count.size();
                                std::cout << done_count << std::endl;
                                qDebug() << "test total: " << test_done_count.size();
                                std::cout << test_done_count << std::endl;
                            }
                            qDebug() << "\033[0m";
                        }

                        task_done = false;
                        int push_point;
                        float push_distance;
                        std::vector<float> action(2);
                        std::vector<float> stepreward(1);
                        stepreward[0] = 0;
                        bool exceeded_limit = false;

                        float before_angle_between, after_angle_between, red_angle, blue_angle;
                        float before_square_angle, after_square_angle;

                        if(step == 0){
                            // Reset env
                            qDebug() << "\033[1;33mReset Environment\033[0m";
                            gCamimage.copyTo(Src);
                            cv::Mat inv_warp_imager;
                            cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                            cv::resize(warped_image, inv_warp_imager, warped_image_size);
                            cv::warpPerspective(inv_warp_imager, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
                            cv::resize(warped_image, warped_image, warped_image_resize);
                            warped_image = background - warped_image;
                            warped_image = ~warped_image;

                            if(train_two_rod){
                                Find_Angle_Between_Rods(before_angle_between, red_angle, blue_angle);
                                // Random reset rod
                                if(before_angle_between < 0.5*PI){
                                    if(use_clock){
                                        if(red_angle < blue_angle){
                                            if(red_angle<PI*0.5 && blue_angle>PI*1.5){
                                                Push_Rod(0, 0.04, use_clock);
                                                Push_Rod(3, 0.04, use_clock);
                                            } else {
                                                Push_Rod(1, 0.04, use_clock);
                                                Push_Rod(2, 0.04, use_clock);
                                            }
                                        } else {
                                            if(blue_angle<PI*0.5 && red_angle>PI*1.5){
                                                Push_Rod(2, 0.04, use_clock);
                                                Push_Rod(1, 0.04, use_clock);
                                            } else {
                                                Push_Rod(3, 0.04, use_clock);
                                                Push_Rod(0, 0.04, use_clock);
                                            }
                                        }
                                    } else {
                                        if(red_angle < blue_angle){
                                            if((0.5*PI<red_angle && red_angle<1.5*PI)
                                               ||(red_angle<0.5*PI && 1.5*PI<blue_angle && blue_angle<2*PI)){
                                                Push_Rod(0, 0.04, use_clock);
                                                Push_Rod(3, 0.04, use_clock);
                                            } else if ((1.5*PI<red_angle && red_angle<2*PI && 1.5*PI<blue_angle && blue_angle<2*PI)
                                                       || (red_angle<0.5*PI && blue_angle<0.5*PI)){
                                                Push_Rod(2, 0.04, use_clock);
                                                Push_Rod(1, 0.04, use_clock);
                                            } else if (blue_angle < PI){
                                                Push_Rod(3, 0.04, use_clock);
                                                if(red_angle < PI*0.5){
                                                    Push_Rod(1, 0.04, use_clock);
                                                } else {
                                                    Push_Rod(0, 0.04, use_clock);
                                                }
                                            }
                                        } else {
                                            if((0.5*PI<blue_angle && blue_angle<1.5*PI)
                                               ||(blue_angle<0.5*PI && 1.5*PI<red_angle && red_angle<2*PI)){
                                                Push_Rod(2, 0.04, use_clock);
                                                Push_Rod(1, 0.04, use_clock);
                                            } else if ((1.5*PI<blue_angle && blue_angle<2*PI && 1.5*PI<red_angle && red_angle<2*PI)
                                                       || (blue_angle<0.5*PI && red_angle<0.5*PI)){
                                                Push_Rod(0, 0.04, use_clock);
                                                Push_Rod(3, 0.04, use_clock);
                                            } else if (red_angle < PI){
                                                Push_Rod(1, 0.04, use_clock);
                                                if(blue_angle < PI*0.5){
                                                    Push_Rod(3, 0.04, use_clock);
                                                } else {
                                                    Push_Rod(2, 0.04, use_clock);
                                                }
                                            }
                                        }
                                    }
                                    Find_Angle_Between_Rods(before_angle_between, red_angle, blue_angle);
                                }
                                while(150.0/180.0*PI < before_angle_between){
                                    int push_point = rand() % 2;
                                    float push_distance = 0.05+(rand() % 21) * 0.001;
                                    Push_Rod(push_point, push_distance, use_clock);
                                    Push_Rod(3-push_point, push_distance, use_clock);
                                    Find_Angle_Between_Rods(before_angle_between, red_angle, blue_angle);
                                }
                            } else {
                                Find_Square_Angle(before_square_angle);
                                // Random reset rod
                                if(before_square_angle > 70.0/180.0*PI && before_square_angle < 110.0/180.0*PI){
                                    int push_point = rand() % 2;
                                    float push_distance = 0.02+(rand() % 21) * 0.001;
                                    if(push_point == 0){
                                        Push_Square(push_point, push_distance, exceeded_limit);
                                    } else {
                                        Push_Square(push_point, push_distance, exceeded_limit);
                                        Push_Square(3-push_point, push_distance-0.02, exceeded_limit);
                                    }
                                    Find_Square_Angle(before_square_angle);
                                }
                            }
                            if(test_mode){
                                qDebug() << "Press 'Enter' to continue";
                                std::cin.ignore();
                            }
                        }

                        gCamimage.copyTo(Src);
                        cv::Mat inv_warp_imager;
                        cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                        cv::resize(warped_image, inv_warp_imager, warped_image_size);
                        cv::warpPerspective(inv_warp_imager, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
                        cv::resize(warped_image, warped_image, warped_image_resize);
                        warped_image = background - warped_image;
                        warped_image = ~warped_image;
                        warped_image.copyTo(before_image);

                        if(train_two_rod){
                            Find_Angle_Between_Rods(before_angle_between, red_angle, blue_angle);

                            std::vector<float> before_state_vector{red_angle/(2.0*PI), blue_angle/(2.0*PI), before_angle_between/(2.0*PI)};
                            before_state = torch::from_blob(before_state_vector.data(), {3}, at::kFloat).detach().clone().to(torch::kCPU);
                            //before_state = torch::from_blob(warped_image.data, { warped_image.rows, warped_image.cols, warped_image.channels() }, at::kByte);
                            //before_state = before_state.permute({ 2, 0, 1 }).to(torch::kF32)/255.0;

                            qDebug() << "\033[1;32mred angle before action: \033[0m" << red_angle/PI*180.0;
                            qDebug() << "\033[1;32mBlue angle before action: \033[0m" << blue_angle/PI*180.0;
                            qDebug() << "\033[1;32mAngle between before action: \033[0m" << before_angle_between/PI*180.0 << "\n";
                        } else {
                            Find_Square_Angle(before_square_angle);

                            std::vector<float> before_state_vector{before_square_angle/(2.0*PI)};
                            before_state = torch::from_blob(before_state_vector.data(), {1}, at::kFloat).detach().clone().to(torch::kCPU);

                            qDebug() << "\033[1;32mSquare angle before action: \033[0m" << before_square_angle/PI*180.0 << "\n";
                        }

                        // Action
                        torch::AutoGradMode enable(true);
                        actor_critic->pi->train();
                        if(test_model){
                            qDebug() << "\033[1;33mTesting\033[0m";
                            auto state = before_state.clone().detach().to(device);
                            torch::AutoGradMode disable(false);
                            actor_critic->pi->eval();
                            action_tensor = actor_critic->act(torch::unsqueeze(state, 0), true);

                            action_tensor = action_tensor.squeeze(0).to(torch::kCPU);
                            std::cout << "\033[1;34mAction predict: \n" << action_tensor << "\033[0m" << std::endl;

                            std::vector<float> tmp_act(action_tensor.data_ptr<float>(), action_tensor.data_ptr<float>() + action_tensor.numel());

                            if(train_two_rod){
                                if(abs(tmp_act[0]) > abs(tmp_act[1])){
                                    if(tmp_act[0]>0){
                                        push_point = 0;
                                    } else {
                                        push_point = 1;
                                    }
                                    push_distance = abs(tmp_act[0])*0.03;
                                } else {
                                    if(tmp_act[1]>0){
                                        push_point = 2;
                                    } else {
                                        push_point = 3;
                                    }
                                    push_distance = abs(tmp_act[1])*0.03;
                                }
                            } else {
                                if(tmp_act[0] > tmp_act[1] && tmp_act[0] > tmp_act[2]){
                                    push_point = 0;
                                    push_distance = tmp_act[0]*0.03;
                                } else if(tmp_act[1] > tmp_act[0] && tmp_act[1] > tmp_act[2]){
                                    push_point = 1;
                                    push_distance = tmp_act[1]*0.03;
                                } else {
                                    push_point = 2;
                                    push_distance = tmp_act[2]*0.03;
                                }
                            }
                        } else if (total_steps < START_STEP) {
                            qDebug() << "\033[1;33mStart exploration\033[0m";
                            // random push point, random push distance from 0mm to 30mm
                            if(train_two_rod){
                                push_point = rand() % 4;
                                push_distance = (rand() % 31) * 0.001;
                                if(push_point == 0){
                                    action[0] = push_distance/0.03;
                                } else if(push_point == 1) {
                                    action[0] = -push_distance/0.03;
                                } else if(push_point == 2) {
                                    action[1] = push_distance/0.03;
                                } else if(push_point == 3) {
                                    action[1] = -push_distance/0.03;
                                }
                                action_tensor = torch::from_blob(action.data(), { 2 }, at::kFloat).to(torch::kCPU);
                            } else {
                                push_point = rand() % 3;
                                push_distance = (rand() % 31) * 0.001;
                                if(push_point == 0){
                                    action[0] = push_distance/0.03;
                                } else if(push_point == 1) {
                                    action[1] = push_distance/0.03;
                                } else if(push_point == 2) {
                                    action[2] = push_distance/0.03;
                                }
                                action_tensor = torch::from_blob(action.data(), { 3 }, at::kFloat).to(torch::kCPU);
                            }
                        } else {
                            qDebug() << "\033[1;33mStart exploitation\033[0m";
                            auto state = before_state.clone().detach().to(device);
                            action_tensor = actor_critic->act(torch::unsqueeze(state, 0), false);

                            action_tensor = action_tensor.squeeze(0).to(torch::kCPU);
                            std::cout << "\033[1;34mAction predict: \n" << action_tensor << "\033[0m" << std::endl;

                            std::vector<float> tmp_act(action_tensor.data_ptr<float>(), action_tensor.data_ptr<float>() + action_tensor.numel());

                            if(train_two_rod){
                                if(abs(tmp_act[0]) > abs(tmp_act[1])){
                                    if(tmp_act[0]>0){
                                        push_point = 0;
                                    } else {
                                        push_point = 1;
                                    }
                                    push_distance = abs(tmp_act[0])*0.03;
                                } else {
                                    if(tmp_act[1]>0){
                                        push_point = 2;
                                    } else {
                                        push_point = 3;
                                    }
                                    push_distance = abs(tmp_act[1])*0.03;
                                }
                            } else {
                                if(tmp_act[0] > tmp_act[1] && tmp_act[0] > tmp_act[2]){
                                    push_point = 0;
                                    push_distance = tmp_act[0]*0.03;
                                } else if(tmp_act[1] > tmp_act[0] && tmp_act[1] > tmp_act[2]){
                                    push_point = 1;
                                    push_distance = tmp_act[1]*0.03;
                                } else {
                                    push_point = 2;
                                    push_distance = tmp_act[2]*0.03;
                                }
                            }
                        }

                        // Task
                        if(train_two_rod){
                            Push_Rod(push_point, push_distance, use_clock);
                        } else {
                            if(before_square_angle > 110.0/180.0*PI && push_point == 0){
                                exceeded_limit = true;
                                qDebug() << "Exceeded limit";
                            }
                            Push_Square(push_point, push_distance, exceeded_limit);
                        }

                        cv::Mat inv_warp_image;
                        gCamimage.copyTo(Src);
                        cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                        cv::resize(warped_image, inv_warp_image, warped_image_size);
                        cv::warpPerspective(inv_warp_image, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
                        cv::resize(warped_image, warped_image, warped_image_resize);
                        warped_image = background - warped_image;
                        warped_image = ~warped_image;
                        warped_image.copyTo(after_image);

                        cv::Mat sub_image;
                        sub_image = after_image - before_image;
                        auto mean = cv::mean(sub_image);
                        //qDebug() << "Pixel color diff mean: "<< mean[0] << " "<<mean[1]<< " "<< mean[2];
                        if(mean[0]<0.15 && mean[1]<0.15 && mean[2]<0.15 && !exceeded_limit){
                            qDebug() << "\033[0;33mNothing Changed(mean<0.15), mean: " << mean[0] << " " << mean[1] << " " << mean[2] << " stop\033[0m";
                            mRunTrainRod = false;
                            break;
                        }

                        // Reward & State
                        if(train_two_rod){
                            Find_Angle_Between_Rods(after_angle_between, red_angle, blue_angle);

                            std::vector<float> after_state_vector{red_angle/(2.0*PI), blue_angle/(2.0*PI), after_angle_between/(2.0*PI)};
                            after_state = torch::from_blob(after_state_vector.data(), {3}, at::kFloat).detach().clone().to(torch::kCPU);
                            //after_state = torch::from_blob(warped_image.data, { warped_image.rows, warped_image.cols, warped_image.channels() }, at::kByte);
                            //after_state = after_state.permute({ 2, 0, 1 }).to(torch::kF32)/255.0;

                            qDebug() << "\033[1;32mred angle after action: \033[0m" << red_angle/PI*180.0;
                            qDebug() << "\033[1;32mBlue angle after action: \033[0m" << blue_angle/PI*180.0;
                            qDebug() << "\033[1;32mAngle between after action: \033[0m" << after_angle_between/PI*180.0;

                            // Reward = +1 for each degree increased and -2 for each step
                            float angle_changed = after_angle_between/PI*180.0 - before_angle_between/PI*180.0;
                            stepreward[0] = angle_changed - 2.0;
                            episode_reward += stepreward[0];

                            qDebug() << "\033[1;32mAngle changed after action: \033[0m" << angle_changed;
                        } else {
                            Find_Square_Angle(after_square_angle);

                            std::vector<float> after_state_vector{after_square_angle/(2.0*PI)};
                            after_state = torch::from_blob(after_state_vector.data(), {1}, at::kFloat).detach().clone().to(torch::kCPU);
                            //after_state = torch::from_blob(warped_image.data, { warped_image.rows, warped_image.cols, warped_image.channels() }, at::kByte);
                            //after_state = after_state.permute({ 2, 0, 1 }).to(torch::kF32)/255.0;

                            qDebug() << "\033[1;32mSquare angle after action: \033[0m" << after_square_angle/PI*180.0;

                            // Reward = +1 for each degree closer to 90 degrees and -2 for each step
                            float distance_changed = abs(before_square_angle/PI*180.0 - 90.0) - abs(after_square_angle/PI*180.0 - 90.0);
                            stepreward[0] = distance_changed - 2.0;
                            episode_reward += stepreward[0];

                            qDebug() << "\033[1;32mDistance changed after action: \033[0m" << distance_changed;
                        }

                        qDebug() << "\033[1;31mReward: " << stepreward[0] << "\033[0m\n";

                        if(train_two_rod){
                            // if it's the last step of the episode or angle is larger than 170 degrees, end
                            if((170.0 < after_angle_between/PI*180.0) || step+1 == maxstep){
                                done[0] = 1;
                                if(170.0 < after_angle_between/PI*180.0){
                                    qDebug() << "\033[1;31mRods are parallel, end episode\033[0m";
                                    if(test_model){
                                        test_done_count.push_back(episode);
                                    } else {
                                        done_count.push_back(episode+1);
                                    }
                                } else {
                                    qDebug() << "\033[1;31mLast step of the episode, end episode\033[0m";
                                }
                            }
                        } else {
                            // if it's the last step of the episode or angle is between 85-95 degrees, end
                            if((after_square_angle/PI*180.0 > 85.0 && after_square_angle/PI*180.0 < 95.0) || step+1 == maxstep){
                                done[0] = 1;
                                if(after_square_angle/PI*180.0 > 85.0 && after_square_angle/PI*180.0 < 95.0){
                                    qDebug() << "\033[1;31mSquare is completed, end episode\033[0m";
                                    if(test_model){
                                        test_done_count.push_back(episode);
                                    } else {
                                        done_count.push_back(episode+1);
                                    }
                                } else {
                                    qDebug() << "\033[1;31mLast step of the episode, end episode\033[0m";
                                }
                            }
                        }

                        if(test_model){
                            auto future_savedata = QtConcurrent::run([before_image = before_image,
                                                                     action_CPU = action_tensor.clone().detach(),
                                                                     stepreward = stepreward,
                                                                     after_image = after_image,
                                                                     drawing = drawing,
                                                                     episode = episode,
                                                                     step = step,
                                                                     this
                                                                     ](){
                                qDebug() << "Saving data";

                                std::vector<float> action_vector(action_CPU.data_ptr<float>(), action_CPU.data_ptr<float>() + action_CPU.numel());

                                QString filename_id = QString(testPath + "/%1").arg(episode);
                                QDir().mkdir(filename_id);
                                QString filename_id2 = QString(filename_id + "/%1").arg(step);
                                QDir().mkdir(filename_id2);

                                QString filename_before_image = QString(filename_id2 + "/before_image.jpg");
                                QByteArray filename_before_imageqb = filename_before_image.toLocal8Bit();
                                const char *filename_before_imagechar = filename_before_imageqb.data();
                                cv::imwrite(filename_before_imagechar, before_image);

                                QString filename_action = QString(filename_id2 + "/action.txt");
                                savedata(filename_action, action_vector);

                                QString filename_reward = QString(filename_id2 + "/reward.txt");
                                savedata(filename_reward, stepreward);

                                QString filename_after_image = QString(filename_id2 + "/after_image.jpg");
                                QByteArray filename_after_imageqb = filename_after_image.toLocal8Bit();
                                const char *filename_after_imagechar = filename_after_imageqb.data();
                                cv::imwrite(filename_after_imagechar, after_image);

                                QString filename_drawing = QString(filename_id2 + "/drawing.jpg");
                                QByteArray filename_drawingqb = filename_drawing.toLocal8Bit();
                                const char *filename_drawingchar = filename_drawingqb.data();
                                cv::imwrite(filename_drawingchar, drawing);
                            });
                            future_savedata.waitForFinished();
                        } else {
                            auto reward_tensor = torch::from_blob(stepreward.data(), { 1 }, torch::kFloat);
                            //std::cout << "reward_tensor: " << reward_tensor << std::endl;
                            done_tensor = torch::from_blob(done.data(), { 1 }, torch::kFloat);
                            //std::cout << "done_tensor: " << done_tensor << std::endl;

                            auto future_savedata = QtConcurrent::run([before_image = before_image,
                                                                     before_state_CPU = before_state.clone().detach(),
                                                                     action_CPU = action_tensor.clone().detach(),
                                                                     reward_CPU = reward_tensor.clone().detach(),
                                                                     done_CPU = done_tensor.clone().detach(),
                                                                     after_image = after_image,
                                                                     after_state_CPU = after_state.clone().detach(),
                                                                     drawing = drawing,
                                                                     datasize = datasize,
                                                                     this
                                                                     ](){

                                qDebug() << "Saving data";

                                std::vector<float> before_state_vector(before_state_CPU.data_ptr<float>(), before_state_CPU.data_ptr<float>() + before_state_CPU.numel());
                                std::vector<float> action_vector(action_CPU.data_ptr<float>(), action_CPU.data_ptr<float>() + action_CPU.numel());
                                std::vector<float> reward_vector(reward_CPU.data_ptr<float>(), reward_CPU.data_ptr<float>() + reward_CPU.numel());
                                std::vector<float> done_vector(done_CPU.data_ptr<float>(), done_CPU.data_ptr<float>() + done_CPU.numel());
                                std::vector<float> after_state_vector(after_state_CPU.data_ptr<float>(), after_state_CPU.data_ptr<float>() + after_state_CPU.numel());

                                QString filename_id = QString(memoryPath + "/%1").arg(datasize);
                                QDir().mkdir(filename_id);

                                QString filename_before_image = QString(filename_id + "/before_image.jpg");
                                QByteArray filename_before_imageqb = filename_before_image.toLocal8Bit();
                                const char *filename_before_imagechar = filename_before_imageqb.data();
                                cv::imwrite(filename_before_imagechar, before_image);

                                torch::Tensor before_image_tensor = torch::from_blob(before_image.data, { before_image.rows, before_image.cols, before_image.channels() }, at::kByte);
                                before_image_tensor = before_image_tensor.permute({ 2, 0, 1 }).to(torch::kF32)/255.0;
                                QString filename_before_image_tensor = QString(filename_id + "/before_image_tensor.pt");
                                torch::save(before_image_tensor, filename_before_image_tensor.toStdString());

                                QString filename_before_state = QString(filename_id + "/before_state.txt");
                                savedata(filename_before_state, before_state_vector);
                                QString filename_before_state_tensor = QString(filename_id + "/before_state_tensor.pt");
                                torch::save(before_state_CPU, filename_before_state_tensor.toStdString());

                                QString filename_action = QString(filename_id + "/action.txt");
                                savedata(filename_action, action_vector);
                                QString filename_action_tensor = QString(filename_id + "/action_tensor.pt");
                                torch::save(action_CPU, filename_action_tensor.toStdString());

                                QString filename_reward = QString(filename_id + "/reward.txt");
                                savedata(filename_reward, reward_vector);
                                QString filename_reward_tensor = QString(filename_id + "/reward_tensor.pt");
                                torch::save(reward_CPU, filename_reward_tensor.toStdString());

                                QString filename_done = QString(filename_id + "/done.txt");
                                savedata(filename_done, done_vector);
                                QString filename_done_tensor = QString(filename_id + "/done_tensor.pt");
                                torch::save(done_CPU, filename_done_tensor.toStdString());

                                QString filename_after_image = QString(filename_id + "/after_image.jpg");
                                QByteArray filename_after_imageqb = filename_after_image.toLocal8Bit();
                                const char *filename_after_imagechar = filename_after_imageqb.data();
                                cv::imwrite(filename_after_imagechar, after_image);

                                torch::Tensor after_image_tensor = torch::from_blob(after_image.data, { after_image.rows, after_image.cols, after_image.channels() }, at::kByte);
                                after_image_tensor = after_image_tensor.permute({ 2, 0, 1 }).to(torch::kF32)/255.0;
                                QString filename_after_image_tensor = QString(filename_id + "/after_image_tensor.pt");
                                torch::save(after_image_tensor, filename_after_image_tensor.toStdString());

                                QString filename_after_state = QString(filename_id + "/after_state.txt");
                                savedata(filename_after_state, after_state_vector);
                                QString filename_after_state_tensor = QString(filename_id + "/after_state_tensor.pt");
                                torch::save(after_state_CPU, filename_after_state_tensor.toStdString());

                                QString filename_drawing = QString(filename_id + "/drawing.jpg");
                                QByteArray filename_drawingqb = filename_drawing.toLocal8Bit();
                                const char *filename_drawingchar = filename_drawingqb.data();
                                cv::imwrite(filename_drawingchar, drawing);
                            });
                            future_savedata.waitForFinished();
                            datasize++;
                        }

                        before_state = after_state;
                    }
                    step++;

                    if(test_model && !LoadOldData){
                        qDebug() << "\033[0;34mTesting model : Step [" << step << "/" << teststep << "] finished\033[0m";
                    } else {
                        total_steps++;

                        std::vector<float> totalsteps;
                        totalsteps.push_back(total_steps);
                        QString filename_totalsteps = QString(memoryPath + "/totalsteps.txt");
                        savedata(filename_totalsteps, totalsteps);

                        if (total_steps > batch_size) {
                            torch::AutoGradMode enable(true);
                            for(int train=0; train<TRAINEVERY; train++){
                                train_number++;
                                if(!LoadOldData){
                                    qDebug() << "\033[1;33mTraining model: [" << train+1 << "/" << TRAINEVERY << "]\033[0m";
                                }
                                int randomdata;
                                if(total_steps > 10000){
                                    randomdata = total_steps - 10000 + rand()%(10000-batch_size+1);
                                } else {
                                    randomdata = rand()%(total_steps-batch_size+1);
                                }
                                //qDebug() << "randomdata: " << randomdata;
                                //qDebug() << "memory size: " << memory.size();
                                std::vector<torch::Tensor> s_data(batch_size), a_data(batch_size), r_data(batch_size), d_data(batch_size), s2_data(batch_size);
                                torch::Tensor s_batch, a_batch, r_batch, d_batch, s2_batch;

                                for (int i = 0; i < batch_size; i++) {
                                    QString filename_id;
                                    filename_id = QString(memoryPath + "/%1").arg(i+randomdata);

                                    QString filename_s = QString(filename_id + "/before_state_tensor.pt");
                                    torch::Tensor tmp_s_data;
                                    torch::load(tmp_s_data, filename_s.toStdString());

                                    QString filename_a = QString(filename_id + "/action_tensor.pt");
                                    torch::Tensor tmp_a_data;
                                    torch::load(tmp_a_data, filename_a.toStdString());

                                    QString filename_r = QString(filename_id + "/reward_tensor.pt");
                                    torch::Tensor tmp_r_data;
                                    torch::load(tmp_r_data, filename_r.toStdString());

                                    QString filename_d = QString(filename_id + "/done_tensor.pt");
                                    torch::Tensor tmp_d_data;
                                    torch::load(tmp_d_data, filename_d.toStdString());

                                    QString filename_s2 = QString(filename_id + "/after_state_tensor.pt");
                                    torch::Tensor tmp_s2_data;
                                    torch::load(tmp_s2_data, filename_s2.toStdString());

                                    s_data[i] = torch::unsqueeze(tmp_s_data.clone().detach(), 0);
                                    a_data[i] = torch::unsqueeze(tmp_a_data.clone().detach(), 0);
                                    r_data[i] = torch::unsqueeze(tmp_r_data.clone().detach(), 0);
                                    d_data[i] = torch::unsqueeze(tmp_d_data.clone().detach(), 0);
                                    s2_data[i] = torch::unsqueeze(tmp_s2_data.clone().detach(), 0);
                                }

                                s_batch = s_data[0]; a_batch = a_data[0]; r_batch = r_data[0]; d_batch = d_data[0]; s2_batch = s2_data[0];
                                for (int i = 1; i < batch_size; i++) {
                                    s_batch = torch::cat({ s_batch, s_data[i] }, 0);
                                    a_batch = torch::cat({ a_batch, a_data[i] }, 0);
                                    r_batch = torch::cat({ r_batch, r_data[i] }, 0);
                                    d_batch = torch::cat({ d_batch, d_data[i] }, 0);
                                    s2_batch = torch::cat({ s2_batch, s2_data[i] }, 0);
                                }
                                s_batch = s_batch.clone().detach().to(device);
                                a_batch = a_batch.clone().detach().to(device);
                                r_batch = r_batch.clone().detach().to(device);
                                d_batch = d_batch.clone().detach().to(device);
                                s2_batch = s2_batch.clone().detach().to(device);

                                // Q-value networks training
                                if(!LoadOldData){
                                    qDebug() << "\033[1;33mTraining Q-value networks\033[0m";
                                }

                                torch::AutoGradMode q_enable(true);

                                torch::Tensor q1 = actor_critic->q1->forward(s_batch, a_batch);
                                torch::Tensor q2 = actor_critic->q2->forward(s_batch, a_batch);

                                torch::AutoGradMode disable(false);
                                // Target actions come from *current* policy
                                policy_output next_state_sample = actor_critic->pi->forward(s2_batch, false, true);
                                torch::Tensor a2_batch = next_state_sample.action;
                                torch::Tensor logp_a2 = next_state_sample.logp_pi;
                                // Target Q-values
                                torch::Tensor q1_pi_target = actor_critic_target->q1->forward(s2_batch, a2_batch);
                                torch::Tensor q2_pi_target = actor_critic_target->q2->forward(s2_batch, a2_batch);
                                torch::Tensor backup = r_batch + GAMMA * (1.0 - d_batch) * (torch::min(q1_pi_target, q2_pi_target) - ALPHA * logp_a2);

                                // MSE loss against Bellman backup
                                torch::AutoGradMode loss_enable(true);
//                                torch::Tensor loss_q1 = torch::mean(pow(q1 - backup, 2)); // JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
//                                torch::Tensor loss_q2 = torch::mean(pow(q2 - backup, 2));
                                torch::Tensor loss_q1 = torch::nn::functional::mse_loss(q1, backup);
                                torch::Tensor loss_q2 = torch::nn::functional::mse_loss(q2, backup);
                                torch::Tensor loss_q = loss_q1 + loss_q2;

                                float loss_q1f = loss_q1.detach().item().toFloat();
                                float loss_q2f = loss_q2.detach().item().toFloat();
                                episode_critic1_loss += loss_q1f;
                                episode_critic2_loss += loss_q2f;

                                critic_optimizer.zero_grad();
                                loss_q.backward();
                                critic_optimizer.step();

                                if(!LoadOldData){
                                    qDebug() << "\033[1;33mCritic optimizer step\033[0m";
                                    qDebug() << "\033[1;33mTraining policy network\033[0m";
                                }

                                // Policy network training
                                for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
                                    actor_critic->q1->parameters()[i].set_requires_grad(false);
                                }
                                for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
                                    actor_critic->q2->parameters()[i].set_requires_grad(false);
                                }

                                policy_output sample = actor_critic->pi->forward(s_batch, false, true);
                                torch::Tensor pi = sample.action;
                                torch::Tensor log_pi = sample.logp_pi;
                                torch::Tensor q1_pi = actor_critic->q1->forward(s_batch, pi);
                                torch::Tensor q2_pi = actor_critic->q2->forward(s_batch, pi);
                                torch::Tensor q_pi = torch::min(q1_pi, q2_pi);

                                // Entropy-regularized policy loss
                                torch::Tensor loss_pi = torch::mean(ALPHA * log_pi - q_pi); // Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

                                float loss_pif = loss_pi.detach().item().toFloat();
                                episode_policy_loss += loss_pif;

                                policy_optimizer.zero_grad();
                                loss_pi.backward();
                                policy_optimizer.step();

                                for(size_t i=0; i < actor_critic->q1->parameters().size(); i++){
                                    actor_critic->q1->parameters()[i].set_requires_grad(true);
                                }
                                for(size_t i=0; i < actor_critic->q2->parameters().size(); i++){
                                    actor_critic->q2->parameters()[i].set_requires_grad(true);
                                }

                                if(!LoadOldData){
                                    qDebug() << "\033[1;33mPolicy optimizer step\033[0m";
                                    qDebug() << "\033[1;33mUpdating target models\033[0m";
                                }

                                // Update target networks
                                torch::AutoGradMode softcopy_disable(false);
                                for (size_t i = 0; i < actor_critic_target->pi->parameters().size(); i++) {
                                    actor_critic_target->pi->parameters()[i].mul_(POLYAK);
                                    actor_critic_target->pi->parameters()[i].add_((1.0 - POLYAK) * actor_critic->pi->parameters()[i]);
                                }
                                for (size_t i = 0; i < actor_critic_target->q1->parameters().size(); i++) {
                                    actor_critic_target->q1->parameters()[i].mul_(POLYAK);
                                    actor_critic_target->q1->parameters()[i].add_((1.0 - POLYAK) * actor_critic->q1->parameters()[i]);
                                }
                                for (size_t i = 0; i < actor_critic_target->q2->parameters().size(); i++) {
                                    actor_critic_target->q2->parameters()[i].mul_(POLYAK);
                                    actor_critic_target->q2->parameters()[i].add_((1.0 - POLYAK) * actor_critic->q2->parameters()[i]);
                                }
                                torch::AutoGradMode softcopy_enable(true);

                                if(!LoadOldData){
                                    qDebug() << "\033[1;31mCritic 1 loss: " << loss_q1f << "\n"
                                             << "Critic 2 loss: " << loss_q2f << "\n"
                                             << "Policy loss: " << loss_pif << "\n"
                                             << "Episode reward: " << episode_reward << "\033[0m";
                                }
                            }
                        }
                        qDebug() << "\033[0;34mEpisode " << episode+1 << ": Step [" << step << "/" << maxstep << "] finished\033[0m\n";
                    }

                    if(LoadOldData && total_steps+1 > datasize){
                        LoadOldData = false;
                        qDebug() << "\033[1;31mDone training old data, start plan\033[0m\n";
                        break;
                    }
                    if(task_done == true){
                        break;
                    }
                    if(test_model && step == teststep){
                        break;
                    }
                }

                // Save
                if(test_model){
                    qDebug() << "\033[0;35m--------------------------------------------" << "\n"
                             << "Reward: " << episode_reward << "\n"
                             << "--------------------------------------------\033[0m";
                    logger.add_scalar("Test_Reward", episode, episode_reward);
                    qDebug() << "\033[0;34mTest model finished\033[0m\n"
                             << "--------------------------------------------";
                    if(!test_mode){
                        test_model = false;
                    }
                } else if (bResetRobot){
                    qDebug() << "\033[0;35m--------------------------------------------" << "\n"
                             << "Restart episode" << episode+1 << "\n"
                             << "--------------------------------------------\033[0m";
                } else {
                    episode++;

                    episode_critic1_loss = episode_critic1_loss / (float)train_number;
                    episode_critic2_loss = episode_critic2_loss / (float)train_number;
                    episode_policy_loss = episode_policy_loss / (float)train_number;

                    qDebug() << "\033[0;35m--------------------------------------------" << "\n"
                        << "Episode: " << episode << "\n"
                        << "Reward: " << episode_reward << "\n"
                        << "Critic 1 Loss: " << episode_critic1_loss << "\n"
                        << "Critic 2 Loss: " << episode_critic2_loss << "\n"
                        << "Policy Loss: " << episode_policy_loss << "\n"
                        << "--------------------------------------------\033[0m";
                    logger.add_scalar("Episode_Reward", episode, episode_reward);
                    logger.add_scalar("Episode_Critic_1_Loss", episode, episode_critic1_loss);
                    logger.add_scalar("Episode_Critic_2_Loss", episode, episode_critic2_loss);
                    logger.add_scalar("Episode_Policy_Loss", episode, episode_policy_loss);

                    int save = SAVEMODELEVERY;
                    if(LoadOldData){
                        save = 25;
                    }
                    if (episode % save == 0) {
                        qDebug() << "Saving models";

                        QString pi_para_path = QString(modelPath + "/pi_para/pi_para_" + QString::number(episode) + ".pt");
                        QString q1_para_path = QString(modelPath + "/q1_para/q1_para_" + QString::number(episode) + ".pt");
                        QString q2_para_path = QString(modelPath + "/q2_para/q2_para_" + QString::number(episode) + ".pt");
                        QString target_pi_para_path = QString(modelPath + "/target_pi_para/target_pi_para_" + QString::number(episode) + ".pt");
                        QString target_q1_para_path = QString(modelPath + "/target_q1_para/target_q1_para_" + QString::number(episode) + ".pt");
                        QString target_q2_para_path = QString(modelPath + "/target_q2_para/target_q2_para_" + QString::number(episode) + ".pt");
                        QString policy_opti_path = QString(modelPath + "/policy_optimizer/policy_optimizer_" + QString::number(episode) + ".pt");
                        QString critic_opti_path = QString(modelPath + "/critic_optimizer/critic_optimizer_" + QString::number(episode) + ".pt");

                        torch::save(actor_critic->pi->parameters(), pi_para_path.toStdString());
                        torch::save(actor_critic->q1->parameters(), q1_para_path.toStdString());
                        torch::save(actor_critic->q2->parameters(), q2_para_path.toStdString());
                        torch::save(actor_critic_target->pi->parameters(), target_pi_para_path.toStdString());
                        torch::save(actor_critic_target->q1->parameters(), target_q1_para_path.toStdString());
                        torch::save(actor_critic_target->q2->parameters(), target_q2_para_path.toStdString());
                        torch::save(policy_optimizer, policy_opti_path.toStdString());
                        torch::save(critic_optimizer, critic_opti_path.toStdString());

                        std::vector<float> save_episode_num;
                        save_episode_num.push_back(episode+1);
                        QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
                        savedata(filename_episode_num, save_episode_num);

                        qDebug() << "Models saved";
                    }

                    qDebug() << "\033[0;34mEpisode " << episode << "finished\033[0m\n"
                             << "--------------------------------------------";
                    if(episode % TESTEVERY == 0){
                        test_model = true;
                    }
                }

                if(!mRunTrainRod && !test_mode){
                    qDebug() << "Quit Train Rod" ;
                    qDebug() << "Saving models";

                    QString pi_para_path = QString(modelPath + "/pi_para/pi_para_" + QString::number(episode) + ".pt");
                    QString q1_para_path = QString(modelPath + "/q1_para/q1_para_" + QString::number(episode) + ".pt");
                    QString q2_para_path = QString(modelPath + "/q2_para/q2_para_" + QString::number(episode) + ".pt");
                    QString target_pi_para_path = QString(modelPath + "/target_pi_para/target_pi_para_" + QString::number(episode) + ".pt");
                    QString target_q1_para_path = QString(modelPath + "/target_q1_para/target_q1_para_" + QString::number(episode) + ".pt");
                    QString target_q2_para_path = QString(modelPath + "/target_q2_para/target_q2_para_" + QString::number(episode) + ".pt");
                    QString policy_opti_path = QString(modelPath + "/policy_optimizer/policy_optimizer_" + QString::number(episode) + ".pt");
                    QString critic_opti_path = QString(modelPath + "/critic_optimizer/critic_optimizer_" + QString::number(episode) + ".pt");

                    torch::save(actor_critic->pi->parameters(), pi_para_path.toStdString());
                    torch::save(actor_critic->q1->parameters(), q1_para_path.toStdString());
                    torch::save(actor_critic->q2->parameters(), q2_para_path.toStdString());
                    torch::save(actor_critic_target->pi->parameters(), target_pi_para_path.toStdString());
                    torch::save(actor_critic_target->q1->parameters(), target_q1_para_path.toStdString());
                    torch::save(actor_critic_target->q2->parameters(), target_q2_para_path.toStdString());
                    torch::save(policy_optimizer, policy_opti_path.toStdString());
                    torch::save(critic_optimizer, critic_opti_path.toStdString());

                    std::vector<float> save_episode_num;
                    save_episode_num.push_back(episode+1);
                    QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
                    savedata(filename_episode_num, save_episode_num);

                    qDebug() << "Models saved";
                    break;
                }
            }
        } catch (const std::exception &e) {
            auto &&msg = torch::GetExceptionString(e);
            qWarning() << msg.c_str();
        } catch (...) {
            qCritical() << "GG";
        }
    });
}


void LP_Plugin_Garment_Manipulation::adjustHeight(double &in, const double &min)
{

}

void LP_Plugin_Garment_Manipulation::FunctionalRender_L(QOpenGLContext *ctx, QSurface *surf, QOpenGLFramebufferObject *fbo, const LP_RendererCam &cam, const QVariant &options)
{
    Q_UNUSED(surf)  //Mostly not used within a Functional.
//    Q_UNUSED(options)   //Not used in this functional.

    if(!gQuit){

        if ( !mInitialized_L ){
            initializeGL_L();
        }

        QMatrix4x4 view = cam->ViewMatrix(),
                   proj = cam->ProjectionMatrix();

        static std::vector<QVector3D> quad1 =
                                      {QVector3D( 0.0f, 0.0f, 0.0f),
                                       QVector3D( 0.0f, 2.5f, 0.0f),
                                       QVector3D(-4.0f, 2.5f, 0.0f),
                                       QVector3D(-4.0f, 0.0f, 0.0f)};

        static std::vector<QVector3D> quad2 =
                                      {QVector3D( 0.0f,-3.0f, 0.0f),
                                       QVector3D( 0.0f, 0.0f, 0.0f),
                                       QVector3D(-3.0f, 0.0f, 0.0f),
                                       QVector3D(-3.0f,-3.0f, 0.0f)};

        static std::vector<QVector3D> quad3 =
                                      {QVector3D( 3.0f,-3.0f, 0.0f),
                                       QVector3D( 3.0f, 0.0f, 0.0f),
                                       QVector3D( 0.0f, 0.0f, 0.0f),
                                       QVector3D( 0.0f,-3.0f, 0.0f)};

        static std::vector<QVector3D> quad4 =
                                      {QVector3D( 4.0f, 0.0f, 0.0f),
                                       QVector3D( 4.0f, 2.5f, 0.0f),
                                       QVector3D( 0.0f, 2.5f, 0.0f),
                                       QVector3D( 0.0f, 0.0f, 0.0f)};

        static std::vector<QVector3D> quad5 =
                                      {QVector3D( 1.5f,-6.0f, 0.0f),
                                       QVector3D( 1.5f,-3.0f, 0.0f),
                                       QVector3D(-1.5f,-3.0f, 0.0f),
                                       QVector3D(-1.5f,-6.0f, 0.0f)};

        static std::vector<QVector2D> tex =
                                      {QVector2D( 1.0f, 0.0f),
                                       QVector2D( 1.0f, 1.0f),
                                       QVector2D( 0.0f, 1.0f),
                                       QVector2D( 0.0f, 0.0f)};



        auto f = ctx->extraFunctions();


        fbo->bind();
        mProgram_L->bind();

        mProgram_L->setUniformValue("m4_mvp", proj * view );
        mProgram_L->enableAttributeArray("a_pos");
        mProgram_L->enableAttributeArray("a_tex");

        mProgram_L->setAttributeArray("a_pos", quad1.data());
        mProgram_L->setAttributeArray("a_tex", tex.data());

        gLock.lockForRead();
        QOpenGLTexture texture1(gCurrentGLFrame.mirrored());
        gLock.unlock();

        texture1.bind();
        mProgram_L->setUniformValue("u_tex", 0);

        f->glDrawArrays(GL_QUADS, 0, 4);

        texture1.release();


        if (gStopFindWorkspace){
            if(!gWarpedImage.isNull()){
                mProgram_L->setAttributeArray("a_pos", quad2.data());
                gLock.lockForRead();
                QOpenGLTexture texture2(gWarpedImage.mirrored());
                gLock.unlock();

                if ( !texture2.create()){
                    qDebug() << "GG";
                    }

                texture2.bind();

                f->glDrawArrays(GL_QUADS, 0, 4);

                texture2.release();
            }

            if(!gEdgeImage.isNull()){
                mProgram_L->setAttributeArray("a_pos", quad3.data());
                gLock.lockForRead();
                QOpenGLTexture texture3(gEdgeImage.mirrored());
                gLock.unlock();

                if ( !texture3.create()){
                    qDebug() << "GG";
                    }

                texture3.bind();

                f->glDrawArrays(GL_QUADS, 0, 4);

                texture3.release();
            }

            if(!gInvWarpImage.isNull()){
                mProgram_L->setAttributeArray("a_pos", quad4.data());
                gLock.lockForRead();
                QOpenGLTexture texture4(gInvWarpImage.mirrored());
                gLock.unlock();

                if ( !texture4.create()){
                    qDebug() << "GG";
                    }

                texture4.bind();

                f->glDrawArrays(GL_QUADS, 0, 4);

                texture4.release();
            }

            if(!gDetectImage.isNull()){
                mProgram_L->setAttributeArray("a_pos", quad5.data());
                gLock.lockForRead();
                QOpenGLTexture texture5(gDetectImage.mirrored());
                gLock.unlock();

                if ( !texture5.create()){
                    qDebug() << "GG";
                    }

                texture5.bind();

                f->glDrawArrays(GL_QUADS, 0, 4);

                texture5.release();
            }
        }


        mProgram_L->disableAttributeArray("a_pos");
        mProgram_L->disableAttributeArray("a_tex");
        mProgram_L->release();
        fbo->release();
    }
}

void LP_Plugin_Garment_Manipulation::FunctionalRender_R(QOpenGLContext *ctx, QSurface *surf, QOpenGLFramebufferObject *fbo, const LP_RendererCam &cam, const QVariant &options)
{
    Q_UNUSED(surf)  //Mostly not used within a Functional.
//    Q_UNUSED(options)   //Not used in this functional.

    if(!gQuit){

        if ( !mInitialized_R ){
            initializeGL_R();
        }

        QMatrix4x4 view = cam->ViewMatrix(),
                   proj = cam->ProjectionMatrix();

        auto f = ctx->extraFunctions();

        f->glEnable(GL_BLEND);

        fbo->bind();
        mProgram_R->bind();


        mProgram_R->setUniformValue("m4_mvp", proj * view );
        mProgram_R->enableAttributeArray("a_pos");
        mProgram_R->enableAttributeArray("a_tex");

        gLock.lockForRead();
        QOpenGLTexture texture1(gCurrentGLFrame);
        gLock.unlock();

        if ( !texture1.create()){
            qDebug() << "GG";
        }
        texture1.bind();
        mProgram_R->setUniformValue("f_pointSize", 2.0f);
        mProgram_R->setUniformValue("u_tex", 0);

        mProgram_R->setAttributeArray("a_pos", mPointCloud.data());
        mProgram_R->setAttributeArray("a_tex", mPointCloudTex.data());

        f->glEnable(GL_PROGRAM_POINT_SIZE);
        f->glDrawArrays(GL_POINTS, 0, mPointCloud.size());

        mProgram_R->setAttributeArray("a_pos", mTestP.data());
        mProgram_R->setUniformValue("f_pointSize", 10.0f);
        f->glDrawArrays(GL_POINTS, 0, mTestP.size());

        mProgram_R->setAttributeArray("a_pos", mGraspP.data());
        mProgram_R->setUniformValue("v4_color", QVector4D(0.f, 0.f, 1.f, 0.8f));
        f->glDrawArrays(GL_POINTS, 0, mGraspP.size());

        mProgram_R->setAttributeArray("a_pos", mReleaseP.data());
        mProgram_R->setUniformValue("v4_color", QVector4D(1.f, 1.f, 0.f, 0.8f));
        f->glDrawArrays(GL_POINTS, 0, mReleaseP.size());

        texture1.release();

        mProgram_R->setUniformValue("m4_mvp", proj * view );

        constexpr float axisLength = 2.f;
        static std::vector<QVector3D> Axes = {QVector3D(0.f, 0.f, 0.f), QVector3D(axisLength, 0.f, 0.f),
                                              QVector3D(0.f, 0.f, 0.f), QVector3D(0.f, axisLength, 0.f),
                                              QVector3D(0.f, 0.f, 0.f), QVector3D(0.f, 0.f, axisLength),
                                              QVector3D(0.f, 0.f, 0.f), QVector3D(axisLength, axisLength, 0.f)};

        mProgram_R->setAttributeArray("a_pos", Axes.data());

        for ( int i=0; i<4; ++i ) {

            switch (i) {
            case 0:
                mProgram_R->setUniformValue("v4_color", QVector4D(1.f, 0.f, 0.f, 0.6f));
                break;
            case 1:
                mProgram_R->setUniformValue("v4_color", QVector4D(0.f, 1.f, 0.f, 0.6f));
                break;
            case 2:
                mProgram_R->setUniformValue("v4_color", QVector4D(0.f, 0.f, 1.f, 0.6f));
                break;
            default:
                mProgram_R->setUniformValue("v4_color", QVector4D(0.f, 1.f, 1.f, 0.6f));
                break;
            }
            f->glLineWidth(3.f);
            f->glDrawArrays(GL_LINES, i*2, 2 );
        }

        mProgram_R->setUniformValue("v4_color", QVector4D(0.f, 0.f, 0.f, 1.f));

//        std::vector<QVector3D> tmpPC;
//        for(int i=0; i<100; i++){
//            for(int j=0; j<50; j++){
//                tmpPC.emplace_back(QVector3D(i*0.1f, j*0.1, 0.0f));
//            }
//        }
//        mProgram_R->setUniformValue("f_pointSize", 1.0f);
//        mProgram_R->setAttributeArray("a_pos", tmpPC.data());
//        f->glDrawArrays(GL_POINTS, 0, tmpPC.size());


        mProgram_R->disableAttributeArray("a_pos");
        mProgram_R->disableAttributeArray("a_tex");
        mProgram_R->release();
        fbo->release();

        f->glDisable(GL_BLEND);
    }
}

void LP_Plugin_Garment_Manipulation::initializeGL_L()
{
        std::string vsh, fsh;

            vsh =
                "attribute vec3 a_pos;\n"       //The position of a point in 3D that used in FunctionRender()
                "attribute vec2 a_tex;\n"
                "uniform mat4 m4_mvp;\n"        //The Model-View-Matrix
                "varying vec2 tex;\n"
                "void main(){\n"
                "   tex = a_tex;\n"
                "   gl_Position = m4_mvp * vec4(a_pos, 1.0);\n" //Output the OpenGL position
                "   gl_PointSize = 10.0;\n"
                "}";
            fsh =
                "uniform sampler2D u_tex;\n"    //Defined the point color variable that will be set in FunctionRender()
                "varying vec2 tex;\n"
                "void main(){\n"
                "   vec4 v4_color = texture2D(u_tex, tex);\n"
                "   gl_FragColor = v4_color;\n" //Output the fragment color;
                "}";

        auto prog = new QOpenGLShaderProgram;   //Intialize the Shader with the above GLSL codes
        prog->addShaderFromSourceCode(QOpenGLShader::Vertex,vsh.c_str());
        prog->addShaderFromSourceCode(QOpenGLShader::Fragment,fsh.data());
        if (!prog->create() || !prog->link()){  //Check whether the GLSL codes are valid
            qDebug() << prog->log();
            return;
        }

        mProgram_L = prog;            //If everything is fine, assign to the member variable

        mInitialized_L = true;
}

void LP_Plugin_Garment_Manipulation::initializeGL_R()
{
    std::string vsh, fsh;

        vsh =
            "attribute vec3 a_pos;\n"       //The position of a point in 3D that used in FunctionRender()
            "attribute vec2 a_tex;\n"
            "uniform mat4 m4_mvp;\n"        //The Model-View-Matrix
            "uniform float f_pointSize;\n"  //Point size determined in FunctionRender()
            "varying vec2 tex;\n"
            "void main(){\n"
            "   gl_Position = m4_mvp * vec4(a_pos, 1.0);\n" //Output the OpenGL position
            "   gl_PointSize = f_pointSize; \n"
            "   tex = a_tex;\n"
            "}";
        fsh =
            "uniform sampler2D u_tex;\n"    //Defined the point color variable that will be set in FunctionRender()
            "uniform vec4 v4_color;\n"
            "varying vec2 tex;\n"
            "void main(){\n"
            "   vec4 color = texture2D(u_tex, tex);\n"
            "   gl_FragColor = v4_color + color;\n" //Output the fragment color;
            "   gl_FragColor.a = v4_color.a;\n"
            "}";

    auto prog = new QOpenGLShaderProgram;   //Intialize the Shader with the above GLSL codes
    prog->addShaderFromSourceCode(QOpenGLShader::Vertex,vsh.c_str());
    prog->addShaderFromSourceCode(QOpenGLShader::Fragment,fsh.data());
    if (!prog->create() || !prog->link()){  //Check whether the GLSL codes are valid
        qDebug() << prog->log();
        return;
    }

    mProgram_R = prog;            //If everything is fine, assign to the member variable

    mInitialized_R = true;
}

void LP_Plugin_Garment_Manipulation::timerEvent(QTimerEvent *event)
{
//    Q_UNUSED(event);
    qDebug() << QString("\033[1;96m[Status] \033[1;32mSaved / \033[1;33mUseless / \033[1;96mReset : \033[1;32m%1 /\033[1;33m%2 /\033[1;96m%3 \033[0m")
                .arg(datanumber, 5, 10, QChar(' '))
                .arg(useless_data, 3, 10, QChar(' '))
                .arg(gRobotResetCount, 2, 10, QChar(' '))
                .toUtf8().data();

    QFile file(QString("%1/statistic.txt").arg(dataPath));
    if ( file.open(QIODevice::WriteOnly)) {
        QTextStream out(&file);

        out << QString("%1 %2 %3\n")
               .arg(datanumber)
               .arg(useless_data)
               .arg(gRobotResetCount);

        file.close();
    } else {
        qWarning() << "GG, statistic cannot be recorded !";
    }
}

void LP_Plugin_Garment_Manipulation::PainterDraw(QWidget *glW)
{
    if( manual_mode ){
        if ( "window_Normal" == glW->objectName()){
            return;
        }
        QPainter painter(glW);
        painter.drawImage(glW->rect(),gWarpedImage,gWarpedImage.rect());
    }
}

QString LP_Plugin_Garment_Manipulation::MenuName()
{
    return tr("menuPlugins");
}

QAction *LP_Plugin_Garment_Manipulation::Trigger()
{
    if ( !mAction ){
        mAction = new QAction("Garment Manipulation");
    }
    return mAction;
}

/**
  Helper function for deciding on int ot float slider
*/
bool filter_slider_ui::is_all_integers(const rs2::option_range& range)
{
    const auto is_integer = [](float f)
    {
        return (fabs(fmod(f, 1)) < std::numeric_limits<float>::min());
    };

    return is_integer(range.min) && is_integer(range.max) &&
        is_integer(range.def) && is_integer(range.step);
}

/**
Constructor for filter_options, takes a name and a filter.
*/
filter_options::filter_options(const std::string name, rs2::filter& flt) :
    filter_name(name),
    filter(flt),
    is_enabled(true)
{
    const std::array<rs2_option, 5> possible_filter_options = {
        RS2_OPTION_FILTER_MAGNITUDE,
        RS2_OPTION_FILTER_SMOOTH_ALPHA,
        RS2_OPTION_MIN_DISTANCE,
        RS2_OPTION_MAX_DISTANCE,
        RS2_OPTION_FILTER_SMOOTH_DELTA
    };

    //Go over each filter option and create a slider for it
    for (rs2_option opt : possible_filter_options)
    {
        if (flt.supports(opt))
        {
            rs2::option_range range = flt.get_option_range(opt);
            supported_options[opt].range = range;
            supported_options[opt].value = range.def;
            supported_options[opt].is_int = filter_slider_ui::is_all_integers(range);
            supported_options[opt].description = flt.get_option_description(opt);
            std::string opt_name = flt.get_option_name(opt);
            supported_options[opt].name = name + "_" + opt_name;
            std::string prefix = "Filter ";
            supported_options[opt].label = opt_name;
        }
    }
}

filter_options::filter_options(filter_options&& other) :
    filter_name(std::move(other.filter_name)),
    filter(other.filter),
    supported_options(std::move(other.supported_options)),
    is_enabled(other.is_enabled.load())
{
}
