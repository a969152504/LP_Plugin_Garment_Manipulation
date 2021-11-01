<<<<<<< HEAD
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

/**
 * @brief BulletPhysics Headers
 */
//#include <BulletSoftBody/btSoftBody.h>
//#include <Bullet3Dynamics/b3CpuRigidBodyPipeline.h>

const QString dataPath("/home/cpii/projects/data");
const QString memoryPath("/home/cpii/projects/memory");
std::vector<double> markervecs_97 = {0.0, 0.0, 0.0}, markercoordinate_98 = {0.0, 0.0, 0.0}, markercoordinate_99 = {0.0, 0.0, 0.0}, markertrans(2), markerposition_98(2), markerposition_99(2), avgHeight;
std::vector<double> grasp_last(3), release_last(3);
std::vector<int> offset(2);
cv::RNG rng(12345);
std::vector<cv::Point2f> roi_corners(4), midpoints(4), dst_corners(4);
std::vector<float> trans(3), markercenter(2);
float gripper_length = 0.244;
constexpr int iBound[2] = { 1000, 9000 };
constexpr uchar uThres = 250;

constexpr double robotDLimit = 0.85;    //Maximum distance the robot can reach
int maxepisode = 10000, maxstep = 20, minisize = 10, batch_size = 8;
const float exploration_rate_decay = 0.9996, GAMMA = 0.99, TAU = 0.001;
float exploration_rate = 1.0;
double lrp = 2e-6, lrc = 1e-3;

std::shared_ptr<QProcess> LP_Plugin_Garment_Manipulation::gProc_RViz;   //Define the static variable

// open the first webcam plugged in the computer
//cv::VideoCapture camera1(4); // Grey: 2, 8. Color: 4, 10.

bool gStopFindWorkspace = false, gPlan = false, gQuit = false;
QFuture<void> gFuture;
QImage gNullImage, gCurrentGLFrame, gEdgeImage, gWarpedImage, gInvWarpImage;
QReadWriteLock gLock;

struct Data {
    torch::Tensor before_state;
    torch::Tensor place_point;
    torch::Tensor reward;
    torch::Tensor done;
    torch::Tensor after_state;
};
std::deque<Data> memory;

// Policy
struct PolicyImpl : torch::nn::Module {
    PolicyImpl()
        :
        fc1(262147, 1024),
        fc2(1024, 512),
        fc3(512, 64),
        fc4(64, 3)
        //dropout(torch::nn::Dropout(torch::nn::DropoutOptions(0.3)))
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
        register_module("fc4", fc4);
        //register_module("dropout", dropout);
    }

    torch::Tensor forward(torch::Tensor input) {
        //std::cout << input.sizes() << std::endl;
        //torch::Tensor x = dropout(input);

        auto x = torch::relu(fc1(input));

        //std::cout << x.sizes() << std::endl;

        x = torch::relu(fc2(x));

        //std::cout << x.sizes() << std::endl;

        x = torch::relu(fc3(x));

        //std::cout << x.sizes() << std::endl;

        x = fc4(x);

        //std::cout << x.sizes() << std::endl;

        x = torch::sigmoid(x);

        return x;
    }

    torch::nn::Linear fc1, fc2, fc3, fc4;
    //torch::nn::Dropout dropout;
};
TORCH_MODULE(Policy);

// Critic
struct CriticImpl : torch::nn::Module {
    CriticImpl()
        :
        fcs1(262147, 1024),
        fcs2(1024, 256),
        fcp1(3, 16),
        fc1(272, 32),
        fc2(32, 1)
        //dropout(torch::nn::Dropout(torch::nn::DropoutOptions(0.3)))
    {
        register_module("fcs1", fcs1);
        register_module("fcs2", fcs2);
        register_module("fcp1", fcp1);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        //register_module("dropout", dropout);
    }

    torch::Tensor forward(torch::Tensor place_point, torch::Tensor state) {
        //std::cout << state.sizes() << std::endl;
        torch::Tensor x = torch::relu(fcs1(state));

//        auto max1 = torch::max(x);
//        std::cout << "max1: " << max1 << std::endl;
        //std::cout << x.sizes() << std::endl;

        //x = dropout(x);
        x = torch::relu(fcs2(x));

//        auto max2 = torch::max(x);
//        std::cout << "max2: " << max2 << std::endl;

        auto y = torch::relu(fcp1(place_point));

//        auto max3 = torch::max(y);
//        std::cout << "max3: " << max3 << std::endl;

        //std::cout << x.sizes() << std::endl;
        //std::cout << place_point.sizes() << std::endl;

        x = torch::cat({ x, y }, 1);

        //std::cout << x.sizes() << std::endl;

        x = torch::relu(fc1(x));

        //auto max4 = torch::max(x);
        //std::cout << "max4: " << max4 << std::endl;

        x = fc2(x);

        //auto max4 = torch::max(x);
        //std::cout << "max4: " << max4 << std::endl;

        //std::cout << x.sizes() << std::endl;

        return x;
    }

    torch::nn::Linear fcs1, fcs2, fcp1, fc1, fc2;
    //torch::nn::Dropout dropout;
};
TORCH_MODULE(Critic);

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; file >> line;) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
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

    layout->addWidget(mLabel);
    layout->addWidget(mLabel2);

    mWidget->setLayout(layout);
    return mWidget.get();
}

class Sleeper : public QThread
{
public:
    static void usleep(unsigned long usecs){QThread::usleep(usecs);}
    static void msleep(unsigned long msecs){QThread::msleep(msecs);}
    static void sleep(unsigned long secs){QThread::sleep(secs);}
};

bool LP_Plugin_Garment_Manipulation::Run()
{
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

    //Reset RViz
    resetRViz();
    //Start the RViz

    //Reset robot position
    resetRobotPosition();

    srand((unsigned)time(NULL));
    useless_data = 0;
    datanumber = 0;
    frame = 0;
    markercount = 0;
    mTableHeight = 0.249;
    warped_image_resize = cv::Size(256, 256);
    gStopFindWorkspace = false;
    gPlan = false;
    gQuit = false;
    mCalAveragePoint = false;
    gFoundBackground = false;
    Transformationmatrix_T2R.setToIdentity();
    Transformationmatrix_T2R.rotate(-45.0, QVector3D(0.f, 0.f, 1.0f));
    Transformationmatrix_T2R.translate(0.46777, 0.63396, 0.f);

    //calibrate();
    //return false;

    rs2::pipeline_profile profile = pipe.start();
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

    gFuture = QtConcurrent::run([this](){        

//        int countx = 0;
//        int county = 360;

        while(!gQuit)
        {            
            // Wait for frames and get them as soon as they are ready
            frames = pipe.wait_for_frames();

            // Our rgb frame
            rs2::frame rgb = frames.get_color_frame();
            pc.map_to(rgb);

            // Let's get our depth frame
            auto depth = frames.get_depth_frame();
            depthw = depth.get_width();
            depthh = depth.get_height();

            // Device information
            depth_i = depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            color_i = rgb.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            d2c_e = depth.get_profile().as<rs2::video_stream_profile>().get_extrinsics_to(rgb.get_profile());
            c2d_e = rgb.get_profile().as<rs2::video_stream_profile>().get_extrinsics_to(depth.get_profile());
            rs2::depth_sensor ds = dev.query_sensors().front().as<rs2::depth_sensor>();
            depth_scale = ds.get_depth_scale();
        //                float fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy, distC1 = j.coeffs[0], distC2 = j.coeffs[1], distC3 = j.coeffs[2], distC4 = j.coeffs[3], distC5 = j.coeffs[4];
        //                qDebug()<< "fx: "<< fx << "fy: "<< fy << "cx: "<< cx << "cy: "<< cy << "coeffs: "<< distC1 << " "<< distC2 << " "<< distC3 << " "<< distC4 << " "<< distC5;
        //                QMatrix4x4 K = {fx,   0.0f,   cx, 0.0f,
        //                                0.0f,   fy,   cy, 0.0f,
        //                                0.0f, 0.0f, 1.0f, 0.0f,
        //                                0.0f, 0.0f, 0.0f, 0.0f};

            // Generate the pointcloud and texture mappings
            points = pc.calculate(depth);
            auto vertices = points.get_vertices();

            // Let's convert them to QImage
            auto q_rgb = realsenseFrameToQImage(rgb);

            cv::Mat camimage = cv::Mat(q_rgb.height(),q_rgb.width(), CV_8UC3, q_rgb.bits());
            cv::cvtColor(camimage, camimage, cv::COLOR_BGR2RGB);

//            qDebug()<< "depthw: "<< depthw <<"depthh: " << depthh<< "q_rgbh: "<<q_rgb.height()<<"q_rgbw: "<<q_rgb.width();

            srcw = camimage.cols;
            srch = camimage.rows;

            camimage.copyTo(gCamimage);

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

            auto tex_coords = points.get_texture_coordinates(); // and texture coordinates

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


            if(mCalAveragePoint && acount<40){
                avgHeight.resize( depthh * depthw );

                auto *_start = avgHeight.data();
                auto mat_Inv = Transformationmatrix_T2C.inv();

                auto _future = QtConcurrent::map(avgHeight, [&](double &h ){
                    auto id = &h - _start;
//                    auto i = id / depthh;
//                    auto j = id % depthh;
                    cv::Point3f Pc = cv::Point3f{mPointCloud[id].x(), -mPointCloud[id].y(), -mPointCloud[id].z()};
                    cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                    cv::Mat_<float> dstMat(mat_Inv * ptMat);
                    float scale = dstMat(0,3);
                    QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                    avgHeight.at(id) += Pt.z();
                });
                _future.waitForFinished();

                acount++;

                if(acount>=40){
                    for(int i=0; i< avgHeight.size(); i++){
                        avgHeight.at(i) /= double(acount);
                    }
                    if(graspx!=0 && graspy!=0){
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

                        for(int i=0; i<20; i++){
                            for(int j=0; j<20; j++){
                                auto id = depthh*(graspx-10+i) + depthh-(graspy-10+j);
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
                                    offset[0] = -10+i;
                                    offset[1] = -10+j;
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
                auto depth = frames.get_depth_frame();
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
                    mLabel->setText("Right click to collect data \n"
                                    "or Press 'R' to start reinforcement learning 1");
                    gPlan = true;
                } else {
                    qCritical() << "Initialize garment POSITION failed.";
                }
            } else if (gPlan) {
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
                                auto depth = frames.get_depth_frame();

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
        }
    } else if ( QEvent::KeyRelease == event->type()){
        auto e = static_cast<QKeyEvent*>(event);

        if ( e->key() == Qt::Key_Space ){
            if (gStopFindWorkspace){
                mRunCollectData = false;
                mRunReinforcementLearning1 = false;
                QProcess exit;
                QStringList exitarg;
                exitarg << "/home/cpii/projects/scripts/exit.sh";
                exit.startDetached("xterm", exitarg);
                mLabel->setText("Right click to collect data \n"
                                "or Press 'R' to start reinforcement learning 1");
            }
        } else if ( e->key() == Qt::Key_T ) {
            mShowInTableCoordinateFrame = !mShowInTableCoordinateFrame;
        } else if ( e->key() == Qt::Key_R && gPlan && mRunReinforcementLearning1 == false){
            mRunReinforcementLearning1 = true;
            mLabel->setText("Training model, press SPACE to stop");
            Reinforcement_Learning_1();
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
          << "source ~/ws_ros2/install/setup.bash" << "\n"
          << "\n"
          << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
          << "\n"
          << "cd tm_robot_gripper/" << "\n"
          << "\n"
          << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
          << "\n"
          << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.1, -0.4, 0.4, -3.14, 0, 0], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
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
}

void LP_Plugin_Garment_Manipulation::loaddata(std::string fileName, std::vector<float> &datas){
    // Open the File
    std::ifstream in(fileName.c_str());
    std::string str;
    float data;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            data = std::stof(str);
            datas.push_back(data);
    }
    //Close The File
    in.close();
}

void LP_Plugin_Garment_Manipulation::findgrasp(std::vector<double> &grasp, cv::Point &grasp_point, cv::Point &center, std::vector<cv::Vec4i> hierarchy){
    int size = 0;
    int close_point = 999;
    int random_number = rand()%100;

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

    cv::Point2f warpedpg = cv::Point2f(grasp_point.x/static_cast<float>(imageWidth)*warped_image_size.width, grasp_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
    cv::Point3f homogeneousg = WarpMatrix.inv() * warpedpg;
    float depth_pointg[2] = {0}, color_pointg[2] = {homogeneousg.x/homogeneousg.z, homogeneousg.y/homogeneousg.z};
    auto depth = frames.get_depth_frame();
    rs2_project_color_pixel_to_depth_pixel(depth_pointg, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_pointg);

    offset[0] = 0; offset[1] = 0;
    graspx = int(depth_pointg[0]);
    graspy = int(depth_pointg[1]);
    mCalAveragePoint = true;
    avgHeight.clear();
    acount = 0;
    while (acount<40){
        Sleeper::msleep(200);
    }
    mCalAveragePoint = false;
    grasp[0] = grasppt.x();
    grasp[1] = grasppt.y();
    grasp[2] = gripper_length + grasppt.z();
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
        auto depth = frames.get_depth_frame();
        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
        int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);


        offset[0] = 0; offset[1] = 0;
        graspx = int(depth_point[0]);
        graspy = int(depth_point[1]);
        mCalAveragePoint = true;
        avgHeight.clear();
        acount = 0;
        while (acount<40){
            Sleeper::msleep(200);
        }
        mCalAveragePoint = false;
        grasp[0] = grasppt.x();
        grasp[1] = grasppt.y();
        grasp[2] = gripper_length + grasppt.z();
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
        auto depth = frames.get_depth_frame();
        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);

        offset[0] = 0; offset[1] = 0;
        graspx = int(depth_point[0]);
        graspy = int(depth_point[1]);
        mCalAveragePoint = true;
        avgHeight.clear();
        acount = 0;
        while (acount<40){
            Sleeper::msleep(200);
        }
        mCalAveragePoint = false;
        grasp[0] = grasppt.x();
        grasp[1] = grasppt.y();
        grasp[2] = gripper_length + grasppt.z();
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
                while (acount<40){
                    Sleeper::msleep(200);
                }
                mCalAveragePoint = false;
                grasp[0] = grasppt.x();
                grasp[1] = grasppt.y();
                grasp[2] = gripper_length + grasppt.z();
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
                auto depth = frames.get_depth_frame();
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
    auto depth = frames.get_depth_frame();
    rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
    int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
    mGraspP.resize(1);
    mGraspP[0] = mPointCloud[P];

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
                  << "source ~/ws_ros2/install/setup.bash" << "\n"
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
            auto depth = frames.get_depth_frame();

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
        auto depth = frames.get_depth_frame();
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
                  << "source ~/ws_ros2/install/setup.bash" << "\n"
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

void LP_Plugin_Garment_Manipulation::Reinforcement_Learning_1(){
    auto rl1current = QtConcurrent::run([this](){
        torch::manual_seed(0);
        if ( !mDetector ) {
            mDetector = std::make_shared<Detector>("/home/cpii/darknet-master/yolo_models/yolov3-df2.cfg", "/home/cpii/darknet-master/yolo_models/yolov3-df2_15000.weights");
        }

        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Training on GPU." << std::endl;
            device = torch::Device(torch::kCUDA);
        }

        torch::autograd::DetectAnomalyGuard detect_anomaly;

        qDebug() << "Creating models";

        auto policy = Policy();
        auto target_policy = Policy();
        auto critic = Critic();
        auto target_critic = Critic();

        policy->to(device);
        critic->to(device);
        target_policy->to(device);
        target_critic->to(device);

        qDebug() << "Creating optimizer";

        torch::optim::Adam policy_optimizer(policy->parameters(), torch::optim::AdamOptions(lrp));
        torch::optim::Adam critic_optimizer(critic->parameters(), torch::optim::AdamOptions(lrc));

        int dataid = 0, episode = 0;

        bool RestoreFromCheckpoint = false;
        const std::string kLogFile = "/home/cpii/projects/log/runs/model2_trial1/tfevents.pb";
        GOOGLE_PROTOBUF_VERIFY_VERSION;
        TensorBoardLogger logger(kLogFile.c_str());

        if (RestoreFromCheckpoint) {
            qDebug() << "Loading models";

            QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
            std::vector<float> saved_episode_num;
            loaddata(filename_episode_num.toStdString(), saved_episode_num);
            episode = saved_episode_num[0]-1;
            maxepisode += episode;

            episodecount = episode;

            QString Pmodelname = "policy_model_" + QString::number(episodecount) + ".pt";
            QString Poptimodelname = "policy_optimizer_" + QString::number(episodecount) + ".pt";
            QString Cmodelname = "critic_model_" + QString::number(episodecount) + ".pt";
            QString Coptimodelname = "critic_optimizer_" + QString::number(episodecount) + ".pt";

            torch::load(policy, Pmodelname.toStdString());
            torch::load(policy_optimizer, Poptimodelname.toStdString());
            torch::load(critic, Cmodelname.toStdString());
            torch::load(critic_optimizer, Coptimodelname.toStdString());

            qDebug() << "Loading memory";
            QString filename_memorysize = QString(memoryPath + "/memorysize.txt");
            std::vector<float> memorysize;
            loaddata(filename_memorysize.toStdString(), memorysize);
            float datasize = memorysize[0];

            QString filename_dataid = QString(memoryPath + "/dataid.txt");
            std::vector<float> saved_dataid;
            loaddata(filename_dataid.toStdString(), saved_dataid);
            dataid = saved_dataid[0]+1;

            QString filename_explor_rate = QString(memoryPath + "/explor_rate.txt");
            std::vector<float> saved_exploration_rate;
            loaddata(filename_explor_rate.toStdString(), saved_exploration_rate);
            exploration_rate = saved_exploration_rate[0];

            for(int i=0; i<datasize; i++){
                QString filename_id = memoryPath + QString("/%1").arg(dataid-datasize+i);

                QString filename_before_state = QString(filename_id + "/before_state.txt");
                std::vector<float> before_state_vector;
                loaddata(filename_before_state.toStdString(), before_state_vector);
                torch::Tensor before_state_tensor = torch::from_blob(before_state_vector.data(), { 262147 }, torch::kFloat);

                QString filename_place_point = QString(filename_id + "/place_point.txt");
                std::vector<float> place_point_vector;
                loaddata(filename_place_point.toStdString(), place_point_vector);
                torch::Tensor place_point_tensor = torch::from_blob(place_point_vector.data(), { 3 }, torch::kFloat);

                QString filename_reward = QString(filename_id + "/reward.txt");
                std::vector<float> reward_vector;
                loaddata(filename_reward.toStdString(), reward_vector);
                torch::Tensor reward_tensor = torch::from_blob(reward_vector.data(), { 1 }, torch::kFloat);

                QString filename_done = QString(filename_id + "/done.txt");
                std::vector<float> done_vector;
                loaddata(filename_done.toStdString(), done_vector);
                torch::Tensor done_tensor = torch::from_blob(done_vector.data(), { 1 }, torch::kFloat);

                QString filename_after_state = QString(filename_id + "/after_state.txt");
                std::vector<float> after_state_vector;
                loaddata(filename_after_state.toStdString(), after_state_vector);
                torch::Tensor after_state_tensor = torch::from_blob(after_state_vector.data(), { 262147 }, torch::kFloat);


                memory.push_back({before_state_tensor.clone(),
                                 place_point_tensor.clone(),
                                 reward_tensor.clone(),
                                 done_tensor.clone(),
                                 after_state_tensor.clone()});

//                std::cout << "before lowest: " << memory[i].before_state.min() << std::endl;
//                std::cout << "before highest: " << memory[i].before_state.max() << std::endl;
//                std::cout << "place_point: " << memory[i].place_point << std::endl;
//                std::cout << "reward: " << memory[i].reward << std::endl;
//                std::cout << "done: " << memory[i].done << std::endl;
//                std::cout << "after lowest: " << memory[i].after_state.min() << std::endl;
//                std::cout << "after highest: " << memory[i].after_state.max() << std::endl;
            }
            qDebug() << "Copying parameters to target models";
            torch::AutoGradMode hardcopy_disable(false);
            for (size_t i = 0; i < target_policy->parameters().size(); i++) {
                target_policy->parameters()[i].copy_(
                    target_policy->parameters()[i] * (1.0 - TAU) + policy->parameters()[i] * TAU);
            }
            for (size_t i = 0; i < target_critic->parameters().size(); i++) {
                target_critic->parameters()[i].copy_(
                    target_critic->parameters()[i] * (1.0 - TAU) + critic->parameters()[i] * TAU);
            }
        } else {
            qDebug() << "Copying parameters to target models";
            torch::AutoGradMode hardcopy_disable(false);
            for(size_t i=0; i < target_policy->parameters().size(); i++){
                target_policy->parameters()[i].copy_(policy->parameters()[i]);
            //    std::cout << "Pt: \n" << target_policy->parameters()[i].sizes() << std::endl;
            //    std::cout << "P: \n" << policy->parameters()[i].sizes() << std::endl;
            }

            for(size_t i=0; i < target_critic->parameters().size(); i++){
                target_critic->parameters()[i].copy_(critic->parameters()[i]);
            //    std::cout << "Ct: \n" << target_critic->parameters()[i].sizes() << std::endl;
            //    std::cout << "C: \n" << critic->parameters()[i].sizes() << std::endl;
            }
        }

        int step = 0, train_number = 0, random, failtimes;
        float episode_reward = 0, episode_critic_loss = 0, episode_policy_loss = 0;
        std::vector<float> done(1);
        torch::Tensor done_tensor;
        auto depth = frames.get_depth_frame();
        cv::Mat warped_image_copy;
        total_reward = 0; total_critic_loss = 0; total_policy_loss = 0;
        bool bResetRobot;

        try {
            while (episode < maxepisode) {
                qDebug() << "--------------------------------------------";
                qDebug() << "\033[0;34mEpisode " << episode+1 << " started\033[0m";

                // Initialize environment
                episode_reward = 0;
                episode_critic_loss = 0;
                episode_policy_loss = 0;
                done[0] = 0;
                step = 0;
                train_number = 0;
                failtimes = 0;
                bResetRobot = false;
                random = rand() % 1001;

                // Reset garment

                gCamimage.copyTo(Src);
                std::vector<double> graspr(3);
                cv::Point grasp_pointr, centerr;
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

                if (contours.size() == 0){
                    qDebug() << "No garment detected!";
                    return;
                }
                findgrasp(graspr, grasp_pointr, centerr, hierarchyr);

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
                          << "source ~/ws_ros2/install/setup.bash" << "\n"
                          << "\n"
                          << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                          << "\n"
                          << "cd tm_robot_gripper/" << "\n"
                          << "\n"
                          << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
        //                  << "\n"
        //                  << "sleep 1" << "\n"
                          << "\n"
                          << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << graspr[0] <<", " << graspr[1] <<", " << graspr[2]+0.1 <<", -3.14, 0, 0], velocity: 1.5, acc_time: 0.5, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                          << "\n"
                          << "sleep 2" << "\n"
                          << "\n"
                          << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << graspr[0] <<", " << graspr[1] <<", " << graspr[2] <<", -3.14, 0, 0], velocity: 1, acc_time: 0.3, blend_percentage: 0, fine_goal: 0}\"" << "\n"
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
                          << "\n"
                          << "sleep 3" << "\n"
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

                    dataid -= 1;
                    memory.pop_back();

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
                Sleeper::sleep(4);


                while (step < maxstep) {
                    //std::cout << "p fc4: \n" << policy->fc4->parameters() << std::endl;
                    //std::cout << "c fc2: \n" << critic->fc2->parameters() << std::endl;
                    qDebug() << "--------------------------------------------";
                    qDebug() << "\033[0;34mEpisode " << episode+1 << ": Step [" << step+1 << "/" << maxstep << "] started\033[0m";
                    qDebug() << "Exploration rate: " << exploration_rate;
                    bool exceed_limit = false;
                    cv::Point center;
                    cv::Mat inv_warp_image;
                    torch::Tensor place_point_tensor;
                    std::vector<float> pick_point(3), place_point(3), src_tableheight(warped_image.cols * warped_image.rows), after_tableheight(warped_image.cols * warped_image.rows);
                    std::vector<double> grasp(3), release(3);
                    cv::Point grasp_point, release_point;
                    torch::Tensor src_tensor, before_state, src_height_tensor, after_height_tensor;
                    float max_height_before = 0, max_height_after = 0, garment_area_before = 0, garment_area_after = 0, conf_before = 0, conf_after = 0;

                    if(step == 0){
                        // Preprocess environment
                        gCamimage.copyTo(Src);
                        cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                        cv::resize(warped_image, inv_warp_image, warped_image_size);
                        cv::warpPerspective(inv_warp_image, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
                        cv::resize(warped_image, warped_image, warped_image_resize);
                        warped_image = background - warped_image;
                        warped_image = ~warped_image;
                        src_tensor = torch::from_blob(warped_image.data, { 3, warped_image.rows, warped_image.cols }, at::kByte);
                        src_tensor = src_tensor.to(at::kFloat) / 255.0f;

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

                        if (contours.size() == 0){
                            qDebug() << "No garment detected!";
                            return;
                        }

                        for(int i=0; i<warped_image.rows; i++){
                            for(int j=0; j<warped_image.cols; j++){
                                auto PT_RGB = warped_image.at<cv::Vec3b>(i, j);
                                int id = i*warped_image.cols+j;
                                if((i >= (0.6*warped_image.rows) && j >= (0.6*warped_image.cols))
                                        || (PT_RGB[0] >= uThres && PT_RGB[1] >= uThres && PT_RGB[2] >= uThres)){
                                    src_tableheight[id] = 0.0;
                                } else {
                                    cv::Point2f warpedp = cv::Point2f(j/static_cast<float>(imageWidth)*warped_image_size.width, i/static_cast<float>(imageHeight)*warped_image_size.height);
                                    cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                                    float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                                    rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
                                    int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
                                    cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                                    cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                                    cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                                    float scale = dstMat(0,3);
                                    QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                                    if(Pt.z() < 0.20){
                                        src_tableheight[id] = Pt.z();
                                    } else {
                                        src_tableheight[id] = 0.0;
                                    }
                                    if(max_height_before < Pt.z() && Pt.z() < 0.20){
                                        max_height_before = Pt.z();
                                        mTestP.resize(1);
                                        mTestP[0] = mPointCloud[P];
                                    }
                                    garment_area_before+=1;
                                }
                            }
                        }
                        src_height_tensor = torch::from_blob(src_tableheight.data(), { src_tableheight.size() }, at::kFloat);

//                        src_height_tensor = torch::randn({256, 256});


                        float angle = 0;
                        cv::Mat rotatedImg;
                        QImage rotatedImgqt;
                        for(int a = 0; a < 36; a++){
                            rotatedImgqt = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888);

                            QMatrix r;

                            r.rotate(angle*10.0);

                            rotatedImgqt = rotatedImgqt.transformed(r);

                            rotatedImg = cv::Mat(rotatedImgqt.height(), rotatedImgqt.width(), CV_8UC3, rotatedImgqt.bits());

                            std::vector<bbox_t> test_result = mDetector->detect(rotatedImg);

                            if(test_result.size()>0){
                                for(auto i=0; i<test_result.size(); i++){
                                    if(test_result[i].obj_id == 1 && conf_before < test_result[i].prob && test_result[i].prob > 0.5){
                                        conf_before = test_result[i].prob;
                                    }
                                }
                            }
                            angle+=1;
                        }

                        findgrasp(grasp, grasp_point, center, hierarchy);

                        if (double(random) / 1000 <= exploration_rate){
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

                                    findgrasp(grasp, grasp_point, center, hierarchy);
                                    findrelease(release, release_point, grasp_point);

                                    cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                                      release_point.y*invImageH*warped_image_size.height);
                                    cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                                    float depth_pointr[2] = {0},
                                          color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                                    auto depth = frames.get_depth_frame();
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
                        torch::Tensor pick_point_tensor = torch::from_blob(pick_point.data(), { 3 }, at::kFloat);
                        auto src_tensor_flatten = torch::flatten(src_tensor);
                        auto src_height_tensor_flatten = torch::flatten(src_height_tensor);
                        before_state = torch::cat({ src_tensor_flatten, src_height_tensor_flatten, pick_point_tensor });
//                        }
//                        torch::Tensor pick_point_tensor = torch::randn({3});
//                        auto src_tensor_flatten = torch::flatten(src_tensor);
//                        auto src_height_tensor_flatten = torch::flatten(src_height_tensor);
//                        before_state = torch::cat({ src_tensor_flatten, src_height_tensor_flatten, pick_point_tensor });
                    } else {
                        before_state = after_state_last.clone().detach();
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
                    policy->train(false);
                    torch::AutoGradMode enable(true);
                    if (double(random) / 1000 <= exploration_rate) {
                        qDebug() << "\033[1;33mStart exploration\033[0m";
                        place_point[0] = release_point.x/static_cast<float>(imageWidth); place_point[1] = release_point.y/static_cast<float>(imageHeight); place_point[2] = release[2]-gripper_length;
                        place_point_tensor = torch::from_blob(place_point.data(), { 3 }, at::kFloat);
//                        place_point_tensor = torch::randn({3});
                        //std::cout << "place_point_tensor: " << place_point_tensor << std::endl;
                    } else {
                        qDebug() << "\033[1;33mStart exploitation\033[0m";
                        auto s = before_state.to(device);
                        place_point_tensor = policy->forward(s);
                        std::cout << "\033[1;34mAction predict: " << place_point_tensor << "\033[0m" << std::endl;
                        place_point_tensor = place_point_tensor.to(torch::kCPU);
                        place_point = std::vector(place_point_tensor.data_ptr<float>(), place_point_tensor.data_ptr<float>() + place_point_tensor.numel());
                        release_point.x = place_point[0]*static_cast<float>(imageWidth); release_point.y = place_point[1]*static_cast<float>(imageHeight);
                        cv::Point2f warpedpr = cv::Point2f(release_point.x/imageWidth*warped_image_size.width,
                                                          release_point.y/imageHeight*warped_image_size.height);
                        cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                        float depth_pointr[2] = {0},
                              color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                        auto depth = frames.get_depth_frame();
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
                        float distance1 = sqrt(grasp[0]*grasp[0]+grasp[1]*grasp[1]+(release[2]+0.05)*(release[2]+0.05));
                        float distance2 = sqrt(release[0]*release[0]+release[1]*release[1]+(release[2]+0.05)*(release[2]+0.05));
                        if(distance1 >= robotDLimit || distance2 >= robotDLimit
                           || release[2] < mTableHeight || release[2] > 0.3){
                            exceed_limit = true;
                        }
                        //qDebug() << "memory size: " << memory.size();
                    }

                    //qDebug() << "width: " << imageWidth << "height: " << imageHeight;
                    qDebug() << "\033[0;32mGrasp: " << grasp_point.x/static_cast<float>(imageWidth) << " "<< grasp_point.y/static_cast<float>(imageHeight) << " "<< grasp[2] << "\033[0m";
                    qDebug() << "\033[0;32mRelease: "<< place_point[0] << " " << place_point[1] << " " << release[2] << "\033[0m";

                    cv::circle( drawing,
                                release_point,
                                12,
                                cv::Scalar( 0, 255, 255 ),
                                3,//cv::FILLED,
                                cv::LINE_AA );
                    cv::arrowedLine(drawing, grasp_point, release_point, cv::Scalar( 0, 255, 0 ), 3, cv::LINE_AA, 0, 0.25);

                    if (exploration_rate > 0.1 && memory.size() > 20) {
                        exploration_rate *= exploration_rate_decay;
                    }

                    // Pick and place task

                    if (exceed_limit == true){
                        // Do nothing
                    } else {
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
                                  << "source ~/ws_ros2/install/setup.bash" << "\n"
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

                            dataid -= 1;
                            memory.pop_back();

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
                    Sleeper::sleep(4);


                    // Reward & State
                    std::vector<float> stepreward(1);
                    torch::Tensor after_state;

                    if(!exceed_limit){
                        gCamimage.copyTo(Src);
                        cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                        cv::resize(warped_image, inv_warp_image, warped_image_size);
                        cv::warpPerspective(inv_warp_image, OriginalCoordinates, WarpMatrix.inv(), cv::Size(srcw, srch)); // do perspective transformation
                        cv::resize(warped_image, warped_image, warped_image_resize);
                        warped_image = background - warped_image;
                        warped_image = ~warped_image;
                        torch::Tensor after_image_tensor = torch::from_blob(warped_image.data, { 3, warped_image.rows, warped_image.cols }, at::kByte);
                        after_image_tensor = after_image_tensor.to(at::kFloat) / 255.0f;

//                      torch::Tensor after_image_tensor = torch::randn({3, 256, 256});

                        for(int i=0; i<warped_image.rows; i++){
                           for(int j=0; j<warped_image.cols; j++){
                               auto PT_RGB = warped_image.at<cv::Vec3b>(i, j);
                               int id = i*warped_image.cols+j;
                               if((i >= (0.6*warped_image.rows) && j >= (0.6*warped_image.cols))
                                       || (PT_RGB[0] >= uThres && PT_RGB[1] >= uThres && PT_RGB[2] >= uThres)){
                                   after_tableheight[id] = 0.0;
                               } else {
                                   cv::Point2f warpedp = cv::Point2f(j/static_cast<float>(imageWidth)*warped_image_size.width, i/static_cast<float>(imageHeight)*warped_image_size.height);
                                   cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                                   float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                                   rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
                                   int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
                                   cv::Point3f Pc = {mPointCloud[P].x(), -mPointCloud[P].y(), -mPointCloud[P].z()};
                                   cv::Mat ptMat = (cv::Mat_<float>(4,1) << Pc.x, Pc.y, Pc.z, 1);
                                   cv::Mat_<float> dstMat(Transformationmatrix_T2C.inv() * ptMat);
                                   float scale = dstMat(0,3);
                                   QVector4D Pt(dstMat(0,0)/scale, dstMat(0,1)/scale, dstMat(0,2)/scale, 1.0f);
                                   if(Pt.z() < 0.20){
                                       after_tableheight[id] = Pt.z();
                                   } else {
                                       after_tableheight[id] = 0.0;
                                   }
                                   if(max_height_after < Pt.z() && Pt.z() < 0.20){
                                       max_height_after = Pt.z();
                                       mTestP.resize(1);
                                       mTestP[0] = mPointCloud[P];
                                   }
                                   garment_area_after+=1;
                               }
                           }
                        }
                        qDebug() << "\033[1;32mMax height after action: \033[0m" << max_height_after;
                        qDebug() << "\033[1;32mGarment area after action: \033[0m" << garment_area_after;
                        after_height_tensor = torch::from_blob(after_tableheight.data(), { after_tableheight.size() }, at::kFloat);

//                      after_height_tensor = torch::randn({256, 256});

                        float angle = 0;
                        cv::Mat rotatedImg;
                        QImage rotatedImgqt;
                        for(int a = 0; a < 36; a++){
                            rotatedImgqt = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888);

                            QMatrix r;

                            r.rotate(angle*10.0);

                            rotatedImgqt = rotatedImgqt.transformed(r);

                            rotatedImg = cv::Mat(rotatedImgqt.height(), rotatedImgqt.width(), CV_8UC3, rotatedImgqt.bits());

                            std::vector<bbox_t> test_result = mDetector->detect(rotatedImg);

                            if(test_result.size()>0){
                                for(auto i =0; i<test_result.size(); i++){
                                    if(test_result[i].obj_id == 1 && conf_after < test_result[i].prob && test_result[i].prob > 0.5){
                                        conf_after = test_result[i].prob;
                                    }
                                }
                            }
                            angle+=1;
                        }
                        qDebug() << "\033[1;32mClasscification confidence level after action: \033[0m" << conf_after;

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

                        if (contours.size() == 0){
                            qDebug() << "No garment detected!";
                            return;
                        }
                        findgrasp(grasp, grasp_point, center, hierarchy);

                        random = rand() % 1001;
                        if (double(random) / 1000 <= exploration_rate){
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

                                    findgrasp(grasp, grasp_point, center, hierarchy);
                                    findrelease(release, release_point, grasp_point);

                                    cv::Point2f warpedpr = cv::Point2f(release_point.x*invImageW*warped_image_size.width,
                                                                      release_point.y*invImageH*warped_image_size.height);
                                    cv::Point3f homogeneousr = WarpMatrix.inv() * warpedpr;
                                    float depth_pointr[2] = {0},
                                          color_pointr[2] = {homogeneousr.x/homogeneousr.z, homogeneousr.y/homogeneousr.z};
                                    auto depth = frames.get_depth_frame();
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

                        gLock.lockForWrite();
                        gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
                        gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
                        gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
                        gLock.unlock();

                        emit glUpdateRequest();

                        std::vector<float> pick_point2(3);
                        pick_point2[0] = grasp_point.x/static_cast<float>(imageWidth); pick_point2[1] = grasp_point.y/static_cast<float>(imageHeight); pick_point2[2] = grasp[2]-gripper_length;
                        torch::Tensor pick_point2_tensor = torch::from_blob(pick_point2.data(), { 3 }, at::kFloat);
                        auto after_image_flatten = torch::flatten(after_image_tensor);
                        auto after_height_tensor_flatten = torch::flatten(after_height_tensor);
                        after_state = torch::cat({ after_image_flatten, after_height_tensor_flatten, pick_point2_tensor });

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
                        garment_area_reward = 500 * ((garment_area_after-garment_area_before)/garment_area_before);
                        conf_reward = 1000 * (conf_after - conf_before);
                        stepreward[0] = height_reward + garment_area_reward + conf_reward;
                        qDebug() << "\033[1;31mReward from height: " << height_reward << " Reward from area: " << garment_area_reward << " Reward from classifier: " << conf_reward << "\033[0m";
                    } else {
                        after_state = before_state.clone().detach();
                    }

                    if (exceed_limit || (max_height_after < 0.2 && conf_after > 0.7)) { // Max height when garment unfolded is about 0.015m, conf level is about 0.75
                        done[0] = 1.0;
                        if(exceed_limit){
                            stepreward[0] += -3000;
                            for(int i=0;i<3;i++){
                                if(place_point[i]<0){
                                    stepreward[0] -= 1000 * (0 - place_point[i]);
                                } else if (place_point[i]>1){
                                    stepreward[0] -= 1000 * (place_point[i] - 1);
                                }
                            }
                            qDebug() << "\033[1;31mExceeds limit, end episode\033[0m";
                        }
                        if(max_height_after < 0.2 && conf_after > 0.7){
                            stepreward[0] += 5000;
                            qDebug() << "\033[1;31mGarment is unfolded, end episode\033[0m";
                        }
                    } else {
                        after_state_last = after_state.clone().detach();
                        for(int i=0; i<3; i++){
                            grasp_last[i] = grasp[i];
                            release_last[i] = release[i];
                        }
                        grasp_point_last = grasp_point;
                        release_point_last = release_point;
                        max_height_last = max_height_after;
                        garment_area_last = garment_area_after;
                        conf_last = conf_after;
                    }

                    qDebug() << "\033[1;31mStep reward: " << stepreward[0] << "\033[0m";
                    episode_reward += stepreward[0];

                    cv::Mat sub_image;
                    sub_image = warped_image_copy - warped_image;
                    auto mean = cv::mean(sub_image);
                    //qDebug() << "Pixel color diff mean: "<< mean[0] << " "<<mean[1]<< " "<< mean[2];
                    if(mean[0]<1.0 && mean[1]<1.0 && mean[2]<1.0 && done[0]!=1.0){
                        qDebug() << "\033[0;33mNothing Changed(mean<1.0), redo step\033[0m";
                        torch::AutoGradMode reset_disable(false);
                        failtimes ++;
                        exploration_rate /= exploration_rate_decay;
                        if(failtimes > 10){
                            bResetRobot = true;
                        }
                        if ( bResetRobot ) {
                            QMetaObject::invokeMethod(this, "resetRViz",
                                                      Qt::BlockingQueuedConnection);

                            qDebug() << "\033[1;31mResetting Rviz.\033[0m";

                            dataid -= 1;
                            memory.pop_back();

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

                    auto reward_tensor = torch::from_blob(stepreward.data(), { 1 }, torch::kFloat);
                    //std::cout << "reward_tensor: " << reward_tensor << std::endl;
                    done_tensor = torch::from_blob(done.data(), { 1 }, torch::kFloat);
                    if (memory.size() >= 10000) {
                        memory.pop_front();
                        memory.push_back({
                            before_state.clone(),
                            place_point_tensor.clone(),
                            reward_tensor.clone(),
                            done_tensor.clone(),
                            after_state.clone(),
                            });
                    } else {
                        memory.push_back({
                            before_state.clone(),
                            place_point_tensor.clone(),
                            reward_tensor.clone(),
                            done_tensor.clone(),
                            after_state.clone(),
                            });
    //                    std::cout << "before_state: " << memory[step].before_state.mean() << std::endl
    //                              << "place_point_tensor: " << memory[step].place_point_tensor << std::endl
    //                              << "reward_tensor: " << memory[step].reward_tensor << std::endl
    //                              << "done_tensor: " << memory[step].done_tensor << std::endl
    //                              << "after_state: " << memory[step].after_state.mean() << std::endl;
                    }


                    auto future_savedata = QtConcurrent::run([before_state_CPU = before_state.clone(),
                                                             place_point_CPU = place_point_tensor.clone(),
                                                             reward_CPU = reward_tensor.clone(),
                                                             done_CPU = done_tensor.clone(),
                                                             after_state_CPU = after_state.clone(),
                                                             dataid = dataid,
                                                             episode = episode,
                                                             this
                                                             ](){


                        std::vector<float> before_state_vector(before_state_CPU.data_ptr<float>(), before_state_CPU.data_ptr<float>() + before_state_CPU.numel());
                        std::vector<float> place_point_vector(place_point_CPU.data_ptr<float>(), place_point_CPU.data_ptr<float>() + place_point_CPU.numel());
                        std::vector<float> reward_vector(reward_CPU.data_ptr<float>(), reward_CPU.data_ptr<float>() + reward_CPU.numel());
                        std::vector<float> done_vector(done_CPU.data_ptr<float>(), done_CPU.data_ptr<float>() + done_CPU.numel());
                        std::vector<float> after_state_vector(after_state_CPU.data_ptr<float>(), after_state_CPU.data_ptr<float>() + after_state_CPU.numel());

                        QString filename_id = QString(memoryPath + "/%1").arg(dataid);
                        QDir().mkdir(filename_id);

                        QString filename_before_state = QString(filename_id + "/before_state.txt");
                        savedata(filename_before_state, before_state_vector);

                        QString filename_place_point = QString(filename_id + "/place_point.txt");
                        savedata(filename_place_point, place_point_vector);

                        QString filename_reward = QString(filename_id + "/reward.txt");
                        savedata(filename_reward, reward_vector);

                        QString filename_done = QString(filename_id + "/done.txt");
                        savedata(filename_done, done_vector);

                        QString filename_after_state = QString(filename_id + "/after_state.txt");
                        savedata(filename_after_state, after_state_vector);

                        std::vector<float> memorysize;
                        memorysize.push_back(memory.size());
                        QString filename_memorysize = QString(memoryPath + "/memorysize.txt");
                        savedata(filename_memorysize, memorysize);

                        std::vector<float> save_dataid;
                        save_dataid.push_back(dataid);
                        QString filename_dataid = QString(memoryPath + "/dataid.txt");
                        savedata(filename_dataid, save_dataid);
                    });

                    dataid++;
                    step++;

                    if (memory.size() > minisize) {
                        policy->train();
                        train_number++;
                        qDebug() << "\033[1;33mTraining model\033[0m";
                        int randomdata = rand()%(memory.size() - batch_size);
                        //qDebug() << "randomdata: " << randomdata;
                        //qDebug() << "memory size: " << memory.size();
                        std::vector<torch::Tensor> s_data(batch_size), a_data(batch_size), r_data(batch_size), d_data(batch_size), s2_data(batch_size);
                        torch::Tensor s_batch, a_batch, r_batch, d_batch, s2_batch;

                        for (int i = 0; i < batch_size; i++) {
                            s_data[i] = torch::unsqueeze(memory[i+randomdata].before_state, 0);
                            a_data[i] = torch::unsqueeze(memory[i+randomdata].place_point, 0);
                            r_data[i] = torch::unsqueeze(memory[i+randomdata].reward, 0);
                            d_data[i] = torch::unsqueeze(memory[i+randomdata].done, 0);
                            s2_data[i] = torch::unsqueeze(memory[i+randomdata].after_state, 0);
                        }

                        s_batch = s_data[0]; a_batch = a_data[0]; r_batch = r_data[0]; d_batch = d_data[0]; s2_batch = s2_data[0];
                        for (int i = 1; i < batch_size; i++) {
                            //std::cout << s_data[i].sizes() << std::endl << s_batch.sizes() << std::endl;
                            s_batch = torch::cat({ s_batch, s_data[i] }, 0);
                            //std::cout << a_data[i].sizes() << std::endl << a_batch.sizes() << std::endl;
                            a_batch = torch::cat({ a_batch, a_data[i] }, 0);
                            //std::cout << r_data[i].sizes() << std::endl << r_batch.sizes() << std::endl;
                            r_batch = torch::cat({ r_batch, r_data[i] }, 0);
                            //std::cout << d_data[i].sizes() << std::endl << d_batch.sizes() << std::endl;
                            d_batch = torch::cat({ d_batch, d_data[i] }, 0);
                            //std::cout << s2_data[i].sizes() << std::endl << s2_batch.sizes() << std::endl;
                            s2_batch = torch::cat({ s2_batch, s2_data[i] }, 0);
                        }
                        s_batch = s_batch.clone().detach().to(device);
                        a_batch = a_batch.clone().detach().to(device);
                        r_batch = r_batch.clone().detach().to(device);
                        d_batch = d_batch.clone().detach().to(device);
                        s2_batch = s2_batch.clone().detach().to(device);

                        // Critic loss
                        qDebug() << "\033[1;33mTraining critic model\033[0m";

                        auto a2_batch = target_policy->forward(s2_batch);
                        //std::cout << "a2_batch: " << a2_batch << std::endl;
                        //std::cout << "s2_batch max: " << s2_batch.max() << std::endl;
                        auto target_q = target_critic->forward(a2_batch, s2_batch);
                        //std::cout << "target_q: " << target_q << std::endl;
                        //std::cout << "r_batch: " << r_batch << std::endl;
                        //std::cout << "d_batch: " << d_batch << std::endl;
                        auto y = r_batch + (1.0 - d_batch) * GAMMA * target_q;
                        //std::cout << "y: " << y << std::endl;
                        //std::cout << "a_batch: " << a_batch << std::endl;
                        //std::cout << "s_batch max: " << s_batch.max() << std::endl;
                        auto q = critic->forward(a_batch, s_batch);
                        //std::cout << "q: " << q << std::endl;

                        torch::Tensor critic_loss = torch::mse_loss(q, y.detach());
                        //std::cout << "Critic loss: " << critic_loss << std::endl;
                        critic_optimizer.zero_grad();
                        critic_loss.backward();
                        critic_optimizer.step();

                        float critic_lossf = critic_loss.item().toFloat();
                        episode_critic_loss += critic_lossf;

                        qDebug() << "\033[1;33mCritic optimizer step\033[0m";

                        // Policy loss
                        qDebug() << "\033[1;33mTraining policy model\033[0m";

                        auto a_predict = policy->forward(s_batch);
                        std::cout << "policy network predict: " << a_predict << std::endl;
                        torch::Tensor policy_loss = critic->forward(a_predict, s_batch);
                        std::cout << "critic network predict: " << policy_loss << std::endl;
                        policy_loss = -policy_loss.mean();
                        //std::cout << "Policy loss: " << policy_loss << std::endl;
                        policy_optimizer.zero_grad();
                        policy_loss.backward();
                        policy_optimizer.step();

                        auto policy_lossf = policy_loss.item().toFloat();
                        episode_policy_loss += policy_lossf;

                        qDebug() << "\033[1;33mPolicy optimizer step\033[0m";

                        // Update target networks
                        qDebug() << "Updating target models";
                        torch::AutoGradMode softcopy_disable(false);
                        for (size_t i = 0; i < target_policy->parameters().size(); i++) {
                            target_policy->parameters()[i].copy_(
                                target_policy->parameters()[i] * (1.0 - TAU) + policy->parameters()[i] * TAU);
                        }
                        for (size_t i = 0; i < target_critic->parameters().size(); i++) {
                            target_critic->parameters()[i].copy_(
                                target_critic->parameters()[i] * (1.0 - TAU) + critic->parameters()[i] * TAU);
                        }

                        qDebug() << "\033[1;31mCritic loss: " << critic_lossf << "\n"
                                 << "Policy loss: " << policy_lossf << "\033[0m";
                    }
                    qDebug() << "\033[0;34mEpisode " << episode+1 << ": Step [" << step << "/" << maxstep << "] finished\033[0m";

                    if (done[0] == 1.0){
                        break;
                    }
                }

                // Save
                episode++;

                episode_critic_loss = episode_critic_loss / (float)train_number;
                episode_policy_loss = episode_policy_loss / (float)train_number;
                //total_reward += episode_reward;
                //total_critic_loss += episode_critic_loss;
                //float avg_critic_loss = total_critic_loss / (float)episode;
                //total_policy_loss += episode_policy_loss;
                //float avg_policy_loss = total_policy_loss / (float)episode;

                qDebug() << "\033[0;35m--------------------------------------------" << "\n"
                    << "Episode: " << episode << "\n"
                    << "Done(1:yes, 0:no): " << done[0] << "\n"
                    << "Reward: " << episode_reward << "\n"
                    //<< "Average Reward: " << total_reward / (float)episode << "\n"
                    << "Critic Loss: " << episode_critic_loss << "\n"
                    //<< "Average Critic Loss: " << avg_critic_loss << "\n"
                    << "Policy Loss: " << episode_policy_loss << "\n"
                    //<< "Average Policy Loss: " << avg_policy_loss << "\n"
                    << "--------------------------------------------\033[0m";
                logger.add_scalar("Episode_Reward", episode, episode_reward);
                //logger.add_scalar("Average_Reward", episode, total_reward / (float)episode);
                logger.add_scalar("Episode_Critic_Loss", episode, episode_critic_loss);
                //logger.add_scalar("Average_Critic_Loss", episode, avg_critic_loss);
                logger.add_scalar("Episode_Policy_Loss", episode, episode_policy_loss);
                //logger.add_scalar("Average_Policy_Loss", episode, avg_policy_loss);

                if (episode % 10 == 0) {
                    qDebug() << "Saving models";
                    QString Pmodelname = "policy_model_" + QString::number(episode) + ".pt";
                    QString Poptimodelname = "policy_optimizer_" + QString::number(episode) + ".pt";
                    QString Cmodelname = "critic_model_" + QString::number(episode) + ".pt";
                    QString Coptimodelname = "critic_optimizer_" + QString::number(episode) + ".pt";
                    torch::save(policy, Pmodelname.toStdString());
                    torch::save(policy_optimizer, Poptimodelname.toStdString());
                    torch::save(critic, Cmodelname.toStdString());
                    torch::save(critic_optimizer, Coptimodelname.toStdString());
                    std::vector<float> save_episode_num;
                    save_episode_num.push_back(episode+1);
                    QString filename_episode_num = QString(memoryPath + "/episode_num.txt");
                    savedata(filename_episode_num, save_episode_num);

                    std::vector<float> save_exploration_rate;
                    save_exploration_rate.push_back(exploration_rate);
                    QString filename_explor_rate = QString(memoryPath + "/explor_rate.txt");
                    savedata(filename_explor_rate, save_exploration_rate);
                    qDebug() << "Models saved";
                }

                qDebug() << "\033[0;34mEpisode " << episode << "finished\033[0m\n"
                         << "--------------------------------------------";

                if(mRunReinforcementLearning1 == false){
                    qDebug() << "Quit Reinforcement Learning 1" ;
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
=======
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

/**
 * @brief BulletPhysics Headers
 */
//#include <BulletSoftBody/btSoftBody.h>
//#include <Bullet3Dynamics/b3CpuRigidBodyPipeline.h>



double pi = M_PI;
cv::Mat gCamimage, Src, warped_image, background;
cv::Matx33f WarpMatrix;
cv::Size warped_image_size;
rs2::pointcloud pc;
rs2::points points;
rs2::device dev;
rs2_intrinsics depth_i, color_i;
rs2_extrinsics d2c_e, c2d_e;
float depth_scale;
int srcw, srch, depthw, depthh, imageWidth, imageHeight, graspp;
double averagegp;
int thresh = 70, frame = 0, markercount = 0, datanumber = 0, acount = 0;
cv::RNG rng(12345);
void Robot_Plan(int, void* );
cv::Mat cameraMatrix, distCoeffs;
std::vector<cv::Vec3d> rvecs, tvecs;
std::vector<cv::Point2f> roi_corners(4), midpoints(4), dst_corners(4);
std::vector<float> trans(3), markercenter(2);
cv::Ptr<cv::aruco::Dictionary> dictionary;
QMatrix4x4 depthtrans, depthinvtrans, depthrotationsx, depthrotationsy, depthrotationszx, depthrotationsinvzx, depthrotationszy, depthrotationsinvzy;


// open the first webcam plugged in the computer
//cv::VideoCapture camera1(4); // Grey: 2, 8. Color: 4, 10.

bool gStopFindWorkspace = false, gPlan = false, gQuit = false;
QFuture<void> gFuture;
QImage gNullImage, gCurrentGLFrame, gEdgeImage, gWarpedImage, gInvWarpImage;
QReadWriteLock gLock;

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


LP_Plugin_Garment_Manipulation::~LP_Plugin_Garment_Manipulation()
{
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
}

QWidget *LP_Plugin_Garment_Manipulation::DockUi()
{
    mWidget = std::make_shared<QWidget>();
    QVBoxLayout *layout = new QVBoxLayout(mWidget.get());

    mLabel = new QLabel("Right click to find the workspace");

    layout->addWidget(mLabel);

    mWidget->setLayout(layout);
    return mWidget.get();
}

class Sleeper : public QThread
{
public:
    static void usleep(unsigned long usecs){QThread::usleep(usecs);}
    static void msleep(unsigned long msecs){QThread::msleep(msecs);}
    static void sleep(unsigned long secs){QThread::sleep(secs);}
};

bool LP_Plugin_Garment_Manipulation::Run()
{
    srand((unsigned)time(NULL));
    datanumber = 0;
    frame = 0;
    markercount = 0;
    gStopFindWorkspace = false;
    gPlan = false;
    gQuit = false;
    mCalAveragePoint = false;
    gFindBackground = false;

    //calibrate();
    //return false;

    rs2::pipeline_profile profile = pipe.start();
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


    gFuture = QtConcurrent::run([this](){        

//        int countx = 0;
//        int county = 360;

        while(!gQuit)
        {
            // Wait for frames and get them as soon as they are ready
            frames = pipe.wait_for_frames();

            // Our rgb frame
            rs2::frame rgb = frames.get_color_frame();
            pc.map_to(rgb);

            // Let's get our depth frame
            auto depth = frames.get_depth_frame();
            depthw = depth.get_width();
            depthh = depth.get_height();

            // Device information
            depth_i = depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            color_i = rgb.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            d2c_e = depth.get_profile().as<rs2::video_stream_profile>().get_extrinsics_to(rgb.get_profile());
            c2d_e = rgb.get_profile().as<rs2::video_stream_profile>().get_extrinsics_to(depth.get_profile());
            rs2::depth_sensor ds = dev.query_sensors().front().as<rs2::depth_sensor>();
            depth_scale = ds.get_depth_scale();
        //                float fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy, distC1 = j.coeffs[0], distC2 = j.coeffs[1], distC3 = j.coeffs[2], distC4 = j.coeffs[3], distC5 = j.coeffs[4];
        //                qDebug()<< "fx: "<< fx << "fy: "<< fy << "cx: "<< cx << "cy: "<< cy << "coeffs: "<< distC1 << " "<< distC2 << " "<< distC3 << " "<< distC4 << " "<< distC5;
        //                QMatrix4x4 K = {fx,   0.0f,   cx, 0.0f,
        //                                0.0f,   fy,   cy, 0.0f,
        //                                0.0f, 0.0f, 1.0f, 0.0f,
        //                                0.0f, 0.0f, 0.0f, 0.0f};

            // Generate the pointcloud and texture mappings
            points = pc.calculate(depth);
            auto vertices = points.get_vertices();

            // Let's convert them to QImage
            auto q_rgb = realsenseFrameToQImage(rgb);

            cv::Mat camimage = cv::Mat(q_rgb.height(),q_rgb.width(), CV_8UC3, q_rgb.bits());
            cv::cvtColor(camimage, camimage, cv::COLOR_BGR2RGB);

//            qDebug()<< "depthw: "<< depthw <<"depthh: " << depthh<< "q_rgbh: "<<q_rgb.height()<<"q_rgbw: "<<q_rgb.width();

            srcw = camimage.cols;
            srch = camimage.rows;

            camimage.copyTo(gCamimage);

            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f>> corners;
            cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
            params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_CONTOUR;
            cv::aruco::detectMarkers(camimage, dictionary, corners, ids, params);


            // if at least one marker detected
            if (ids.size() > 0) {

                cv::aruco::drawDetectedMarkers(camimage, corners, ids);

                cv::aruco::estimatePoseSingleMarkers(corners, 0.041, cameraMatrix, distCoeffs, rvecs, tvecs);

                // Get location of the table
                    std::vector<int> detected_markers(3);

                    // draw axis for each marker
                    for(auto i=0; i<ids.size(); i++){
                        cv::aruco::drawAxis(camimage, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
                        if(markercount<=200 && !gStopFindWorkspace){
                            if(ids[i] == 97){
                                detected_markers[0] = 97;
                                std::vector< cv::Point3f> table_corners_3d;
                                std::vector< cv::Point2f> table_corners_2d;
                                table_corners_3d.push_back(cv::Point3f(-0.02, 0.94,  0.0));
                                table_corners_3d.push_back(cv::Point3f( 0.98, 0.94,  0.0));
                                table_corners_3d.push_back(cv::Point3f( 0.98,-0.06,  0.0));
                                table_corners_3d.push_back(cv::Point3f(-0.02,-0.06,  0.0));
                                cv::projectPoints(table_corners_3d, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, table_corners_2d);
                                roi_corners[0].x = roi_corners[0].x + table_corners_2d[0].x;
                                roi_corners[0].y = roi_corners[0].y + table_corners_2d[0].y;
                                roi_corners[1].x = roi_corners[1].x + table_corners_2d[1].x;
                                roi_corners[1].y = roi_corners[1].y + table_corners_2d[1].y;
                                roi_corners[2].x = roi_corners[2].x + table_corners_2d[2].x;
                                roi_corners[2].y = roi_corners[2].y + table_corners_2d[2].y;
                                roi_corners[3].x = roi_corners[3].x + table_corners_2d[3].x;
                                roi_corners[3].y = roi_corners[3].y + table_corners_2d[3].y;
                                markercount = markercount + 1;
                            } else if (ids[i] == 98){
                                detected_markers[1] = 98;
//                                std::vector< cv::Point3f> table_corners_3d;
//                                std::vector< cv::Point2f> table_corners_2d;
//                                table_corners_3d.push_back(cv::Point3f(-0.02, 0.045,  0.0));
//                                table_corners_3d.push_back(cv::Point3f( 0.98, 0.045,  0.0));
//                                table_corners_3d.push_back(cv::Point3f( 0.98,-0.955,  0.0));
//                                table_corners_3d.push_back(cv::Point3f(-0.02,-0.955,  0.0));
//                                cv::projectPoints(table_corners_3d, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, table_corners_2d);
//                                roi_corners[0].x = roi_corners[0].x + table_corners_2d[0].x;
//                                roi_corners[0].y = roi_corners[0].y + table_corners_2d[0].y;
//                                roi_corners[1].x = roi_corners[1].x + table_corners_2d[1].x;
//                                roi_corners[1].y = roi_corners[1].y + table_corners_2d[1].y;
//                                roi_corners[2].x = roi_corners[2].x + table_corners_2d[2].x;
//                                roi_corners[2].y = roi_corners[2].y + table_corners_2d[2].y;
//                                roi_corners[3].x = roi_corners[3].x + table_corners_2d[3].x;
//                                roi_corners[3].y = roi_corners[3].y + table_corners_2d[3].y;
//                                markercount = markercount + 1;
                            } else if (ids[i] == 99){
//                                std::vector< cv::Point3f> table_corners_3d;
//                                std::vector< cv::Point2f> table_corners_2d;
//                                table_corners_3d.push_back(cv::Point3f(-0.98, 0.04, 0.0));
//                                table_corners_3d.push_back(cv::Point3f( 0.02, 0.04, 0.0));
//                                table_corners_3d.push_back(cv::Point3f( 0.02,-0.96, 0.0));
//                                table_corners_3d.push_back(cv::Point3f(-0.98,-0.96, 0.0));
//                                cv::projectPoints(table_corners_3d, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, table_corners_2d);
//                                roi_corners[0].x = roi_corners[0].x + table_corners_2d[0].x;
//                                roi_corners[0].y = roi_corners[0].y + table_corners_2d[0].y;
//                                roi_corners[1].x = roi_corners[1].x + table_corners_2d[1].x;
//                                roi_corners[1].y = roi_corners[1].y + table_corners_2d[1].y;
//                                roi_corners[2].x = roi_corners[2].x + table_corners_2d[2].x;
//                                roi_corners[2].y = roi_corners[2].y + table_corners_2d[2].y;
//                                roi_corners[3].x = roi_corners[3].x + table_corners_2d[3].x;
//                                roi_corners[3].y = roi_corners[3].y + table_corners_2d[3].y;
//                                markercount = markercount + 1;
                            } else if (ids[i] == 0){
                                detected_markers[2] = 0;
//                                std::vector< cv::Point3f> table_corners_3d;
//                                std::vector< cv::Point2f> table_corners_2d;
//                                table_corners_3d.push_back(cv::Point3f(-0.52, 0.02, 0.0));
//                                table_corners_3d.push_back(cv::Point3f( 0.48, 0.02, 0.0));
//                                table_corners_3d.push_back(cv::Point3f( 0.48,-0.98, 0.0));
//                                table_corners_3d.push_back(cv::Point3f(-0.52,-0.98, 0.0));
//                                cv::projectPoints(table_corners_3d, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, table_corners_2d);
//                                roi_corners[0].x = roi_corners[0].x + table_corners_2d[0].x;
//                                roi_corners[0].y = roi_corners[0].y + table_corners_2d[0].y;
//                                roi_corners[1].x = roi_corners[1].x + table_corners_2d[1].x;
//                                roi_corners[1].y = roi_corners[1].y + table_corners_2d[1].y;
//                                roi_corners[2].x = roi_corners[2].x + table_corners_2d[2].x;
//                                roi_corners[2].y = roi_corners[2].y + table_corners_2d[2].y;
//                                roi_corners[3].x = roi_corners[3].x + table_corners_2d[3].x;
//                                roi_corners[3].y = roi_corners[3].y + table_corners_2d[3].y;
//                                markercount = markercount + 1;
                            }
                            if(markercount >= 200){
                                qDebug()<< "Alignment Done";
                            }
                        }
                    }

                    if(detected_markers[0] == 97 && detected_markers[1] == 98 && detected_markers[2] == 0 && frame <= 200){
                        for(auto i=0; i<ids.size(); i++){
                            if(ids[i] == 97){
                                for(auto j=i; j<ids.size(); j++){
                                    if(ids[j] == 98){
                                        for(auto k=j; k<ids.size(); k++){
                                            if(ids[k] == 0){
                                                frame = frame+1;

                                                markercenter[0] = corners[j][0].x;
                                                markercenter[1] = corners[j][0].y;

                                                float tmp_depth_point[2] = {0}, tmp_depth_pointi[2] = {0}, tmp_depth_pointj[2] = {0}, tmp_depth_pointk[2] = {0}, tmp_color_point[2] = {markercenter[0], markercenter[1]}, tmp_color_pointi[2] = {corners[i][0].x, corners[i][0].y}, tmp_color_pointj[2] = {corners[j][0].x, corners[j][0].y}, tmp_color_pointk[2] = {corners[k][0].x, corners[k][0].y};
                                                rs2_project_color_pixel_to_depth_pixel(tmp_depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_point);
                                                rs2_project_color_pixel_to_depth_pixel(tmp_depth_pointi, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_pointi);
                                                rs2_project_color_pixel_to_depth_pixel(tmp_depth_pointj, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_pointj);
                                                rs2_project_color_pixel_to_depth_pixel(tmp_depth_pointk, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_pointk);

                                                trans[0] = trans[0]*(frame-1) + vertices[(int)tmp_depth_point[0]+(int)tmp_depth_point[1]*depthw].z;
                                                trans[1] = trans[1]*(frame-1) - atan((vertices[(int)tmp_depth_pointk[0]+(int)tmp_depth_pointk[1]*depthw].z - vertices[(int)tmp_depth_pointj[0]+(int)tmp_depth_pointj[1]*depthw].z) / (vertices[(int)tmp_depth_pointk[0]+(int)tmp_depth_pointk[1]*depthw].y - vertices[(int)tmp_depth_pointj[0]+(int)tmp_depth_pointj[1]*depthw].y));
                                                trans[2] = trans[2]*(frame-1) - atan((vertices[(int)tmp_depth_pointi[0]+(int)tmp_depth_pointi[1]*depthw].z - vertices[(int)tmp_depth_pointj[0]+(int)tmp_depth_pointj[1]*depthw].z) / (vertices[(int)tmp_depth_pointi[0]+(int)tmp_depth_pointi[1]*depthw].x - vertices[(int)tmp_depth_pointj[0]+(int)tmp_depth_pointj[1]*depthw].x));

                                                for(int l=0; l<3; l++){
                                                    trans[l] = trans[l]/frame;
                                                }
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }              


//                    qDebug() <<"t1: "<< trans[1]<<"t2: "<< trans[2]<<"t3: "<< trans[3]<<"t4: "<< trans[4];
//                    qDebug()<< "mx: "<< markercenter[0]<<"my: "<<markercenter[1]<<"app: "<<markercenter[1]*depthw+markercenter[0];
//                    qDebug()<<"apx: "<<ap[(int)markercenter[1]*depthw+(int)markercenter[0]].x<<"apy: "<<ap[(int)markercenter[1]*depthw+(int)markercenter[0]].y<<"apz: "<< ap[(int)markercenter[1]*depthw+(int)markercenter[0]].z;
                    float depth_point[2] = {0}, color_point[2] = {markercenter[0], markercenter[1]};
                    rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);

                    depthtrans = {1.0f, 0.0f, 0.0f, -vertices[(int)depth_point[1]*depthw+(int)depth_point[0]].x,
                                  0.0f, 1.0f, 0.0f,  vertices[(int)depth_point[1]*depthw+(int)depth_point[0]].y,
                                  0.0f, 0.0f, 1.0f,                                                     trans[0],
                                  0.0f, 0.0f, 0.0f,                                                         1.0f};

                    depthinvtrans = {1.0f, 0.0f, 0.0f,  vertices[(int)depth_point[1]*depthw+(int)depth_point[0]].x,
                                     0.0f, 1.0f, 0.0f, -vertices[(int)depth_point[1]*depthw+(int)depth_point[0]].y,
                                     0.0f, 0.0f, 1.0f,                                                         0.0f,
                                     0.0f, 0.0f, 0.0f,                                                          1.0f};

                    depthrotationsx = {1.0f,          0.0f,           0.0f, 0.0f,
                                       0.0f, cos(trans[1]), -sin(trans[1]), 0.0f,
                                       0.0f, sin(trans[1]),  cos(trans[1]), 0.0f,
                                       0.0f,          0.0f,           0.0f, 1.0f};

                    depthrotationsy = { cos(trans[2]), 0.0f, sin(trans[2]), 0.0f,
                                                 0.0f, 1.0f,          0.0f, 0.0f,
                                       -sin(trans[2]), 0.0f, cos(trans[2]), 0.0f,
                                                 0.0f, 0.0f,          0.0f, 1.0f};
                }

                // Draw PointCloud
                mPointCloud.resize(depthw * depthh);
                mPointCloudTex.resize(depthw * depthh);

                auto tex_coords = points.get_texture_coordinates(); // and texture coordinates

                for ( int i=0; i<depthw; ++i ){
                    for ( int j=0; j<depthh; ++j ){
                        if (vertices[(depthh-j)*depthw-(depthw-i)].z){
                            mPointCloud[i*depthh + j] = QVector3D(vertices[(depthh-j)*depthw-(depthw-i)].x, -vertices[(depthh-j)*depthw-(depthw-i)].y, -vertices[(depthh-j)*depthw-(depthw-i)].z);
                            mPointCloudTex[i*depthh + j] = QVector2D(tex_coords[(depthh-j)*depthw-(depthw-i)].u, tex_coords[(depthh-j)*depthw-(depthw-i)].v);
                            if(ids.size() > 0){
                                mPointCloud[i*depthh + j] = depthinvtrans * depthrotationsy * depthrotationsx * depthtrans * mPointCloud[i*depthh + j];
                            }
                        }
                    }
                }

//                float depth_point1[2] = {0}, color_point1[2] = {corners[0][0].x, corners[0][0].y};
//                rs2_project_color_pixel_to_depth_pixel(depth_point1, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point1);
//                qDebug() << "97: " <<mPointCloud[(int)depth_point1[0]*depthh+(depthh - (int)depth_point1[1])].z();

//                float depth_point2[2] = {0}, color_point2[2] = {corners[1][0].x, corners[1][0].y};
//                rs2_project_color_pixel_to_depth_pixel(depth_point2, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point2);
//                qDebug() << "98: " <<mPointCloud[(int)depth_point2[0]*depthh+(depthh - (int)depth_point2[1])].z();

//                float depth_point3[2] = {0}, color_point3[2] = {corners[3][0].x, corners[3][0].y};
//                rs2_project_color_pixel_to_depth_pixel(depth_point3, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point3);
//                qDebug() << "0: " <<mPointCloud[(int)depth_point3[0]*depthh+(depthh - (int)depth_point3[1])].z();

                // Test point
//                cv::Point2f ball;
//                ball.x = 200;
//                ball.y = 500;

//                cv::circle( camimage,
//                            ball,
//                            15,
//                            cv::Scalar( 0, 0, 255 ),
//                            cv::FILLED,
//                            cv::LINE_8 );


//                float depth_point[2] = {0}, color_point[2] = {ball.x, ball.y};
//                rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
//                int P = int(depth_point[0])*depthh + (depthh-int(depth_point[1]));

//                qDebug()<< "resultx: "<<result.x << "resulty: " << result.y << "resultx2: "<< result2[0] << "resulty2: "<< result2[1] << "P: "<< P;
//                qDebug()<< "resultx: "<<result.x/srcw << "resulty: " << result.y/srch;
//                qDebug()<< "depth_point: "<< depth_point[0]/depthw<< " "<< depth_point[1]/depthh;

//                assert(P < mPointCloud.size());
//                qDebug() << ball.x / srcw << ", " << ball.y / srch << ":" << result.x / depthw << ", " << result.y / depthh ;
//                qDebug() << "texcorx: " << mPointCloudTex[P].x() << "texcory: "<< mPointCloudTex[P].y() << "\n";

//                assert(depthh*depthw == mPointCloud.size());

//                mTestP.resize(1);
//                mTestP[0] = mPointCloud[P];
//                countx = countx + 2;
//                if(countx == 1280){
//                    county = county + 1;
//                    countx = 0;
//                }

                if(mCalAveragePoint && acount<100){
                    averagegp = averagegp + mPointCloud[graspp].z();
                    acount++;
                }

                // And finally we'll emit our signal
                gLock.lockForWrite();
                gCurrentGLFrame = QImage((uchar*) camimage.data, camimage.cols, camimage.rows, camimage.step, QImage::Format_BGR888);
                gLock.unlock();
                emit glUpdateRequest();
    }
    });
    return false;
}

bool LP_Plugin_Garment_Manipulation::eventFilter(QObject *watched, QEvent *event)
{
    if ( QEvent::MouseButtonRelease == event->type()){
        auto e = static_cast<QMouseEvent*>(event);

        if ( e->button() == Qt::RightButton ){
            if (markercount==0){
                qDebug("No marker data!");
            } else if (!gFindBackground){
                roi_corners[0].x = round(roi_corners[0].x / markercount);
                roi_corners[0].y = round(roi_corners[0].y / markercount);
                roi_corners[1].x = round(roi_corners[1].x / markercount);
                roi_corners[1].y = round(roi_corners[1].y / markercount);
                roi_corners[2].x = round(roi_corners[2].x / markercount);
                roi_corners[2].y = round(roi_corners[2].y / markercount);
                roi_corners[3].x = round(roi_corners[3].x / markercount);
                roi_corners[3].y = round(roi_corners[3].y / markercount);
                Robot_Plan( 0, 0 );
                gFindBackground = true;
                mLabel->setText("Right click to plan");
            } else if (!gStopFindWorkspace) {
                gStopFindWorkspace = true;
                Robot_Plan( 0, 0 );
                mLabel->setText("Right click to get the garment");
            } else if (!gPlan) {
                gPlan = true;
                QProcess *openrviz, *plan = new QProcess();
                QStringList openrvizarg, planarg;

                openrvizarg << "/home/cpii/projects/scripts/openrviz.sh";
                planarg << "/home/cpii/projects/scripts/move.sh";

                openrviz->startDetached("xterm", openrvizarg);

                Sleeper::sleep(3);

                plan->startDetached("xterm", planarg);
                mLabel->setText("Right click to collect data");
            } else if (gPlan) {
                mLabel->setText("Press SPACE to quit");
                if ( !mRunCollectData ) {
                    mRunCollectData = true;
                    auto future = QtConcurrent::run([this](){
                        while(mRunCollectData){
                            Robot_Plan(0, 0);

                            QProcess *unfold = new QProcess();
                            QStringList unfoldarg;

                            unfoldarg << "/home/cpii/projects/scripts/unfold.sh";

                            unfold->startDetached("xterm", unfoldarg);

                            Sleeper::sleep(40);

                            // Save data
                            gCamimage.copyTo(Src);
                            cv::warpPerspective(Src, warped_image, WarpMatrix, warped_image_size); // do perspective transformation
                            cv::resize(warped_image, warped_image, cv::Size(500,500));
                            warped_image = background - warped_image;
                            std::vector<std::string> points;
                            for (auto i=0; i<mPointCloud.size(); i++){
                                points.push_back(std::to_string(mPointCloud[i].z()));
                            }
                            std::vector<cv::Point2f> OriTablePoints;
                            for(int i=0; i<warped_image.cols; i++){
                                for(int j=0; j<warped_image.rows; j++){
                                    cv::Point2f warpedp = cv::Point2f(i/static_cast<float>(imageWidth)*warped_image_size.width, j/static_cast<float>(imageHeight)*warped_image_size.height);
                                    cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                                    float color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                                    OriTablePoints.emplace_back(cv::Point2f(color_point[0], color_point[1]));
                                }
                            }
                            std::vector<std::string> tablepoints;
                    //        mTestP.resize(OriTablePoints.size());
                            for (int i=0; i<OriTablePoints.size(); i++){
                                float tmp_depth_point[2] = {0}, tmp_color_point[2] = {OriTablePoints[i].x, OriTablePoints[i].y};
                                auto depth = frames.get_depth_frame();
                                rs2_project_color_pixel_to_depth_pixel(tmp_depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_point);
                                int tmpTableP = int(tmp_depth_point[0])*depthh + (depthh-int(tmp_depth_point[1]));
                    //            mTestP[i] = mPointCloud[tmpTableP];
                                tablepoints.push_back(std::to_string(mPointCloud[tmpTableP].z()));
                            }

                            // Save points
                            QString filename_points = QString("/home/cpii/projects/data/%1/after_points.txt").arg(datanumber);
                            QByteArray filename_pointsc = filename_points.toLocal8Bit();
                            const char *filename_pointscc = filename_pointsc.data();
                            std::ofstream output_file(filename_pointscc);
                            std::ostream_iterator<std::string> output_iterator(output_file, "\n");
                            std::copy(points.begin(), points.end(), output_iterator);

                            // Save table points
                            QString filename_tablepoints = QString("/home/cpii/projects/data/%1/after_tablepoints.txt").arg(datanumber);
                            QByteArray filename_tablepointsc = filename_tablepoints.toLocal8Bit();
                            const char *filename_tablepointscc = filename_tablepointsc.data();
                            std::ofstream output_file2(filename_tablepointscc);
                            std::ostream_iterator<std::string> output_iterator2(output_file2, "\n");
                            std::copy(tablepoints.begin(), tablepoints.end(), output_iterator2);

                            // Save Src
                            QString filename_Src = QString("/home/cpii/projects/data/%1/after_Src.jpg").arg(datanumber);
                            QByteArray filename_Srcc = filename_Src.toLocal8Bit();
                            const char *filename_Srccc = filename_Srcc.data();
                            cv::imwrite(filename_Srccc, Src);

                            // Save warped image
                            QString filename_warped = QString("/home/cpii/projects/data/%1/after_warped_image.jpg").arg(datanumber);
                            QByteArray filename_warpedc = filename_warped.toLocal8Bit();
                            const char *filename_warpedcc = filename_warpedc.data();
                            cv::imwrite(filename_warpedcc, warped_image);

                            datanumber++;
                        }
                        qDebug() << "Quit CollectData()";
                    });
                } else {
                    qDebug() << "Collecting Data, press SPACE to stop";
                }
            }
        }
    } else if ( QEvent::KeyRelease == event->type()){
        auto e = static_cast<QKeyEvent*>(event);

        if ( e->key() == Qt::Key_Space ){
            if (gStopFindWorkspace && gPlan){
                mRunCollectData = false;
                QProcess *exit = new QProcess();
                QStringList exitarg;
                exitarg << "/home/cpii/projects/scripts/exit.sh";
                exit->startDetached("xterm", exitarg);
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


void LP_Plugin_Garment_Manipulation::Robot_Plan(int, void* )
{
    if(!gFindBackground){
        // Find the table
        midpoints[0] = (roi_corners[0] + roi_corners[1]) / 2;
        midpoints[1] = (roi_corners[1] + roi_corners[2]) / 2;
        midpoints[2] = (roi_corners[2] + roi_corners[3]) / 2;
        midpoints[3] = (roi_corners[3] + roi_corners[0]) / 2;
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

    cv::resize(warped_image, warped_image, cv::Size(500,500));

    if(!gFindBackground){
        background = warped_image;
        // Save background
        QString filename_Src = QString("/home/cpii/projects/data/background.jpg");
        QByteArray filename_Srcc = filename_Src.toLocal8Bit();
        const char *filename_Srccc = filename_Srcc.data();
        cv::imwrite(filename_Srccc, background);
        return;
    }
    warped_image = background - warped_image;

    gLock.lockForWrite();
    gWarpedImage = QImage((uchar*) warped_image.data, warped_image.cols, warped_image.rows, warped_image.step, QImage::Format_BGR888).copy();
    gLock.unlock();
    emit glUpdateRequest();

    gLock.lockForWrite();
    gInvWarpImage = QImage((uchar*) OriginalCoordinates.data, OriginalCoordinates.cols, OriginalCoordinates.rows, OriginalCoordinates.step, QImage::Format_BGR888).copy();
    gLock.unlock();
    emit glUpdateRequest();

    // Find contours
    cv::Mat src_gray;
    cv::cvtColor( warped_image, src_gray, cv::COLOR_BGR2GRAY );
    cv::blur( src_gray, src_gray, cv::Size(3,3) );

    cv::Mat canny_output;
    cv::Canny( src_gray, canny_output, thresh, thresh*2 );
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

    for( size_t i = 0; i< contours.size(); i++ ){
        cv::Scalar color = cv::Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        cv::drawContours( drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0 );

        for (size_t j = 0; j < contours[i].size(); j++){
//                std::cout << "\n" << i << " " << j << "Points with coordinates: x = " << contours[i][j].x << " y = " << contours[i][j].y;

            if ((static_cast<double>(contours[i][j].x) / imageWidth > 0.6
                    && static_cast<double>(contours[i][j].y) / imageHeight > 0.6 )
                    || (sqrt(pow((static_cast<double>(contours[i][j].x) / imageWidth - 0.85), 2) + pow((static_cast<double>(contours[i][j].y) / imageHeight - 0.85), 2)) > 0.85)
                    || (static_cast<double>(contours[i][j].x) / imageWidth > 0.90
                    && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)
                    || (static_cast<double>(contours[i][j].x) / imageWidth < 0.10
                    && static_cast<double>(contours[i][j].y) / imageHeight > 0.90)
                    || (static_cast<double>(contours[i][j].x) / imageWidth > 0.475
                    && static_cast<double>(contours[i][j].x) / imageWidth < 0.525
                    && static_cast<double>(contours[i][j].y) / imageHeight < 0.05))
            { // Filter out the robot arm and markers
                    size = size - 1;
            } else {
                center.x = center.x + contours[i][j].x;
                center.y = center.y + contours[i][j].y;
            }
        }
        size = size + contours[i].size();
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
                     || (sqrt(pow((static_cast<double>(contours[i][j].x) / imageWidth - 0.85), 2) + pow((static_cast<double>(contours[i][j].y) / imageHeight - 0.85), 2)) > 0.85)
                     || (static_cast<double>(contours[i][j].x) / imageWidth > 0.90
                     && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)
                     || (static_cast<double>(contours[i][j].x) / imageWidth < 0.10
                     && static_cast<double>(contours[i][j].y) / imageHeight > 0.90)
                     || (static_cast<double>(contours[i][j].x) / imageWidth > 0.475
                     && static_cast<double>(contours[i][j].x) / imageWidth < 0.525
                     && static_cast<double>(contours[i][j].y) / imageHeight < 0.05)){ // Filter out the robot arm and markers
                } else if (sqrt(pow((static_cast<double>(contours[i][j].x) - center.x), 2) + pow((static_cast<double>(contours[i][j].y) - center.y), 2)) < close_point){
                    close_point = sqrt(pow((static_cast<double>(contours[i][j].x) - center.x), 2) + pow((static_cast<double>(contours[i][j].y) - center.y), 2));
                    grasp_point.x = contours[i][j].x;
                    grasp_point.y = contours[i][j].y;
                    grasp[0] = 0.707106781*(static_cast<double>(grasp_point.y) / static_cast<double>(imageHeight)) + 0.707106781*(static_cast<double>(grasp_point.x) / static_cast<double>(imageWidth)) - 1.180868325;
                    grasp[1] = 0.707106781*(static_cast<double>(grasp_point.x) / static_cast<double>(imageWidth)) - 0.707106781*(static_cast<double>(grasp_point.y) / static_cast<double>(imageHeight));
                }
            }
        }

//        std::vector<int> ids;
//        std::vector<std::vector<cv::Point2f>> corners;
//        cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
//        params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_CONTOUR;
//        cv::aruco::detectMarkers(warped_image, dictionary, corners, ids, params);
//        for(int i=0; i<ids.size(); i++){
//            if(ids[i] == 81){
//                grasp[0] = 0.707106781*(static_cast<double>(corners[i][0].y) / static_cast<double>(imageHeight)) + 0.707106781*(static_cast<double>(corners[i][0].x) / static_cast<double>(imageWidth)) - 1.180868325;
//                grasp[1] = 0.707106781*(static_cast<double>(corners[i][0].x) / static_cast<double>(imageWidth)) - 0.707106781*(static_cast<double>(corners[i][0].y) / static_cast<double>(imageHeight));
//                grasp_point.x = corners[i][0].x;
//                grasp_point.y = corners[i][0].y;
//            }
//        }

    } else {
        int random_number = rand()%250;
        for( size_t i = 0; i< contours.size(); i++ ){
            for (size_t j = 0; j < contours[i].size(); j++){
                if ((static_cast<double>(contours[i][j].x) / imageWidth > 0.6
                     && static_cast<double>(contours[i][j].y) / imageHeight > 0.6 )
                     || (sqrt(pow((static_cast<double>(contours[i][j].x) / imageWidth - 0.85), 2) + pow((static_cast<double>(contours[i][j].y) / imageHeight - 0.85), 2)) > 0.85)
                     || (static_cast<double>(contours[i][j].x) / imageWidth > 0.90
                     && static_cast<double>(contours[i][j].y) / imageHeight < 0.10)
                     || (static_cast<double>(contours[i][j].x) / imageWidth < 0.10
                     && static_cast<double>(contours[i][j].y) / imageHeight > 0.90)
                     || (static_cast<double>(contours[i][j].x) / imageWidth > 0.475
                     && static_cast<double>(contours[i][j].x) / imageWidth < 0.525
                     && static_cast<double>(contours[i][j].y) / imageHeight < 0.05)){ // Filter out the robot arm and markers
                } else if (abs(sqrt(pow((static_cast<double>(contours[i][j].x) - center.x), 2) + pow((static_cast<double>(contours[i][j].y) - center.y), 2)) - random_number) < close_point){
                     close_point = abs(sqrt(pow((static_cast<double>(contours[i][j].x) - center.x), 2) + pow((static_cast<double>(contours[i][j].y) - center.y), 2)) - random_number);
                     grasp_point.x = contours[i][j].x;
                     grasp_point.y = contours[i][j].y;
                     grasp[0] = 0.707106781*(static_cast<double>(grasp_point.y) / static_cast<double>(imageHeight)) + 0.707106781*(static_cast<double>(grasp_point.x) / static_cast<double>(imageWidth)) - 1.180868325;
                     grasp[1] = 0.707106781*(static_cast<double>(grasp_point.x) / static_cast<double>(imageWidth)) - 0.707106781*(static_cast<double>(grasp_point.y) / static_cast<double>(imageHeight));

                }
            }
        }
        int random_release_dir = rand()%361;
        int random_release_dis = 50+rand()%150;
//        int c = 0;
        release_point = {(grasp_point.x+random_release_dis*cos(static_cast<double>(random_release_dir)/180*pi)), (grasp_point.y+random_release_dis*sin(static_cast<double>(random_release_dir)/180*pi))};
        while((release_point.x / static_cast<double>(imageWidth) > 0.6
              && release_point.y / static_cast<double>(imageHeight) > 0.6 )
              || (sqrt(pow((release_point.x / static_cast<double>(imageWidth) - 0.85), 2) + pow((release_point.y / static_cast<double>(imageHeight) - 0.85), 2)) > 0.85)
              || (release_point.x / static_cast<double>(imageWidth) > 0.90)
              || (release_point.x / static_cast<double>(imageWidth) < 0.10)
              || (release_point.y / static_cast<double>(imageHeight) > 0.90)
              || (release_point.y / static_cast<double>(imageHeight) < 0.10))
        { // Limit the release point
//            c++;
            random_release_dir = rand()%361;
            random_release_dis = 50+rand()%150;
            release_point = {(grasp_point.x+random_release_dis*cos(static_cast<double>(random_release_dir)/180*pi)), (grasp_point.y+random_release_dis*sin(static_cast<double>(random_release_dir)/180*pi))};
//            qDebug()<<"x: "<< release_point.x << "y: "<< release_point.y;
        }
//        qDebug() << c;
        release[0] = 0.707106781*(static_cast<double>(release_point.y) / static_cast<double>(imageHeight)) + 0.707106781*(static_cast<double>(release_point.x) / static_cast<double>(imageWidth)) - 1.202081528;
        release[1] = 0.707106781*(static_cast<double>(release_point.x) / static_cast<double>(imageWidth)) - 0.707106781*(static_cast<double>(release_point.y) / static_cast<double>(imageHeight));
        cv::circle( drawing,
                    release_point,
                    15,
                    cv::Scalar( 0, 255, 0 ),
                    cv::FILLED,
                    cv::LINE_8 );
    }

    cv::circle( drawing,
                grasp_point,
                15,
                cv::Scalar( 0, 0, 255 ),
                cv::FILLED,
                cv::LINE_8 );

    gLock.lockForWrite();
    gEdgeImage = QImage((uchar*) drawing.data, drawing.cols, drawing.rows, drawing.step, QImage::Format_BGR888).copy();
    gLock.unlock();
    emit glUpdateRequest();

    // Find the depth of grasp point
    cv::Point2f warpedp = cv::Point2f(grasp_point.x/static_cast<float>(imageWidth)*warped_image_size.width, grasp_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
    cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
    float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
    auto depth = frames.get_depth_frame();
    rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
    int P = int(depth_point[0])*depthh + depthh-int(depth_point[1]);

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

    acount = 0;
    averagegp = 0;
    graspp = P;
    mCalAveragePoint = true;
    while (acount<30){
        Sleeper::sleep(1);
    }
    mCalAveragePoint = false;
    grasp[2] = 0.249 + static_cast<double>(averagegp)/acount;

    if(grasp[2] <= 0.250){
        grasp[2] = 0.250;
    }

//    qDebug()<< grasp[2];

    mGraspP.resize(1);
    mGraspP[0] = mPointCloud[P];

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
                  << "source ~/ws_ros2/install/setup.bash" << "\n"
                  << "\n"
                  << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                  << "\n"
                  << "cd tm_robot_gripper/" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
                  << "\n"
                  << "sleep 3" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+0.1 <<", -3.14, 0, 0], velocity: 1, acc_time: 0.1, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 7" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2] <<", -3.14, 0, 0], velocity: 1, acc_time: 0.1, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 3" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
                  << "\n"
                  << "sleep 3" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [-0.4, 0, 0.7, -3.14, 0, 0], velocity: 1, acc_time: 0.1, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 3" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\""<< "\n"
                  << "\n"
                  << "sleep 3" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.2, -0.5, 0.4, -3.14, 0, 0], velocity: 1, acc_time: 1, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n";
        } else {
           qDebug("file open error");
        }
        file.close();
    } else {
        std::vector<std::string> points;
        for (auto i=0; i<mPointCloud.size(); i++){
            points.push_back(std::to_string(mPointCloud[i].z()));
        }
        std::vector<cv::Point2f> OriTablePoints;
        for(int i=0; i<warped_image.cols; i++){
            for(int j=0; j<warped_image.rows; j++){
                cv::Point2f warpedp = cv::Point2f(i/static_cast<float>(imageWidth)*warped_image_size.width, j/static_cast<float>(imageHeight)*warped_image_size.height);
                cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
                float color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
                OriTablePoints.emplace_back(cv::Point2f(color_point[0], color_point[1]));
            }
        }
        std::vector<std::string> tablepoints;
//        mTestP.resize(OriTablePoints.size());
        for (int i=0; i<OriTablePoints.size(); i++){
            float tmp_depth_point[2] = {0}, tmp_color_point[2] = {OriTablePoints[i].x, OriTablePoints[i].y};
            auto depth = frames.get_depth_frame();
            rs2_project_color_pixel_to_depth_pixel(tmp_depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, tmp_color_point);
            int tmpTableP = int(tmp_depth_point[0])*depthh + (depthh-int(tmp_depth_point[1]));
//            mTestP[i] = mPointCloud[tmpTableP];
            tablepoints.push_back(std::to_string(mPointCloud[tmpTableP].z()));
        }

        cv::Point2f warpedp = cv::Point2f(release_point.x/static_cast<float>(imageWidth)*warped_image_size.width, release_point.y/static_cast<float>(imageHeight)*warped_image_size.height);
        cv::Point3f homogeneous = WarpMatrix.inv() * warpedp;
        float depth_point[2] = {0}, color_point[2] = {homogeneous.x/homogeneous.z, homogeneous.y/homogeneous.z};
        auto depth = frames.get_depth_frame();
        rs2_project_color_pixel_to_depth_pixel(depth_point, reinterpret_cast<const uint16_t*>(depth.get_data()), depth_scale, 0.1, 10.0, &depth_i, &color_i, &c2d_e, &d2c_e, color_point);
        int PP = int(depth_point[0])*depthh + depthh-int(depth_point[1]);
        mReleaseP.resize(1);
        mReleaseP[0] = mPointCloud[PP];

        float height = (rand()%20 + 5)*0.01;

        // Save data
        // Create Directory
        QString filename_dir = QString("/home/cpii/projects/data/%1").arg(datanumber);
        QDir dir;
        dir.mkpath(filename_dir);

        // Save points
        QString filename_points = QString("/home/cpii/projects/data/%1/before_points.txt").arg(datanumber);
        QByteArray filename_pointsc = filename_points.toLocal8Bit();
        const char *filename_pointscc = filename_pointsc.data();
        std::ofstream output_file(filename_pointscc);
        std::ostream_iterator<std::string> output_iterator(output_file, "\n");
        std::copy(points.begin(), points.end(), output_iterator);

        // Save table points
        QString filename_tablepoints = QString("/home/cpii/projects/data/%1/before_tablepoints.txt").arg(datanumber);
        QByteArray filename_tablepointsc = filename_tablepoints.toLocal8Bit();
        const char *filename_tablepointscc = filename_tablepointsc.data();
        std::ofstream output_file2(filename_tablepointscc);
        std::ostream_iterator<std::string> output_iterator2(output_file2, "\n");
        std::copy(tablepoints.begin(), tablepoints.end(), output_iterator2);

        // Save Src
        QString filename_Src = QString("/home/cpii/projects/data/%1/before_Src.jpg").arg(datanumber);
        QByteArray filename_Srcc = filename_Src.toLocal8Bit();
        const char *filename_Srccc = filename_Srcc.data();
        cv::imwrite(filename_Srccc, Src);

        // Save warped image
        QString filename_warped = QString("/home/cpii/projects/data/%1/before_warped_image.jpg").arg(datanumber);
        QByteArray filename_warpedc = filename_warped.toLocal8Bit();
        const char *filename_warpedcc = filename_warpedc.data();
        cv::imwrite(filename_warpedcc, warped_image);

        // Save grasp and release points
        QString filename_grp = QString("/home/cpii/projects/data/%1/grasp_release_points.txt").arg(datanumber);
        QFile filep(filename_grp);
        if(filep.open(QIODevice::ReadWrite)) {
            QTextStream streamp(&filep);
            streamp << grasp_point.x << "\n"
                    << grasp_point.y << "\n"
                    << release_point.x << "\n"
                    << release_point.y << "\n"
                    << grasp[2]+height;
        }

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
                  << "source ~/ws_ros2/install/setup.bash" << "\n"
                  << "\n"
                  << "source ~/tm_robot_gripper/install/setup.bash" << "\n"
                  << "\n"
                  << "cd tm_robot_gripper/" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\"" << "\n"
                  << "\n"
                  << "sleep 3" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+0.1 <<", -3.14, 0, 0], velocity: 1, acc_time: 0.1, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 7" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2] <<", -3.14, 0, 0], velocity: 1, acc_time: 0.1, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 3" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 0}\"" << "\n"
                  << "\n"
                  << "sleep 3" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << grasp[0] <<", " << grasp[1] <<", " << grasp[2]+height <<", -3.14, 0, 0], velocity: 1, acc_time: 0.1, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 3" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [" << release[0] <<", " << release[1] <<", " << grasp[2]+0.1 <<", -3.14, 0, 0], velocity: 1, acc_time: 0.1, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n"
                  << "sleep 7" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_io tmr_msgs/srv/SetIO \"{module: 1, type: 1, pin: 0, state: 1}\""<< "\n"
                  << "\n"
                  << "sleep 3" << "\n"
                  << "\n"
                  << "ros2 service call /tmr/set_positions tmr_msgs/srv/SetPositions \"{motion_type: 2, positions: [0.2, -0.5, 0.4, -3.14, 0, 0], velocity: 1, acc_time: 1, blend_percentage: 0, fine_goal: 0}\"" << "\n"
                  << "\n";
        } else {
           qDebug("file open error");
        }
        file.close();
    }
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
        mProgram_R->setUniformValue("f_pointSize", 1.0f);
        mProgram_R->setUniformValue("u_tex", 0);

        mProgram_R->setAttributeArray("a_pos", mPointCloud.data());
        mProgram_R->setAttributeArray("a_tex", mPointCloudTex.data());

        f->glEnable(GL_PROGRAM_POINT_SIZE);
        f->glDrawArrays(GL_POINTS, 0, mPointCloud.size());

        mProgram_R->setAttributeArray("a_pos", mTestP.data());
        mProgram_R->setUniformValue("f_pointSize", 20.0f);
        f->glDrawArrays(GL_POINTS, 0, mTestP.size());

        mProgram_R->setAttributeArray("a_pos", mGraspP.data());
        mProgram_R->setUniformValue("f_pointSize", 20.0f);
        f->glDrawArrays(GL_POINTS, 0, mGraspP.size());

        mProgram_R->setAttributeArray("a_pos", mReleaseP.data());
        mProgram_R->setUniformValue("f_pointSize", 20.0f);
        f->glDrawArrays(GL_POINTS, 0, mReleaseP.size());

        texture1.release();

        std::vector<QVector3D> tmpPC;
        for(int i=0; i<100; i++){
            for(int j=0; j<50; j++){
                tmpPC.emplace_back(QVector3D(i*0.1f, j*0.1, 0.0f));
            }
        }
        mProgram_R->setUniformValue("f_pointSize", 1.0f);
        mProgram_R->setAttributeArray("a_pos", tmpPC.data());
        f->glDrawArrays(GL_POINTS, 0, tmpPC.size());


        mProgram_R->disableAttributeArray("a_pos");
        mProgram_R->disableAttributeArray("a_tex");
        mProgram_R->release();
        fbo->release();
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

    mProgram_R = prog;            //If everything is fine, assign to the member variable

    mInitialized_R = true;
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
>>>>>>> a85d1eeb58a4c32e00cb2801458a21871f605d8c
