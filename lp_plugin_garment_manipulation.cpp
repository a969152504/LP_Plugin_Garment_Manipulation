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
