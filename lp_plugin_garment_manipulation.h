#ifndef LP_PLUGIN_LP_Plugin_Garment_Manipulation_H
#define LP_PLUGIN_LP_Plugin_Garment_Manipulation_H

#include <torch/torch.h>
#include <torch/script.h>
#include <tensorboard_logger.h>

#include "LP_Plugin_Garment_Manipulation_global.h"

#include "plugin/lp_actionplugin.h"

#include <QObject>
#include <QOpenGLBuffer>
#include <QCheckBox>
#include <QVector2D>
#include <QVector3D>
#include <QMatrix4x4>

#include "opencv2/aruco.hpp"
#include "opencv2/aruco/charuco.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/dnn.hpp"

#include <text/ocr.hpp>
#include <text/textDetector.hpp>
#include <text/swt_text_detection.hpp>

#include <tesseract/baseapi.h>
#include <allheaders.h>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include <yolo_v2_class.hpp>

#include <math.h>
#include <QProcess>
#include <QFile>
#include <QTextStream>
#include <QThread>
#include <QPushButton>

class QLabel;
class LP_ObjectImpl;
class QOpenGLShaderProgram;

/**
 * @brief The LP_Plugin_Garment_Manipulation class
 * Ros2 TM robot arm garment manipulation
 */

#define LP_Plugin_Garment_Manipulation_iid "cpii.rp5.SmartFashion.LP_Plugin_Garment_Manipulation/0.1"

class LP_Plugin_Garment_Manipulation : public LP_ActionPlugin
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID LP_Plugin_Garment_Manipulation_iid)
    Q_INTERFACES(LP_ActionPlugin)

public:
    virtual ~LP_Plugin_Garment_Manipulation();

        // LP_Functional interface
        QWidget *DockUi() override;
        bool Run() override;
        bool eventFilter(QObject *watched, QEvent *event) override;
        bool saveCameraParams(const std::string &filename, cv::Size imageSize, float aspectRatio, int flags,
                              const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, double totalAvgErr);

        // LP_ActionPlugin interface
        QString MenuName();
        QAction *Trigger();

signals:


        // LP_Functional interface
public slots:

        void FunctionalRender_L(QOpenGLContext *ctx, QSurface *surf, QOpenGLFramebufferObject *fbo, const LP_RendererCam &cam, const QVariant &options) override;
        void FunctionalRender_R(QOpenGLContext *ctx, QSurface *surf, QOpenGLFramebufferObject *fbo, const LP_RendererCam &cam, const QVariant &options) override;
        void calibrate();
        void update_data(rs2::frame_queue& data, rs2::frame& colorized_depth, rs2::points& points, rs2::pointcloud& pc, rs2::colorizer& color_map);
        void resetRViz();
        bool resetRobotPosition();
        void Robot_Plan(int, void* );
        void Generate_Data();
        void Env_reset(float &Rz, bool &bResetRobot, int &datasize, bool &end_training);
        void RobotReset_RunCamera();
        void Reinforcement_Learning_1();
        void Reinforcement_Learning_2();
        void Train_rod();
        void trans_old_data(int datasize, QString location, QString savepath);
        void findgrasp(std::vector<double> &grasp, cv::Point &grasp_point, float &Rz, std::vector<cv::Vec4i> hierarchy);
        void findrelease(std::vector<double> &release, cv::Point &release_point, cv::Point grasp_point);
        void Find_Rod_Angle(std::string type, cv::Point& point, float& angle);
        void Find_Angle_Between_Rods(float& angle_between_rods, float& yellow_angle, float& blue_angle);
        void Find_Square_Angle(float& Square_Angle);
        void Push_Down(cv::Point push_point, float& rod_angle, float push_distance, std::vector<float>& start_point, std::vector<float>& end_point);
        void Push_Up(cv::Point push_point, float& rod_angle, float push_distance, std::vector<float>& start_point, std::vector<float>& end_point);
        void Push_to_Center(cv::Point push_point, float& rod_angle, float push_distance, std::vector<float>& start_point, std::vector<float>& end_point);
        void Push_Rod(int push_point, float push_distance, bool use_clock);
        void Push_Square(int push_point, float push_distance, bool exceeded_limit);
        void savedata(QString fileName, std::vector<float> datas);
        void loaddata(std::string fileName, std::vector<float> &datas);
        void textbox_draw(cv::Mat src, std::vector<cv::Rect>& groups, std::vector<float>& probs, std::vector<int>& indexes);
        void findSquares( const cv::Mat& image, std::vector<std::vector<cv::Point>>& squares );
        void findLines( const cv::Mat image);
        void getIndex(std::vector<float> v, float K, int& index);

private:
        bool mInitialized_L = false;
        bool mInitialized_R = false;
        bool gCameraDisplay = true;
        bool mRunCollectData = false;
        bool mRunReinforcementLearning1 = false;
        bool mRunReinforcementLearning2 = false;
        bool mRunTrainRod = false;
        bool mGenerateData = false;
        bool mGetPlacePoint = false;
        bool mCalAveragePoint = false;
        bool mFindmarker = false;
        int mTarget_marker = 0;
        cv::Point mMarkerPosi;
        bool mCalAvgMat = false;
        bool gFoundBackground = false;
        bool mShowInTableCoordinateFrame = false;
        double mTableHeight = 0.0;
        bool manual_mode = true;
        bool use_filter = true;

        std::shared_ptr<QWidget> mWidget;
        QLabel *mLabel = nullptr;
        QLabel *mLabel2 = nullptr;
        QPushButton *mButton0 = nullptr;
        QPushButton *mButton1 = nullptr;
        QPushButton *mButton2 = nullptr;
        QPushButton *mButton3 = nullptr;
        QOpenGLShaderProgram *mProgram_L = nullptr,
                             *mProgram_R = nullptr;

        static std::shared_ptr<QProcess> gProc_RViz;

        // Realsense configuration structure, it will define streams that need to be opened
//        rs2::config cfg;

        std::string cam_num1 = "117222250105";
        std::string cam_num2 = "117122250165";

        rs2::pipeline pipe;

        // Frames returned by our pipeline, they will be packed in this structure
        rs2::frameset frames;

        std::vector<QVector3D> mPointCloud, mPointCloudColor, mGraspP, mReleaseP, mTestP;
        std::vector<QVector2D> mPointCloudTex;

        double pi = M_PI;
        cv::Mat gCamimage, Src, warped_image, background, saved_warped_image, drawing, OriginalCoordinates;
        std::vector<std::vector<cv::Point>> contours;
        cv::Matx33f WarpMatrix;
        cv::Matx44f Transformationmatrix_T2C;
        QMatrix4x4 Transformationmatrix_T2R;
        QVector3D grasppt;
        cv::Size warped_image_size, warped_image_resize;
        rs2::pointcloud pc;
        rs2::points points;
        rs2::device dev;
        // Declare depth colorizer for pretty visualization of depth data
        rs2::colorizer color_map;
        // Declaring two concurrent queues that will be used to enqueue and dequeue frames from different threads
        rs2::frame_queue original_data;
        rs2::frame_queue filtered_data;
        // Declare pointcloud objects, for calculating pointclouds and texture mappings
        rs2::pointcloud original_pc;
        rs2::pointcloud filtered_pc;
        // Declare objects that will hold the calculated pointclouds and colored frames
        // We save the last set of data to minimize flickering of the view
        rs2::frame colored_depth;
        rs2::frame filtered_frame;
        rs2::points original_points;
        rs2::points filtered_points;

        rs2_intrinsics depth_i, color_i;
        rs2_extrinsics d2c_e, c2d_e;
        float depth_scale, height;
        int srcw, srch, depthw, depthh, imageWidth, imageHeight, graspx, graspy;
        cv::Point grasp_point_last, release_point_last;
        torch::Tensor after_state_last, after_pick_point_last;
        cv::Mat after_image_last;
        float max_height_last, conf_last, garment_area_last, Rz_last;
        int thresh = 100, frame = 0, markercount = 0, datanumber = 0, acount = 0, useless_data = 0, gRobotResetCount = 0;
        cv::Mat cameraMatrix, distCoeffs;
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::Ptr<cv::aruco::Dictionary> dictionary, dictionary2;
        QMatrix4x4 depthtrans, depthinvtrans, depthrotationsx, depthrotationsy, depthrotationszx, depthrotationsinvzx, depthrotationszy, depthrotationsinvzy;
        float total_reward, total_critic_loss, total_policy_loss;
        int episodecount;
        QVector2D place_pointxy;

        std::shared_ptr<Detector> mDetector;

        void adjustHeight(double &in, const double &min);
        /**
         * @brief initializeGL initalize any OpenGL resource
         */
        void initializeGL_L();
        void initializeGL_R();


        // QObject interface
        protected:
        void timerEvent(QTimerEvent *event); //Reserved

        // LP_Functional interface
        public slots:
            void PainterDraw(QWidget *glW);
        };


#endif // LP_PLUGIN_LP_Plugin_Garment_Manipulation_H
