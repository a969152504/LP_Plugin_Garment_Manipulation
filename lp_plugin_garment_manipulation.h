#ifndef LP_PLUGIN_LP_Plugin_Garment_Manipulation_H
#define LP_PLUGIN_LP_Plugin_Garment_Manipulation_H


#include <torch/torch.h>
#include <torch/script.h>

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

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
<<<<<<< HEAD

#include <yolo_v2_class.hpp>

#include <tensorboard_logger.h>
=======
>>>>>>> a85d1eeb58a4c32e00cb2801458a21871f605d8c

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
        void resetRViz();
        bool resetRobotPosition();
        void Robot_Plan(int, void* );
        void Reinforcement_Learning_1();
        void findgrasp(std::vector<double> &grasp, cv::Point &grasp_point, cv::Point &center, std::vector<cv::Vec4i> hierarchy);
        void findrelease(std::vector<double> &release, cv::Point &release_point, cv::Point grasp_point);
        void savedata(QString fileName, std::vector<float> datas);
        void loaddata(std::string fileName, std::vector<float> &datas);

private:
        bool mInitialized_L = false;
        bool mInitialized_R = false;
        bool gCameraDisplay = true;
        bool mRunCollectData = false;
<<<<<<< HEAD
        bool mRunReinforcementLearning1 = false;
        bool mCalAveragePoint = false;
        bool gFoundBackground = false;
        bool mShowInTableCoordinateFrame = false;
        double mTableHeight = 0.0;

        std::shared_ptr<QWidget> mWidget;
        QLabel *mLabel = nullptr;
        QLabel *mLabel2 = nullptr;
=======
        bool mCalAveragePoint = false;
        bool gFindBackground = false;
        std::shared_ptr<QWidget> mWidget;
        QLabel *mLabel = nullptr;
>>>>>>> a85d1eeb58a4c32e00cb2801458a21871f605d8c
        QPushButton *mbutton1 = nullptr;
        QOpenGLShaderProgram *mProgram_L = nullptr,
                             *mProgram_R = nullptr;

        static std::shared_ptr<QProcess> gProc_RViz;

        // Realsense configuration structure, it will define streams that need to be opened
//        rs2::config cfg;

        rs2::pipeline pipe;


        // Frames returned by our pipeline, they will be packed in this structure
        rs2::frameset frames;

        std::vector<QVector3D> mPointCloud, mPointCloudColor, mGraspP, mReleaseP, mTestP;
        std::vector<QVector2D> mPointCloudTex;

<<<<<<< HEAD
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
        rs2_intrinsics depth_i, color_i;
        rs2_extrinsics d2c_e, c2d_e;
        float depth_scale, height;
        int srcw, srch, depthw, depthh, imageWidth, imageHeight, graspx, graspy;
        cv::Point grasp_point_last, release_point_last;
        torch::Tensor after_state_last;
        float max_height_last, conf_last, garment_area_last;
        int thresh = 100, frame = 0, markercount = 0, datanumber = 0, acount = 0, useless_data = 0, gRobotResetCount = 0;
        cv::Mat cameraMatrix, distCoeffs;
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::Ptr<cv::aruco::Dictionary> dictionary;
        QMatrix4x4 depthtrans, depthinvtrans, depthrotationsx, depthrotationsy, depthrotationszx, depthrotationsinvzx, depthrotationszy, depthrotationsinvzy;
        float total_reward, total_critic_loss, total_policy_loss;
        int episodecount;

        std::shared_ptr<Detector> mDetector;

        void adjustHeight(double &in, const double &min);
=======
>>>>>>> a85d1eeb58a4c32e00cb2801458a21871f605d8c
        /**
         * @brief initializeGL initalize any OpenGL resource
         */
void initializeGL_L();
void initializeGL_R();

<<<<<<< HEAD

// QObject interface
protected:
void timerEvent(QTimerEvent *event);
=======
>>>>>>> a85d1eeb58a4c32e00cb2801458a21871f605d8c
};


#endif // LP_PLUGIN_LP_Plugin_Garment_Manipulation_H
