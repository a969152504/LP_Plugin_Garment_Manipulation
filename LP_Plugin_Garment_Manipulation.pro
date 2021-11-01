QT += gui widgets

TEMPLATE = lib
DEFINES += LP_PLUGIN_GARMENT_MANIPULATION_LIBRARY OPENCV

CONFIG += c++17

QMAKE_POST_LINK=$(MAKE) install

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    lp_plugin_garment_manipulation.cpp

HEADERS += \
    LP_Plugin_Garment_Manipulation_global.h \
    lp_plugin_garment_manipulation.h

# Default rules for deployment.
unix {
    target.path = /usr/lib
}
# Default rules for deployment.
target.path = $$OUT_PWD/../App/plugins/$$TARGET

!isEmpty(target.path): INSTALLS += target


win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../Model/release/ -lModel
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../Model/debug/ -lModel
else:unix:!macx: LIBS += -L$$OUT_PWD/../Model/ -lModel

INCLUDEPATH += $$PWD/../Model
DEPENDPATH += $$PWD/../Model

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../Functional/release/ -lFunctional
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../Functional/debug/ -lFunctional
else:unix:!macx: LIBS += -L$$OUT_PWD/../Functional/ -lFunctional

INCLUDEPATH += $$PWD/../Functional
DEPENDPATH += $$PWD/../Functional

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../OpenMesh/lib/ -lOpenMeshCore
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../OpenMesh/lib/ -lOpenMeshCored
else:unix:!macx: LIBS += -L$$PWD/../../OpenMesh/lib/ -lOpenMeshCore

INCLUDEPATH += $$PWD/../../OpenMesh/include
DEPENDPATH += $$PWD/../../OpenMesh/include

win32:CONFIG(release, debug|release): {
    LIBS += -L$$PWD/../../OpenCV/install/release/lib/ \
       -lopencv_core450 \
       -lopencv_videoio450 \
       -lopencv_imgproc450 \
       -lopencv_imgcodecs450 \
       -lopencv_dnn450

    INCLUDEPATH += $$PWD/../../OpenCV/install/release/include
    DEPENDPATH += $$PWD/../../OpenCV/install/release/include
}else:unix:!macx: {
    LIBS += -L$$PWD/../../OpenCV/install/lib/  \
        -lopencv_core \
        -lopencv_videoio \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -lopencv_dnn \
        -lopencv_aruco

INCLUDEPATH += $$PWD/../../../opencv/opencv-4.5.2/install/include/opencv4
DEPENDPATH += $$PWD/../../../opencv/opencv-4.5.2/install/include/opencv4
}


unix:!macx: LIBS += -L$$PWD/../../../librealsense-master/build/ -lrealsense2

INCLUDEPATH += $$PWD/../../../librealsense-master/include/librealsense2
DEPENDPATH += $$PWD/../../../librealsense-master/include/librealsense2

INCLUDEPATH += $$PWD/../../../librealsense-master/examples
DEPENDPATH += $$PWD/../../../librealsense-master/examples

unix:!macx: LIBS += -L$$PWD/../../../bullet3/build_cmake/install/lib/ \
    -lBulletSoftBody \
    -lBulletCollision \
    -lBulletDynamics \
    -lBulletRobotics

INCLUDEPATH += $$PWD/../../../bullet3/build_cmake/install/include/bullet
DEPENDPATH += $$PWD/../../../bullet3/build_cmake/install/include/bullet

INCLUDEPATH += $$PWD/../../../darknet-master/include/
DEPENDPATH += $$PWD/../../../darknet-master/include/

INCLUDEPATH += $$PWD/../../../libtorch/include/
DEPENDPATH += $$PWD/../../../libtorch/include/

INCLUDEPATH += $$PWD/../../../libtorch/include/torch/csrc/api/include
DEPENDPATH += $$PWD/../../../libtorch/include/torch/csrc/api/include

INCLUDEPATH += $$PWD/../../../tensorboard_logger-master/include
DEPENDPATH += $$PWD/../../../tensorboard_logger-master/include

INCLUDEPATH += $$PWD/../../../tensorboard_logger-master/build
DEPENDPATH += $$PWD/../../../tensorboard_logger-master/build

unix:!macx: LIBS += -L$$PWD/../../../tensorboard_logger-master/build \
   -ltensorboard_logger
