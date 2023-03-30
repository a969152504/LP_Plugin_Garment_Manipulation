#ifndef TEXTILES_H
#define TEXTILES_H

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/photo.hpp"


class TEXTILES
{
public:
    TEXTILES();
    ~TEXTILES(void);

    cv::Mat Generate_depthmap(cv::Mat depth_mat);

    cv::Mat Segmentation_background_subtraction(cv::Mat image);
    std::vector<cv::Point> Segmentation_compute_approximated_polygon(cv::Mat mask);

    cv::Mat DepthMapClustering_preprocess(cv::Mat depth_image, cv::Mat mask);
    cv::Mat DepthMapClustering_cluster_similar_regions(cv::Mat preprocessed_depth_image);

    std::vector<cv::Point> PickandPlacePoints_calculate_unfold_paths(cv::Mat labeled_image, std::vector<cv::Point> approximated_polygon);
    std::vector<int> PickandPlacePoints_calculate_bumpiness(cv::Mat labeled_image, std::vector<cv::Point> unfold_paths);
    std::vector<cv::Point> PickandPlacePoints_calculate_pick_and_place_points(cv::Mat labeled_image, std::vector<cv::Point> unfold_paths, std::vector<int> bumpiness);

    cv::Mat draw_segmentation_stage(cv::Mat img_src, cv::Mat mask, std::vector<cv::Point> approximated_polygon);
    cv::Mat draw_clustering_stage(cv::Mat image_src, cv::Mat labeled_image);
};

#endif // TEXTILES_H
