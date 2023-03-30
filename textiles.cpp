#include "textiles.h"
#include "math.h"
#include "qdebug.h"


TEXTILES::TEXTILES()
{

}

TEXTILES::~TEXTILES()
{

}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

std::vector<int> unique(cv::Mat input, bool sort = false)
{
    qDebug() << QString::fromStdString(type2str(input.type()));
    std::vector<int> out;
    for (int row = 0; row < input.rows; row++)
    {
        for (int col = 0; col < input.cols; col++)
        {
            if(row==0 && col==0){
                out.push_back(input.at<int>(col, row));
            } else {
                if(std::find(out.begin(), out.end(), input.at<int>(col, row)) == out.end()){
                    out.push_back(input.at<int>(col, row));
                }
            }
        }
    }

    if(sort){
        std::sort(out.begin(), out.end());
    }

    return out;
}

double findMedian(std::vector<int> a)
{
    int n = a.size();

    // If size of the arr[] is even
    if (n % 2 == 0) {

        // Applying nth_element
        // on n/2th index
        std::nth_element(a.begin(),
                         a.begin() + n / 2,
                         a.end());

        // Applying nth_element
        // on (n-1)/2 th index
        std::nth_element(a.begin(),
                         a.begin() + (n - 1) / 2,
                         a.end());

        // Find the average of value at
        // index N/2 and (N-1)/2
        return (double)(a[(n - 1) / 2] + a[n / 2]) / 2.0;
    }

    // If size of the arr[] is odd
    else {

        // Applying nth_element
        // on n/2
        std::nth_element(a.begin(),
                         a.begin() + n / 2,
                         a.end());

        // Value at index (N/2)th
        // is the median
        return (double)a[n / 2];
    }
}

void findneighbour(cv::Mat &markers, cv::Mat &tmp, int row, int col, int mark){
    markers.at<uchar>(col, row) = mark;
    tmp.at<uchar>(col, row) = 1;
    //qDebug() << row << " " << col;
    if((row-1) >= 0){
        if(markers.at<uchar>(col, row-1) == 1 && tmp.at<uchar>(col, row-1) == 0){
            qDebug() << "0";
            findneighbour(markers, tmp, row-1, col, mark);
        }
    }
    if((row+1) < markers.rows){
        if(markers.at<uchar>(col, row+1) == 1 && tmp.at<uchar>(col, row+1) == 0){
            qDebug() << "1";
            findneighbour(markers, tmp, row+1, col, mark);
        }
    }
    if((col-1) >= 0){
        if(markers.at<uchar>(col-1, row) == 1 && tmp.at<uchar>(col-1, row) == 0){
            qDebug() << "2";
            findneighbour(markers, tmp, row, col-1, mark);
        }
    }
    if((col+1) < markers.cols){
        if(markers.at<uchar>(col+1, row) == 1 && tmp.at<uchar>(col+1, row) == 0){
            qDebug() << "3";
            findneighbour(markers, tmp, row, col+1, mark);
        }
    }
}

cv::Mat TEXTILES::Generate_depthmap(cv::Mat depth_mat)
{

}

cv::Mat TEXTILES::Segmentation_background_subtraction(cv::Mat image)
{
    cv::Mat image_hsv, image_blur;
    cv::cvtColor(image, image_hsv, cv::COLOR_BGR2HSV);
    cv::GaussianBlur(image_hsv, image_blur, cv::Size(5, 5), 0);

    // define range of white color in HSV
    // change it according to your need !
    const int sensitivity = 20;
    cv::Scalar lower_white = cv::Scalar(0, 0, 255-sensitivity);
    cv::Scalar upper_white = cv::Scalar(255, sensitivity, 255);

    // Threshold the HSV image to get only white colors
    cv::Mat mask;
    cv::inRange(image_blur, lower_white, upper_white, mask);

    // Filter result using morphological operations (closing)
    cv::Mat filtered_mask_close;
    cv::Mat filtered_mask_open;
    cv::Mat kernel = cv::Mat::ones(5, 5, CV_8UC1);
    cv::morphologyEx(mask, filtered_mask_close, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 5);
    cv::morphologyEx(filtered_mask_close, filtered_mask_open, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 8);

    cv::Mat threshold_mask;
    cv::threshold(filtered_mask_open, threshold_mask, 0, 255, cv::THRESH_BINARY_INV);

    cv::imshow("image", image);
    //cv::imshow("image_blur", image_blur);
    //cv::imshow("mask", mask);
    //cv::imshow("threshold_mask", threshold_mask);
    //cv::waitKey(-1);

    return threshold_mask;
}

std::vector<cv::Point> TEXTILES::Segmentation_compute_approximated_polygon(cv::Mat mask)
{
    /* Calculate the approximated polygon that describes the garment
       :param mask: Segmentation mask where white is garment and black is background
       :return: Garment Approximated Polygon (as a vector of 2D points)
    */

    // Get clothes outline with largest area
    std::vector<std::vector<cv::Point>> garment_outlines;
    cv::findContours(mask, garment_outlines, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> largest_outline = *std::max_element(garment_outlines.begin(),
                                                               garment_outlines.end(),
                                                               [](std::vector<cv::Point> const& lhs, std::vector<cv::Point> const& rhs)
                                                               {
                                                                   return cv::contourArea(lhs, false) < cv::contourArea(rhs, false);
                                                               });

    // Simplify outline to approximated polygon:
    double perimeter = cv::arcLength(largest_outline, true);
    std::vector<cv::Point> approximated_polygon;
    cv::approxPolyDP(largest_outline, approximated_polygon, 0.010 * perimeter, true);

    //std::cout << approximated_polygon << std::endl;

    return approximated_polygon;
}

cv::Mat TEXTILES::DepthMapClustering_preprocess(cv::Mat depth_image, cv::Mat mask)
{
    /* Removes the background information and normalizes the depth image range to 8-bit unsigned.
       :param depth_image:
       :param mask: Segmentation mask where white is garment and black is background
       :return: Depth image normalized and converted to 8-bit unsigned
    */

    float background = 10000.0;  // Define a depth value for background pixels
    cv::Mat masked_depth_image;
    depth_image.copyTo(masked_depth_image);

    float min_value = background, max_value = 0.0;

    for(int row = 0; row<masked_depth_image.rows; row++){
        for(int col = 0; col<masked_depth_image.cols; col++){
            if(mask.at<uchar>(col, row) == 255){
                // Get range of values (min, max, range)
                if(masked_depth_image.at<float>(col, row) < min_value){
                    min_value = masked_depth_image.at<float>(col, row);
                }
                if (masked_depth_image.at<float>(col, row) > max_value) {
                    max_value = masked_depth_image.at<float>(col, row);
                }
            } else {
                masked_depth_image.at<float>(col, row) = background;
            }
        }
    }

    float range_value = max_value - min_value;

    //qDebug() << "min_value: " << min_value << "\n"
    //         << "max_value: " << max_value << "\n"
    //         << "range_value: " << range_value;

    // Normalize using depth range
    cv::Mat scaled_depth_map;
    masked_depth_image.copyTo(scaled_depth_map);

    for(int row = 0; row<scaled_depth_map.rows; row++){
        for(int col = 0; col<scaled_depth_map.cols; col++){
            if(scaled_depth_map.at<float>(col, row) != background){
                scaled_depth_map.at<float>(col, row) = (scaled_depth_map.at<float>(col, row) - min_value) * (255.0/range_value);
            } else {
                scaled_depth_map.at<float>(col, row) = 255.0;
            }
        }
    }
    scaled_depth_map.convertTo(scaled_depth_map, CV_8UC1);

    return scaled_depth_map;
}

cv::Mat TEXTILES::DepthMapClustering_cluster_similar_regions(cv::Mat preprocessed_depth_image)
{
    /* Apply clustering algorithm to group similar regions. This uses watershed currently.
       :param preprocessed_depth_image: Depth image normalized and converted to 8-bit unsigned
       :param mask: segmentation mask for the depth map
       :return: Labeled image after the clustering process. Each label is the median value of each cluster
    */

    // denoise image
    cv::Mat denoised, denoised_equalize;
    //denoised = restoration.denoise_tv_chambolle(scaled_depth_map);
    //denoised = util.img_as_ubyte(denoised);
    cv::fastNlMeansDenoising(preprocessed_depth_image, denoised, 10);
    cv::equalizeHist(denoised, denoised_equalize);

    // find continuous region (low gradient) --> markers
    cv::Mat markers;
    cv::Laplacian(denoised_equalize, markers, CV_8U, 5);

    for(int row=0; row<markers.rows; row++){
        for(int col=0; col<markers.cols; col++){
            if(markers.at<uchar>(col, row) < 3){
                markers.at<uchar>(col, row) = 1;
            } else {
                markers.at<uchar>(col, row) = 0;
            }
        }
    }

    // Declare what you need
    cv::FileStorage file0("/home/cpii/Desktop/Textiles/data3/markers0.ext", cv::FileStorage::WRITE);
    // Write to file!
    file0 << "matName" << markers;

    cv::Mat tmp = cv::Mat::zeros(markers.size(), CV_8UC1);
    int mark = 1;
    for(int row=0; row<markers.rows; row++){
        for(int col=0; col<markers.cols; col++){
            if(tmp.at<uchar>(col, row)==0){
                if(markers.at<uchar>(col, row)==1){
                    findneighbour(markers, tmp, row, col, mark);
                    mark++;
                } else {
                    tmp.at<uchar>(col, row) = 1;
                }
            }
        }
    }

    // Declare what you need
    cv::FileStorage filet("/home/cpii/Desktop/Textiles/data3/tmp.ext", cv::FileStorage::WRITE);
    // Write to file!
    filet << "matName" << tmp;

    // Declare what you need
    cv::FileStorage file1("/home/cpii/Desktop/Textiles/data3/markers1.ext", cv::FileStorage::WRITE);
    // Write to file!
    file1 << "matName" << markers;

    // local gradient
    cv::Mat gradient;
    cv::Laplacian(denoised, gradient, CV_8U, 5);

    // labels
    cv::watershed(gradient, markers);

    //markers.convertTo(markers, CV_8UC1);

    // Change labels by median value of each region
    std::vector<int> unique_labels = unique(markers, true);
    qDebug() << unique_labels;
    cv::Mat avg = cv::Mat::zeros(markers.size(), CV_8UC1);

    for(int i = 0; i < unique_labels.size(); i++){
        std::vector<int> values;
        for (int row = 0; row < preprocessed_depth_image.rows; row++) {
            for (int col = 0; col < preprocessed_depth_image.cols; col++) {
                if (markers.at<uchar>(col, row) == unique_labels[i]) {
                    values.push_back(preprocessed_depth_image.at<uchar>(col, row));
                }
            }
        }
        double avg_value = findMedian(values);
        for (int row = 0; row < avg.rows; row++) {
            for (int col = 0; col < avg.cols; col++) {
                if (markers.at<uchar>(col, row) == unique_labels[i]) {
                    avg.at<uchar>(col, row) = avg_value;
                }
            }
        }
    }

    return avg;
}

std::vector<cv::Point> TEXTILES::PickandPlacePoints_calculate_unfold_paths(cv::Mat labeled_image, std::vector<cv::Point> approximated_polygon)
{

}

std::vector<int> TEXTILES::PickandPlacePoints_calculate_bumpiness(cv::Mat labeled_image, std::vector<cv::Point> unfold_paths)
{

}

std::vector<cv::Point> TEXTILES::PickandPlacePoints_calculate_pick_and_place_points(cv::Mat labeled_image, std::vector<cv::Point> unfold_paths, std::vector<int> bumpiness)
{

}

cv::Mat TEXTILES::draw_segmentation_stage(cv::Mat img_src, cv::Mat mask, std::vector<cv::Point> approximated_polygon)
{
    cv::Mat rgb_mask;
    mask.convertTo(rgb_mask, CV_8U);
    cv::cvtColor(rgb_mask, rgb_mask, cv::COLOR_GRAY2BGR);

    cv::Mat drawing;
    cv::addWeighted(img_src, 0.4, rgb_mask, 0.1, 0, drawing);

    for(int i=0; i<approximated_polygon.size(); i++){
        cv::circle( drawing,
                    approximated_polygon[i],
                    4,
                    cv::Scalar( 0, 0, 255 ),
                    3,
                    cv::FILLED );
        if(i==0){
            cv::line(drawing, approximated_polygon[approximated_polygon.size()-1], approximated_polygon[0], cv::Scalar(0,0,255), 2, cv::LINE_AA);
        } else {
            cv::line(drawing, approximated_polygon[i-1], approximated_polygon[i], cv::Scalar(0,0,255), 2, cv::LINE_AA);
        }
    }

    return drawing;
}

cv::Mat TEXTILES::draw_clustering_stage(cv::Mat image_src, cv::Mat labeled_image)
{
    cv::Mat drawing;

    cv::applyColorMap(labeled_image, drawing, cv::COLORMAP_VIRIDIS);

    return drawing;
}
