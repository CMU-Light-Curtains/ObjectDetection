#ifndef PYLC_LIB_HPP
#define PYLC_LIB_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <pybind11/eigen.h>
#include <opencv2/opencv.hpp>
#include <opencv/cxeigen.hpp>
#include <converters.hpp>
#include <sim_class.hpp>
#include <chrono>
#include <spline_class.hpp>

namespace py = pybind11;

struct Sorted{
    std::string camera_name, laser_name;
    int input_index;
};

void processPointsJoint(std::shared_ptr<DatumProcessor>& datumProcessor, std::vector<std::shared_ptr<Input>>& inputs, py::dict& input_names, std::vector<py::dict>& soi, std::shared_ptr<Output>& output, bool get_cloud = false){

    // Resize
    output->clouds.resize(soi.size());
    output->images_multi.resize(soi.size());

    // Resort
    std::vector<Sorted> sortedVec;
    sortedVec.resize(soi.size());
    for(int i=0; i<soi.size(); i++){
        std::string camera_name = std::string(py::str(soi[i]["C"]));
        std::string laser_name = std::string(py::str(soi[i]["L"]));
        int input_index = py::int_(input_names[camera_name.c_str()]);
        sortedVec[input_index].camera_name = camera_name;
        sortedVec[input_index].laser_name = laser_name;
        sortedVec[input_index].input_index = input_index;
    }

    // Compute
    //#pragma omp parallel for shared(inputs, output, sortedVec)
    for(int i=0; i<sortedVec.size(); i++){
        Sorted& sorted = sortedVec[i];
        std::shared_ptr<Input>& input = inputs[i];

        // Get Depth Map
        cv::Mat depth_img = input->depth_image;

        // Resize local
        output->images_multi[i].resize(input->design_pts_multi.size());

        // Process
        std::vector<PointXYZI> combined_cloud;
        // pcl::PointCloud<pcl::PointXYZRGB> combined_cloud;
        for(int j=0; j<input->design_pts_multi.size(); j++){
            const Eigen::MatrixXf& m = input->design_pts_multi[j];

            cv::Mat& image = output->images_multi[i][j];
            std::vector<PointXYZI> cloud;
            // pcl::PointCloud<pcl::PointXYZRGB> cloud;
            datumProcessor->processPointsT(m, depth_img, sorted.camera_name, sorted.laser_name, image, cloud, get_cloud);
            // combined_cloud += cloud;
            combined_cloud.insert(combined_cloud.end(), cloud.begin(), cloud.end());
        }

        // Convert pcl::PointCloud to Eigen::MatrixXf.
        output->clouds[i] = Eigen::Matrix<float, Eigen::Dynamic, 4> (combined_cloud.size(), 4);
        for (int pt_index = 0; pt_index < combined_cloud.size(); pt_index++) {
            PointXYZI point = combined_cloud[pt_index];
            output->clouds[i](pt_index, 0) = point[0];
            output->clouds[i](pt_index, 1) = point[1];
            output->clouds[i](pt_index, 2) = point[2];
            output->clouds[i](pt_index, 3) = point[3];
        }
        // output->clouds[i] = combined_cloud.getMatrixXfMap(6, 8, 0);

        // Write Cloud
        // pcl::toROSMsg(combined_cloud, output->clouds[i]);
    }
}

PYBIND11_MODULE(pylc_lib, m) {

    // Datum Object
    py::class_<Datum, std::shared_ptr<Datum>>(m, "Datum")
    .def(py::init<>())
            .def_readwrite("type", &Datum::type)
            .def_readwrite("camera_name", &Datum::camera_name)
            .def_readwrite("rgb_matrix", &Datum::rgb_matrix)
            .def_readwrite("depth_matrix", &Datum::depth_matrix)
            .def_readwrite("world_to_rgb", &Datum::world_to_rgb)
            .def_readwrite("world_to_depth", &Datum::world_to_depth)
            .def_readwrite("cam_to_laser", &Datum::cam_to_laser)
            .def_readwrite("cam_to_world", &Datum::cam_to_world)
            .def_readwrite("fov", &Datum::fov)
            .def_readwrite("laser_name", &Datum::laser_name)
            .def_readwrite("distortion", &Datum::distortion)
            .def_readwrite("imgh", &Datum::imgh)
            .def_readwrite("imgw", &Datum::imgw)
            .def_readwrite("limit", &Datum::limit)
            .def_readwrite("galvo_m", &Datum::galvo_m)
            .def_readwrite("galvo_b", &Datum::galvo_b)
            .def_readwrite("maxADC", &Datum::maxADC)
            .def_readwrite("thickness", &Datum::thickness)
            .def_readwrite("divergence", &Datum::divergence)
            .def_readwrite("laser_limit", &Datum::laser_limit)
            .def_readwrite("laser_timestep", &Datum::laser_timestep)
    ;

    // Input Object
    py::class_<Input, std::shared_ptr<Input>>(m, "Input")
            .def(py::init<>())
            .def_readwrite("camera_name", &Input::camera_name)
            .def_readwrite("rgb_image", &Input::rgb_image)
            .def_readwrite("depth_image", &Input::depth_image)
            .def_readwrite("design_pts", &Input::design_pts)
            .def_readwrite("design_pts_multi", &Input::design_pts_multi)
            .def_readwrite("surface_pts", &Input::surface_pts)
            .def_readwrite("design_pts_conv", &Input::design_pts_conv)
            ;

    // Output Object
    py::class_<Output, std::shared_ptr<Output>>(m, "Output")
            .def(py::init<>())
            .def_readwrite("clouds", &Output::clouds)
            .def_readwrite("images_multi", &Output::images_multi)
            .def_readwrite("output_pts", &Output::output_pts)
            .def_readwrite("laser_rays", &Output::laser_rays)
            .def_readwrite("angles", &Output::angles)
            .def_readwrite("velocities", &Output::velocities)
            .def_readwrite("accels", &Output::accels)
            .def_readwrite("spline", &Output::spline)
            .def_readwrite("output_pts_set", &Output::output_pts_set)
            .def_readwrite("spline_set", &Output::spline_set)
            ;

    // DatumProcessor
    py::class_<DatumProcessor, std::shared_ptr<DatumProcessor>>(m, "DatumProcessor")
            .def(py::init<>())
            .def("setSensors", &DatumProcessor::setSensors)
            .def("processTest", &DatumProcessor::processTest)
            .def("evalPaths", &DatumProcessor::evalPaths)
            .def("evalPath", &DatumProcessor::evalPath)
            ;

    m.def("makeRandom", &makeRandom, "makeRandom");
    m.def("fitBSpline", &fitBSpline, "fitBSpline");
    m.def("convolve", &convolve, "convolve");
    m.def("processPointsJoint", &processPointsJoint, "processPointsJoint");

    #ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
    #else
        m.attr("__version__") = "dev";
    #endif
}

#endif


//std::vector<int> strides(bufferdim.size());
//if (!strides.empty())
//{
//    strides.back() = sizeof(float);
//    for (auto i = (int)strides.size()-2 ; i > -1 ; i--)
//        strides[i] = strides[i+1] * bufferdim[i+1];
//}
