#ifndef CONVERTERS_H
#define CONVERTERS_H

#include <memory>
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

namespace py = pybind11;

// Numpy - cv::Mat interop
namespace pybind11 { namespace detail {

        template <> struct type_caster<cv::Mat> {
        public:

            PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

            // Cast numpy to cv::Mat
            bool load(handle src, bool)
            {
                /* Try a default converting into a Python */
                //array b(src, true);
                array b = reinterpret_borrow<array>(src);
                buffer_info info = b.request();

                int ndims = info.ndim;

                decltype(CV_32F) dtype;
                size_t elemsize;
                if (info.format == format_descriptor<float>::format()) {
                    if (ndims == 3) {
                        dtype = CV_32FC3;
                    } else {
                        dtype = CV_32FC1;
                    }
                    elemsize = sizeof(float);
                } else if (info.format == format_descriptor<double>::format()) {
                    if (ndims == 3) {
                        dtype = CV_64FC3;
                    } else {
                        dtype = CV_64FC1;
                    }
                    elemsize = sizeof(double);
                } else if (info.format == format_descriptor<unsigned char>::format()) {
                    if (ndims == 3) {
                        dtype = CV_8UC3;
                    } else {
                        dtype = CV_8UC1;
                    }
                    elemsize = sizeof(unsigned char);
                } else {
                    throw std::logic_error("Unsupported type");
                    return false;
                }

                std::vector<int> shape = {(int)info.shape[0], (int)info.shape[1]};

                value = cv::Mat(cv::Size(shape[1], shape[0]), dtype, info.ptr, cv::Mat::AUTO_STEP);
                return true;
            }

            // Cast cv::Mat to numpy
            static handle cast(const cv::Mat &m, return_value_policy, handle defval)
            {
                std::string format = format_descriptor<unsigned char>::format();
                size_t elemsize = sizeof(unsigned char);
                int dim, channels;
                switch(m.type()) {
                    case CV_8U:
                        format = format_descriptor<unsigned char>::format();
                        elemsize = sizeof(unsigned char);
                        dim = 2;
                        channels = 0;
                        break;
                    case CV_8UC3:
                        format = format_descriptor<unsigned char>::format();
                        elemsize = sizeof(unsigned char);
                        dim = 3;
                        channels = 3;
                        break;
                    case CV_32F:
                        format = format_descriptor<float>::format();
                        elemsize = sizeof(float);
                        dim = 2;
                        channels = 0;
                        break;
                    case CV_64F:
                        format = format_descriptor<double>::format();
                        elemsize = sizeof(double);
                        dim = 2;
                        channels = 0;
                        break;
                    case CV_32FC4:
                        format = format_descriptor<float>::format();
                        elemsize = sizeof(float);
                        dim = 3;
                        channels = 4;
                        break;
                    default:
                        throw std::logic_error("Unsupported type");
                }

                std::vector<size_t> bufferdim;
                std::vector<size_t> strides;
                if (dim == 2) {
                    bufferdim = {(size_t) m.rows, (size_t) m.cols};
                    strides = {elemsize * (size_t) m.cols, elemsize};
                } else if (dim == 3) {
                    bufferdim = {(size_t) m.rows, (size_t) m.cols, (size_t) channels};
                    strides = {(size_t) elemsize * m.cols * channels, (size_t) elemsize * channels, (size_t) elemsize};
                }
                return array(buffer_info(
                        m.data,         /* Pointer to buffer */
                        elemsize,       /* Size of one scalar */
                        format,         /* Python struct-style format descriptor */
                        dim,            /* Number of dimensions */
                        bufferdim,      /* Buffer dimensions */
                        strides         /* Strides (in bytes) for each index */
                )).release();
            }

        };
}} // namespace pybind11::detail


#endif