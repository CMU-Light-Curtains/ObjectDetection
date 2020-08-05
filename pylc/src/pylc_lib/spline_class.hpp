#ifndef SPLINECLASS_H
#define SPLINECLASS_H

#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <opencv2/opencv.hpp>
#include <opencv/cxeigen.hpp>
#include <chrono>

#include <alglibinternal.h>
#include <interpolation.h>
#include <math.h>
#include <malloc.h>
#include <random>

void test(){
    alglib::real_1d_array X, Z;
}

bool sortX(const Eigen::Vector2f &a, const Eigen::Vector2f &b)
{
    return a(0) < b(0);
}

bool sortY(const Eigen::Vector2f &a, const Eigen::Vector2f &b)
{
    return a(1) < b(1);
}

std::vector<Eigen::Vector2f> getPointsFromVectors(const Eigen::Vector2f &x, const Eigen::Vector2f &z) {
    Eigen::MatrixXf pts_m(x.rows(), x.cols()+z.cols());
    pts_m << x, z;
    long num_pts = x.size();
    std::vector<Eigen::Vector2f> pts(num_pts);
    for(int i=0; i<num_pts; i++)
    {
        pts[i] = pts_m.row(i);
    }
    return pts;
}

bool closestDistance(Eigen::MatrixXf& design_pts, Eigen::MatrixXf& targets, float threshold=0.1, bool debug=false){
    for(int i=0; i<targets.rows(); i++){
        float lowest_l2 = std::numeric_limits<float>::infinity();
        //Eigen::Vector2f closest_point;
        for(int j=0; j<design_pts.rows(); j++){
            if(std::isnan(design_pts(j,0))) continue;
            float l2 = sqrt(pow(targets(i,0)-design_pts(j,0),2) + pow(targets(i,1)-design_pts(j,2),2));
            if(l2 < lowest_l2){
                lowest_l2 = l2;
                //closest_point = Eigen::Vector2f(design_pts(j,0), design_pts(j,2));
            }
            //std::cout << "[" << design_pts(j,0) << ", " << design_pts(j,2) << "]" << std::endl;
        }
        if(lowest_l2 > threshold) return true;
    }
    return false;
}

Eigen::MatrixXf makeRandomWithInputs(Eigen::Vector2f p0, const std::vector<Eigen::Vector2f>& inputs, float height, float width, int num_random_pts, float smoothing_constant, int sample_pts, bool hack=false)
{
    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_real_distribution<float> dist_x(p0(0), p0(0)+width); // uniform, unbiased
    std::uniform_real_distribution<float> dist_z(p0(1), p0(1)+height); // uniform, unbiased

    std::vector<Eigen::Vector2f> pts;
    for(int i =0; i< num_random_pts; i++)
    {
        if(hack) break;
        float x = dist_x(engine);
        float z = dist_z(engine);
        pts.emplace_back(Eigen::Vector2f(x,z));
    }

    // Handle Inputs
    for(const auto& inp : inputs){
        std::normal_distribution<float> dist_x{inp[0],0.3};
        std::normal_distribution<float> dist_z{inp[1],0.3};
        float x = dist_x(engine);
        float z = dist_z(engine);
        pts.emplace_back(Eigen::Vector2f(x,z));
    }

    std::sort(pts.begin(), pts.end(), sortX); //sort points from

    std::vector<double> x_sorted, z_sorted;
    for(const auto& pt: pts)
    {
        x_sorted.push_back(pt(0));
        z_sorted.push_back(pt(1));
    }

    alglib::real_1d_array X, Z;
    X.setcontent(x_sorted.size(), &(x_sorted[0]));
    Z.setcontent(z_sorted.size(), &(z_sorted[0]));
    alglib::ae_int_t info;
    alglib::spline1dinterpolant s;
    alglib::spline1dfitreport rep;
    double rho = smoothing_constant;
    alglib::spline1dfitpenalized(X, Z, 50, rho, info, s, rep);

    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(sample_pts, p0(0), p0(0)+width);
    Eigen::VectorXf z(x.size());

    for(int i=0; i < x.size(); i++)
    {
        z(i) = alglib::spline1dcalc(s,x(i));
    }

    Eigen::MatrixXf pts_out;
    pts_out.resize(x.size(), 2);
    for(int i=0; i<x.size(); i++){
        pts_out(i, 0) = x(i);
        pts_out(i, 1) = z(i);
    }

    return pts_out;
}

Eigen::MatrixXf makeRandom(Eigen::Vector2f p0, float height, float width, int num_random_pts, float smoothing_constant, int sample_pts)
{
    std::random_device seeder;
    std::mt19937 engine(seeder());
    std::uniform_real_distribution<float> dist_x(p0(0), p0(0)+width); // uniform, unbiased
    std::uniform_real_distribution<float> dist_z(p0(1), p0(1)+height); // uniform, unbiased

    std::vector<Eigen::Vector2f> pts;
    for(int i =0; i< num_random_pts; i++)
    {
        float x = dist_x(engine);
        float z = dist_z(engine);
        pts.emplace_back(Eigen::Vector2f(x,z));
    }

    std::sort(pts.begin(), pts.end(), sortX); //sort points from

    std::vector<double> x_sorted, z_sorted;
    for(const auto& pt: pts)
    {
        x_sorted.push_back(pt(0));
        z_sorted.push_back(pt(1));
    }

    alglib::real_1d_array X, Z;
    X.setcontent(x_sorted.size(), &(x_sorted[0]));
    Z.setcontent(z_sorted.size(), &(z_sorted[0]));
    alglib::ae_int_t info;
    alglib::spline1dinterpolant s;
    alglib::spline1dfitreport rep;
    double rho = smoothing_constant;
    alglib::spline1dfitpenalized(X, Z, 50, rho, info, s, rep);

    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(sample_pts, p0(0), p0(0)+width);
    Eigen::VectorXf z(x.size());

    for(int i=0; i < x.size(); i++)
    {
        z(i) = alglib::spline1dcalc(s,x(i));
    }

    Eigen::MatrixXf pts_out;
    pts_out.resize(x.size(), 2);
    for(int i=0; i<x.size(); i++){
        pts_out(i, 0) = x(i);
        pts_out(i, 1) = z(i);
    }

    return pts_out;
}

Eigen::MatrixXf makeSpline(std::vector<Eigen::Vector2f> pts, Eigen::Vector2f p0, float width, float smoothing_constant, int sample_pts, float basis, bool swap=false)
{
    if(swap)
        std::sort(pts.begin(), pts.end(), sortY); //sort points from
    else
        std::sort(pts.begin(), pts.end(), sortX); //sort points from

    std::vector<double> x_sorted, z_sorted;
    for(const auto& pt: pts)
    {
        if(swap){
            x_sorted.push_back(pt(1));
            z_sorted.push_back(pt(0));
        }else{
            x_sorted.push_back(pt(0));
            z_sorted.push_back(pt(1));
        }
    }

    alglib::real_1d_array X, Z;
    X.setcontent(x_sorted.size(), &(x_sorted[0]));
    Z.setcontent(z_sorted.size(), &(z_sorted[0]));
    alglib::ae_int_t info;
    alglib::spline1dinterpolant s;
    alglib::spline1dfitreport rep;
    double rho = smoothing_constant;
    alglib::spline1dfitpenalized(X, Z, basis, rho, info, s, rep);

    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(sample_pts, p0(0), p0(0)+width);
    Eigen::VectorXf z(x.size());

    for(int i=0; i < x.size(); i++)
    {
        z(i) = alglib::spline1dcalc(s,x(i));
    }

    Eigen::MatrixXf pts_out;
    pts_out.resize(x.size(), 2);
    for(int i=0; i<x.size(); i++){
        if(swap){
            pts_out(i, 1) = x(i);
            pts_out(i, 0) = z(i);
        }else{
            pts_out(i, 0) = x(i);
            pts_out(i, 1) = z(i);
        }
    }

    return pts_out;
}

void removeNan(std::vector<float>& x){
    x.erase(std::remove_if(std::begin(x), std::end(x), [](const float& value) { return std::isnan(value); }), std::end(x));
}

int convolve_sse(float* in, int input_length,
                 float* kernel,	int kernel_length, float* out)
{
    float* in_padded = (float*)(alloca(sizeof(float) * (input_length + 8)));

    __m128* kernel_many = (__m128*)(alloca(16 * kernel_length));
    __m128 block;

    __m128 prod;
    __m128 acc;

    // surrounding zeroes, before and after
    _mm_storeu_ps(in_padded, _mm_set1_ps(0));
    memcpy(&in_padded[4], in, sizeof(float) * input_length);
    _mm_storeu_ps(in_padded + input_length + 4, _mm_set1_ps(0));

    // Repeat each kernal value across a 4-vector
    int i;
    for (i = 0; i < kernel_length; i++) {
        kernel_many[i] = _mm_set1_ps(kernel[i]); // broadcast
    }

    for (i = 0; i < input_length + kernel_length - 4; i += 4) {

        // Zero the accumulator
        acc = _mm_setzero_ps();

        int startk = i > (input_length - 1) ? i - (input_length - 1) : 0;
        int endk = (i + 3) < kernel_length ? (i + 3) : kernel_length - 1;

        /* After this loop, we have computed 4 output samples
        * for the price of one.
        * */
        for (int k = startk; k <= endk; k++) {

            // Load 4-float data block. These needs to be an unaliged
            // load (_mm_loadu_ps) as we step one sample at a time.
            block = _mm_loadu_ps(in_padded + 4 + i - k);
            prod = _mm_mul_ps(block, kernel_many[k]);

            // Accumulate the 4 parallel values
            acc = _mm_add_ps(acc, prod);
        }
        _mm_storeu_ps(out + i, acc);
    }

    // Left-overs
    for (; i < input_length + kernel_length - 1; i++) {

        out[i] = 0.0;
        int startk = i >= input_length ? i - input_length + 1 : 0;
        int endk = i < kernel_length ? i : kernel_length - 1;
        for (int k = startk; k <= endk; k++) {
            out[i] += in[i - k] * kernel[k];
        }
    }

    return 0;
}

int convolve_naive(float* in, int input_length,
                   float* kernel,	int kernel_length, float* out)
{
    for (int i = 0; i < input_length + kernel_length - 1; i++) {
        out[i] = 0.0;
        int startk = i >= input_length ? i - input_length + 1 : 0;
        int endk = i < kernel_length ? i : kernel_length - 1;
        for (int k = startk; k <= endk; k++) {
            out[i] += in[i - k] * kernel[k];
        }
    }

    return 0;
}

std::vector<float> convolve(std::vector<float>& input, std::vector<float>& kernel, int mode = 0){
    std::vector<float> output;
    int M = input.size();
    int N = kernel.size();
    output.resize(M + N - 1);
    convolve_sse(input.data(), M, kernel.data(), N, output.data());

    if(mode == 0) return output;
    else if(mode == 1){
        output.erase(output.begin(), output.begin() + N-1);
        for(int i=0; i<N-1; i++) output.pop_back();
        return output;
    }

    return output;

}

std::vector<float> runningMean(std::vector<float>& x, int N){
    std::vector<float> cumsum = {0};
    std::vector<float> moving_aves;
    for(int i=1; i<x.size()+1; i++){
        cumsum.emplace_back(cumsum[i-1] + x[i-1]);
        if(i >= N){
            float moving_ave = (cumsum[i] - cumsum[i-N])/(float)N;
            moving_aves.emplace_back(moving_ave);
        }
    }
    return moving_aves;
}

float squaredSum(std::vector<float>& x){
    float sum = 0;
    for(auto& item : x) sum += pow(item,2);
    return sum;
}

std::vector<float> peakFinding(std::vector<float>& x, int N){
    std::vector<std::pair<int, float>> array;

    for(int i=1; i<x.size()-1; i++){
        float left_pt;
        float mid_pt;
        float right_pt;
        if(i >= N && i < x.size()-N){
            left_pt = x[i-N];
            mid_pt = x[i];
            right_pt = x[i+N];
        }else if(i >=1 && i < N){
            left_pt = x[0];
            mid_pt = x[i];
            right_pt = x[i+N];
        }else if(i >= x.size()-N && i < x.size()-1){
            left_pt = x[i-N];
            mid_pt = x[i];
            right_pt = x[x.size()-1];
        }

        auto left_dist = mid_pt - left_pt;
        auto right_dist = right_pt - mid_pt;

        if(copysignf(1.0, left_dist) != copysignf(1.0, right_dist)){
            auto dist = std::min(fabs(left_dist), fabs(right_dist));
            if(dist > 0.12) array.emplace_back(std::pair<int,float>(i, dist));
        }
    }

    std::vector<float> peaks;
    std::vector<std::pair<int, float>> clusters;
    std::vector<std::pair<int, float>> curr_cluster;

    for(int i=0; i<array.size(); i++){
        bool still_adding = false;
        if(curr_cluster.size() == 0){
            curr_cluster.emplace_back(array[i]);
            still_adding = true;
        }else if(abs(curr_cluster.back().first - array[i].first) < N){
            curr_cluster.emplace_back(array[i]);
            still_adding = true;
        }

        if(!still_adding || (i+1 == array.size())){
            float maxsize = 0;
            std::pair<int, float> best_item;
            for(auto& item : curr_cluster){
                if(item.second > maxsize){
                    maxsize = item.second;
                    best_item = item;
                }
            }
            clusters.emplace_back(best_item);
            peaks.emplace_back(maxsize);
            curr_cluster.clear();
        }
    }

    return peaks;
}

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1, bool include=false) {
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    if(include) values.push_back(stop);
    return values;
}

void setCol(Eigen::MatrixXf& matrix, int colnum, float val){
    for(int i=0; i<matrix.rows(); i++){
        matrix(i, colnum) = val;
    }
}

void eigenXSort(Eigen::MatrixXf& matrix)
{
    std::vector<Eigen::VectorXf> vec;
    for (int64_t i = 0; i < matrix.rows(); ++i)
        vec.push_back(matrix.row(i));
    std::sort(vec.begin(), vec.end(), [](Eigen::VectorXf const& t1, Eigen::VectorXf const& t2){ return t1(0) < t2(0); } );
    for (int64_t i = 0; i < matrix.rows(); ++i)
        matrix.row(i) = vec[i];
};

std::vector<int> eigenAngleSort(Eigen::MatrixXf& matrix)
{
    std::vector<int> idx(matrix.rows());
    iota(idx.begin(), idx.end(), 0);
    std::vector<float> angs;
    for (int64_t i = 0; i < matrix.rows(); ++i) {
        float angle = -((atan2f(matrix(i,1), matrix(i,0)) * 180 / M_PI) - 90);
        angs.emplace_back(angle);
    }
    std::sort(idx.begin(), idx.end(),
         [&angs](int i1, int i2) {return angs[i1] < angs[i2];});
    Eigen::MatrixXf matrix_copy = matrix;
    for(int i=0; i<idx.size(); i++){
        matrix.row(i) = matrix_copy.row(idx[i]);
    }
    return idx;
};

std::vector<std::vector<int>> splitVector(const std::vector<int>& idx, int split_count){
    int each_contains = (int)(((float)idx.size() / (float)split_count));
    int remainder = idx.size() % split_count;
    std::vector<std::vector<int>> splits;
    for(int j=0; j<split_count; j++){
        if(j == split_count - 1)
            splits.emplace_back(std::vector<int>(idx.begin() + j*each_contains, idx.begin() + j*each_contains + each_contains + remainder));
        else
            splits.emplace_back(std::vector<int>(idx.begin() + j*each_contains, idx.begin() + j*each_contains + each_contains));
    }
    return splits;
}

template <class T>
void reorder(std::vector<T>& vA, std::vector<size_t>& vOrder)
{
    assert(vA.size() == vOrder.size());
    // for all elements to put in place
    for( size_t i = 0; i < vA.size(); ++i )
    {
        // while vOrder[i] is not yet in place
        // every swap places at least one element in it's proper place
        while(       vOrder[i] !=   vOrder[vOrder[i]] )
        {
            std::swap( vA[vOrder[i]], vA[vOrder[vOrder[i]]] );
            std::swap(    vOrder[i],     vOrder[vOrder[i]] );
        }
    }
}

template< class T >
void reorder_naive(std::vector<T>& vA, const std::vector<size_t>& vOrder)
{
    assert(vA.size() == vOrder.size());
    std::vector<T> vCopy = vA; // Can we avoid this?
    for(int i = 0; i < vOrder.size(); ++i)
        vA[i] = vCopy[ vOrder[i] ];
}

std::vector<std::vector<std::vector<int>>> generateContPerms(int index_count, int split_count){
    struct custom_sort
    {
        inline bool operator() (const std::vector<int>& struct1, const std::vector<int>& struct2)
        {
            return (struct1[0] < struct2[0]);
        }
    };
    // Generate all continious permutations
    std::vector<int> idx(index_count);
    iota(idx.begin(), idx.end(), 0);
    std::vector<std::vector<int>> permutations;
    do {
        permutations.emplace_back(idx);
    } while (std::next_permutation(idx.begin(), idx.end()));
    std::vector<std::vector<std::vector<int>>> contiguous_perms;
    std::map<std::string, int> hash;
    for(const auto& perm : permutations){
        std::vector<std::vector<int>> splits = splitVector(perm, split_count);
        // Check if each split is continuous
        bool sorted = true;
        for(const auto& split : splits){
            if(!std::is_sorted(split.begin(), split.end())) sorted = false;
        }
        // Sort my splits
        std::sort(splits.begin(), splits.end(), custom_sort());
        if(sorted) {
            // Generate Key
            std::string s = "";
            for(const auto& split : splits){
                for(const auto& v : split) s+=std::to_string(v);
                s+="-";
            }
            if(hash.count(s)) continue;
            else hash[s] = 1;
            contiguous_perms.emplace_back(splits);
        }
    }
    return contiguous_perms;
}

void removeRow(Eigen::MatrixXf& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

Eigen::MatrixXf customSort(const Eigen::MatrixXf& matrix, const std::vector<int>& custom_order){
    if(matrix.rows() == custom_order.size()) {
        Eigen::MatrixXf matrix_copy = matrix;
        for (int i = 0; i < custom_order.size(); i++) {
            matrix_copy.row(i) = matrix.row(custom_order[i]);
        }
        return matrix_copy;
    }else if(matrix.rows() > custom_order.size()){
        Eigen::MatrixXf matrix_copy;
        matrix_copy.resize(custom_order.size(), matrix.cols());
        for (int i = 0; i < custom_order.size(); i++) {
            matrix_copy.row(i) = matrix.row(custom_order[i]);
        }
        return matrix_copy;
    }else{
        throw std::runtime_error("custom_sort error");
    }
}

std::string getKey(const std::vector<int>& custom_order){
    std::string s = "";
    for(const auto& i : custom_order){
        s += std::to_string(i);
        s += "-";
    }
    return s;
}

void removeRows(Eigen::MatrixXf& matrix, std::vector<int>& to_remove){
    int count = 0;
    for(auto& row : to_remove){
        removeRow(matrix, row-count);
        count++;
    }
}

template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
    T h = (b - a) / static_cast<T>(N-1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}

Eigen::MatrixXf fitBSpline(const Eigen::MatrixXf& input_pts_x, int mode = 0, bool hack=1){
    // Hack
    Eigen::MatrixXf input_pts = input_pts_x;

//    // HACK
//    if(input_pts.rows() == 1){
//        input_pts.resize(3,3);
//        ROS_ERROR("Hack");
//        input_pts.row(0) = input_pts_x.row(0);
//        input_pts.row(1) = input_pts_x.row(0);
//        input_pts.row(2) = input_pts_x.row(0);
////        input_pts.row(3) = input_pts_x.row(0);
////        input_pts.row(4) = input_pts_x.row(0);
//
//        input_pts(0,0) -= 0.1;
//        input_pts(1,0) -= 0.0;
//        input_pts(2,0) += 0.1;
//
//        input_pts(1,1) -= 0.1;
//
////        input_pts(0,1) += 0.1;
////        input_pts(1,1) -= 0.1;
////        input_pts(2,1) -= 0.0;
////        input_pts(3,1) -= 0.1;
////        input_pts(4,1) += 0.1;
////
////        input_pts(0,0) -= 0.1;
////        input_pts(1,0) -= 0.1;
////        input_pts(2,0) -= 0.0;
////        input_pts(3,0) += 0.1;
////        input_pts(4,0) += 0.1;
//
//        std::cout << input_pts << std::endl;
//    }

    // Copy
    Eigen::MatrixXf inputs = input_pts;

    if(hack){
        // Reverse Rows
        inputs = input_pts.colwise().reverse();

        // Add two control points at end
        Eigen::Vector3f c1 = Eigen::Vector3f((inputs(0,0)+inputs(1,0))/2.,(inputs(0,1)+inputs(1,1))/2.,(inputs(0,2)+inputs(1,2))/2.);
        int in = inputs.rows()-1;
        Eigen::Vector3f c2 = Eigen::Vector3f((inputs(in,0)+inputs(in-1,0))/2.,(inputs(in,1)+inputs(in-1,1))/2.,(inputs(in,2)+inputs(in-1,2))/2.);
        inputs.conservativeResize(inputs.rows()+2, inputs.cols());
        inputs.row(inputs.rows()-2) = c1;
        inputs.row(inputs.rows()-1) = c2;
    }

    // Initialize entry
    int n = inputs.rows() - 2;
    int n1 = n+1;
    std::vector<float> dx(n, 0);
    std::vector<float> dy(n, 0);

    // First and Last Derivs
    if(mode == 0){
        dx[0] = inputs(n, 0) - inputs(0, 0);
        dy[0] = inputs(n, 1) - inputs(0, 1);
        dx[n-1] = -(inputs(n1, 0) - inputs(n-1, 0));
        dy[n-1] = -(inputs(n1, 1) - inputs(n-1, 1));
    }else if(mode == 1){
        float DIV = 3.;
        dx[0] = (inputs(1, 0) - inputs(0, 0))/DIV;
        dy[0] = (inputs(1, 1) - inputs(0, 1))/DIV;
        dx[n-1] = (inputs(n-1, 0) - inputs(n-2, 0))/DIV;
        dy[n-1] = (inputs(n-1, 1) - inputs(n-2, 1))/DIV;
    }

    // Fill other control derivs
    std::vector<float> Ax(n, 0);
    std::vector<float> Ay(n, 0);
    std::vector<float> Bi(n, 0);
    Bi[1] = -1./inputs(1, 2);
    Ax[1] = -(inputs(2, 0) - inputs(0, 0) - dx[0])*Bi[1];
    Ay[1] = -(inputs(2, 1) - inputs(0, 1) - dy[0])*Bi[1];
    for(int i=2; i<n-1; i++){
        Bi[i] = -1/(inputs(i,2) + Bi[i-1]);
        Ax[i] = -(inputs(i+1,0) - inputs(i-1,0) - Ax[i-1])*Bi[i];
        Ay[i] = -(inputs(i+1,1) - inputs(i-1,1) - Ay[i-1])*Bi[i];
    }
    for(int i=n-2; i>0; i--){
        dx[i] = Ax[i] + dx[i+1]*Bi[i];
        dy[i] = Ay[i] + dy[i+1]*Bi[i];
    }

    // Interpolate
    std::vector<Eigen::Vector2f> paths = {Eigen::Vector2f(inputs(0,0), inputs(0,1))};
    for(int i=0; i<n-1; i++){
        // Distance
        float dist = sqrt(pow((inputs(i,0) - inputs(i+1,0)),2) + pow((inputs(i,1) - inputs(i+1,1)),2));
        float count = (float)((int)(dist/0.01));

        // Interpolate
        float extend = 0.1;
        auto ts = linspace<float>(0.-extend, 1+extend-(1./count), count);
        for(auto t: ts){
            float t1 = 1-t;
            float t12 = t1*t1;
            float t2 = t*t;
            float B0 = t1*t12;
            float B1 = 3*t*t12;
            float B2 = 3*t2*t1;
            float B3 = t*t2;
            float X = (inputs(i,0)*B0 + (inputs(i,0) + dx[i])*B1 +
                 (inputs(i+1,0) - dx[i+1])*B2 + inputs(i+1,0)*B3);
            float Y = (inputs(i,1)*B0 + (inputs(i,1) + dy[i])*B1 +
                 (inputs(i+1,1) - dy[i+1])*B2 + inputs(i+1,1)*B3);
            paths.emplace_back(Eigen::Vector2f(X,Y));
        }
    }

    // Special Case for 1 item
    if(paths.size() == 1){
        Eigen::Vector2f lVec = paths[0];
        Eigen::Vector2f rVec = paths[0];
        lVec(0) += 0.2;
        rVec(0) -= 0.2;
        paths.emplace_back(lVec);
        paths.emplace_back(rVec);
    }

    if(hack){
        std::reverse(paths.begin(), paths.end());
    }

    // Convert to eigen
    Eigen::MatrixXf outputs;
    outputs.resize(paths.size(),2);
    for(int i=0; i<paths.size(); i++){
        outputs(i,0) = paths[i](0);
        outputs(i,1) = paths[i](1);
    }

    return outputs;
}

#endif