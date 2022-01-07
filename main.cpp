#include <iostream>
#include <opencv2/opencv.hpp>
#include <hyperpose/hyperpose.hpp>
#include <chrono>

using namespace hyperpose;
//using namespace std;

int main() {
    std::cout << "Hello, World!" << std::endl;
    const cv::Size network_resolution{432, 368};
    const dnn::onnx onnx_model{"/home/rick/Code/Scratches/test/22-01-07-HyperPose-Example/res/openpifpaf-resnet50-HW=368x432.onnx"};

    // * Input video.
    auto capture = cv::VideoCapture("../videos/2022-1-7-15-52-7.avi");

    // * Output video.
    auto writer = cv::VideoWriter(
            "../videos/output001.avi", capture.get(cv::CAP_PROP_FOURCC), capture.get(cv::CAP_PROP_FPS),
            network_resolution);

    // * Create TensorRT engine.
    dnn::tensorrt engine(onnx_model, network_resolution);

    // * post-processing: Using paf.
    parser::pifpaf parser(engine.input_size().height, engine.input_size().width);
    cv::Mat frame;

    while (capture.isOpened()) {
        std::vector<cv::Mat> batch;
        for (int i = 0; i < engine.max_batch_size(); ++i) {
            cv::Mat mat;
            capture >> mat;
            if (mat.empty())
                break;
            batch.push_back(mat);
        }

        if (batch.empty())
            break;

        // * TensorRT Inference.
        auto start = std::chrono::high_resolution_clock::now();
        auto feature_map_packets = engine.inference(batch);
        auto end = std::chrono::high_resolution_clock::now();

        // * Paf.
        std::vector<std::vector<human_t>> pose_vectors;
        pose_vectors.reserve(feature_map_packets.size());
        for (auto&& packet : feature_map_packets)
            pose_vectors.push_back(parser.process(packet[0], packet[1]));

        // * Visualization
        for (size_t i = 0; i < batch.size(); ++i) {
            cv::resize(batch[i], batch[i], network_resolution);
            for (auto&& pose : pose_vectors[i])
                draw_human(batch[i], pose);
            writer << batch[i];

            cv::imshow("img", batch[i]);
            cv::waitKey(1);
        }

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
    }

    return 0;
}
