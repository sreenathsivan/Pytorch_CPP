#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <torch/script.h>
using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

//custom datast class
class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    std::vector<std::tuple<std::vector<float>, int>> data; // Data and labels
public:
    CustomDataset(torch::Tensor X_train,torch::Tensor y_train) {
        Convert(X_train,y_train);
    }
    void Convert(torch::Tensor X_train,torch::Tensor y_train) {
        // std::cout<<X_train;
    int rows = X_train.size(0);
    int cols = X_train.size(1);
    // Loop through the elements of the tensor
    for (int i = 0; i < rows; ++i) {
          std::vector<float> features;
        std::vector<int> labels;
        for (int j = 0; j < cols; ++j) {
            // std::cout << "Element at index (" << i << ", " << j << "): " << X_train[i][j].item<float>() << std::endl;
            features.push_back(X_train[i][j].item<float>());
        }
        labels.push_back(y_train[i].item<int>());
        data.push_back(std::make_tuple(features, labels[0]));
    }
    // std::cout<<features;
}
    torch::data::Example<> get(size_t index) override {
    std::vector<float> features = std::get<0>(data[index]);
    int label = std::get<1>(data[index]);
        // Convert features to a Torch tensor
        auto tensor_features = torch::tensor(features, torch::kFloat32);
        auto tensor_label = torch::tensor(label, torch::kInt64);
        return { tensor_features, tensor_label };
    }
    torch::optional<size_t> size() const override {
        return data.size();
    };
};

class Net : public torch::nn::Module {
public:
    Net() {
        
        fc1 = register_module("fc1", torch::nn::Linear(4, 16)); 
        fc2 = register_module("fc2", torch::nn::Linear(16, 16));
        fc3 = register_module("fc3", torch::nn::Linear(16, 3)); 
    }

    // forward pass
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.view({x.size(0), -1}))); 
        // std::cout<<x;
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return torch::log_softmax(x, 1); 
    }

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

//read data function
std::pair<std::vector<float>, std::vector<int>> read_data(std::string data_src);

int main() {
    //read data 
    std::string iris_data_path = "iris.csv";
    auto data = read_data(iris_data_path);
    std::vector<float> features=data.first;
    std::vector<int> labels = data.second;

    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    torch::Tensor X = torch::from_blob(features.data(), {150, 4});
    torch::Tensor y = torch::tensor(labels, torch::kInt32);
    const float trainRatio = 0.8;
    const int dataSize = X.size(0);


    const int trainSize = static_cast<int>(trainRatio * dataSize);
    const int testSize = dataSize - trainSize;

    std::cout<<trainSize<<"\n";
    std::cout<<testSize<<"\n";

    std::vector<int> indices(dataSize);
    std::iota(indices.begin(), indices.end(), 0);  // Fill indices with 0, 1, ..., dataSize-1
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    std::vector<int> trainIndices(indices.begin(), indices.begin() + trainSize);
    std::vector<int> testIndices(indices.begin() + trainSize, indices.end());

    //split  dataset
    torch::Tensor X_train = X.index_select(0, torch::tensor(trainIndices));
    torch::Tensor X_test = X.index_select(0, torch::tensor(testIndices));
    torch::Tensor y_train = y.index_select(0, torch::tensor(trainIndices));
    torch::Tensor y_test = y.index_select(0, torch::tensor(testIndices));


    auto traindata=CustomDataset(X_train,y_train).map(torch::data::transforms::Stack<>());
    auto testdata=CustomDataset(X_test,y_test).map(torch::data::transforms::Stack<>());
    auto num_train_samples = traindata.size().value();
    auto num_test_samples = testdata.size().value();
    auto trainloader = torch::data::make_data_loader(std::move(traindata), torch::data::DataLoaderOptions().batch_size(12).workers(1));
    auto testloader  = torch::data::make_data_loader(std::move(testdata), torch::data::DataLoaderOptions().batch_size(12).workers(1));

    const int64_t input_size = 4;
    const int64_t num_classes = 3;
    const int64_t hidden_size = 16;
    const int64_t batch_size = 12;
    const size_t num_epochs = 200;
    const double learning_rate = 0.001;

    //model instance
    auto model = std::make_shared<Net>();
    model->to(device);

    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));
    torch::nn::CrossEntropyLoss criterion;

    // Training loop
    std::cout << "Training...\n";
    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        model->train();
        size_t batch_index = 0;
        for (auto& batch : *trainloader) {
            auto data = batch.data;
            auto target = batch.target;
            optimizer.zero_grad();
            auto output = model->forward(data);
            // std::cout<<output<<"\n";    
            auto loss = criterion(output, target);
             // Debugging output and target shapes and types
            // std::cout << "Output shape: " << output.sizes() << "\n";
            // std::cout << "Target shape: " << target.sizes() << "\n";
            // std::cout << "Output dtype: " << output.dtype() << "\n";
            // std::cout << "Target dtype: " << target.dtype() << "\n";
            // std::cout<<loss;
            auto predictions = output.argmax(1);
            // std::cout<<predictions;
            loss.backward();
            optimizer.step();
            if (++batch_index % 10 == 0) {
                std::cout << "Epoch [" << epoch << "/" << num_epochs << "], Batch [" << batch_index << "], Loss: " << loss.item<float>() << '\n';
            }
        }
    }   
    std::cout << "Training finished!\n";



    float train_loss = 0;
    int train_correct = 0;
    int train_total = 0;
    int rows=0;
        for (const auto& batch : *trainloader) {
            // std::cout<<++row<<"\n";
        auto data = batch.data;
        auto target = batch.target;

        auto output = model->forward(data);
        // std::cout<<data;
        train_loss += criterion(output, target).item<float>();
        auto predictions = output.argmax(1);
        train_correct += predictions.eq(target).sum().item<int>();
        train_total += target.size(0);
    }
    //calculating train accuracy 
    std::cout << "Train Loss: " << train_loss << std::endl;
    std::cout << "Train Accuracy: " << (static_cast<float>(train_correct) / train_total) * 100 << "%" << std::endl;
    
   std::cout << "Testing...\n";
    // Evaluation
    model->eval();
    float test_loss = 0;
    int correct = 0;
    int total = 0;
    for (const auto& batch : *testloader) {
        auto data = batch.data;
        auto target = batch.target;
        auto output = model->forward(data);
        test_loss += criterion(output, target).item<float>();
        auto predictions = output.argmax(1);
        correct += predictions.eq(target).sum().item<int>();
        total += target.size(0);
    }

    std::cout << "Testing finished!\n";
    //calulating test accuracy
    std::cout << "Test Loss: " << test_loss << std::endl;
    std::cout << "Accuracy: " << (static_cast<float>(correct) / total) * 100 << "%" << std::endl;

    // const std::string model_save_path = "model.pt";
    // // Save the model
    // torch::save(model, model_save_path);

return 0;


}

std::pair<std::vector<float>, std::vector<int>> read_data(std::string data_src) {
    std::ifstream data_stream(data_src);
    if (!data_stream.good()) {
        std::cerr << "Could not open file " << data_src << std::endl;
        return {};
    }

    std::vector<float> features;
    std::vector<int> labels;

    std::string line;
    bool firstLine = true;
    while (std::getline(data_stream, line)) {
        if (firstLine) {
            firstLine = false;
            continue;
        }

        std::vector<float> feature_row;
        std::string label_string;
        std::stringstream ss(line);
        std::string feature;

        int count = 0;
        while (std::getline(ss, feature, ',')) {
            if (count < 4) {
                features.push_back(std::stof(feature));
            } else {
                label_string = feature;
            }
            count++;
        }

        label_string.erase(std::remove(label_string.begin(), label_string.end(), '\"'), label_string.end());

        int label;
        if (label_string == "Setosa") {
            label = 0;
        } else if (label_string == "Versicolor") {
            label = 1;
        } else if (label_string == "Virginica") {
            label = 2;
        }

        // features.push_back(feature_row);
        labels.push_back(label);
    }

    return std::make_pair(features, labels);
}


