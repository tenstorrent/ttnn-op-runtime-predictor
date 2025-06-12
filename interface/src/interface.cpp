//skeleton code for mlp side query interface
#include "../include/interface.hpp"

auto load_mlpack_model(const std::string& model_path, int input_size, std::vector<int> hidden_layers){

    //init model architecture
    FFN<MeanSquaredError, RandomInitialization> model;
    model.InputDimensions() = std::vector<size_t>({static_cast<unsigned long>(input_size)});

    int num_hidden_layers = hidden_layers.size();
    for(int i = 0; i < num_hidden_layers; i++){
        model.Add<Linear>(hidden_layers[i]);
        model.Add<ReLU>();
    }
    model.Add<Linear>(1);

    //load model from path
    data::Load(model_path, "model", model);
   return model;
}