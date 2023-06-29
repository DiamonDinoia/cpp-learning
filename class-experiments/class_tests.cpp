#include <iostream>
#include <map>
#include <string>
#include <typeindex>
#include <vector>

struct Layer {
    enum layer_t {LAYER, LINEAR} ;
    virtual std::string type() { return "Layer"; }

};

struct Linear : public Layer {
    std::string type() override { return "Linear"; }
};

std::ostream& operator<<(std::ostream& out, Layer::layer_t value);

void pointer_example(){
    std::vector<Layer*> layers;
    layers.emplace_back(new Layer());
    layers.emplace_back(new Linear());
    for (auto& layer : layers) {
        std::cout << layer->type() << std::endl;
    }
}

void value_example () {
    std::vector<Layer> layers;
    layers.emplace_back();
    layers.emplace_back(Linear());
    for (auto& layer : layers) {
        std::cout << layer.type() << std::endl;
        const char *typemsg;
        std::type_index ti(typeid(layer));
        if (ti == std::type_index(typeid(Layer))) {
            typemsg = "type Layer";
        } else if (ti == std::type_index(typeid(Linear))) {
            typemsg = "type Linear";
        } else {
            typemsg = "unknown";
        }
        std::cout << typemsg << std::endl;
    }

}

int main(const int argc, const char* const* argv) {
    pointer_example();
    value_example();
    return 0;
}

std::ostream& operator<<(std::ostream& out, const Layer::layer_t value){
    static std::map<Layer::layer_t, std::string> strings;
    if (strings.size() == 0){
#define INSERT_ELEMENT(p) strings[p] = #p
            INSERT_ELEMENT(Layer::layer_t::LAYER);
            INSERT_ELEMENT( Layer::layer_t::LINEAR);
#undef INSERT_ELEMENT
    }

    return out << strings[value];
}
// enum print function
