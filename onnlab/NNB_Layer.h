#pragma once
#include "BasicLayerI.h"

#include <vector>

namespace nn
{
	// A layer is a group of neurons in which the order of calling functions for updating the values of neurons does not matter.
	class NNB_Layer : public interfaces::BasicLayerInterface {
		std::vector<interfaces::NeuronBasicInterface *> neurons;
		bool hasTrainable;

	public:
		NNB_Layer() {
			hasTrainable = false;
		};

		NNB_Layer(std::initializer_list<interfaces::NeuronBasicInterface *> neurons):neurons(neurons) {
			hasTrainable = false;
			for (const auto item : neurons) {
				if (item->IsTrainable()) {
					hasTrainable = true;
					break;
				}
			}
		};

		~NNB_Layer() override { };

		const std::vector<interfaces::NeuronBasicInterface *> &Neurons() override {
			return neurons;
		}

		bool HasTrainable() override {
			return hasTrainable;
		}

		void AddNeuron(interfaces::NeuronBasicInterface *neuron) override {
			neurons.push_back(neuron);
			if (!hasTrainable && neuron->IsTrainable()) {
				hasTrainable = true;
			}
		}
	};
}
