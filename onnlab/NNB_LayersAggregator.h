#pragma once
#include "BasicLayerI.h"

#include <vector>

namespace nn
{
	// A layer is a group of neurons in which the order of calling functions for updating the values of neurons does not matter.
	class NNB_LayersAggregator : public interfaces::BasicLayerInterface {
		std::vector<interfaces::NBI *> neurons;
		bool hasTrainable;

	public:
		NNB_LayersAggregator() {
			hasTrainable = false;
		};

		NNB_LayersAggregator(std::initializer_list<interfaces::BasicLayerInterface *> layers) {
			hasTrainable = false;
			unsigned count = 0;
			for (const auto layer : layers) {
				count += layer->Neurons().size();
				if (!hasTrainable && layer->HasTrainable()) {
					hasTrainable = true;
				}
			}
			neurons.reserve(count);
			for (const auto layer : layers) {
				for (const auto neuron : layer->Neurons()) {
					neurons.push_back(neuron);
				}
			}
		};

		~NNB_LayersAggregator() override { };

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

		void AddLayer(interfaces::BasicLayerInterface *layer) {
			neurons.reserve(neurons.size() + layer->Neurons().size());
			if (!hasTrainable && layer->HasTrainable()) {
				hasTrainable = true;
			}
			for (const auto neuron : layer->Neurons()) {
				neurons.push_back(neuron);
			}
		}
	};
}
