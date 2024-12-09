#pragma once
#include "BasicLayerI.h"

#include <cstdlib>
#include <concepts>
#include <vector>
#include <span>
#include <functional>

namespace nn
{
	template<typename T>
	concept NeuronInherit = std::is_base_of<nn::interfaces::NBI, T>::value;

	template<NeuronInherit NeuronT>
	class NeuronHoldingStaticLayer : public interfaces::BasicLayerInterface {
		NeuronT *storage_base, *storage_end;
		std::vector<interfaces::NBI *> neuronsI;
		std::span<NeuronT> obj_view;
		bool hasTrainable;

	public:
		using NeuronEmplacer = void(NeuronT *const mem_ptr, unsigned index);

		NeuronHoldingStaticLayer(unsigned neurons_count, std::function<NeuronEmplacer> emplacer) {
			storage_base = reinterpret_cast<NeuronT *>(malloc(neurons_count * sizeof(NeuronT))); // Yes, I know about possible nullptr, BUT it is only lab code
			storage_end = storage_base + neurons_count;
			neuronsI.reserve(neurons_count);

			hasTrainable = false;
			NeuronT *mem_ptr = storage_base;
			for (unsigned idx = 0; idx != neurons_count; ++idx) {
				emplacer(mem_ptr, idx);

				neuronsI.emplace_back(dynamic_cast<nn::interfaces::NBI *>(mem_ptr));
				if (!hasTrainable && neuronsI.back()->IsTrainable()) {
					hasTrainable = true;
				}
				++mem_ptr;
			}
			obj_view = std::span<NeuronT>{ storage_base, neurons_count };
		};

		~NeuronHoldingStaticLayer() override {
			for (NeuronT *curr = storage_base; curr != storage_end; ++curr) {
				dynamic_cast<nn::interfaces::NBI *>(curr)->~NeuronBasicInterface();
			}
			free(storage_base);
		};

		const std::vector<interfaces::NeuronBasicInterface *> &Neurons() override {
			return neuronsI;
		}

		const std::span<NeuronT> &NeuronsInside() {
			return obj_view;
		}

		bool HasTrainable() override {
			return hasTrainable;
		}

		void AddNeuron(interfaces::NeuronBasicInterface *neuron) override {
			throw std::exception("AddNeuron not allowed in NeuronHoldingStaticLayer!");
		}
	};
}
