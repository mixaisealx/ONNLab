#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicWghOptI.h"

#include <limits>
#include <stdexcept>

namespace nn
{
	template<float weight>
	class NNB_StraightConnectionW : public interfaces::ConnectionBasicInterface, public interfaces::BasicWeightOptimizableInterface {
		interfaces::NeuronBasicInterface *from;
		interfaces::NeuronBasicInterface *to;

		NNB_StraightConnectionW(const NNB_StraightConnectionW &) = delete;
		NNB_StraightConnectionW &operator=(const NNB_StraightConnectionW &) = delete;
	public:
		NNB_StraightConnectionW(interfaces::NeuronBasicInterface *from, interfaces::NeuronBasicInterface *to): from(from), to(to) {
			interfaces::BatchNeuronBasicI *bfrom = dynamic_cast<interfaces::BatchNeuronBasicI *>(from);
			interfaces::BatchNeuronBasicI *bto = dynamic_cast<interfaces::BatchNeuronBasicI *>(to);
			unsigned fromb = (bfrom ? bfrom->GetCurrentBatchSize() : 1);
			unsigned tob = (bto ? bto->GetCurrentBatchSize() : 1);
			if (fromb != std::numeric_limits<unsigned>::max() &&
				tob != std::numeric_limits<unsigned>::max() &&
				fromb != tob) {
				throw std::runtime_error("Different batch sizes is not allowed!");
			}
			NBI_AddOutputConnection(from, this);
			NBI_AddInputConnection(to, this);
		}
		~NNB_StraightConnectionW() override {
			if (from && to) {
				NBI_RemoveOutputConnection(from, this);
				NBI_RemoveInputConnection(to, this);
				from = to = nullptr;
			}
		}

		interfaces::NeuronBasicInterface *From() override {
			return from;
		}

		interfaces::NeuronBasicInterface *To() override {
			return to;
		}

		float Weight() override {
			return weight;
		}

		void Weight(float) override {
			// Compatibility plug
			//throw std::runtime_error("Logic error! StraightConnection weight can not be set (it is always 1.0)!");
		}

		void WeightOptimReset() override {
			// Compatibility plug
		}
		void WeightOptimDoUpdate(float gradient) override {
			// Compatibility plug
		}
	};

	using NNB_StraightConnection = NNB_StraightConnectionW<1.0f>;
}
