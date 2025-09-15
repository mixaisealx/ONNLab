#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicWghOptI.h"
#include "OptimizerI.h"
#include "BatchNeuronBasicI.h"

#include <limits>
#include <stdexcept>

namespace nn
{
	template<interfaces::OptimizerInherit OptimizerT>
	class NNB_Connection : public interfaces::ConnectionBasicInterface, public interfaces::BasicWeightOptimizableInterface {
		interfaces::NeuronBasicInterface *from;
		interfaces::NeuronBasicInterface *to;
		OptimizerT *optimizer;
		float weight;
		OptimizerT::State optimizer_context;

		NNB_Connection(const NNB_Connection &) = delete;
		NNB_Connection &operator=(const NNB_Connection &) = delete;
	public:
		NNB_Connection(interfaces::NeuronBasicInterface *from, interfaces::NeuronBasicInterface *to, OptimizerT *optimizer, float initial_weight = 0.0f): from(from), to(to), optimizer(optimizer), weight(initial_weight) {
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
			optimizer->Reset(&optimizer_context);
		}

		~NNB_Connection() override {
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

		void Weight(float weight) override {
			this->weight = weight;
		}

		void WeightOptimReset() override {
			optimizer->Reset(&optimizer_context);
		}

		void WeightOptimDoUpdate(float gradient) override {
			weight -= optimizer->CalcDelta(gradient, &optimizer_context);
		}
	};
}
