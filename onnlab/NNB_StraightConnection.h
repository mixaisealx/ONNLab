#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicWghOptI.h"

namespace nn
{
	class NNB_StraightConnection : public interfaces::ConnectionBasicInterface, public interfaces::BasicWeightOptimizableInterface {
		interfaces::NeuronBasicInterface *from;
		interfaces::NeuronBasicInterface *to;

		NNB_StraightConnection(const NNB_StraightConnection &) = delete;
		NNB_StraightConnection &operator=(const NNB_StraightConnection &) = delete;
	public:
		NNB_StraightConnection(interfaces::NeuronBasicInterface *from, interfaces::NeuronBasicInterface *to): from(from), to(to) {
			NBI_AddOutputConnection(from, this);
			NBI_AddInputConnection(to, this);
		}
		~NNB_StraightConnection() override {
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
			return 1.0f;
		}

		void Weight(float) override {
			// Compatibility plug
			//throw std::exception("Logic error! StraightConnection weight can not be set (it is always 1.0)!");
		}

		void WeightOptimReset() override {
			// Compatibility plug
		}
		void WeightOptimDoUpdate(float delta) override {
			// Compatibility plug
		}
	};
}
