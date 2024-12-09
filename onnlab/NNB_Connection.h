#pragma once
#include "NNBasicsInterfaces.h"
#include "BasicWghOptI.h"

namespace nn
{
	class NNB_Connection : public interfaces::ConnectionBasicInterface, public interfaces::BasicWeightOptimizableInterface {
		interfaces::NeuronBasicInterface *from;
		interfaces::NeuronBasicInterface *to;
		float weight;
		float optimizer_learning_rate;

		NNB_Connection(const NNB_Connection &) = delete;
		NNB_Connection &operator=(const NNB_Connection &) = delete;
	public:
		NNB_Connection(interfaces::NeuronBasicInterface *from, interfaces::NeuronBasicInterface *to, float optimizer_learning_rate = 0.1f, float weight = 0.0f): from(from), to(to), weight(weight), optimizer_learning_rate(optimizer_learning_rate){
			NBI_AddOutputConnection(from, this);
			NBI_AddInputConnection(to, this);
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
			// Nothing to do
		}

		void WeightOptimDoUpdate(float delta) override {
			weight -= optimizer_learning_rate * delta * from->OwnLevel();
		}
	};
}
