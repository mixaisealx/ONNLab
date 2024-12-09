#pragma once
#include "NNB_Connection.h"

namespace nn
{
	class NNB_Connection_spyable : public NNB_Connection {
		NNB_Connection_spyable(const NNB_Connection_spyable &) = delete;
		NNB_Connection_spyable &operator=(const NNB_Connection_spyable &) = delete;

	public:
		NNB_Connection_spyable(interfaces::NeuronBasicInterface *from, interfaces::NeuronBasicInterface *to, float optimizer_learning_rate = 0.1f, float weight = 0.0f):
			NNB_Connection(from, to, optimizer_learning_rate, weight) {
		}

		float last_delta = 0;

		void WeightOptimDoUpdate(float delta) override {
			NNB_Connection::WeightOptimDoUpdate(delta);
			last_delta = delta;
		}
	};
}
