#pragma once
#include "NNB_Connection.h"

namespace nn
{
	template<interfaces::OptimizerInherit OptimizerT>
	class NNB_Connection_spyable : public NNB_Connection<OptimizerT> {
		NNB_Connection_spyable(const NNB_Connection_spyable &) = delete;
		NNB_Connection_spyable &operator=(const NNB_Connection_spyable &) = delete;

	public:
		NNB_Connection_spyable(interfaces::NeuronBasicInterface *from, interfaces::NeuronBasicInterface *to, float optimizer_learning_rate = 0.1f, float weight = 0.0f):
			NNB_Connection(from, to, optimizer_learning_rate, weight) {
		}

		float last_gradient = 0;

		void WeightOptimDoUpdate(float gradient) override {
			NNB_Connection::WeightOptimDoUpdate(gradient);
			last_gradient = gradient;
		}
	};
}
