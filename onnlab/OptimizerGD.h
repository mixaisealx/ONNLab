#pragma once
#include "OptimizerI.h"

namespace nn::optimizers
{
	class GradientDescendent : public nn::interfaces::OptimizerI {
		float learning_rate;
	public:
		struct State { }; // Empty state

		GradientDescendent(float learning_rate = 0.1f):learning_rate(learning_rate) { }

		void Reset(void *state) override { }

		float CalcDelta(float gradient, void *state = nullptr) override {
			return learning_rate * gradient;
		}

		float GetLearningRate() override {
			return learning_rate;
		}

		void SetLearningRate(float lr = 0.1f) override {
			learning_rate = lr;
		}
	};
}
