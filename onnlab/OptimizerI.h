#pragma once
#include "NNBasicsInterfaces.h"

namespace nn::interfaces
{
	class OptimizerI {
	public:
		virtual void Reset(void* state) = 0;
		virtual float CalcDelta(float gradient, void *state) = 0;

		virtual float GetLearningRate() = 0;
		virtual void SetLearningRate(float learning_rate) = 0;
	};

	template<typename T>
	concept OptimizerInherit = std::is_base_of<OptimizerI, T>::value;
}
