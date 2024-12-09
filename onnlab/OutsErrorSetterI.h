#pragma once
#include <vector>

#include "NNBasicsInterfaces.h"

namespace nn::interfaces
{
	class OutsErrorSetterI {
	public:
		class ErrorCalculatorI {
		public:
			ErrorCalculatorI() = default;
			virtual ~ErrorCalculatorI() = default;
			// Step 1
			virtual void InitState(unsigned vector_size) = 0;
			// Step 2 (called n times)
			virtual void ProcessNeuronError(float neuron_out, float perfect_out) = 0;
			// Step 3
			virtual void DoCalc() = 0;
			// Step 4 (called n times, expected to receive the results in the order of receipt in step 2)
			virtual float GetNeuronPartialError() = 0;
			// Like step 1, but can't be called before 1.
			virtual void ResetState() = 0;
		};
		OutsErrorSetterI() = default;
		virtual ~OutsErrorSetterI() = default;

		virtual void FillupError(std::vector<interfaces::NeuronBasicInterface *> &outputs, const std::vector<float> &perfect_result) = 0;
	};
}
