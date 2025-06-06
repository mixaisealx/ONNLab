#pragma once
#include "NNBasicsInterfaces.h"

namespace nn::interfaces
{
	class ErrorCalculatorI {
	public:
		ErrorCalculatorI() = default;
		virtual ~ErrorCalculatorI() = default;
		// Step 1
		virtual void ResetState() = 0;
		// Step 2 (called n times)
		virtual void ProcessNeuronError(float neuron_out, float perfect_out) = 0;
		// Step 3
		virtual void DoCalc() = 0;
		// Step 3.5 (calculete the loss if needed)
		virtual float CalcLoss() = 0;
		// Step 4 (called n times, expected to receive the results in the order of receipt in step 2)
		virtual float GetNeuronPartialError() = 0;
	};
}
