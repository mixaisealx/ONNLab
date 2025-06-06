#pragma once

namespace nn::interfaces {
	class BasicBackPropogableInterface {
	public:
		virtual void BackPropResetError() = 0;
		virtual void BackPropAccumulateError(float error, unsigned data_channel = 0) = 0;
		virtual float BackPropGetFinalError(unsigned data_channel = 0) = 0;
	};

	class MaccBackPropogableInterface : virtual public BasicBackPropogableInterface {
	public:
		virtual float RealAccumulatorValue(unsigned data_channel = 0) = 0;
		virtual float SurrogateAccumulatorValue(unsigned data_channel = 0) = 0;
	};

	class ZeroGradBackPropogableInterface : virtual public BasicBackPropogableInterface {
	public:
		virtual float HiddenActivationFunctionDerivative(float x, float backprop_error) const = 0;
		virtual float BackPropErrorFactor(float accumulator_value) = 0;
	};

	class BackPropMetaLayerMark { }; // A layer without neurons in standard meaning
}