#pragma once

namespace nn::interfaces
{
	class LimitedNeuronI {
	public:
		virtual float UpperLimitValue() const = 0;
		virtual float LowerLimitValue() const = 0;
	};
}